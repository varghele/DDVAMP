import torch
import torch.nn as nn
from typing import Optional, Tuple
from train.utils.EarlyStopping import EarlyStopping
from train.utils.record_result import record_result
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from deeptime.decomposition.deep import VAMPNet
from components.models.RevVAMPNet import RevVAMPNet


def train_vamp(
        train_loader: torch.utils.data.DataLoader,
        n_epochs: int,
        args,
        #vampnet: Optional[VAMPNet] = None,
        vampnet, # Fix to pass correct net
        lobe: Optional[nn.Module] = None,
        vlu: Optional[nn.Module] = None,
        vls: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        validation_loader: Optional[torch.utils.data.DataLoader] = None,
        loader_train_all: Optional[torch.utils.data.DataLoader] = None,
        log_pth: str = None
) -> Tuple[RevVAMPNet, int]:
    """Train the VAMPNet model with optional VAMPCE pre-training.

    Args:
        train_loader: DataLoader for training data
        n_epochs: Number of epochs for training
        vampnet: VAMPNet model instance
        lobe: GraphVAMPNet model instance
        vlu: VAMP-U model instance
        vls: VAMP-S model instance
        device : Optional[torch.device] Device to run computations on
        validation_loader: Optional validation data loader
        loader_train_all: Optional full training set loader
        args: arguments passed from main training function
        log_pth: Path where results are stored

    Returns:
        Tuple of (trained model, total training epochs)
    """

    n_OOM = 0
    pre_epoch = args.pre_train_epoch

    def validate_model(model, val_loader):
        """Run validation and return mean score."""
        with torch.no_grad():
            scores = [model.validate((batch[0].to(device), batch[1].to(device)))
                     for batch in val_loader]
            return torch.mean(torch.stack(scores))

    def save_checkpoint(epoch, model_state):
        """Save model checkpoint."""
        if args.save_checkpoints and epoch % 50 == 49:
            torch.save(model_state, f"{args.save_folder}/logs_{epoch}.pt")

    # VAMPCE pre-training phase
    #----------------------------------------------------------------------------
    if args.score_method == 'VAMPCE':
        print("Training vanilla VAMPNet model...")

        # Disable auxiliary networks during pre-training
        vampnet.vampu.requires_grad_(False)
        vampnet.vamps.requires_grad_(False)
        vampnet.score_method = 'VAMP2'

        # Initialize early stopping for pre-training
        early_stopping = EarlyStopping(
            save_path=args.save_folder,
            file_name='best_pre_lobe',
            delta=1e-4,
            patience=100
        )
        best_dict = None

        # Pre-training loop
        for epoch in tqdm(range(pre_epoch)):
            try:
                # Training step
                for batch_0, batch_t in train_loader:
                    torch.cuda.empty_cache()
                    vampnet.partial_fit((batch_0.to(device), batch_t.to(device)))

                # Validation step
                if validation_loader is not None:
                    # Calculate validation score
                    mean_score = validate_model(vampnet, validation_loader)
                    early_stopping(mean_score.item(), {'state_dict': vampnet.lobe.state_dict()})

                    if early_stopping.is_best:
                        best_dict = vampnet.lobe.state_dict()

                    if early_stopping.early_stop:
                        print("Early stopping pre-training")
                        break

                    # Log progress
                    if epoch % 10 == 9:
                        record_result(
                            f"Pre-training step: {epoch}, validation score: {mean_score.item():.4f}, "
                            f"best score: {early_stopping.val_loss_min:.4f}",
                            log_pth
                        )

            except RuntimeError as e:
                print(f"Epoch {epoch}: Runtime error!")
                print(e, e.args[0])
                n_OOM += 1

        # Re-enable auxiliary networks and load best model
        vampnet.vampu.requires_grad_(True)
        vampnet.vamps.requires_grad_(True)
        vampnet.score_method = 'VAMPCE'

        if best_dict:
            vampnet.lobe.load_state_dict(best_dict)
            vampnet.lobe_timelagged.load_state_dict(deepcopy(best_dict))
        print(f"VAMP2 best score: {early_stopping.val_loss_min:.4f}")

    # Train auxiliary networks
    #----------------------------------------------------------------------------
    print("Training auxiliary networks...")
    early_stopping = EarlyStopping(
        args.save_folder,
        file_name='best_pre',
        delta=1e-4,
        patience=100
    )
    best_dict = None

    # Freeze main network weights
    vampnet.lobe.requires_grad_(False)
    vampnet.lobe_timelagged.requires_grad_(False)

    # Calculate state probabilities for all data
    def calculate_state_probabilities(train_loader, num_classes):
        """Calculate state probabilities for all data points."""
        data_size = sum(batch[0].shape[0] for batch, _ in train_loader)
        state_probs = np.zeros((data_size, num_classes))
        state_probs_tau = np.zeros((data_size, num_classes))

        n_iter = 0
        with torch.no_grad():
            for batch_0, batch_t in train_loader:
                torch.cuda.empty_cache()
                batch_size = len(batch_0)
                state_probs[n_iter:n_iter + batch_size] = vampnet.transform(batch_0)
                state_probs_tau[n_iter:n_iter + batch_size] = vampnet.transform(batch_t, instantaneous=False)
                n_iter += batch_size

        return state_probs, state_probs_tau

    # Calculate initial state probabilities
    state_probs, state_probs_tau = calculate_state_probabilities(
        train_loader,
        int(args.num_classes)
    )
    vampnet.update_auxiliary_weights([state_probs, state_probs_tau], optimize_S=True)

    # Train auxiliary networks
    for epoch in tqdm(range(pre_epoch)):
        try:
            # Training step
            torch.cuda.empty_cache()
            vampnet.train_US([state_probs, state_probs_tau])

            # Validation step
            if validation_loader is not None:
                mean_score = validate_model(vampnet, validation_loader)

                # Update early stopping
                model_state = {
                    'epoch': pre_epoch,
                    'state_dict': lobe.state_dict(),
                    'vlu_dict': vlu.state_dict(),
                    'vls_dict': vls.state_dict(),
                }
                early_stopping(mean_score.item(), model_state)

                # Log progress
                if epoch % 10 == 9:
                    record_result(
                        f"Auxiliary training epoch: {epoch}, "
                        f"validation score: {mean_score.item():.4f}",
                        log_pth
                    )

                # Save best model
                if early_stopping.is_best:
                    best_dict = {
                        'vlu_dict': vlu.state_dict(),
                        'vls_dict': vls.state_dict()
                    }

                # Check early stopping
                if early_stopping.early_stop:
                    print("Early stopping triggered for auxiliary networks")
                    break

        except RuntimeError as e:
            print(f"Runtime error at epoch {epoch}:")
            print(e, e.args[0])
            n_OOM += 1

    # Clean up
    del state_probs
    del state_probs_tau

    # Print results and restore model state
    print(f"VAMPCE score: {early_stopping.val_loss_min:.4f}")

    # Re-enable main network training
    vampnet.lobe.requires_grad_(True)
    vampnet.lobe_timelagged.requires_grad_(True)

    # Load best auxiliary network weights if available
    if best_dict:
        vampnet.vampu.load_state_dict(best_dict['vlu_dict'])
        vampnet.vamps.load_state_dict(best_dict['vls_dict'])

    # Log final results
    record_result(
        f"Pre-training complete - epochs: {pre_epoch}, "
        f"final score: {early_stopping.val_loss_min:.4f}",
        log_pth
    )

    # Update learning rate for next phase
    for param_group in vampnet.optimizer.param_groups:
        param_group['lr'] = 0.2

    # Train full network
    # ----------------------------------------------------------------------------
    print("Training full network...")
    all_train_epo = 0
    best_model = None
    early_stopping = EarlyStopping(
        args.save_folder,
        file_name='best_allnet',
        delta=1e-4,
        patience=200
    )

    def save_checkpoint(epoch, model_state):
        """Save model checkpoint."""
        if args.save_checkpoints and epoch % 50 == 9:
            torch.save(model_state, f"{args.save_folder}/logs_{epoch}.pt")

    for epoch in tqdm(range(n_epochs)):
        # Adjust learning rate at epoch 100
        if epoch == 100:
            for param_group in vampnet.optimizer.param_groups:
                param_group['lr'] = 0.2

        try:
            # Training step
            batch_count = 0
            for batch_0, batch_t in train_loader:
                torch.cuda.empty_cache()
                vampnet.partial_fit((batch_0.to(device), batch_t.to(device)))
                all_train_epo += 1
                batch_count += 1

            # Validation step
            if validation_loader is not None:
                # Calculate validation score
                mean_score = validate_model(vampnet, validation_loader)
                vampnet._validation_scores.append((vampnet._step, mean_score.item()))

                # Update early stopping
                model_state = {
                    'epoch': pre_epoch,
                    'state_dict': lobe.state_dict(),
                    'vlu_dict': vlu.state_dict(),
                    'vls_dict': vls.state_dict(),
                }
                early_stopping(mean_score.item(), model_state)

                # Save best model if improved
                if early_stopping.is_best:
                    best_model = {
                        'epoch': pre_epoch,
                        'lobe': lobe.state_dict(),
                        'vlu': vlu.state_dict(),
                        'vls': vls.state_dict()
                    }

                # Check early stopping condition
                if early_stopping.early_stop:
                    print("Early stopping triggered for full network training")
                    break

                # Save checkpoints and log progress
                if epoch % 10 == 9:
                    save_checkpoint(epoch, model_state)

                    # Calculate and log training metrics
                    train_mean = np.mean(vampnet.train_scores[-batch_count - 1:-1][0, 1])
                    record_result(
                        f"Epoch: {epoch}, "
                        f"validation score: {mean_score.item():.4f}, "
                        f"training mean: {train_mean:.4f}, "
                        f"best score: {early_stopping.val_loss_min:.4f}",
                        log_pth
                    )

        except RuntimeError as e:
            print(f"Runtime error at epoch {epoch}:")
            print(e, e.args[0])
            n_OOM += 1

    # Load best model if available
    if best_model:
        print(f"Loading best model (score: {early_stopping.val_loss_min:.4f})")
        vampnet.lobe.load_state_dict(best_model['lobe'])
        vampnet.lobe_timelagged.load_state_dict(deepcopy(best_model['lobe']))
        vampnet.vampu.load_state_dict(best_model['vlu'])
        vampnet.vamps.load_state_dict(best_model['vls'])

    return vampnet.fetch_model(), all_train_epo


