import torch
import torch.nn as nn
from typing import Optional
from train.utils.EarlyStopping import EarlyStopping
from train.utils.record_result import record_result
from tqdm import tqdm
import numpy as np


def train_US(
        train_loader: torch.utils.data.DataLoader,
        n_epochs: int,
        args,
        vampnet: Optional[nn.Module] = None,
        lobe: Optional[nn.Module] = None,
        vlu: Optional[nn.Module] = None,
        vls: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        validation_loader=None,
        loader_train_all=None,
        log_pth: str = None,
        model_name="best_us"
):
    """Train the auxiliary networks (U and S) of the VAMPNet model.

    Args:
        train_loader: DataLoader for training data
        n_epochs: Number of epochs for training
        vampnet: VAMPNet model instance
        lobe: GraphVAMPNet model instance
        vlu: VAMP-U model instance
        vls: VAMP-S model instance
        validation_loader: Optional validation data loader
        loader_train_all: Optional full training set loader
        model_name: Name for saving the model checkpoints
    """
    n_OOM = 0

    def validate_model(model, val_loader):
        """Run validation and return mean score."""
        with torch.no_grad():
            scores = [model.validate((batch[0].to(device), batch[1].to(device)))
                     for batch in val_loader]
            return torch.mean(torch.stack(scores))

    print("Training auxiliary networks...")
    early_stopping = EarlyStopping(
        args.save_folder,
        file_name=model_name,
        delta=1e-4,
        patience=100
    )

    # Freeze main network weights
    vampnet.lobe.requires_grad_(False)
    vampnet.lobe_timelagged.requires_grad_(False)

    # Calculate state probabilities
    def calculate_state_probabilities():
        data_size = sum(batch[0].shape[0] for batch, _ in train_loader)
        state_probs = np.zeros((data_size, int(args.num_classes)))
        state_probs_tau = np.zeros((data_size, int(args.num_classes)))

        n_iter = 0
        with torch.no_grad():
            for batch_0, batch_t in train_loader:
                torch.cuda.empty_cache()
                batch_size = len(batch_0)
                state_probs[n_iter:n_iter + batch_size] = vampnet.transform(batch_0)
                state_probs_tau[n_iter:n_iter + batch_size] = vampnet.transform(
                    batch_t, instantaneous=False
                )
                n_iter += batch_size

        return state_probs, state_probs_tau

    # Initialize state probabilities
    state_probs, state_probs_tau = calculate_state_probabilities()
    vampnet.update_auxiliary_weights([state_probs, state_probs_tau], optimize_S=True)

    # Train auxiliary networks
    best_dict = None
    for epoch in tqdm(n_epochs):
        try:
            # Training step
            torch.cuda.empty_cache()
            vampnet.train_US([state_probs, state_probs_tau])

            # Validation step
            if validation_loader is not None:
                mean_score = validate_model(vampnet, validation_loader)

                # Update early stopping
                model_state = {
                    'epoch': n_epochs,
                    'state_dict': lobe.state_dict(),
                    'vlu_dict': vlu.state_dict(),
                    'vls_dict': vls.state_dict(),
                }
                early_stopping(mean_score.item(), model_state)

                # Log progress
                if epoch % 50 == 9:
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

    # Print results
    print(f"Final VAMPCE score: {early_stopping.val_loss_min:.4f}")

    # Re-enable main network training
    vampnet.lobe.requires_grad_(True)
    vampnet.lobe_timelagged.requires_grad_(True)

    # Load best model if available
    if best_dict:
        vampnet.vampu.load_state_dict(best_dict['vlu_dict'])
        vampnet.vamps.load_state_dict(best_dict['vls_dict'])

    # Log final results
    record_result(
        f"Auxiliary training complete - epochs: {n_epochs}, "
        f"final score: {early_stopping.val_loss_min:.4f}",
        log_pth
    )

    # Update learning rate for next phase
    vampnet.set_optimizer_lr(0.2)
