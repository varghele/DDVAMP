import torch
from typing import Optional, Tuple, Union
from args import buildParser


def train(
        train_loader: torch.utils.data.DataLoader,
        n_epochs: int,
        validation_loader: Optional[torch.utils.data.DataLoader] = None,
        loader_train_all: Optional[torch.utils.data.DataLoader] = None
) -> Tuple[VAMPNet, int]:
    """
    Train the VAMPNet model with optional VAMPCE pre-training.

    Args:
        train_loader: DataLoader for training data
        n_epochs: Number of epochs for training
        validation_loader: Optional validation data loader
        loader_train_all: Optional full training set loader

    Returns:
        Tuple of (trained model, total training epochs)
    """
    n_OOM = 0
    pre_epoch = args.pre_train_epoch

    # VAMPCE pre-training phase
    if args.score_method == 'VAMPCE':
        # Train vanilla VAMPNet
        print("Training vanilla VAMPNet model...")
        vampnet.vampu.requires_grad_(False)
        vampnet.vamps.requires_grad_(False)
        vampnet.score_method = 'VAMP2'

        early_stopping = EarlyStopping(args.save_folder, file_name='best_pre_lobe', delta=1e-4, patience=100)
        best_dict = _train_vamp2_phase(vampnet, train_loader, validation_loader, pre_epoch, early_stopping)

        # Train auxiliary network
        print("Training auxiliary network...")
        vampnet.score_method = 'VAMPCE'
        vampnet.lobe.requires_grad_(False)
        vampnet.lobe_timelagged.requires_grad_(False)

        # Calculate state probabilities
        state_probs, state_probs_tau = _calculate_state_probabilities(vampnet, train_loader, device)
        vampnet.update_auxiliary_weights([state_probs, state_probs_tau], optimize_S=True)

        # Train U and S networks
        early_stopping = EarlyStopping(args.save_folder, file_name='best_pre', delta=1e-4, patience=100)
        best_dict = _train_auxiliary_networks(vampnet, state_probs, state_probs_tau,
                                              validation_loader, pre_epoch, early_stopping)

    # Train full network
    print("Training complete network...")
    all_train_epo = 0
    early_stopping = EarlyStopping(args.save_folder, file_name='best_allnet', delta=1e-4, patience=200)

    for epoch in tqdm(range(n_epochs)):
        if epoch == 100:
            vampnet.set_optimizer_lr(0.2)

        try:
            all_train_epo = _train_epoch(vampnet, train_loader, validation_loader,
                                         early_stopping, epoch, all_train_epo)

            if early_stopping.early_stop:
                print("Early stopping all network train")
                break

        except RuntimeError as e:
            print(f"Epoch {epoch}: Runtime error!")
            print(e, e.args[0])
            n_OOM += 1

        # Save checkpoints
        if args.save_checkpoints and epoch % 50 == 49:
            _save_checkpoint(vampnet, epoch)

    return vampnet.fetch_model(), all_train_epo


def _train_epoch(model, train_loader, validation_loader, early_stopping, epoch, all_train_epo):
    """Handle single training epoch."""
    now_train_num = 0
    for batch_0, batch_t in train_loader:
        torch.cuda.empty_cache()
        model.partial_fit((batch_0.to(device), batch_t.to(device)))
        all_train_epo += 1
        now_train_num += 1

    if validation_loader is not None:
        mean_score = _validate_model(model, validation_loader)
        _update_early_stopping(model, mean_score, early_stopping, epoch)

    return all_train_epo


def _calculate_state_probabilities(model, train_loader, device):
    """Calculate state probabilities for all training data."""
    data_size = sum(batch_0.shape[0] for batch_0, _ in train_loader)
    state_probs = np.zeros((data_size, int(args.num_classes)))
    state_probs_tau = np.zeros((data_size, int(args.num_classes)))

    n_iter = 0
    with torch.no_grad():
        for batch_0, batch_t in train_loader:
            torch.cuda.empty_cache()
            batch_size = len(batch_0)
            state_probs[n_iter:n_iter + batch_size] = model.transform(batch_0)
            state_probs_tau[n_iter:n_iter + batch_size] = model.transform(batch_t, instantaneous=False)
            n_iter += batch_size

    return state_probs, state_probs_tau

