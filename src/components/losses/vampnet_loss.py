import torch
from src.components.scores.vamp_score import vamp_score


def vampnet_loss(data: torch.Tensor,
                 data_lagged: torch.Tensor,
                 method: str = 'VAMP2',
                 epsilon: float = 1e-6,
                 mode: str = 'trunc') -> torch.Tensor:
    """
    Compute the VAMPNet loss function as the negative VAMP score.

    Parameters
    ----------
    data : torch.Tensor
        Instantaneous data batch
    data_lagged : torch.Tensor
        Time-lagged data batch
    method : str, default='VAMP2'
        VAMP scoring method to use. Options: ['VAMP1', 'VAMP2', 'VAMPE']
    epsilon : float, default=1e-6
        Small constant for numerical stability
    mode : str, default='trunc'
        Mode for eigenvalue truncation. Options: ['trunc', 'regularize']

    Returns
    -------
    torch.Tensor
        Negative VAMP score (loss value)

    Notes
    -----
    The loss is computed as the negative VAMP score to convert the maximization
    problem into a minimization problem suitable for gradient descent.

    References
    ----------
    .. [1] Wu, H., Mardt, A., Pasquali, L., & Noe, F. (2018). Deep generative
           Markov state models. NeurIPS.
    """
    # Validate inputs
    if not isinstance(data, torch.Tensor) or not isinstance(data_lagged, torch.Tensor):
        raise TypeError("Data inputs must be torch.Tensor objects")

    if data.shape != data_lagged.shape:
        raise ValueError("Data and time-lagged data must have the same shape")

    if method not in ['VAMP1', 'VAMP2', 'VAMPE', 'VAMPCE']:
        raise ValueError(f"Unknown scoring method: {method}")

    if mode not in ['trunc', 'regularize']:
        raise ValueError(f"Unknown mode: {mode}")

    try:
        # Compute VAMP score
        score = vamp_score(data, data_lagged,
                           method=method,
                           epsilon=epsilon,
                           mode=mode)

        # Return negative score for minimization
        return -1.0 * score

    except RuntimeError as e:
        raise RuntimeError(f"Error computing VAMP score: {str(e)}")