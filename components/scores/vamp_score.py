import torch
from deeptime.decomposition.deep import *
from deeptime.util.torch import disable_TF32, multi_dot

def vamp_score(data: torch.Tensor,
               data_lagged: torch.Tensor,
               method: str = 'VAMP2',
               epsilon: float = 1e-6,
               mode: str = 'trunc') -> torch.Tensor:
    """
    Compute the VAMP score based on data and corresponding time-shifted data.

    Parameters
    ----------
    data : torch.Tensor
        (N, d)-dimensional torch tensor containing instantaneous data
    data_lagged : torch.Tensor
        (N, d)-dimensional torch tensor containing time-lagged data
    method : str, default='VAMP2'
        VAMP scoring method. Options: ['VAMP1', 'VAMP2', 'VAMPE', 'VAMPCE']
    epsilon : float, default=1e-6
        Cutoff parameter for small eigenvalues or regularization parameter
    mode : str, default='trunc'
        Mode for handling small eigenvalues. Options: ['trunc', 'regularize']

    Returns
    -------
    torch.Tensor
        Computed VAMP score. Includes +1 contribution from constant singular function.

    Notes
    -----
    The score computation includes different methods:
    - VAMP1: Nuclear norm of Koopman matrix
    - VAMP2: Squared Frobenius norm of Koopman matrix
    - VAMPE: Eigendecomposition-based score
    - VAMPCE: Custom scoring for VAMP with cross-entropy
    """
    # Validate inputs
    valid_methods = ['VAMP1', 'VAMP2', 'VAMPE', 'VAMPCE']
    valid_modes = ['trunc', 'regularize']

    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}', supported are {valid_methods}")

    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}', supported are {valid_modes}")

    if not torch.is_tensor(data) or not torch.is_tensor(data_lagged):
        raise TypeError("Data inputs must be torch.Tensor objects")

    if data.shape != data_lagged.shape:
        raise ValueError(f"Data shapes must match but were {data.shape} and {data_lagged.shape}")

    try:
        # Compute score based on method
        if method == 'VAMP1':
            koopman = koopman_matrix(data, data_lagged, epsilon=epsilon, mode=mode)
            score = torch.norm(koopman, p='nuc')

        elif method == 'VAMP2':
            koopman = koopman_matrix(data, data_lagged, epsilon=epsilon, mode=mode)
            score = torch.pow(torch.norm(koopman, p='fro'), 2)

        elif method == 'VAMPE':
            # Compute covariances
            c00, c0t, ctt = covariances(data, data_lagged, remove_mean=True)

            # Compute inverse square roots
            c00_sqrt_inv = sym_inverse(c00, epsilon=epsilon, return_sqrt=True, mode=mode)
            ctt_sqrt_inv = sym_inverse(ctt, epsilon=epsilon, return_sqrt=True, mode=mode)

            # Compute Koopman operator
            koopman = multi_dot([c00_sqrt_inv, c0t, ctt_sqrt_inv]).t()

            # SVD decomposition
            u, s, v = torch.svd(koopman)
            mask = s > epsilon

            # Apply mask and compute transformed matrices
            u = torch.mm(c00_sqrt_inv, u[:, mask])
            v = torch.mm(ctt_sqrt_inv, v[:, mask])
            s = s[mask]

            # Compute score
            u_t = u.t()
            v_t = v.t()
            s = torch.diag(s)
            score = torch.trace(
                2. * multi_dot([s, u_t, c0t, v]) -
                multi_dot([s, u_t, c00, u, s, v_t, ctt, v])
            )

        elif method == 'VAMPCE':
            score = torch.trace(data[0])

        else:
            raise ValueError(f"Method {method} not implemented")

        return 1 + score

    except RuntimeError as e:
        raise RuntimeError(f"Error computing VAMP score: {str(e)}")
