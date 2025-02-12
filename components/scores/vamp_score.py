import torch
from deeptime.decomposition.deep import *
from deeptime.util.torch import multi_dot

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
    # Add diagnostic logging
    diagnostic_info = {
        'data_stats': {
            'min': data.min().item(),
            'max': data.max().item(),
            'mean': data.mean().item(),
            'has_nan': torch.isnan(data).any().item(),
            'has_inf': torch.isinf(data).any().item()
        },
        'data_lagged_stats': {
            'min': data_lagged.min().item(),
            'max': data_lagged.max().item(),
            'mean': data_lagged.mean().item(),
            'has_nan': torch.isnan(data_lagged).any().item(),
            'has_inf': torch.isinf(data_lagged).any().item()
        }
    }

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
        if method in ['VAMP1', 'VAMP2']:
            koopman = koopman_matrix(data, data_lagged, epsilon=epsilon, mode=mode)
            diagnostic_info['koopman_stats'] = {
                'min': koopman.min().item(),
                'max': koopman.max().item(),
                'mean': koopman.mean().item(),
                'has_nan': torch.isnan(koopman).any().item(),
                'has_inf': torch.isinf(koopman).any().item()
            }

            score = torch.norm(koopman, p='nuc') if method == 'VAMP1' else torch.pow(torch.norm(koopman, p='fro'), 2)

        elif method == 'VAMPE':
            # Compute Covariance
            c00, c0t, ctt = covariances(data, data_lagged, remove_mean=True)
            diagnostic_info['covariance_stats'] = {
                'c00': {'min': c00.min().item(), 'max': c00.max().item()},
                'c0t': {'min': c0t.min().item(), 'max': c0t.max().item()},
                'ctt': {'min': ctt.min().item(), 'max': ctt.max().item()}
            }

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
            score = -1.0 * score
            return score

        else:
            raise ValueError(f"Method {method} not implemented")

        final_score = 1 + score

        if torch.isinf(final_score):
            raise ValueError(f"Infinite score detected. Diagnostic information:\n{diagnostic_info}")

        return final_score

    except Exception as e:
        raise RuntimeError(f"Error in VAMP score computation: {str(e)}\nDiagnostic information:\n{diagnostic_info}")
