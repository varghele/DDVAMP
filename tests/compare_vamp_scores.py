import torch
import numpy as np
from typing import Tuple
import pytest

from components.scores.vamp_score import vamp_score as vamp_score_new
from deeptime.decomposition.deep import *
from deeptime.util.torch import disable_TF32, multi_dot


valid_score_methods = ('VAMP1', 'VAMP2', 'VAMPE', 'VAMPCE')


def vamp_score_old(data: torch.Tensor, data_lagged: torch.Tensor, method='VAMP2', epsilon: float = 1e-6, mode='trunc'):
    r"""Computes the VAMP score based on data and corresponding time-shifted data.

    Parameters
    ----------
    data : torch.Tensor
        (N, d)-dimensional torch tensor
    data_lagged : torch.Tensor
        (N, k)-dimensional torch tensor
    method : str, default='VAMP2'
        The scoring method. See :meth:`score <deeptime.decomposition.CovarianceKoopmanModel.score>` for details.
    epsilon : float, default=1e-6
        Cutoff parameter for small eigenvalues, alternatively regularization parameter.
    mode : str, default='trunc'
        Regularization mode for Hermetian inverse. See :meth:`sym_inverse`.

    Returns
    -------
    score : torch.Tensor
        The score. It contains a contribution of :math:`+1` for the constant singular function since the
        internally estimated Koopman operator is defined on a decorrelated basis set.
    """
    assert method in valid_score_methods, f"Invalid method '{method}', supported are {valid_score_methods}"
    assert data.shape == data_lagged.shape, f"Data and data_lagged must be of same shape but were {data.shape} " \
                                            f"and {data_lagged.shape}."
    out = None
    if method == 'VAMP1':
        koopman = koopman_matrix(data, data_lagged, epsilon=epsilon, mode=mode)
        out = torch.norm(koopman, p='nuc')
    elif method == 'VAMP2':
        koopman = koopman_matrix(data, data_lagged, epsilon=epsilon, mode=mode)
        out = torch.pow(torch.norm(koopman, p='fro'), 2)
    elif method == 'VAMPE':
        c00, c0t, ctt = covariances(data, data_lagged, remove_mean=True)
        c00_sqrt_inv = sym_inverse(c00, epsilon=epsilon, return_sqrt=True, mode=mode)
        ctt_sqrt_inv = sym_inverse(ctt, epsilon=epsilon, return_sqrt=True, mode=mode)
        koopman = multi_dot([c00_sqrt_inv, c0t, ctt_sqrt_inv]).t()

        u, s, v = torch.svd(koopman)
        mask = s > epsilon

        u = torch.mm(c00_sqrt_inv, u[:, mask])
        v = torch.mm(ctt_sqrt_inv, v[:, mask])
        s = s[mask]

        u_t = u.t()
        v_t = v.t()
        s = torch.diag(s)
        out = torch.trace(
            2. * multi_dot([s, u_t, c0t, v]) - multi_dot([s, u_t, c00, u, s, v_t, ctt, v])
        )
    elif method == 'VAMPCE':
        #out = torch.trace(data)
        out = torch.trace(data[0])
        assert out is not None
        return out * -1.0
    assert out is not None
    return 1 + out


def generate_test_data(batch_size: int, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random test data."""
    torch.manual_seed(42)  # For reproducibility
    data = torch.randn(batch_size, dim)
    data_lagged = torch.randn(batch_size, dim)
    return data, data_lagged


def test_vamp_scores():
    """Test both VAMP score implementations."""

    # Test parameters
    test_cases = [
        {'batch_size': 10, 'dim': 5},
        {'batch_size': 100, 'dim': 20},
        {'batch_size': 1000, 'dim': 50}
    ]

    methods = ['VAMP1', 'VAMP2', 'VAMPE', 'VAMPCE']
    modes = ['trunc', 'regularize']
    epsilons = [1e-6, 1e-8]

    for case in test_cases:
        print(f"\nTesting with batch_size={case['batch_size']}, dim={case['dim']}")

        data, data_lagged = generate_test_data(case['batch_size'], case['dim'])

        for method in methods:
            for mode in modes:
                for eps in epsilons:
                    try:
                        # Compute scores with both implementations
                        score_old = vamp_score_old(
                            data, data_lagged,
                            method=method,
                            epsilon=eps,
                            mode=mode
                        )

                        score_new = vamp_score_new(
                            data, data_lagged,
                            method=method,
                            epsilon=eps,
                            mode=mode
                        )

                        # Compare results
                        if torch.is_tensor(score_old) and torch.is_tensor(score_new):
                            is_close = torch.allclose(score_old, score_new, rtol=1e-5)
                        else:
                            is_close = np.allclose(score_old, score_new, rtol=1e-5)

                        print(f"""
                        Method: {method}, Mode: {mode}, Epsilon: {eps}
                        Old score: {score_old}
                        New score: {score_new}
                        Match: {is_close}
                        """)

                        assert is_close, f"""
                        Scores don't match for:
                        method={method}, mode={mode}, epsilon={eps}
                        Old: {score_old}
                        New: {score_new}
                        """

                    except Exception as e:
                        print(f"""
                        Error in test:
                        method={method}, mode={mode}, epsilon={eps}
                        Error: {str(e)}
                        """)
                        raise e


if __name__ == "__main__":
    test_vamp_scores()