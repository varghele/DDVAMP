import torch
import numpy as np


def matrix_inverse(matrix: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """
    Calculate the inverse of a square matrix using eigendecomposition.

    Parameters
    ----------
    matrix : torch.Tensor
        Square matrix to invert
    epsilon : float, optional
        Threshold for eigenvalue cutoff to ensure numerical stability,
        by default 1e-10

    Returns
    -------
    torch.Tensor
        Inverse of the input matrix

    Notes
    -----
    The function uses eigendecomposition to compute the inverse, which is
    numerically stable for symmetric matrices. Eigenvalues below epsilon
    are excluded to prevent division by near-zero values.
    """
    # Move matrix to CPU and convert to numpy for eigendecomposition
    matrix_cpu = matrix.detach().to('cpu')
    eigenvals, eigenvecs = np.linalg.eigh(matrix_cpu.numpy())

    # Filter out small eigenvalues for numerical stability
    mask = eigenvals > epsilon
    eigenvals_filtered = eigenvals[mask]
    eigenvecs_filtered = eigenvecs[:, mask]

    # Compute inverse using eigendecomposition
    inverse = eigenvecs_filtered @ np.diag(1.0 / eigenvals_filtered) @ eigenvecs_filtered.T

    return torch.from_numpy(inverse)

def covariances_E(chi_lag_0: torch.Tensor, chi_lag_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate instantaneous and time-lagged covariance matrices.

    Parameters
    ----------
    chi_lag_0 : torch.Tensor
        Data tensor at time t
    chi_lag_t : torch.Tensor
        Data tensor at time t + τ

    Returns
    -------
    C0inv : torch.Tensor
        Inverse of instantaneous covariance matrix
    Ctau : torch.Tensor
        Time-lagged covariance matrix

    Notes
    -----
    Both input tensors must have the same shape. The covariance matrices
    are normalized by the number of samples.
    """
    # Compute normalization factor
    n_samples = chi_lag_0.shape[0]
    norm = 1.0 / n_samples

    # Calculate covariance matrices
    C0 = norm * chi_lag_0.T @ chi_lag_0    # Instantaneous covariance
    Ctau = norm * chi_lag_0.T @ chi_lag_t  # Time-lagged covariance

    # Compute inverse of instantaneous covariance
    C0inv = matrix_inverse(C0)

    return C0inv, Ctau


def _compute_pi(transition_matrix: torch.Tensor) -> np.ndarray:
    """
    Calculate the stationary distribution of a transition matrix.

    Parameters
    ----------
    transition_matrix : torch.Tensor
        Square transition matrix (K) representing state transitions

    Returns
    -------
    np.ndarray
        Normalized stationary distribution (π) satisfying π = Kπ

    Notes
    -----
    The stationary distribution is computed by finding the eigenvector
    corresponding to the eigenvalue closest to 1, then normalizing it
    to sum to 1. This ensures the resulting vector is a valid
    probability distribution.
    """
    # Convert to numpy and transpose for eigenvector calculation
    K_transpose = transition_matrix.T.numpy()

    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(K_transpose)

    # Find index of eigenvalue closest to 1
    stationary_idx = np.argmin((eigenvals - 1.0) ** 2)

    # Extract corresponding eigenvector
    pi = eigenvecs[:, stationary_idx]

    # Normalize to ensure sum equals 1
    pi_normalized = pi / np.sum(pi, keepdims=True)

    return pi_normalized

