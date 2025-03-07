import torch
import numpy as np
from typing import Optional, Union, Callable, Tuple, List
from deeptime.util.torch import eigh, multi_dot


def symeig_reg(mat, epsilon: float = 1e-6, mode='regularize', eigenvectors=True) \
        -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
    r""" Solves a eigenvector/eigenvalue decomposition for a hermetian matrix also if it is rank deficient.

    Parameters
    ----------
    mat : torch.Tensor
        the hermetian matrix
    epsilon : float, default=1e-6
        Cutoff for eigenvalues.
    mode : str, default='regularize'
        Whether to truncate eigenvalues if they are too small or to regularize them by taking the absolute value
        and adding a small positive constant. :code:`trunc` leads to truncation, :code:`regularize` leads to epsilon
        being added to the eigenvalues after taking the absolute value
    eigenvectors : bool, default=True
        Whether to compute eigenvectors.

    Returns
    -------
    (eigval, eigvec) : Tuple[torch.Tensor, Optional[torch.Tensor]]
        Eigenvalues and -vectors.
    """
    assert mode in sym_inverse.valid_modes, f"Invalid mode {mode}, supported are {sym_inverse.valid_modes}"

    if mode == 'regularize':
        identity = torch.eye(mat.shape[0], dtype=mat.dtype, device=mat.device)
        mat = mat + epsilon * identity

    # Calculate eigvalues and potentially eigvectors
    eigval, eigvec = eigh(mat)

    if eigenvectors:
        eigvec = eigvec.transpose(0, 1)

    if mode == 'trunc':
        # Filter out Eigenvalues below threshold and corresponding Eigenvectors
        mask = eigval > epsilon
        eigval = eigval[mask]
        if eigenvectors:
            eigvec = eigvec[mask]
    elif mode == 'regularize':
        # Calculate eigvalues and eigvectors
        eigval = torch.abs(eigval)
    elif mode == 'clamp':
        eigval = torch.clamp_min(eigval, min=epsilon)

    return eigval, eigvec


def covariances(x: "torch.Tensor", y: "torch.Tensor", remove_mean: bool = True):
    """Computes instantaneous and time-lagged covariances matrices.

    Parameters
    ----------
    x : (T, n) torch.Tensor
        Instantaneous data.
    y : (T, n) torch.Tensor
        Time-lagged data.
    remove_mean: bool, default=True
        Whether to remove the mean of x and y.

    Returns
    -------
    cov_00 : (n, n) torch.Tensor
        Auto-covariance matrix of x.
    cov_0t : (n, n) torch.Tensor
        Cross-covariance matrix of x and y.
    cov_tt : (n, n) torch.Tensor
        Auto-covariance matrix of y.

    See Also
    --------
    deeptime.covariance.Covariance : Estimator yielding these kind of covariance matrices based on raw numpy arrays
                                     using an online estimation procedure.
    """
    assert x.shape == y.shape, "x and y must be of same shape"
    batch_size = x.shape[0]

    if remove_mean:
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

    # Calculate the cross-covariance
    y_t = y.transpose(0, 1)
    x_t = x.transpose(0, 1)
    cov_01 = 1 / (batch_size - 1) * torch.matmul(x_t, y)
    # Calculate the auto-correlations
    cov_00 = 1 / (batch_size - 1) * torch.matmul(x_t, x)
    cov_11 = 1 / (batch_size - 1) * torch.matmul(y_t, y)

    return cov_00, cov_01, cov_11


def matrix_inverse(matrix: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """
    Calculate the inverse of a square matrix using eigendecomposition.

    This function computes the inverse by:
    1. Converting the matrix to CPU
    2. Computing eigenvalues and eigenvectors
    3. Filtering out small eigenvalues
    4. Computing inverse using eigendecomposition

    Parameters
    ----------
    matrix : torch.Tensor
        Square matrix to invert
    epsilon : float, default=1e-10
        Threshold for eigenvalue filtering

    Returns
    -------
    torch.Tensor
        Inverse of the input matrix

    Raises
    ------
    ValueError
        If matrix is not square or is singular
    """
    # Input validation
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")

    # Move matrix to CPU and convert to numpy
    matrix_cpu = matrix.detach().to('cpu').numpy()

    # Compute eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(matrix_cpu)

    # Filter small eigenvalues
    valid_indices = eigenvals > epsilon
    if not np.any(valid_indices):
        raise ValueError("Matrix is singular or near-singular")

    filtered_vals = eigenvals[valid_indices]
    filtered_vecs = eigenvecs[:, valid_indices]

    # Compute inverse using eigendecomposition
    return filtered_vecs @ np.diag(1.0 / filtered_vals) @ filtered_vecs.T


def covariances_E(chi_instant: torch.Tensor, chi_lagged: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate instantaneous and time-lagged covariance matrices.

    Computes the inverse of the instantaneous covariance matrix and
    the time-lagged covariance matrix for VAMP analysis.

    Parameters
    ----------
    chi_instant : torch.Tensor
        Instantaneous data matrix (time t)
    chi_lagged : torch.Tensor
        Time-lagged data matrix (time t + τ)

    Returns
    -------
    C0_inv : torch.Tensor
        Inverse of instantaneous covariance matrix
    C_tau : torch.Tensor
        Time-lagged covariance matrix

    Notes
    -----
    Normalization is performed by dividing by the number of samples.
    """
    # Calculate normalization factor
    n_samples = chi_instant.shape[0]
    norm_factor = 1.0 / n_samples

    # Compute covariance matrices
    C0 = norm_factor * chi_instant.T @ chi_instant  # Instantaneous covariance
    C_tau = norm_factor * chi_instant.T @ chi_lagged  # Time-lagged covariance

    # Compute inverse of instantaneous covariance
    C0_inv = matrix_inverse(C0)

    return C0_inv, C_tau


def _compute_pi(transition_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the stationary distribution of a transition matrix.

    Finds the eigenvector corresponding to eigenvalue 1 (or closest to 1)
    and normalizes it to obtain the stationary distribution.

    Parameters
    ----------
    transition_matrix : np.ndarray
        Square transition matrix (K) representing state transitions

    Returns
    -------
    np.ndarray
        Normalized stationary distribution vector (π)

    Notes
    -----
    The stationary distribution π satisfies π = Kᵀπ, making it the
    eigenvector of Kᵀ with eigenvalue 1.
    """
    # Compute eigendecomposition of transpose
    eigenvals, eigenvecs = np.linalg.eig(transition_matrix.T)

    # Find index of eigenvalue closest to 1
    stationary_index = np.argmin((eigenvals - 1.0) ** 2)

    # Extract corresponding eigenvector
    stationary_dist = eigenvecs[:, stationary_index]

    # Normalize to ensure sum = 1
    normalized_dist = stationary_dist / np.sum(stationary_dist, keepdims=True)

    return normalized_dist


def sym_inverse(mat, epsilon: float = 1e-6, return_sqrt=False, mode='regularize'):
    """ Utility function that returns the inverse of a matrix, with the
    option to return the square root of the inverse matrix.

    Parameters
    ----------
    mat: numpy array with shape [m,m]
        Matrix to be inverted.
    epsilon : float
        Cutoff for eigenvalues.
    return_sqrt: bool, optional, default = False
        if True, the square root of the inverse matrix is returned instead
    mode: str, default='trunc'
        Whether to truncate eigenvalues if they are too small or to regularize them by taking the absolute value
        and adding a small positive constant. :code:`trunc` leads to truncation, :code:`regularize` leads to epsilon
        being added to the eigenvalues after taking the absolute value

    Returns
    -------
    x_inv: numpy array with shape [m,m]
        inverse of the original matrix
    """
    eigval, eigvec = symeig_reg(mat, epsilon, mode)

    # Build the diagonal matrix with the filtered eigenvalues or square
    # root of the filtered eigenvalues according to the parameter
    if return_sqrt:
        diag = torch.diag(torch.sqrt(1. / eigval))
    else:
        diag = torch.diag(1. / eigval)

    return multi_dot([eigvec.t(), diag, eigvec])


sym_inverse.valid_modes = ('trunc', 'regularize', 'clamp')


def koopman_matrix(x: "torch.Tensor", y: "torch.Tensor", epsilon: float = 1e-6, mode: str = 'trunc',
                   c_xx: Optional[Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]] = None) -> "torch.Tensor":
    r""" Computes the Koopman matrix

    .. math:: K = C_{00}^{-1/2}C_{0t}C_{tt}^{-1/2}

    based on data over which the covariance matrices :math:`C_{\cdot\cdot}` are computed.

    Parameters
    ----------
    x : torch.Tensor
        Instantaneous data.
    y : torch.Tensor
        Time-lagged data.
    epsilon : float, default=1e-6
        Cutoff parameter for small eigenvalues.
    mode : str, default='trunc'
        Regularization mode for Hermetian inverse. See :meth:`sym_inverse`.
    c_xx : tuple of torch.Tensor, optional, default=None
        Tuple containing c00, c0t, ctt if already computed.

    Returns
    -------
    K : torch.Tensor
        The Koopman matrix.
    """
    if c_xx is not None:
        c00, c0t, ctt = c_xx
    else:
        c00, c0t, ctt = covariances(x, y, remove_mean=True)
    c00_sqrt_inv = sym_inverse(c00, return_sqrt=True, epsilon=epsilon, mode=mode)
    ctt_sqrt_inv = sym_inverse(ctt, return_sqrt=True, epsilon=epsilon, mode=mode)
    return multi_dot([c00_sqrt_inv, c0t, ctt_sqrt_inv]).t()
