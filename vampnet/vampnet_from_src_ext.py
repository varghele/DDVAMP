import time

import torch
from typing import Optional, Union, Callable, Tuple, List
import torch.nn as nn
import numpy as np
from deeptime.base import Model, Transformer, EstimatorTransformer
from deeptime.base_torch import DLEstimatorMixin
from deeptime.util.torch import map_data, eigh, multi_dot, disable_TF32
from tqdm import tqdm


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
        raise RuntimeError(f"Error computing VAMP score: {str(e)} | Method {method}")

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


class VAMPU(nn.Module):
    """
    VAMP-U (Variational Approach for Markov Processes - Unitary) module.
    Implements the unitary transformation part of the VAMP algorithm.
    """

    def __init__(self,
                 units: int,
                 activation, #: nn.Module,
                 device: Optional[torch.device] = None):
        """
        Initialize VAMP-U module.

        Parameters
        ----------
        units : int
            Number of output units
        activation : nn.Module
            Activation function to use
        device : Optional[torch.device]
            Device to place the module on (CPU/GPU)
        """
        super().__init__()
        self.M = units
        self.activation = activation
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize u_kernel with uniform values
        self._u_kernel = nn.Parameter(
            (1. / self.M) * torch.ones((self.M,), device=self.device),
            requires_grad=True
        ) # u_var

    @property
    def u_kernel(self) -> nn.Parameter:
        """
        Get the u_kernel parameter.

        Returns
        -------
        torch.nn.Parameter
            The u_kernel parameter
        """
        return self._u_kernel

    def compute_output_shape(self, input_shape: List[int]) -> List:
        """
        Compute the output shape of the module.

        Parameters
        ----------
        input_shape : List[int]
            Input shape

        Returns
        -------
        List
            List of output shapes for all outputs
        """
        return [self.M] * 2 + [(self.M, self.M)] * 4 + [self.M]

    def _tile(self, x: torch.Tensor, n_batch: int) -> torch.Tensor:
        """
        Tile a tensor along the batch dimension.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to tile
        n_batch : int
            Number of times to tile

        Returns
        -------
        torch.Tensor
            Tiled tensor
        """
        x_exp = torch.unsqueeze(x, dim=0)
        shape = x.shape
        return x_exp.repeat(n_batch, *([1] * len(shape)))

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass of the VAMP-U module.

        Parameters
        ----------
        x : Tuple[torch.Tensor, torch.Tensor]
            Tuple of (instantaneous, time-lagged) data

        Returns
        -------
        List[torch.Tensor]
            List containing [u, v, C00, C11, C01, sigma, mu]
        """
        # Extract data
        chi_t, chi_tau = x
        n_batch = chi_t.shape[0]
        norm = 1. / n_batch

        # Compute correlations
        chi_tau_t = chi_tau.t()
        corr_tau = norm * torch.matmul(chi_tau_t, chi_tau)
        chi_mean = torch.mean(chi_tau, dim=0, keepdim=True)

        # Apply activation and compute kernel
        ac_u_kernel = self.activation(self._u_kernel).to(self.device)
        kernel_u = torch.unsqueeze(ac_u_kernel, dim=0)

        # Compute u vector and related quantities
        u = kernel_u / torch.sum(chi_mean * kernel_u, dim=1, keepdim=True)
        u_t = u.t()
        v = torch.matmul(corr_tau, u_t)
        mu = norm * torch.matmul(chi_tau, u_t)

        # Compute covariance matrices
        cmu_t = (chi_tau * mu).t()
        sigma = torch.matmul(cmu_t, chi_tau)
        gamma = chi_tau * torch.matmul(chi_tau, u_t)
        gamma_t = gamma.t()

        # Compute time-lagged covariances
        chi_t_t = chi_t.t()
        C00 = norm * torch.matmul(chi_t_t, chi_t)
        C11 = norm * torch.matmul(gamma_t, gamma)
        C01 = norm * torch.matmul(chi_t_t, gamma)

        # Return all computed quantities
        return [
            self._tile(var, n_batch) for var in (u, v, C00, C11, C01, sigma)
        ] + [mu]

    def reset_weights(self):
        """Reset the u_kernel weights to initial values."""
        with torch.no_grad():
            self._u_kernel.data = (1. / self.M) * torch.ones_like(self._u_kernel)


class VAMPS(nn.Module):
    """
    VAMP-S (Variational Approach for Markov Processes - Symmetric) module.
    Implements the symmetric transformation part of the VAMP algorithm.
    """

    def __init__(self,
                 units: int,
                 activation, #: nn.Module,
                 order: int = 20,
                 renorm: bool = False,
                 device: Optional[torch.device] = None):
        """
        Initialize VAMP-S module.

        Parameters
        ----------
        units : int
            Number of output units
        activation : nn.Module
            Activation function to use
        order : int
            Order of the approximation
        renorm : bool
            Whether to renormalize weights
        device : Optional[torch.device]
            Device to place the module on (CPU/GPU)
        """
        super().__init__()
        self.M = units
        self.activation = activation
        self.renorm = renorm
        self.order = order
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize s_kernel with small random values
        self._s_kernel = nn.Parameter(
            0.1 * torch.ones((self.M, self.M), device=self.device),
            requires_grad=True
        ) # S_var
        self._init_weight = None

    @property
    def s_kernel(self) -> nn.Parameter:
        """
        Get the s_kernel parameter.

        Returns
        -------
        torch.nn.Parameter
            The s_kernel parameter
        """
        return self._s_kernel

    def reset_weights(self):
        """Reset weights to initial values if available."""
        if self._init_weight is None:
            self._init_weight = self._s_kernel.clone().detach()
        else:
            with torch.no_grad():
                self._s_kernel.copy_(self._init_weight)

    def compute_output_shape(self, input_shape: List[int]) -> List:
        """
        Compute the output shape of the module.

        Parameters
        ----------
        input_shape : List[int]
            Input shape

        Returns
        -------
        List
            List of output shapes
        """
        return [(self.M, self.M)] * 2 + [self.M] + [(self.M, self.M)]

    def forward(self, x: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> List[torch.Tensor]:
        """
        Forward pass of the VAMP-S module.

        Parameters
        ----------
        x : Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]
            Input tensors, either [v, C00, C11, C01, sigma] or
            [chi_t, chi_tau, u, v, C00, C11, C01, sigma]

        Returns
        -------
        List[torch.Tensor]
            List containing [vamp_e_tile, K_tile, probs/zeros, S_tile]
        """
        if len(x) == 5:
            v, C00, C11, C01, sigma = x
            chi_t = chi_tau = u = None
        else:
            chi_t, chi_tau, u, v, C00, C11, C01, sigma = x
            u = u[0]

        n_batch = v.shape[0]
        norm = 1. / n_batch
        C00, C11, C01 = C00[0], C11[0], C01[0]
        sigma, v = sigma[0], v[0]

        # Compute kernel and its transformations
        kernel_w = self.activation(self._s_kernel).to(self.device)
        kernel_w_t = kernel_w.t()
        w1 = kernel_w + kernel_w_t
        w_norm = w1 @ v

        # Handle renormalization if needed
        if self.renorm:
            quasi_inf_norm = lambda x: torch.max(torch.abs(x))
            w1 = w1 / quasi_inf_norm(w_norm)
            w_norm = w1 @ v

        # Compute final matrices
        w2 = (1 - torch.squeeze(w_norm)) / torch.squeeze(v)
        S = w1 + torch.diag(w2)
        S_t = S.t()

        # Compute probabilities if full input is provided
        if chi_t is not None:
            u_t = u.t()
            chi_tau_t = chi_tau.t()
            q = (norm * torch.matmul(S, chi_tau_t).t() * torch.matmul(chi_tau, u_t))
            probs = torch.sum(chi_t * q, dim=1)
        else:
            probs = torch.zeros((n_batch, self.M), device=self.device)

        # Compute final outputs
        K = S @ sigma
        vamp_e = S_t @ C00 @ S @ C11 - 2 * S_t @ C01

        # Tile outputs for batch dimension
        vamp_e_tile = torch.tile(torch.unsqueeze(vamp_e, dim=0), [n_batch, 1, 1])
        K_tile = torch.tile(torch.unsqueeze(K, dim=0), [n_batch, 1, 1])
        S_tile = torch.tile(torch.unsqueeze(S, dim=0), [n_batch, 1, 1])

        return [vamp_e_tile, K_tile, probs, S_tile]


class VAMPNetModel(Transformer, Model):
    r"""
    A VAMPNet model which can be fit to data optimizing for one of the implemented VAMP scores.

    Parameters
    ----------
    lobe : torch.nn.Module
        One of the lobes of the VAMPNet. See also :class:`deeptime.util.torch.MLP`.
    lobe_timelagged : torch.nn.Module, optional, default=None
        The timelagged lobe. Can be left None, in which case the lobes are shared.
    dtype : data type, default=np.float32
        The data type for which operations should be performed. Leads to an appropriate cast within fit and
        transform methods.
    device : device, default=None
        The device for the lobe(s). Can be None which defaults to CPU.

    See Also
    --------
    VAMPNet : The corresponding estimator.
    """

    def __init__(self, lobe: "torch.nn.Module", lobe_timelagged: Optional["torch.nn.Module"] = None,
                 dtype=np.float32, device=None):
        super().__init__()
        self._lobe = lobe
        self._lobe_timelagged = lobe_timelagged if lobe_timelagged is not None else lobe

        if dtype == np.float32:
            self._lobe = self._lobe.float()
            self._lobe_timelagged = self._lobe_timelagged.float()
        elif dtype == np.float64:
            self._lobe = self._lobe.double()
            self._lobe_timelagged = self._lobe_timelagged.double()
        self._dtype = dtype
        self._device = device

    @property
    def lobe(self) -> "torch.nn.Module":
        r""" The instantaneous lobe.

        Returns
        -------
        lobe : nn.Module
        """
        return self._lobe

    @property
    def lobe_timelagged(self) -> "torch.nn.Module":
        r""" The timelagged lobe. Might be equal to :attr:`lobe`.

        Returns
        -------
        lobe_timelagged : nn.Module
        """
        return self._lobe_timelagged

    def transform(self, data, instantaneous: bool = True, **kwargs):
        r""" Transforms data through the instantaneous or time-shifted network lobe.

        Parameters
        ----------
        data : numpy array or torch tensor
            The data to transform.
        instantaneous : bool, default=True
            Whether to use the instantaneous lobe or the time-shifted lobe for transformation.
        **kwargs
            Ignored kwargs for api compatibility.

        Returns
        -------
        transform : array_like
            List of numpy array or numpy array containing transformed data.
        """
        if instantaneous:
            self._lobe.eval()
            net = self._lobe
        else:
            self._lobe_timelagged.eval()
            net = self._lobe_timelagged
        out = []
        for data_tensor in map_data(data, device=self._device, dtype=self._dtype):
            out.append(net(data_tensor).cpu().numpy())
        return out if len(out) > 1 else out[0]


class RevVAMPNet(EstimatorTransformer, DLEstimatorMixin, nn.Module):
    r""" Implementation of VAMPNets. :footcite:`mardt2018vampnets`
    These networks try to find an optimal featurization of data based on a VAMP score :footcite:`wu2020variational`
    by using neural networks as featurizing transforms which are equipped with a loss that is the negative VAMP score.
    This estimator is also a transformer and can be used to transform data into the optimized space.
    From there it can either be used to estimate Markov state models via making assignment probabilities
    crisp (in case of softmax output distributions) or to estimate the Koopman operator
    using the :class:`VAMP <deeptime.decomposition.VAMP>` estimator.

    Parameters
    ----------
    lobe : torch.nn.Module
        A neural network module which maps input data to some (potentially) lower-dimensional space.
    lobe_timelagged : torch.nn.Module, optional, default=None
        Neural network module for timelagged data, in case of None the lobes are shared (structure and weights).
    device : torch device, default=None
        The device on which the torch modules are executed.
    optimizer : str or Callable, default='Adam'
        An optimizer which can either be provided in terms of a class reference (like `torch.optim.Adam`) or
        a string (like `'Adam'`). Defaults to Adam.
    learning_rate : float, default=5e-4
        The learning rate of the optimizer.
    score_method : str, default='VAMP2'
        The scoring method which is used for optimization.
    score_mode : str, default='regularize'
        The mode under which inverses of positive semi-definite matrices are estimated. Per default, the matrices
        are perturbed by a small constant added to the diagonal. This makes sure that eigenvalues are not too
        small. For a complete list of modes, see :meth:`sym_inverse`.
    epsilon : float, default=1e-6
        The strength of the regularization under which matrices are inverted. Meaning depends on the score_mode,
        see :meth:`sym_inverse`.
    dtype : dtype, default=np.float32
        The data type of the modules and incoming data.

    See Also
    --------
    deeptime.decomposition.VAMP

    References
    ----------
    .. footbibliography::
    """
    _MUTABLE_INPUT_DATA = True

    def __init__(self, lobe: "torch.nn.Module",
                 lobe_timelagged: Optional["torch.nn.Module"] = None,
                 vampu: Optional[nn.Module] = None,
                 vamps: Optional[nn.Module] = None,
                 device=None, optimizer: Union[str, Callable] = 'Adam',
                 learning_rate: float = 5e-4,
                 activation_vampu: Optional["torch.nn.Module"] = None,
                 activation_vamps: Optional["torch.nn.Module"] = None,
                 num_classes: int = 1,
                 tau: int = 20,
                 score_method: str = 'VAMP2',
                 score_mode: str = 'regularize',
                 epsilon: float = 1e-6,
                 dtype=np.float32):

        # Initialize parent classes
        EstimatorTransformer.__init__(self)
        DLEstimatorMixin.__init__(self)
        nn.Module.__init__(self)

        # Register networks as modules
        self.lobe = lobe
        self.lobe_timelagged = lobe_timelagged or lobe
        self.add_module('lobe', self.lobe)
        self.add_module('lobe_timelagged', self.lobe_timelagged)

        # Set up configuration
        self.valid_score_methods = ('VAMP1', 'VAMP2', 'VAMPE', 'VAMPCE')
        self.score_method = score_method
        self.score_mode = score_mode
        self._step = 0
        self._epsilon = epsilon
        self.device = device
        self.learning_rate = learning_rate
        self.dtype = dtype
        self._train_scores = []
        self._validation_scores = []

        # Initialize VAMP networks
        #self._vampu = VAMPU(units=num_classes, activation=activation_vampu, device=device)
        #self._vamps = VAMPS(units=num_classes, activation=activation_vamps, device=device)
        self._vampu = vampu
        self._vamps = vamps
        self.add_module('_vampu', self._vampu)
        self.add_module('_vamps', self._vamps)

        # Initialize caches and state
        self._k_cache = {}
        self.network_lag = tau
        self._lag = tau
        self._K = None
        self.data = None
        self.LAG_EPOCH = 1000
        self.LAST = -1

        # Setup optimizer based on score method
        if score_method == "VAMPCE":
            assert self._vampu is not None and self._vamps is not None, "vampu and vamps module must be defined"
            all_params = (list(self.lobe.parameters()) +
                          list(self.lobe_timelagged.parameters()) +
                          list(self._vampu.parameters()) +
                          list(self._vamps.parameters()))
            self.setup_optimizer(optimizer, all_params)
        else:
            all_params = (list(self.lobe.parameters()) +
                          list(self.lobe_timelagged.parameters()))
            self.setup_optimizer(optimizer, all_params)

        # Use DLEstimatorMixin's method to set up optimizer
        if isinstance(optimizer, str):
            optimizer_cls = getattr(torch.optim, optimizer)
            self._optimizer = optimizer_cls(all_params, lr=learning_rate)
        else:
            self._optimizer = optimizer(all_params, lr=learning_rate)

    @property
    def optimizer(self):
        return self._optimizer

    def set_optimizer_lr(self, new_lr: float) -> None:
        """
        Set a new learning rate for all parameter groups in the optimizer.

        Args:
            new_lr (float): New learning rate value to set

        Example:
            vampnet.set_optimizer_lr(0.2)  # Sets learning rate to 0.2
        """
        if not hasattr(self, 'optimizer'):
            raise AttributeError("No optimizer found. Initialize optimizer before setting learning rate.")

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        # Optional: log the learning rate change
        if hasattr(self, '_step'):
            print(f"Step {self._step}: Learning rate set to {new_lr}")

    def parameters(self, recurse: bool = True):
        """Return an iterator over module parameters.

        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Returns:
            Iterator over parameters
        """
        if self.score_method == "VAMPCE":
            params = (list(self.lobe.parameters(recurse)) +
                      list(self.lobe_timelagged.parameters(recurse)) +
                      list(self._vampu.parameters(recurse)) +
                      list(self._vamps.parameters(recurse)))
        else:
            params = (list(self.lobe.parameters(recurse)) +
                      list(self.lobe_timelagged.parameters(recurse)))

        for param in params:
            yield param

    @property
    def K(self) -> np.ndarray:
        """The estimated Koopman operator."""
        if self._K is None: # or self._reestimated: MARKER
            self._K = np.ones((1, 1))

        return self._K

    @property
    def lag(self) -> int:
        """
        The model lag time.

        Returns
        -------
        int
            Current lag time
        """
        return self._lag

    @lag.setter
    def lag(self, lag: int):
        """
        Update the model lag time for ITS calculation.

        Parameters
        ----------
        lag : int
            New lag time to update the model to
        """
        # Reset VAMP-S weights
        self._vamps.reset_weights()

        # Update auxiliary weights with current data
        data = self.data
        self.update_auxiliary_weights(data, optimize_u=False, optimize_S=True, reset_weights=False)

        # Train VAMP networks
        self.train_US(data, train_u=False, out_log=True)

        # First training phase - fixed U
        for _ in tqdm(range(self.LAG_EPOCH)):
            self.train_US(data, train_u=False)

        # Second training phase - update both U and S
        for _ in tqdm(range(self.LAG_EPOCH)):
            self.train_US(data)

        # Final training step with logging
        self.train_US(data, out_log=True)
        print(f"new lag {lag} ok")

        # Update internal state
        self._lag = lag
        self._reestimated = True

    @property
    def vampu(self) -> nn.Module:
        """
        The VAMP-U network module.

        Returns
        -------
        nn.Module
            VAMP-U neural network module
        """
        return self._vampu

    @property
    def vamps(self) -> nn.Module:
        """
        The VAMP-S network module.

        Returns
        -------
        nn.Module
            VAMP-S neural network module
        """
        return self._vamps

    @property
    def train_scores(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._train_scores)

    @property
    def validation_scores(self) -> np.ndarray:
        r""" The collected validation scores. First dimension contains the step, second dimension the score.
        Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._validation_scores)

    @property
    def epsilon(self) -> float:
        r""" Regularization parameter for matrix inverses.

        :getter: Gets the currently set parameter.
        :setter: Sets a new parameter. Must be non-negative.
        :type: float
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        assert value >= 0
        self._epsilon = value

    @property
    def score_method(self) -> str:
        r""" Property which steers the scoring behavior of this estimator.

        :getter: Gets the current score.
        :setter: Sets the score to use.
        :type: str
        """
        return self._score_method

    @score_method.setter
    def score_method(self, value: str):
        assert value in self.valid_score_methods, f"Tried setting an unsupported scoring method '{value}', " \
                                             f"available are {self.valid_score_methods}."
        self._score_method = value

    @property
    def lobe(self) -> "torch.nn.Module":
        r""" The instantaneous lobe of the VAMPNet.

        :getter: Gets the instantaneous lobe.
        :setter: Sets a new lobe.
        :type: torch.nn.Module
        """
        return self._lobe

    @lobe.setter
    def lobe(self, value: "torch.nn.Module"):
        self._lobe = value
        if self.dtype == np.float32:
            self._lobe = self._lobe.float()
        else:
            self._lobe = self._lobe.double()
        self._lobe = self._lobe.to(device=self.device)

    @property
    def lobe_timelagged(self) -> "torch.nn.Module":
        r""" The timelagged lobe of the VAMPNet.

        :getter: Gets the timelagged lobe. Can be the same a the instantaneous lobe.
        :setter: Sets a new lobe. Can be None, in which case the instantaneous lobe is shared.
        :type: torch.nn.Module
        """
        return self._lobe_timelagged

    @lobe_timelagged.setter
    def lobe_timelagged(self, value: Optional["torch.nn.Module"]):
        if value is None:
            value = self.lobe
        else:
            if self.dtype == np.float32:
                value = value.float()
            else:
                value = value.double()
        self._lobe_timelagged = value
        self._lobe_timelagged = self._lobe_timelagged.to(device=self.device)

    def check_gradients(self):
        """
        Check for NaN or infinity values in gradients of all parameters.
        Raises:
            ValueError: If any NaN or infinity values are detected in gradients.
        """
        gradient_issues = {}

        def check_grad(name, parameter):
            if parameter.grad is not None:
                if torch.isnan(parameter.grad).any():
                    gradient_issues[name] = {'issue': 'NaN', 'location': parameter.grad}
                if torch.isinf(parameter.grad).any():
                    gradient_issues[name] = {'issue': 'Inf', 'location': parameter.grad}

        # Check all network components
        for name, param in self.lobe.named_parameters():
            check_grad(f'lobe.{name}', param)

        for name, param in self.lobe_timelagged.named_parameters():
            check_grad(f'lobe_timelagged.{name}', param)

        if self.score_method == "VAMPCE":
            for name, param in self._vampu.named_parameters():
                check_grad(f'vampu.{name}', param)
            for name, param in self._vamps.named_parameters():
                check_grad(f'vamps.{name}', param)

        if gradient_issues:
            error_msg = "Gradient issues detected:\n"
            for name, issue in gradient_issues.items():
                error_msg += f"Parameter {name}: {issue['issue']} values detected\n"
            raise ValueError(error_msg)

    def stabilize_training(self, loss_value):
        """
        Stabilize training by handling various numerical issues including vanishing gradients.
        """
        # 1. Initial loss value checks
        if torch.isnan(loss_value) or torch.isinf(loss_value):
            raise ValueError("Loss value is NaN or infinite")

        # 2. Use gradient scaling for mixed precision and stability
        scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None

        try:
            # 3. Compute backward pass with scaled gradients
            if scaler:
                scaler.scale(loss_value).backward()
            else:
                loss_value.backward()

            # 4. Check for vanishing gradients
            grad_norm = 0.0
            num_params_with_grad = 0
            for name, param in self.named_parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item()
                    num_params_with_grad += 1

            if num_params_with_grad > 0:
                avg_grad_norm = grad_norm / num_params_with_grad

                # If gradients are too small, scale them up
                if avg_grad_norm < 1e-8:
                    scale_factor = 1e-4 / (avg_grad_norm + 1e-12)
                    for param in self.parameters():
                        if param.grad is not None:
                            param.grad.data.mul_(scale_factor)

            # 5. Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            # 6. Optimizer step with scaled gradients
            if scaler:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()

            # 7. Learning rate adjustment if gradients are consistently small
            if hasattr(self, 'grad_history'):
                self.grad_history.append(avg_grad_norm)
                if len(self.grad_history) > 10:
                    self.grad_history.pop(0)
                    if np.mean(self.grad_history) < 1e-6:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= 1.5
            else:
                self.grad_history = [avg_grad_norm]

        except RuntimeError as e:
            self.optimizer.zero_grad()
            raise RuntimeError(f"Backward pass failed: {str(e)}")

    def partial_fit(self, data, train_score_callback: Callable[[int, "torch.Tensor"], None] = None):
        r""" Performs a partial fit on data. This does not perform any batching.

        Parameters
        ----------
        data : tuple or list of length 2, containing instantaneous and timelagged data
            The data to train the lobe(s) on.
        train_score_callback : callable, optional, default=None
            An optional callback function which is evaluated after partial fit, containing the current step
            of the training (only meaningful during a :meth:`fit`) and the current score as torch Tensor.

        Returns
        -------
        self : VAMPNet
            Reference to self.
        """
        #self.check_gradients()
        if self.dtype == np.float32:
            self.lobe = self.lobe.float()
            self.lobe_timelagged = self.lobe_timelagged.float()
        elif self.dtype == np.float64:
            self.lobe = self.lobe.double()
            self.lobe_timelagged = self.lobe_timelagged.double()

        self.train()
        self.lobe.train()
        self.lobe_timelagged.train()

        assert isinstance(data, (list, tuple)) and len(data) == 2, \
            "Data must be a list or tuple of batches belonging to instantaneous " \
            "and respective time-lagged data."

        batch_0, batch_t = data[0], data[1]

        if isinstance(data[0], np.ndarray):
            batch_0 = torch.from_numpy(data[0].astype(self.dtype)).to(device=self.device)
        if isinstance(data[1], np.ndarray):
            batch_t = torch.from_numpy(data[1].astype(self.dtype)).to(device=self.device)

        # Ensure inputs require gradients
        #batch_0, batch_t = batch_data[0].to(self.device), batch_data[1].to(self.device)
        batch_0.requires_grad_(True)
        batch_t.requires_grad_(True)

        self.optimizer.zero_grad()
        x_0 = self.lobe(batch_0)
        x_t = self.lobe_timelagged(batch_t)

        if self.score_method == 'VAMPCE':
            self._vampu.train()
            self._vamps.train()
            self._vampu.u_kernel.retain_grad()
            self._vamps.s_kernel.retain_grad()
            (u_out, v_out, C00_out, C11_out, C01_out, sigma_out, mu_out) = self._vampu([x_0, x_t])
            Ve_out, K_out, p_out, S_out = self._vamps([x_0, x_t, u_out, v_out, C00_out,C11_out, C01_out, sigma_out])
            self._K = K_out[0]
            loss_value = vampnet_loss(Ve_out, Ve_out, method=self.score_method, epsilon=self.epsilon,
                                      mode=self.score_mode)
        else:
            loss_value = vampnet_loss(x_0, x_t, method=self.score_method, epsilon=self.epsilon, mode=self.score_mode)
        # Stabilized Optimizer with Grad Scaling to prevent explosion or Vanish
        self.stabilize_training(loss_value)
        torch.cuda.synchronize()

        if train_score_callback is not None:
            lval_detached = loss_value.detach()
            train_score_callback(self._step, -lval_detached)
        self._train_scores.append((self._step, (-loss_value).item()))
        self._step += 1

        return self

    def validate(self, validation_data: Tuple["torch.Tensor"]) -> "torch.Tensor":
        r""" Evaluates the currently set lobe(s) on validation data and returns the value of the configured score.

        Parameters
        ----------
        validation_data : Tuple of torch Tensor containing instantaneous and timelagged data
            The validation data.

        Returns
        -------
        score : torch.Tensor
            The value of the score.
        """
        with disable_TF32():
            self.lobe.eval()
            self.lobe_timelagged.eval()
            if self.vamps is not None:
                self.vamps.eval()
                self.vampu.eval()

            with torch.no_grad():
                val = self.lobe(validation_data[0])
                val_t = self.lobe_timelagged(validation_data[1])
                if self.score_method == "VAMPCE":
                    (u_out, v_out, C00_out, C11_out, C01_out, sigma_out, mu_out) = self._vampu([val, val_t])
                    Ve_out, K_out, p_out, S_out = self._vamps([val, val_t, u_out, v_out,
                                                               C00_out, C11_out, C01_out, sigma_out])
                    score_value = vamp_score(Ve_out, Ve_out, method=self.score_method, mode=self.score_mode,
                                             epsilon=self.epsilon)
                else:
                    score_value = vamp_score(val, val_t, method=self.score_method, mode=self.score_mode,
                                             epsilon=self.epsilon)
                return score_value

    def update_auxiliary_weights(self, data, optimize_u: bool = True, optimize_S: bool = False,
                                 reset_weights: bool = True):
        """
        Update the weights for the auxiliary VAMP-U and VAMP-S networks.

        Parameters
        ----------
        data : tuple
            Tuple containing (chi_0, chi_t) data tensors
        optimize_u : bool, default=True
            Whether to optimize the VAMP-U weights
        optimize_S : bool, default=False
            Whether to optimize the VAMP-S weights
        reset_weights : bool, default=True
            Currently unused parameter for weight reset functionality

        Returns
        -------
        None
        """
        # Convert input data to tensors and move to device
        batch_0, batch_t = data[0], data[1]
        chi_0 = torch.Tensor(batch_0).to(self.device)
        chi_t = torch.Tensor(batch_t).to(self.device)

        # Calculate covariance matrices
        C0inv, Ctau = covariances_E(chi_0, chi_t)

        # Get current VAMP outputs
        (u_outd, v_outd, C00_outd, C11_outd,
         C01_outd, sigma_outd, mu_outd) = self._vampu([chi_0, chi_t])

        Ve_out, K_out, p_out, S_out = self._vamps([
            chi_0, chi_t, u_outd, v_outd, C00_outd,
            C11_outd, C01_outd, sigma_outd])

        # Calculate Koopman operator
        K = torch.Tensor(C0inv) @ Ctau.to('cpu')
        self._K = K_out[0]

        # Update VAMP-U weights if requested
        if optimize_u:
            pi = _compute_pi(K)
            u_kernel = np.log(np.abs(C0inv @ pi))
            for param in self.vampu.parameters():
                with torch.no_grad():
                    param[:] = torch.Tensor(u_kernel)

        # Update VAMP-S weights if requested
        if optimize_S:
            (u_out, v_out, C00_out, C11_out,
             C01_out, sigma, mu_out) = self.vampu([chi_0, chi_t])
            sigma_inv = matrix_inverse(sigma[0])
            S_nonrev = K @ sigma_inv
            S_rev = 0.5 * (S_nonrev + S_nonrev.t())
            s_kernel = np.log(np.abs(0.5 * S_rev))
            for param in self.vamps.parameters():
                with torch.no_grad():
                    param[:] = torch.Tensor(s_kernel)

    def train_US(self, data: Tuple[torch.Tensor, torch.Tensor],
                 lr_rate: float = 1e-3,
                 train_u: bool = True,
                 out_log: bool = False) -> None:
        """
        Train the VAMP-U and VAMP-S networks.

        Parameters
        ----------
        data : Tuple[torch.Tensor, torch.Tensor]
            Tuple of (instantaneous, time-lagged) data
        lr_rate : float, default=1e-3
            Learning rate for optimization
        train_u : bool, default=True
            Whether to train the VAMP-U network
        out_log : bool, default=False
            Whether to print loss values during training
        """
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # Monitor gradients
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN gradients detected in {name}")
        # Freeze main network parameters
        self.lobe.requires_grad_(False)
        self.lobe_timelagged.requires_grad_(False)

        # Configure VAMP-U training
        if train_u:
            self._vampu.train()
            self._vampu.requires_grad_(True)
            self._vampu.u_kernel.retain_grad()
        else:
            self._vampu.requires_grad_(False)

        # Configure VAMP-S training
        self._vamps.train()
        self._vamps.s_kernel.retain_grad()

        # Prepare data
        x_0, x_t = data[0], data[1]
        x_0 = torch.Tensor(x_0).to(self.device)
        x_t = torch.Tensor(x_t).to(self.device)

        # Forward pass
        self.optimizer.zero_grad()

        # VAMP-U forward pass
        u_out, v_out, C00_out, C11_out, C01_out, sigma_out, mu_out = self._vampu([x_0, x_t])

        # VAMP-S forward pass
        Ve_out, K_out, p_out, S_out = self._vamps([
            x_0, x_t, u_out, v_out, C00_out,
            C11_out, C01_out, sigma_out
        ])

        # Store Koopman operator
        self._K = K_out[0]

        # Compute and backpropagate loss
        loss_value = vampnet_loss(Ve_out, Ve_out,
                                  method=self.score_method,
                                  epsilon=self.epsilon,
                                  mode=self.score_mode)
        loss_value.backward()
        self.optimizer.step()

        # Optional loss logging
        if out_log:
            print(f"Loss: {loss_value.item():.6f}")

        # Restore network states
        self.lobe.requires_grad_(True)
        self.lobe_timelagged.requires_grad_(True)
        self.lobe.train()
        self.lobe_timelagged.train()

        if not train_u:
            self._vampu.requires_grad_(True)
            self._vampu.train()

    def estimate_koopman(self, lag: int) -> np.ndarray:
        """
        Estimate the Koopman operator for a given lag time.

        Uses cached results if available to avoid recomputation.

        Parameters
        ----------
        lag : int
            Lag time for the Koopman operator estimation

        Returns
        -------
        np.ndarray
            Estimated Koopman operator matrix for the specified lag time
        """
        # Return cached result if available
        if lag in self._k_cache:
            return self._k_cache[lag]

        # Update lag time and compute Koopman operator
        self.lag = lag
        koopman_op = self._K.detach().cpu().numpy()

        # Cache result for future use
        self._k_cache[lag] = koopman_op

        return koopman_op

    def estimate_koopman_op(self, trajectories: Union[List[np.ndarray], np.ndarray],
                            tau: int) -> np.ndarray:
        """
        Estimate the Koopman operator from trajectory data.

        Parameters
        ----------
        trajectories : Union[List[np.ndarray], np.ndarray]
            Either a list of trajectories or a single trajectory array
        tau : int
            Time lag for the Koopman operator estimation

        Returns
        -------
        np.ndarray
            Estimated Koopman operator matrix
        """
        # Process input trajectories
        if isinstance(trajectories, list):
            # Concatenate multiple trajectories
            instant_data = np.concatenate([t[:-tau] for t in trajectories], axis=0)
            lagged_data = np.concatenate([t[tau:] for t in trajectories], axis=0)
        else:
            # Single trajectory
            instant_data = trajectories[:-tau]
            lagged_data = trajectories[tau:]

        # Convert to tensors and move to device
        instant_data = torch.Tensor(instant_data).to(self.device)
        lagged_data = torch.Tensor(lagged_data).to(self.device)

        # VAMP-U forward pass
        u_out, v_out, C00_out, C11_out, C01_out, sigma_out, mu_out = self._vampu([
            instant_data, lagged_data
        ])

        # VAMP-S forward pass
        Ve_out, K_out, p_out, S_out = self._vamps([
            instant_data, lagged_data,
            u_out, v_out, C00_out, C11_out, C01_out, sigma_out
        ])

        # Extract and convert Koopman operator to numpy array
        koopman_op = K_out[0].detach().cpu().numpy()

        return koopman_op

    def its(self, lags: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Calculate implied timescales for a sequence of lag times.

        Parameters
        ----------
        lags : np.ndarray
            Array of lag times to analyze
        dt : float, default=1.0
            Time step between frames

        Returns
        -------
        np.ndarray
            Array of shape (n_output-1, n_lags) containing implied timescales
            for each lag time and eigenvalue
        """
        # Initialize output array (excluding stationary eigenvalue)
        n_timescales = self.n_output - 1
        implied_timescales = np.empty((n_timescales, len(lags)))

        # Calculate implied timescales for each lag time
        for i, lag in enumerate(lags):
            # Get Koopman operator for current lag
            koopman_op = self.estimate_koopman(lag)

            # Calculate eigenvalues of real part of Koopman operator
            eigenvals, _ = np.linalg.eig(np.real(koopman_op))

            # Sort eigenvalues by magnitude and exclude stationary eigenvalue
            sorted_eigenvals = np.sort(np.abs(eigenvals))[:-1]

            # Calculate implied timescales: -lag*dt/ln(λ)
            implied_timescales[:, i] = -lag * dt / np.log(sorted_eigenvals)

        # Reset lag time to default
        self.reset_lag()

        return implied_timescales

    def get_its(self, trajectory: np.ndarray, lags: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Calculate implied timescales (ITS) for a sequence of lag times.

        Parameters
        ----------
        trajectory : np.ndarray
            Input trajectory data
        lags : np.ndarray
            Array of lag times to analyze
        dt : float, default=1.0
            Time step between frames

        Returns
        -------
        np.ndarray
            Array of shape (n_output-1, n_lags) containing implied timescales
            for each lag time and eigenvalue
        """
        # Initialize output array (excluding stationary eigenvalue)
        n_timescales = self.n_output - 1
        implied_timescales = np.empty((n_timescales, len(lags)))

        # Calculate implied timescales for each lag time
        for i, lag in enumerate(lags):
            # Get Koopman operator for current lag
            koopman_op = self.estimate_koopman_op(trajectory, lag)

            # Calculate and sort eigenvalues
            eigenvals, _ = np.linalg.eig(np.real(koopman_op))
            sorted_eigenvals = np.sort(np.abs(np.real(eigenvals)))[:self.LAST]

            # Calculate implied timescales: -lag*dt/ln(λ)
            implied_timescales[:, i] = -lag * dt / np.log(sorted_eigenvals)

        # Reset lag time to default
        self.reset_lag()

        return implied_timescales

    def reset_lag(self) -> None:
        """
        Reset the model's lag time to its original network lag value.

        This method restores the lag time parameter to the value that was
        initially set during network configuration.
        """
        self.lag = self.network_lag

    def fit(self, data_loader: "torch.utils.data.DataLoader", n_epochs=1, validation_loader=None,
            train_score_callback: Callable[[int, "torch.Tensor"], None] = None,
            validation_score_callback: Callable[[int, "torch.Tensor"], None] = None,
            progress=None, **kwargs):
        r""" Fits a VampNet on data.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The data to use for training. Should yield a tuple of batches representing
            instantaneous and time-lagged samples.
        n_epochs : int, default=1
            The number of epochs (i.e., passes through the training data) to use for training.
        validation_loader : torch.utils.data.DataLoader, optional, default=None
            Validation data, should also be yielded as a two-element tuple.
        train_score_callback : callable, optional, default=None
            Callback function which is invoked after each batch and gets as arguments the current training step
            as well as the score (as torch Tensor).
        validation_score_callback : callable, optional, default=None
            Callback function for validation data. Is invoked after each epoch if validation data is given
            and the callback function is not None. Same as the train callback, this gets the 'step' as well as
            the score.
        progress : context manager, optional, default=None
            Progress bar (eg tqdm), defaults to None.
        **kwargs
            Optional keyword arguments for scikit-learn compatibility

        Returns
        -------
        self : VAMPNet
            Reference to self.
        """
        from deeptime.util.platform import handle_progress_bar
        progress = handle_progress_bar(progress)
        self._step = 0

        # and train
        with disable_TF32():
            for _ in progress(range(n_epochs), desc="VAMPNet epoch", total=n_epochs, leave=False):
                for batch_0, batch_t in data_loader:
                    self.partial_fit((batch_0.to(device=self.device), batch_t.to(device=self.device)),
                                     train_score_callback=train_score_callback)
                if validation_loader is not None:
                    with torch.no_grad():
                        scores = []
                        for val_batch in validation_loader:
                            scores.append(
                                self.validate((val_batch[0].to(device=self.device), val_batch[1].to(device=self.device)))
                            )
                        mean_score = torch.mean(torch.stack(scores))
                        self._validation_scores.append((self._step, mean_score.item()))
                        if validation_score_callback is not None:
                            validation_score_callback(self._step, mean_score)
        return self

    def fetch_model(self) -> VAMPNetModel:
        r""" Yields the current model. """
        from copy import deepcopy
        lobe = deepcopy(self.lobe)
        if self.lobe == self.lobe_timelagged:
            lobe_timelagged = lobe
        else:
            lobe_timelagged = deepcopy(self.lobe_timelagged)
        return VAMPNetModel(lobe, lobe_timelagged, dtype=self.dtype, device=self.device)
