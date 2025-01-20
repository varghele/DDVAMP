import torch
import torch.nn as nn
from typing import List, Tuple, Optional


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
