import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union


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

        # Save last output for analysis
        self.last_output = None

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

    def forward(self, x: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, List[torch.Tensor]]) -> List[
        torch.Tensor]:
        """
        Forward pass of the VAMP-U module.

        Parameters
        ----------
        x : Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, List[torch.Tensor]]
            Either a tuple/list of (instantaneous, time-lagged) data or a single tensor

        Returns
        -------
        List[torch.Tensor]
            List containing [u, v, C00, C11, C01, sigma, mu]
        """
        # Get device from kernel
        device = self._u_kernel.device

        # Handle different input types
        if isinstance(x, (tuple, list)):
            if len(x) != 2:
                raise ValueError(f"Expected 2 tensors, got {len(x)}")
            chi_t, chi_tau = x[0], x[1]
        else:
            chi_t = chi_tau = x

        # Ensure inputs are tensors and on correct device
        if not isinstance(chi_t, torch.Tensor):
            chi_t = torch.tensor(chi_t)
        if not isinstance(chi_tau, torch.Tensor):
            chi_tau = torch.tensor(chi_tau)

        chi_t = chi_t.to(device)
        chi_tau = chi_tau.to(device)

        n_batch = chi_t.shape[0]
        norm = 1. / n_batch

        chi_tau_t = chi_tau.t()
        corr_tau = norm * torch.matmul(chi_tau_t, chi_tau)
        chi_mean = torch.mean(chi_tau, dim=0, keepdim=True)

        ac_u_kernel = self.activation(self._u_kernel)  # Already on correct device
        kernel_u = torch.unsqueeze(ac_u_kernel, dim=0)

        # Ensure intermediate computations are on correct device
        chi_mean = chi_mean.to(device)
        kernel_u = kernel_u.to(device)

        u = kernel_u / torch.sum(chi_mean * kernel_u, dim=1, keepdim=True)
        u_t = u.t()
        v = torch.matmul(corr_tau, u_t)
        mu = norm * torch.matmul(chi_tau, u_t)

        cmu_t = (chi_tau * mu).t()
        sigma = torch.matmul(cmu_t, chi_tau)
        gamma = chi_tau * torch.matmul(chi_tau, u_t)
        gamma_t = gamma.t()

        chi_t_t = chi_t.t()
        C00 = norm * torch.matmul(chi_t_t, chi_t)
        C11 = norm * torch.matmul(gamma_t, gamma)
        C01 = norm * torch.matmul(chi_t_t, gamma)

        # Ensure output tensors are on correct device
        out = [
                  self._tile(var, n_batch).to(device) for var in (u, v, C00, C11, C01, sigma)
              ] + [mu.to(device)]

        self.last_output = out
        return out

    def reset_weights(self):
        """Reset the u_kernel weights to initial values."""
        with torch.no_grad():
            self._u_kernel.data = (1. / self.M) * torch.ones_like(self._u_kernel)
