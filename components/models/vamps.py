import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union
import numpy as np


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

        self.last_output = None

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

    def forward(self, x: Union[List[torch.Tensor], Tuple[torch.Tensor, ...], torch.Tensor]) -> Union[
        List[torch.Tensor], torch.Tensor]:
        """
        Forward pass of the VAMP-S module.

        Parameters
        ----------
        x : Union[List[torch.Tensor], Tuple[torch.Tensor, ...], torch.Tensor]
            Either a single tensor for direct transformation,
            or input tensors [v, C00, C11, C01, sigma] or
            [chi_t, chi_tau, u, v, C00, C11, C01, sigma]

        Returns
        -------
        Union[List[torch.Tensor], torch.Tensor]
            Either transformed single tensor or
            List containing [vamp_e_tile, K_tile, probs/zeros, S_tile]
        """
        # Get device
        device = self._s_kernel.device

        # Handle single input case (direct transformation)
        if isinstance(x, (torch.Tensor, np.ndarray)):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.to(device)
            kernel_w = self.activation(self._s_kernel)
            # Reshape kernel for broadcasting
            kernel_w = kernel_w.view(1, -1)  # Make it (1, n_features)
            if len(x.shape) == 1:
                x = x.view(-1, 1)  # Make it (n_samples, 1)
            return x * kernel_w.to(device)

        # Handle tuple/list input case
        if isinstance(x, (list, tuple)):
            if len(x) == 5:
                v, C00, C11, C01, sigma = x
                chi_t = chi_tau = u = None
            elif len(x) == 8:
                chi_t, chi_tau, u, v, C00, C11, C01, sigma = x
                u = u[0]
            else:
                raise ValueError(f"Expected list/tuple of length 5 or 8, got length {len(x)}")
        else:
            raise ValueError(f"Unsupported input type: {type(x)}")

        # Original functionality for list/tuple inputs
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
        kernel_w = self.activation(self._s_kernel).to(device)
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
            probs = torch.zeros((n_batch, self.M), device=device)

        # Compute final outputs
        K = S @ sigma
        vamp_e = S_t @ C00 @ S @ C11 - 2 * S_t @ C01

        # Tile outputs for batch dimension
        vamp_e_tile = torch.tile(torch.unsqueeze(vamp_e, dim=0), [n_batch, 1, 1])
        K_tile = torch.tile(torch.unsqueeze(K, dim=0), [n_batch, 1, 1])
        S_tile = torch.tile(torch.unsqueeze(S, dim=0), [n_batch, 1, 1])

        out = [vamp_e_tile, K_tile, probs, S_tile]
        self.last_output = out

        return out

