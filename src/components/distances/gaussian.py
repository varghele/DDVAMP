import torch
from typing import Optional
from torch import Tensor


class GaussianDistance(object):
    """
    Expands distances using Gaussian basis functions.

    This class creates a set of Gaussian basis functions to expand distance values
    into a higher dimensional space, useful for creating distance-based features
    in molecular graphs.
    """

    def __init__(
            self,
            dmin: float,
            dmax: float,
            step: float,
            var: Optional[float] = None,
            device: Optional[torch.device] = None
    ):
        """
        Initialize Gaussian distance expansion.

        Parameters
        ----------
        dmin : float
            Minimum distance between atoms to be considered
        dmax : float
            Maximum distance between atoms to be considered
        step : float
            Step size for the gaussian filter centers
        var : Optional[float]
            Variance of the Gaussian functions. If None, uses step size
        device : Optional[torch.device]
            Device to place the filters on. If None, uses CPU

        Raises
        ------
        AssertionError
            If dmin >= dmax or range is smaller than step
        """
        # Validate inputs
        assert dmin < dmax, f"dmin ({dmin}) must be less than dmax ({dmax})"
        assert dmax - dmin > step, f"Range ({dmax - dmin}) must be greater than step ({step})"

        # Set device
        self.device = device if device is not None else torch.device('cpu')

        # Create filter centers
        self.filter = torch.arange(dmin, dmax + step, step, device=self.device)
        self.num_features = len(self.filter)

        # Set variance
        self.var = var if var is not None else step

    def expand(self, distances: Tensor) -> Tensor:
        """
        Apply Gaussian distance filter to distances.

        Parameters
        ----------
        distances : torch.Tensor
            Distance matrix of shape [batch_size, num_atoms, num_neighbors]

        Returns
        -------
        torch.Tensor
            Expanded distances of shape [batch_size, num_atoms, num_neighbors, num_features]
        """
        # Move inputs to correct device
        distances = distances.to(self.device)

        # Expand dimensions for broadcasting
        distances = distances.unsqueeze(-1)

        # Calculate Gaussian expansion
        return torch.exp(
            -(distances - self.filter) ** 2 / self.var ** 2
        )

    def __repr__(self) -> str:
        """String representation of the GaussianDistance object."""
        return (f"GaussianDistance(dmin={self.filter[0]:.2f}, "
                f"dmax={self.filter[-1]:.2f}, "
                f"step={self.filter[1] - self.filter[0]:.2f}, "
                f"var={self.var:.2f}, "
                f"num_features={self.num_features})")

