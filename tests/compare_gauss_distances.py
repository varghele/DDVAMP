import torch
import pytest
import numpy as np
from typing import Tuple
from components.distances.gaussian import GaussianDistance
from forked.RevGraphVAMP.layers import GaussianDistance as OriginalGaussianDistance


def create_test_data() -> Tuple[torch.Tensor, dict]:
    """Create test data for comparing GaussianDistance implementations."""
    # Test parameters
    batch_size = 4
    num_atoms = 10
    num_neighbors = 8

    # Create random distance matrix
    distances = torch.rand(batch_size, num_atoms, num_neighbors) * 10.0

    # Test parameters
    params = {
        'dmin': 0.0,
        'dmax': 10.0,
        'step': 0.5,
        'var': 0.5
    }

    return distances, params


def test_gaussian_distances():
    """Test and compare both GaussianDistance implementations."""
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create test data
    distances, params = create_test_data()

    # Initialize both implementations
    orig_gaussian = OriginalGaussianDistance(**params)
    new_gaussian = GaussianDistance(**params)

    # Get outputs (ensure both are on CPU)
    orig_output = orig_gaussian.expand(distances).cpu()
    new_output = new_gaussian.expand(distances).cpu()

    # Basic shape check
    assert orig_output.shape == new_output.shape, "Outputs have different shapes"

    # Compare values
    diff = torch.abs(orig_output - new_output)
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()

    # Print statistics
    print(f"\nShape: {orig_output.shape}")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Max difference: {max_diff:.6f}")

    # Assertions
    assert mean_diff < 1e-6, f"Mean difference too large: {mean_diff}"
    assert max_diff < 1e-6, f"Max difference too large: {max_diff}"

    print("All tests passed successfully!")


if __name__ == "__main__":
    pytest.main([__file__])
