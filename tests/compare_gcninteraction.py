import torch
import torch.nn as nn
import pytest
from src.components.layers import GCNInteraction
from forked.RevGraphVAMP.layers import InteractionBlock


def print_tensor_stats(name: str, tensor: torch.Tensor) -> None:
    """Print detailed statistics for a tensor."""
    print(f"\n{name} statistics:")
    print(f"Shape: {tensor.shape}")
    print(f"Mean: {tensor.mean().item():.6f}")
    print(f"Std: {tensor.std().item():.6f}")
    print(f"Min: {tensor.min().item():.6f}")
    print(f"Max: {tensor.max().item():.6f}")
    print(f"Abs Mean: {tensor.abs().mean().item():.6f}")


def compare_layer_outputs(name: str, output_new: torch.Tensor, output_orig: torch.Tensor,
                          rtol: float = 1e-3, atol: float = 1e-3) -> None:
    """Compare outputs from both implementations."""
    print(f"\n=== {name} Comparison ===")
    print_tensor_stats("New implementation", output_new)
    print_tensor_stats("Original implementation", output_orig)

    diff = (output_new - output_orig).abs()
    print(f"\nDifference statistics:")
    print(f"Mean diff: {diff.mean().item():.6f}")
    print(f"Max diff: {diff.max().item():.6f}")
    print(f"Relative diff: {(diff / (output_orig.abs() + 1e-7)).mean().item():.6f}")


def test_interaction_implementations():
    """Test and compare both interaction implementations."""
    # Set random seed and device
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test parameters
    batch_size = 32
    n_atoms = 100
    n_neighbors = 32
    n_gaussians = 25
    n_filters = 64

    # Create random input data
    features = torch.randn(batch_size, n_atoms, n_filters).to(device)
    rbf_expansion = torch.randn(batch_size, n_atoms, n_neighbors, n_gaussians).to(device)
    neighbor_list = torch.randint(0, n_atoms, (batch_size, n_atoms, n_neighbors)).to(device)

    # Initialize both implementations
    interaction_new = GCNInteraction(
        n_inputs=n_filters,
        n_gaussians=n_gaussians,
        n_filters=n_filters,
        activation=nn.Tanh()
    ).to(device)

    interaction_orig = InteractionBlock(
        n_inputs=n_filters,
        n_gaussians=n_gaussians,
        n_filters=n_filters,
        activation=nn.Tanh()
    ).to(device)

    # Forward pass
    print("\n=== Forward Pass ===")
    output_new, attn_new = interaction_new(features, rbf_expansion, neighbor_list)
    output_orig, attn_orig = interaction_orig(features, rbf_expansion, neighbor_list)

    # Move tensors to CPU for comparison
    output_new = output_new.cpu()
    output_orig = output_orig.cpu()
    attn_new = attn_new.cpu() if attn_new is not None else None
    attn_orig = attn_orig.cpu() if attn_orig is not None else None

    # Compare outputs
    compare_layer_outputs("Output", output_new, output_orig)

    # Compare attention if available
    if attn_new is not None and attn_orig is not None:
        compare_layer_outputs("Attention", attn_new, attn_orig)

    # Tests with detailed error messages
    try:
        # 1. Shape tests
        assert output_new.shape == output_orig.shape, \
            f"Output shapes differ: new={output_new.shape}, original={output_orig.shape}"

        if attn_new is not None and attn_orig is not None:
            assert attn_new.shape == attn_orig.shape, \
                f"Attention shapes differ: new={attn_new.shape}, original={attn_orig.shape}"

        # 2. Value range tests
        assert -1 <= output_new.mean() <= 1, \
            f"New implementation mean {output_new.mean():.6f} outside [-1, 1]"
        assert -1 <= output_orig.mean() <= 1, \
            f"Original implementation mean {output_orig.mean():.6f} outside [-1, 1]"

        # 3. Attention tests
        if attn_new is not None and attn_orig is not None:
            for attn, name in [(attn_new, "new"), (attn_orig, "original")]:
                attn_sum = attn.sum(dim=-1)
                assert torch.allclose(attn_sum, torch.ones_like(attn_sum), rtol=1e-5), \
                    f"{name} implementation attention weights don't sum to 1 " \
                    f"(mean={attn_sum.mean():.6f}, std={attn_sum.std():.6f})"

        # 4. Layer component comparison
        print("\n=== Layer Components ===")
        print("New implementation:")
        print(interaction_new)
        print("\nOriginal implementation:")
        print(interaction_orig)

        print("\nAll tests passed successfully!")

    except AssertionError as e:
        print(f"\nTest failed: {str(e)}")
        raise


if __name__ == "__main__":
    pytest.main([__file__])
