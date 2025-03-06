import torch
import torch.nn as nn
from src.components.layers.cfconv import CFConv
from forked.RevGraphVAMP.layers import ContinuousFilterConv


def test_cfconv_implementations():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test parameters
    batch_size = 32
    n_atoms = 100
    n_neighbors = 32
    n_gaussians = 25
    n_filters = 64

    # Create random input data - all on the same device
    features = torch.randn(batch_size, n_atoms, n_filters).to(device)
    rbf_expansion = torch.randn(batch_size, n_atoms, n_neighbors, n_gaussians).to(device)
    neighbor_list = torch.randint(0, n_atoms, (batch_size, n_atoms, n_neighbors)).to(device)

    # Initialize both implementations and move to device
    cfconv_1 = CFConv(n_gaussians=n_gaussians,
                      n_filters=n_filters,
                      cutoff=5.0,
                      activation=nn.Tanh(),
                      use_attention=True).to(device)

    cfconv_2 = ContinuousFilterConv(n_gaussians=n_gaussians,
                                    n_filters=n_filters,
                                    activation=nn.Tanh()).to(device)

    # Forward pass
    output_1, attn_1 = cfconv_1(features, rbf_expansion, neighbor_list)
    output_2, attn_2 = cfconv_2(features, rbf_expansion, neighbor_list)

    # Move tensors to CPU for comparison
    output_1 = output_1.cpu()
    output_2 = output_2.cpu()
    attn_1 = attn_1.cpu()
    attn_2 = attn_2.cpu()

    # Tests
    print(f"Output shapes: 1={output_1.shape}, 2={output_2.shape}")
    print(f"Attention shapes: 1={attn_1.shape}, 2={attn_2.shape}")

    # 1. Check output shapes
    assert output_1.shape == output_2.shape, \
        f"Output shapes differ: {output_1.shape} vs {output_2.shape}"

    # 2. Check attention shapes
    assert attn_1.shape == attn_2.shape, \
        f"Attention shapes differ: {attn_1.shape} vs {attn_2.shape}"

    # 3. Check if outputs are in reasonable ranges
    print(f"Output means: 1={output_1.mean():.4f}, 2={output_2.mean():.4f}")
    print(f"Output stds: 1={output_1.std():.4f}, 2={output_2.std():.4f}")

    # Check if means are within a reasonable range (-1 to 1 since we use tanh)
    assert -1 <= output_1.mean() <= 1, f"Output 1 mean {output_1.mean()} outside reasonable range"
    assert -1 <= output_2.mean() <= 1, f"Output 2 mean {output_2.mean()} outside reasonable range"

    # 4. Check if attention weights sum to approximately 1 for each atom
    attn_1_sum = attn_1.sum(dim=-1)
    attn_2_sum = attn_2.sum(dim=-1)
    print(f"Attention sum ranges: 1=[{attn_1_sum.min():.4f}, {attn_1_sum.max():.4f}], "
          f"2=[{attn_2_sum.min():.4f}, {attn_2_sum.max():.4f}]")

    assert torch.allclose(attn_1_sum, torch.ones_like(attn_1_sum), rtol=1e-5), \
        "First implementation attention weights don't sum to 1"
    assert torch.allclose(attn_2_sum, torch.ones_like(attn_2_sum), rtol=1e-5), \
        "Second implementation attention weights don't sum to 1"

    print("All tests passed successfully!")


if __name__ == "__main__":
    test_cfconv_implementations()
