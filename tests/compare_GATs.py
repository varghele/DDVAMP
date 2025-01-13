import torch
import torch.nn as nn
import pytest
from components.layers.gat import PyGGAT
from forked.RevGraphVAMP.layers import GATLayer


def test_gat_implementations():
    """Compare custom GATLayer with PyG's GATConv wrapper implementation."""

    # Set random seed for reproducibility
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test parameters
    batch_size = 4
    num_nodes = 10
    c_in = 16
    c_out = 32
    num_heads = 4

    # Initialize both implementations
    custom_gat = GATLayer(
        c_in=c_in,
        c_out=c_out,
        num_heads=num_heads,
        concat_heads=True,
        alpha=0.2
    ).to(device)

    pyg_gat = PyGGAT(
        c_in=c_in,
        c_out=c_out,
        num_heads=num_heads,
        concat_heads=True,
        alpha=0.2
    ).to(device)

    def print_tensor_stats(name, tensor):
        """Helper function to print tensor statistics"""
        print(f"\n{name} statistics:")
        print(f"Shape: {tensor.shape}")
        print(f"Mean: {tensor.mean().item():.6f}")
        print(f"Std: {tensor.std().item():.6f}")
        print(f"Min: {tensor.min().item():.6f}")
        print(f"Max: {tensor.max().item():.6f}")

    # Create test inputs
    node_feats = torch.randn(batch_size, num_nodes, c_in).to(device)

    # Create adjacency matrix with self-loops
    adj_matrix = torch.ones(batch_size, num_nodes, num_nodes).to(device)

    # Create edge_index for PyG (from adj_matrix)
    edge_index = adj_matrix[0].nonzero().t().to(device)

    # Test custom implementation
    print("\n=== Testing Custom GAT Implementation ===")
    custom_out, custom_attn = custom_gat(node_feats, adj_matrix, return_attn_probs=True)
    print_tensor_stats("Custom output", custom_out)
    print_tensor_stats("Custom attention", custom_attn)

    # Test PyG implementation
    print("\n=== Testing PyG GAT Implementation ===")
    pyg_out = pyg_gat(node_feats[0], edge_index)  # Use first batch for PyG
    print_tensor_stats("PyG output", pyg_out)

    # Detailed comparison tests
    print("\n=== Comparison Tests ===")

    # 1. Shape tests
    print("\nShape comparison:")
    print(f"Custom output: {custom_out.shape}")
    print(f"PyG output: {pyg_out.shape}")

    assert custom_out.shape == (batch_size, num_nodes, c_out), \
        f"Custom output shape {custom_out.shape} doesn't match expected {(batch_size, num_nodes, c_out)}"

    assert pyg_out.shape == (num_nodes, c_out), \
        f"PyG output shape {pyg_out.shape} doesn't match expected {(num_nodes, c_out)}"

    # 2. Value range tests
    print("\nValue range comparison:")
    custom_stats = {
        'mean': custom_out[0].mean().item(),
        'std': custom_out[0].std().item(),
        'min': custom_out[0].min().item(),
        'max': custom_out[0].max().item()
    }

    pyg_stats = {
        'mean': pyg_out.mean().item(),
        'std': pyg_out.std().item(),
        'min': pyg_out.min().item(),
        'max': pyg_out.max().item()
    }

    print("\nCustom implementation statistics:")
    for k, v in custom_stats.items():
        print(f"{k}: {v:.6f}")

    print("\nPyG implementation statistics:")
    for k, v in pyg_stats.items():
        print(f"{k}: {v:.6f}")

    # 3. Attention tests
    print("\nAttention weights test:")
    attention_sum = custom_attn.sum(dim=2)
    print(f"Attention sum mean: {attention_sum.mean():.6f}")
    print(f"Attention sum std: {attention_sum.std():.6f}")

    assert torch.allclose(attention_sum,
                          torch.ones_like(attention_sum),
                          rtol=1e-5), "Attention weights don't sum to 1"

    # 4. Gradient flow test
    print("\nGradient flow test:")

    # Test custom implementation gradients
    custom_loss = custom_out.mean()
    custom_loss.backward()

    custom_grads = {name: param.grad.abs().mean().item()
                    for name, param in custom_gat.named_parameters()}

    # Reset gradients and test PyG implementation
    pyg_gat.zero_grad()
    pyg_loss = pyg_out.mean()
    pyg_loss.backward()

    pyg_grads = {name: param.grad.abs().mean().item()
                 for name, param in pyg_gat.named_parameters()}

    print("\nCustom implementation gradients:")
    for name, grad in custom_grads.items():
        print(f"{name}: {grad:.6f}")

    print("\nPyG implementation gradients:")
    for name, grad in pyg_grads.items():
        print(f"{name}: {grad:.6f}")

    print("\nAll tests passed successfully!")


if __name__ == "__main__":
    pytest.main([__file__])
