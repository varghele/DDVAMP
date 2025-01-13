import torch
import torch.nn as nn
import numpy as np
from components.layers.gcn_interaction import GCNInteraction


def test_gcn_interaction():
    """
    Test the GCNInteraction implementation for correctness.
    """
    # Set random seed and device
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test parameters
    batch_size = 4
    n_atoms = 10
    n_neighbors = 8
    n_gaussians = 16
    n_filters = 32

    # Create test inputs on device
    features = torch.randn(batch_size, n_atoms, n_filters).to(device)
    rbf_expansion = torch.randn(batch_size, n_atoms, n_neighbors, n_gaussians).to(device)
    neighbor_list = torch.randint(0, n_atoms, (batch_size, n_atoms, n_neighbors)).to(device)

    # Initialize model on device
    model = GCNInteraction(
        n_inputs=n_filters,
        n_gaussians=n_gaussians,
        n_filters=n_filters,
        activation=nn.Tanh()
    ).to(device)

    def print_tensor_stats(name, tensor):
        """Helper function to print tensor statistics"""
        print(f"\n{name} statistics:")
        print(f"Shape: {tensor.shape}")
        print(f"Mean: {tensor.mean().item():.4f}")
        print(f"Std: {tensor.std().item():.4f}")
        print(f"Min: {tensor.min().item():.4f}")
        print(f"Max: {tensor.max().item():.4f}")

    # Test 1: Forward pass and shape check
    print("\n=== Test 1: Forward Pass ===")
    output, attention = model(features, rbf_expansion, neighbor_list)

    print_tensor_stats("Input features", features)
    print_tensor_stats("Output", output)
    print_tensor_stats("Attention", attention)

    assert output.shape == features.shape, f"Output shape {output.shape} doesn't match input shape {features.shape}"
    assert attention.shape == (batch_size, n_atoms, n_neighbors), f"Incorrect attention shape: {attention.shape}"

    # Test 2: Layer-by-layer check
    print("\n=== Test 2: Layer-by-layer Check ===")

    # Initial dense layer
    initial_output = model.initial_dense(features)
    print_tensor_stats("Initial dense output", initial_output)
    assert not torch.isnan(initial_output).any(), "NaN values in initial dense output"

    # CFConv layer
    conv_output, conv_attention = model.cfconv(initial_output, rbf_expansion, neighbor_list)
    print_tensor_stats("CFConv output", conv_output)
    print_tensor_stats("CFConv attention", conv_attention)
    assert not torch.isnan(conv_output).any(), "NaN values in CFConv output"

    # Output dense layer
    final_output = model.output_dense(conv_output)
    print_tensor_stats("Final dense output", final_output)
    assert not torch.isnan(final_output).any(), "NaN values in final output"

    # Test 3: Attention weights check
    print("\n=== Test 3: Attention Weights Check ===")
    attention_sum = attention.sum(dim=-1)
    print_tensor_stats("Attention sum", attention_sum)

    # Check if attention weights sum to approximately 1
    assert torch.allclose(attention_sum,
                          torch.ones_like(attention_sum),
                          rtol=1e-5, atol=1e-5), "Attention weights don't sum to 1"

    # Test 4: Gradient flow check
    print("\n=== Test 4: Gradient Flow Check ===")
    loss = output.mean()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        grad_norm = param.grad.norm().item()
        print(f"{name} gradient norm: {grad_norm:.4f}")
        assert not torch.isnan(param.grad).any(), f"NaN gradients in {name}"

    # Test 5: Value range check
    print("\n=== Test 5: Value Range Check ===")
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"
    assert output.abs().mean() < 1.0, "Output values seem too large"

    print("\nAll tests passed successfully!")


if __name__ == "__main__":
    test_gcn_interaction()
