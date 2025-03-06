import pytest
import torch
import torch.nn as nn


def create_test_data(batch_size=32, num_atoms=100, num_neighbors=32):
    """Create test data for GraphVAMPNet."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create distances (first half) - float type
    distances = torch.abs(torch.randn(batch_size, num_atoms, num_neighbors)).to(device)

    # Create neighbor indices (second half) - explicitly as int64
    neighbors = torch.randint(0, num_atoms, (batch_size, num_atoms, num_neighbors),
                              dtype=torch.int64, device=device)

    # Create parameters
    params = {
        'num_atoms': num_atoms,
        'num_neighbors': num_neighbors,
        'n_classes': 4,
        'n_conv': 3,
        'h_a': 32,
        'h_g': 64,
        'dmin': 0.0,
        'dmax': 10.0,
        'step': 0.1,
        'conv_type': 'SchNet',
        'num_heads': 4,
        'residual': True,
        'use_backbone_atoms': False,
        'attention_pool': False
    }

    return distances, neighbors, params


def ensure_module_on_device(module, device):
    """Recursively ensure all submodules and parameters are on the specified device."""
    for child in module.children():
        if isinstance(child, nn.Module):
            child.to(device)
            ensure_module_on_device(child, device)

    for param in module.parameters(recurse=False):
        param.data = param.data.to(device)


def test_graph_vamp_net():
    """Test GraphVampNet functionality."""
    torch.manual_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    try:
        # Create test data
        distances, neighbors, params = create_test_data()

        # Import and initialize model
        from src.components.models.GraphVAMPNet import GraphVampNet
        model = GraphVampNet(**params)

        # Move model to device and ensure all submodules are on device
        model = model.to(device)
        ensure_module_on_device(model, device)

        # Print device information
        print("\nDevice Information:")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Distances device: {distances.device}")
        print(f"Neighbors device: {neighbors.device}")

        # Test forward pass
        with torch.no_grad():
            # Create input data
            data = torch.cat([
                distances,
                neighbors.to(dtype=torch.float32)  # Convert to float for concatenation
            ], dim=-1).to(device)

            # Get edge features
            edge_features = model.gauss.expand(distances).to(device)

            # Get initial embeddings
            atom_emb = model._get_initial_embeddings(neighbors.shape[0]).to(device)

            print("\nTensor Information:")
            print(f"Edge features shape: {edge_features.shape}, device: {edge_features.device}")
            print(f"Neighbor list shape: {neighbors.shape}, device: {neighbors.device}")
            print(f"Atom embeddings shape: {atom_emb.shape}, device: {atom_emb.device}")

            # Test each convolution layer
            for i, conv in enumerate(model.convs):
                print(f"\nTesting conv layer {i}:")
                # Ensure conv layer and all its components are on device
                conv.to(device)
                ensure_module_on_device(conv, device)

                # Verify filter generator is on device
                if hasattr(conv, 'cfconv'):
                    print(f"Filter generator device: {next(conv.cfconv.filter_generator.parameters()).device}")

                # Forward pass through conv layer
                conv_out, attn = conv(
                    features=atom_emb,
                    rbf_expansion=edge_features,
                    neighbor_list=neighbors  # Keep as int64
                )
                print(f"Conv output device: {conv_out.device}")
                atom_emb = conv_out if not model.residual else (atom_emb + conv_out)

            # Full forward pass
            output = model(data)
            print(f"\nFinal output shape: {output.shape}")
            print(f"Output device: {output.device}")

            # Test with embeddings and attention
            emb, attn = model(data, return_emb=True, return_attn=True)
            print(f"Embedding shape: {emb.shape}")
            print(f"Embedding device: {emb.device}")

        # Basic checks
        assert output.shape == (data.shape[0], params['n_classes']), "Wrong output shape"
        assert torch.allclose(output.sum(dim=1), torch.ones(data.shape[0], device=device)), \
            "Probabilities don't sum to 1"
        assert emb.shape == (data.shape[0], params['h_g']), "Wrong embedding shape"

        print("\nAll basic checks passed!")

    except Exception as e:
        import traceback
        print(f"Test failed with error: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        pytest.fail(f"Test failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__])
