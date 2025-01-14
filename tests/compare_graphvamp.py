import pytest
import torch
import torch.nn as nn
import os
import sys
from typing import Tuple

# Add the forked repository to the Python path
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FORKED_PATH = os.path.join(REPO_PATH, 'forked', 'RevGraphVAMP')
sys.path.insert(0, FORKED_PATH)


def create_test_data(batch_size=32, num_atoms=100, num_neighbors=32) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Create test data for both GraphVAMPNet implementations."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"cuda is {'is' if torch.cuda.is_available() else 'is not'} available")

    # Create distances and neighbors
    distances = torch.abs(torch.randn(batch_size, num_atoms, num_neighbors)).to(device)
    neighbors = torch.randint(0, num_atoms, (batch_size, num_atoms, num_neighbors),
                              dtype=torch.int64, device=device)

    # Parameters matching both implementations
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
        'attention_pool': False,
        'seq_file': None,
        'dont_pool_backbone': False
    }

    return distances, neighbors, params


def test_graph_vamp_net_comparison():
    """Test both GraphVAMPNet implementations and compare their outputs."""
    torch.manual_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    try:
        # Set up command line arguments first
        from args import buildParser
        parser = buildParser()
        # Parse with empty list to avoid reading actual command line args
        args = parser.parse_args([])

        # Update args with our test parameters
        args.num_atoms = 100
        args.num_neighbors = 32
        args.num_classes = 4
        args.n_conv = 3
        args.h_a = 32
        args.h_g = 64
        args.dmin = 0.0
        args.dmax = 10.0
        args.step = 0.1
        args.conv_type = 'SchNet'
        args.num_heads = 4
        args.residual = True
        args.use_backbone_atoms = False
        args.attention_pool = False
        args.seq_file = None
        args.dont_pool_backbone = False

        # Add args to module namespace
        import builtins
        builtins.args = args

        # Create test data
        distances, neighbors, params = create_test_data()
        data = torch.cat([distances, neighbors.float()], dim=-1).to(device)

        # Import models
        from components.models.GraphVAMPNet import GraphVampNet as ModifiedGraphVampNet
        from model import GraphVampNet as OriginalGraphVampNet

        # Initialize models
        modified_model = ModifiedGraphVampNet(**params).to(device)
        original_model = OriginalGraphVampNet().to(device)

        print("\nModel Information:")
        print(f"Modified model device: {next(modified_model.parameters()).device}")
        print(f"Original model device: {next(original_model.parameters()).device}")

        # Copy weights for fair comparison
        with torch.no_grad():
            for (name1, param1), (name2, param2) in zip(
                    modified_model.named_parameters(),
                    original_model.named_parameters()
            ):
                param2.data.copy_(param1.data)

        # Test forward passes
        with torch.no_grad():
            # Regular forward pass
            modified_output = modified_model(data)
            original_output = original_model(data)

            print("\nOutput Information:")
            print(f"Modified output shape: {modified_output.shape}")
            print(f"Original output shape: {original_output.shape}")

            # Test with embeddings and attention
            modified_emb, modified_attn = modified_model(data, return_emb=True, return_attn=True)
            original_emb, original_attn = original_model(data, return_emb=True, return_attn=True)

            print("\nEmbedding Information:")
            print(f"Modified embedding shape: {modified_emb.shape}")
            print(f"Original embedding shape: {original_emb.shape}")

        # Compare outputs
        assert modified_output.shape == original_output.shape, "Output shapes don't match"
        assert torch.allclose(modified_output, original_output, atol=1e-5), "Outputs don't match"

        # Compare embeddings
        assert modified_emb.shape == original_emb.shape, "Embedding shapes don't match"
        assert torch.allclose(modified_emb, original_emb, atol=1e-5), "Embeddings don't match"

        print("\nAll comparison checks passed!")

    except Exception as e:
        import traceback
        print(f"Test failed with error: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        pytest.fail(f"Test failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([])  # Don't pass file argument to avoid argparse error
