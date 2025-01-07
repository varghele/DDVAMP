import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.nn import GATConv
import time
import numpy as np
from models.layers import GATLayer



# Your custom GAT implementation
class CustomGAT(nn.Module):
    def __init__(self, c_in, c_out, num_heads=1, concat_heads=True, alpha=0.2):
        super().__init__()
        self.gat = GATLayer(c_in, c_out, num_heads, concat_heads, alpha)

    def forward(self, x, adj_matrix):
        return self.gat(x, adj_matrix)[0]  # Only return node features, not attention


# PyG GAT implementation wrapper
class PyGGAT(nn.Module):
    def __init__(self, c_in, c_out, num_heads=1, concat_heads=True, alpha=0.2):
        super().__init__()
        out_channels = c_out // num_heads if concat_heads else c_out
        self.gat = GATConv(c_in, out_channels,
                           heads=num_heads,
                           concat=concat_heads,
                           negative_slope=alpha,
                           add_self_loops=True)  # Match custom implementation

        # Reset parameters to match custom initialization
        with torch.no_grad():
            nn.init.xavier_uniform_(self.gat.lin.weight, gain=1.414)

    def forward(self, x, edge_index):
        # Add normalization to match custom implementation
        out = self.gat(x, edge_index)
        return out / torch.sqrt(torch.tensor(self.gat.heads))  # Scale output


def compare_gat_implementations():
    # Test parameters
    batch_size = 32
    num_nodes = 100
    c_in = 64
    c_out = 128
    num_heads = 4

    # Generate random input data
    x = torch.randn(batch_size, num_nodes, c_in)
    adj_matrix = torch.zeros(batch_size, num_nodes, num_nodes)

    # Create random sparse connections (50% density)
    for b in range(batch_size):
        random_edges = torch.rand(num_nodes, num_nodes) < 0.5
        #random_edges = torch.ones(num_nodes, num_nodes)
        adj_matrix[b] = random_edges

    # Convert to edge_index format for PyG
    edge_index = adj_matrix[0].nonzero().t()

    # Initialize models
    custom_gat = CustomGAT(c_in, c_out, num_heads)
    pyg_gat = PyGGAT(c_in, c_out, num_heads)

    # Timing and memory comparison
    def measure_performance(model, inputs, name):
        start_time = time.time()
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()

        if name == "Custom":
            output = model(inputs[0], inputs[1])
        else:
            output = model(inputs[0].reshape(-1, c_in), inputs[1])
            output = output.reshape(batch_size, num_nodes, -1)

        end_memory = torch.cuda.memory_allocated()
        end_time = time.time()

        return {
            'time': end_time - start_time,
            'memory': (end_memory - start_memory) / 1024 ** 2,  # MB
            'output_shape': output.shape,
            'output_mean': output.mean().item(),
            'output_std': output.std().item()
        }

    # Run comparisons
    custom_inputs = (x, adj_matrix)
    pyg_inputs = (x, edge_index)

    custom_results = measure_performance(custom_gat, custom_inputs, "Custom")
    pyg_results = measure_performance(pyg_gat, pyg_inputs, "PyG")

    # Print results
    print("=== Performance Comparison ===")
    print("\nCustom GAT:")
    print(f"Time: {custom_results['time']:.4f} seconds")
    print(f"Memory: {custom_results['memory']:.2f} MB")
    print(f"Output shape: {custom_results['output_shape']}")
    print(f"Output stats: mean={custom_results['output_mean']:.4f}, std={custom_results['output_std']:.4f}")

    print("\nPyG GAT:")
    print(f"Time: {pyg_results['time']:.4f} seconds")
    print(f"Memory: {pyg_results['memory']:.2f} MB")
    print(f"Output shape: {pyg_results['output_shape']}")
    print(f"Output stats: mean={pyg_results['output_mean']:.4f}, std={pyg_results['output_std']:.4f}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run comparison
    compare_gat_implementations()
