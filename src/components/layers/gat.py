import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


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