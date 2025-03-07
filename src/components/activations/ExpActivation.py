import torch
import torch.nn as nn


class ExpActivation(nn.Module):
    """Exponential activation function as a proper nn.Module."""
    def forward(self, x):
        return torch.exp(x)
