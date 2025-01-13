from components.layers.linear import LinearLayer
from components.layers.cfconv import CFConv
import torch
import torch.nn as nn


class GCNInteraction(nn.Module):
    """
    Graph Convolutional Network (GCN) interaction block.

    An interaction block consists of:
    1. Atom-wise linear layer without activation
    2. Continuous filter convolution with optional attention
    3. Atom-wise linear layer with activation
    4. Atom-wise linear layer without activation

    The output forms an additive residual connection with the original input features.

    Parameters
    ----------
    n_inputs : int
        Number of input features
    n_gaussians : int
        Number of gaussians used in the radial basis function
    n_filters : int
        Number of filters for the continuous filter convolution
    activation : nn.Module, optional
        Activation function for the atom-wise layers, default=nn.Tanh()
    use_attention : bool, optional
        Whether to use attention in the CFConv layer, default=True
    """

    def __init__(self, n_inputs: int, n_gaussians: int, n_filters: int,
                 activation: nn.Module = nn.Tanh(), use_attention: bool = True):
        super().__init__()

        # Initial atom-wise layer without activation
        self.initial_dense = nn.Sequential(
            *LinearLayer.create(
                d_in=n_inputs,
                d_out=n_filters,
                bias=False,
                activation=None
            )
        )

        # Continuous filter convolution
        self.cfconv = CFConv(
            n_gaussians=n_gaussians,
            n_filters=n_filters,
            activation=activation,
            use_attention=use_attention
        )

        # Output layers
        output_layers = (
            LinearLayer.create(
                d_in=n_filters,
                d_out=n_filters,
                bias=True,
                activation=activation
            ) +
            LinearLayer.create(
                d_in=n_filters,
                d_out=n_filters,
                bias=True
            )
        )
        self.output_dense = nn.Sequential(*output_layers)

    def forward(self, features: torch.Tensor, rbf_expansion: torch.Tensor,
                neighbor_list: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute interaction block forward pass.

        Parameters
        ----------
        features : torch.Tensor
            Input features from embedding or interaction layer [batch_size, n_atoms, n_features]
        rbf_expansion : torch.Tensor
            Radial basis function expansion of distances [batch_size, n_atoms, n_neighbors, n_gaussians]
        neighbor_list : torch.Tensor
            Indices of neighboring atoms [batch_size, n_atoms, n_neighbors]

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - output_features: Output features [batch_size, n_atoms, n_filters]
            - attention: Attention weights from CFConv layer (None if use_attention=False)
        """
        # Initial dense layer
        init_features = self.initial_dense(features)

        # Continuous filter convolution
        conv_output, attention = self.cfconv(init_features, rbf_expansion, neighbor_list)

        # Output dense layers
        output_features = self.output_dense(conv_output)

        return output_features, attention
