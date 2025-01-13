import torch
import torch.nn as nn
import torch.nn.functional as F
from components.layers.linear import LinearLayer


class CFConv(nn.Module):
    """
    Continuous Filter Convolution layer as described in SchNet (Schütt et al., 2018) with optional attention mechanism.
    Attention mechanism and hyperbolic tangent come from GraphVAMPNet (Ghorbani et al., 2022).
    See also RevGraphVAMP: (Huang et al. 2024)

    This layer performs the following operations:
    1. Generates continuous filters from radial basis function expansions
    2. Applies these filters to neighboring atom features
    3. Aggregates the filtered features either through attention or summation

    Parameters
    ----------
    n_gaussians : int
        Number of Gaussian functions used in the radial basis expansion
    n_filters : int
        Number of filters to be generated and output features
    cutoff : float, optional
        Interaction cutoff distance, by default 5.0
    activation : nn.Module, optional
        Activation function for the filter generator network, by default nn.Tanh()
    use_attention : bool, optional
        Whether to use attention mechanism for feature aggregation, by default True

    Attributes
    ----------
    filter_generator : nn.Sequential
        Neural network that generates filters from RBF expansions
    nbr_filter : nn.Parameter
        Learnable parameter for attention mechanism (only if use_attention=True)
    use_attention : bool
        Flag indicating whether attention mechanism is used
    cutoff : float
        Cutoff distance for interactions
    """

    def __init__(self, n_gaussians, n_filters, cutoff=5.0, activation=nn.Tanh(), use_attention=True):
        super().__init__()

        # Create filter generator using LinearLayer
        filter_layers = (
            LinearLayer.create(
                d_in=n_gaussians,
                d_out=n_filters,
                bias=True,
                activation=activation,
                weight_init='xavier'
            ) +
            LinearLayer.create(
                d_in=n_filters,
                d_out=n_filters,
                bias=True,
                weight_init='xavier'
            )
        )
        self.filter_generator = nn.Sequential(*filter_layers)

        # Initialize attention components
        self.use_attention = use_attention
        if use_attention:
            self.nbr_filter = nn.Parameter(torch.Tensor(n_filters, 1))
            nn.init.xavier_uniform_(self.nbr_filter, gain=1.414)

        self.cutoff = cutoff

    def forward(self, features, rbf_expansion, neighbor_list):
        """
        Forward pass of the continuous filter convolution layer.

        Parameters
        ----------
        features : torch.Tensor
            Input atom features of shape [batch_size, n_atoms, n_features]
        rbf_expansion : torch.Tensor
            Radial basis function expansion of interatomic distances
            of shape [batch_size, n_atoms, n_neighbors, n_gaussians]
        neighbor_list : torch.Tensor
            Indices of neighboring atoms of shape [batch_size, n_atoms, n_neighbors]

        Returns
        -------
        tuple
            - aggregated_features : torch.Tensor
                Convolved and aggregated features of shape [batch_size, n_atoms, n_filters]
            - nbr_filter : torch.Tensor or None
                Attention weights if use_attention=True, None otherwise
        """
        # Ensure all inputs are on the same device
        device = features.device
        rbf_expansion = rbf_expansion.to(device)
        neighbor_list = neighbor_list.to(device)

        batch_size, n_atoms, n_neighbors = neighbor_list.size()

        # Generate continuous filters from RBF expansion
        conv_filter = self.filter_generator(rbf_expansion)
        conv_filter = conv_filter.to(device)

        # Ensure neighbor_list is int64 and properly shaped for gathering
        neighbor_list = neighbor_list.to(torch.int64)
        neighbor_list = neighbor_list.reshape(-1, n_atoms * n_neighbors, 1)
        neighbor_list = neighbor_list.expand(-1, -1, features.size(2))

        # Gather features of neighboring atoms
        neighbor_features = torch.gather(features, 1, neighbor_list)
        neighbor_features = neighbor_features.reshape(batch_size, n_atoms, n_neighbors, -1)

        # Apply continuous filters to neighbor features
        conv_features = neighbor_features * conv_filter

        # Aggregate features using either attention or summation
        if self.use_attention:
            # Move nbr_filter to correct device
            nbr_filter = self.nbr_filter.to(device)
            # Compute attention weights and apply them
            attention_weights = torch.matmul(conv_features, nbr_filter).squeeze(-1)
            attention_weights = F.softmax(attention_weights, dim=-1)
            aggregated_features = torch.einsum('bij,bijc->bic', attention_weights, conv_features)
            return aggregated_features, attention_weights
        else:
            # Simple summation aggregation
            aggregated_features = conv_features.sum(dim=2)
            return aggregated_features, None
