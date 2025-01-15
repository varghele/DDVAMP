import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from components.layers.gat import PyGGAT
from components.layers.gcn_interaction import GCNInteraction
from components.distances.gaussian import GaussianDistance
from args.args import buildParser


class GraphVampNet(nn.Module):
    """
    Graph Neural Network for VAMP analysis of molecular dynamics.

    Supports multiple graph convolution types:
    - GraphConvLayer
    - NeighborMultiHeadAttention
    - GATLayer
    - SchNet
    """

    def __init__(self, args=None):
        """
        Initialize GraphVampNet with multiple graph convolution options.

        Parameters
        ----------
        args : argparse.Namespace, optional
            Namespace containing model parameters. If None, default parameters will be used.
        """
        super(GraphVampNet, self).__init__()  # Fixed super() call

        # Initialize args if not provided
        if args is None:
            parser = buildParser()
            args = parser.parse_args([])

        # Store parameters from args
        self.num_atoms = getattr(args, 'num_atoms', 100)  # Add default values
        self.num_neighbors = getattr(args, 'num_neighbors', 5)
        self.n_classes = getattr(args, 'num_classes', 5)
        self.n_conv = getattr(args, 'n_conv', 3)
        self.h_a = getattr(args, 'h_a', 64)
        self.h_g = getattr(args, 'h_g', 32)
        self.residual = getattr(args, 'residual', True)
        self.use_backbone_atoms = getattr(args, 'use_backbone_atoms', False)
        self.attention_pool = getattr(args, 'attention_pool', True)
        self.conv_type = getattr(args, 'conv_type', 'SchNet')
        self.num_heads = getattr(args, 'num_heads', 4)
        self.dmin = getattr(args, 'dmin', 0.0)
        self.dmax = getattr(args, 'dmax', 10.0)
        self.step = getattr(args, 'step', 0.1)

        # Initialize Gaussian distance basis
        self.gauss = GaussianDistance(self.dmin, self.dmax, self.step)
        self.h_b = self.gauss.num_features

        # Initialize embeddings based on configuration
        if getattr(args, 'use_pre_trained', False) and hasattr(args, 'pre_trained_weights_file'):
            self._init_pretrained_embeddings(args.pre_trained_weights_file)
        elif hasattr(args, 'seq_file') and args.seq_file:
            self._init_sequence_embeddings(args.seq_file)
        else:
            self._init_random_embeddings('normal')  # Default to 'normal' initialization

        # Initialize convolution layers based on type
        if self.conv_type == 'SchNet':
            self.convs = nn.ModuleList([
                GCNInteraction(
                    n_inputs=self.h_a,
                    n_gaussians=self.h_b,
                    n_filters=self.h_a,
                    activation=nn.Tanh(),
                    use_attention=True
                ) for _ in range(self.n_conv)
            ])
        elif self.conv_type == 'GATLayer':
            self.convs = nn.ModuleList([
                PyGGAT(
                    c_in=self.h_a,
                    c_out=self.h_a,
                    num_heads=self.num_heads,
                    concat_heads=True,
                    alpha=0.2
                ) for _ in range(self.n_conv)
            ])
        else:
            raise ValueError(f"Unsupported convolution type: {self.conv_type}")

        # Initialize output layers
        if self.h_g is not None:
            self.amino_emb = nn.Linear(self.h_a, self.h_g)
            self.fc_classes = nn.Linear(self.h_g, self.n_classes)
        else:
            self.fc_classes = nn.Linear(self.h_a, self.n_classes)

        # Initialize pooling attention if needed
        if self.attention_pool:
            self.attn_pool = PyGGAT(
                c_in=self.h_a,
                c_out=self.h_a,
                num_heads=1,
                concat_heads=True,
                alpha=0.2
            )

    def _init_random_embeddings(self, init_type='normal'):
        """Initialize random atom embeddings.

        Parameters
        ----------
        init_type : str
            Type of initialization ('normal' or 'uniform')

        Raises
        ------
        ValueError
            If init_type is not 'normal' or 'uniform'
        """
        if init_type == 'normal':
            self.atom_embeddings = nn.Parameter(torch.randn(self.num_atoms, self.h_a))
        elif init_type == 'uniform':
            self.atom_embeddings = nn.Parameter(torch.rand(self.num_atoms, self.h_a))
        else:
            raise ValueError(f"Unknown initialization type: {init_type}")

    def _init_pretrained_embeddings(self, weights_file):
        """Initialize embeddings from pretrained weights file.

        Parameters
        ----------
        weights_file : str
            Path to the pretrained weights file

        Raises
        ------
        ValueError
            If atom_embeddings not found in weights file
        """
        weights = torch.load(weights_file)
        if 'atom_embeddings' in weights:
            self.atom_embeddings = nn.Parameter(weights['atom_embeddings'])
        else:
            raise ValueError(f"No atom embeddings found in weights file: {weights_file}")

    def _init_sequence_embeddings(self, seq_file):
        """Initialize embeddings from sequence file.

        Parameters
        ----------
        seq_file : str
            Path to the sequence file
        """
        # Load sequence data from file
        with open(seq_file, 'r') as f:
            sequences = f.readlines()

        # Process sequences and create embeddings
        # This is a placeholder - implement according to your sequence format
        embeddings = torch.randn(self.num_atoms, self.h_a)
        self.atom_embeddings = nn.Parameter(embeddings)

    def _get_initial_embeddings(self, batch_size):
        """Get initial atom embeddings for batch.

        Parameters
        ----------
        batch_size : int
            Size of the batch

        Returns
        -------
        torch.Tensor
            Initial embeddings expanded for the batch
        """
        return self.atom_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

    def pooling(self, x):
        """Global pooling operation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to pool [batch_size, num_atoms, features]

        Returns
        -------
        torch.Tensor
            Pooled tensor [batch_size, features]
        """
        if self.attention_pool:
            return self.attn_pool(x)

        # Average over the atoms dimension (dim=1)
        return torch.mean(x, dim=1)

    def _init_conv_layers(self, conv_type: str, num_heads: int) -> nn.ModuleList:
        """
        Initialize convolution layers based on type.

        Parameters
        ----------
        conv_type : str
            Type of convolution layer ('SchNet' or 'GATLayer')
        num_heads : int
            Number of attention heads for GAT layers

        Returns
        -------
        nn.ModuleList
            List of convolution layers

        Raises
        ------
        ValueError
            If conv_type is not supported
        """
        if conv_type == 'SchNet':
            return nn.ModuleList([
                GCNInteraction(
                    n_inputs=self.h_a,
                    n_gaussians=self.h_b,
                    n_filters=self.h_a,
                    activation=nn.Tanh(),
                    use_attention=True
                ) for _ in range(self.n_conv)
            ])

        elif conv_type == 'GATLayer':
            return nn.ModuleList([
                PyGGAT(
                    c_in=self.h_a,
                    c_out=self.h_a,  # Keep output dim same as input
                    num_heads=num_heads,
                    concat_heads=True,
                    alpha=0.2
                ) for _ in range(self.n_conv)
            ])

        """elif conv_type == 'GraphConvLayer':
            return nn.ModuleList([
                GraphConvLayer(
                    self.h_a,
                    self.h_b
                ) for _ in range(self.n_conv)
            ])

        elif conv_type == 'NeighborMultiHeadAttention':
            return nn.ModuleList([
                NeighborMultiHeadAttention(
                    self.h_a,
                    self.h_b,
                    self.num_heads
                ) for _ in range(self.n_conv)
            ])"""

        raise ValueError(f"Unsupported convolution type: {conv_type}")

    def _init_output_layers(self):
        """Initialize output layers."""
        if self.h_g is not None:
            self.amino_emb = nn.Linear(self.h_a, self.h_g)
            self.fc_classes = nn.Linear(self.h_g, self.n_classes)
        else:
            self.fc_classes = nn.Linear(self.h_a, self.n_classes)

    def forward(
            self,
            data: torch.Tensor,
            return_emb: bool = False,
            return_attn: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the network.

        Parameters
        ----------
        data : torch.Tensor
            Input data containing distances and neighbor lists
            Shape: [batch_size, num_atoms, 2*num_neighbors]
        return_emb : bool
            Whether to return embeddings
        return_attn : bool
            Whether to return attention weights

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Class probabilities [batch_size, n_classes] or (embeddings, attention weights)
        """
        # Split input data
        M = data.shape[-1]
        nbr_adj_dist = data[..., :M // 2]
        nbr_adj_list = data[..., M // 2:].to(torch.int64)

        # Get edge features
        edge_features = self.gauss.expand(nbr_adj_dist)

        # Get initial atom embeddings
        atom_emb = self._get_initial_embeddings(nbr_adj_list.shape[0])

        # Apply convolutions
        attn_weights = None
        for conv in self.convs:
            conv_out, attn = conv(
                features=atom_emb,
                rbf_expansion=edge_features,
                neighbor_list=nbr_adj_list
            )
            if self.residual:
                atom_emb = atom_emb + conv_out
            else:
                atom_emb = conv_out
            if return_attn:
                attn_weights = attn

        # Apply activation
        atom_emb = F.relu(atom_emb)

        # Pool if using backbone atoms
        if self.use_backbone_atoms:
            atom_emb = self.pool_amino(atom_emb)

        # Global pooling
        graph_emb = self.pooling(atom_emb)  # Now returns [batch_size, features]

        # Final embeddings
        if self.h_g is not None:
            graph_emb = self.amino_emb(graph_emb)

        # Get class probabilities
        logits = self.fc_classes(graph_emb)
        probs = F.softmax(logits, dim=-1)

        if return_emb:
            return graph_emb, attn_weights
        return probs
