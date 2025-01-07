import torch
import torch.nn as nn
import torch.nn.functional as F


class SchNet(nn.Module):
    def __init__(self, n_atom_basis=128, n_filters=128, n_interactions=3, cutoff=5.0):
        super(SchNet, self).__init__()

        # Atom embedding
        self.atom_embedding = nn.Embedding(100, n_atom_basis)  # Max atomic number set to 100

        # Interaction blocks
        self.interactions = nn.ModuleList([
            InteractionBlock(n_atom_basis, n_filters, cutoff)
            for _ in range(n_interactions)
        ])

        # Output network
        self.output_network = OutputNetwork(n_atom_basis)

    def forward(self, Z, positions, batch_idx):
        x = self.atom_embedding(Z)

        for interaction in self.interactions:
            x = interaction(x, positions, batch_idx)

        return self.output_network(x)


class InteractionBlock(nn.Module):
    def __init__(self, n_atom_basis, n_filters, cutoff):
        super(InteractionBlock, self).__init__()

        self.filter_network = FilterNetwork(n_filters, cutoff)
        self.cfconv = CFConv(n_atom_basis, n_filters)
        self.dense = nn.Sequential(
            nn.Linear(n_atom_basis, n_atom_basis),
            nn.ReLU(),
            nn.Linear(n_atom_basis, n_atom_basis)
        )

    def forward(self, x, positions, batch_idx):
        v = self.cfconv(x, positions, self.filter_network, batch_idx)
        v = self.dense(v)
        return x + v


class FilterNetwork(nn.Module):
    def __init__(self, n_filters, cutoff):
        super(FilterNetwork, self).__init__()
        self.n_filters = n_filters
        self.cutoff = cutoff

        self.filter_net = nn.Sequential(
            nn.Linear(1, n_filters),
            nn.ReLU(),
            nn.Linear(n_filters, n_filters)
        )

    def forward(self, distances):
        # Apply cutoff
        distances = distances.unsqueeze(-1)
        filters = self.filter_net(distances)
        return filters * self.cutoff_fn(distances)

    def cutoff_fn(self, distances):
        return torch.cos(distances * math.pi / (2 * self.cutoff)) * (distances <= self.cutoff)


class CFConv(nn.Module):
    def __init__(self, n_atom_basis, n_filters):
        super(CFConv, self).__init__()
        self.dense = nn.Linear(n_filters, n_atom_basis)

    def forward(self, x, positions, filter_network, batch_idx):
        # Compute pairwise distances
        distances = compute_distances(positions, batch_idx)

        # Get filters
        filters = filter_network(distances)

        # Continuous convolution
        y = self.dense(filters)
        return y


class OutputNetwork(nn.Module):
    def __init__(self, n_atom_basis):
        super(OutputNetwork, self).__init__()
        self.output_net = nn.Sequential(
            nn.Linear(n_atom_basis, n_atom_basis // 2),
            nn.ReLU(),
            nn.Linear(n_atom_basis // 2, 1)
        )

    def forward(self, x):
        return self.output_net(x)
