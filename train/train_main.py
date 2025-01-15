import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from args import buildParser
from utils.unflatten import unflatten
from utils.count_parameters import count_parameters

from deeptime.util.data import TrajectoryDataset
import numpy as np
from components.models.GraphVAMPNet import GraphVampNet
from components.models.vamps import VAMPS
from components.models.vampu import VAMPU
from components.models.RevVAMPNet import RevVAMPNet

def setup_device():
    """Set up and return the appropriate device (CPU/CUDA)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    return device


def load_data(file_path):
    """Load and prepare data from files."""
    # Define file paths
    data_info_file = f"{file_path}red_5nbrs_1ns_datainfo_min.npy"
    dist_file = f"{file_path}red_5nbrs_1ns_dist_min.npy"
    nbr_data_file = f"{file_path}red_5nbrs_1ns_inds_min.npy"

    # Load data
    data_info = np.load(data_info_file, allow_pickle=True).item()
    traj_length = data_info['length']
    dists1 = np.load(dist_file)
    inds1 = np.load(nbr_data_file)

    return traj_length, dists1, inds1


def prepare_dataset(dists1, inds1, traj_length, args):
    """Prepare dataset from distance and index data."""
    # Unflatten arrays
    dist_sp = [r for r in unflatten(dists1, traj_length)]
    inds_sp = [r for r in unflatten(inds1, traj_length)]

    # Combine distances and indices
    data = []
    for i in range(len(dist_sp)):
        mydists1 = torch.from_numpy(dist_sp[i])
        myinds1 = torch.from_numpy(inds_sp[i])
        data.append(torch.cat((mydists1, myinds1), axis=-1))

    # Create dataset
    dataset = TrajectoryDataset.from_trajectories(lagtime=args.tau, data=data)

    return dataset


def create_dataloaders(dataset, args):
    """Create training and validation dataloaders."""
    # Split dataset
    n_val = int(len(dataset) * args.val_frac)
    train_data, val_data = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])

    # Create dataloaders
    loader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    loader_train_all = DataLoader(train_data, batch_size=len(dataset), shuffle=True)

    return loader_train, loader_val, loader_train_all


def initialize_model(args, device):
    """Initialize the RevVAMPNet model and its components."""
    # Initialize base network with proper arguments
    lobe = GraphVampNet()
    lobe_timelagged = deepcopy(lobe).to(device=device)
    lobe = lobe.to(device)

    print('Number of parameters', count_parameters(lobe))

    # Initialize VAMP networks
    vlu = VAMPU(args.num_classes, activation=torch.exp)
    vls = VAMPS(args.num_classes, activation=torch.exp, renorm=True)

    # Create RevVAMPNet
    vampnet = RevVAMPNet(
        lobe=lobe,
        lobe_timelagged=lobe_timelagged,
        learning_rate=args.lr,
        device=device,
        optimizer='Adam',
        score_method=args.score_method,
        vampu=vlu,
        vamps=vls
    )

    return vampnet


def main():
    # Setup
    device = setup_device()
    args = buildParser().parse_args()

    # Set file path
    file_path = '../datasets/traj_revgraphvamp_org/intermediate/'

    # Load data
    traj_length, dists1, inds1 = load_data(file_path)

    # Prepare dataset
    dataset = prepare_dataset(dists1, inds1, traj_length, args)

    # Create dataloaders
    loader_train, loader_val, loader_train_all = create_dataloaders(dataset, args)

    # Initialize model
    model = initialize_model(args, device)



    print(f"Model score method: {args.score_method}")
    print(f"Dataset type: {type(loader_train)}")


if __name__ == "__main__":
    main()
