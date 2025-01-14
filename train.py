import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from deeptime.util.data import TrajectoryDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
import warnings
import pickle

from args import buildParser
from components.models.GraphVAMPNet import GraphVampNet
from components.models.RevVAMPNet import RevVAMPNet
from components.scores.vamps import VAMPS
from components.scores.vampu import VAMPU
from utils_vamp import EarlyStopping, unflatten, plot_its, plot_ck_test, get_its, get_ck_test

# Ignore deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


def setup_device():
    """Set up and return the appropriate device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    return device


def setup_directories(args):
    """Create necessary directories and save metadata."""
    if not os.path.exists(args.save_folder):
        print('Creating folder for saving checkpoints')
        os.makedirs(args.save_folder)

    # Save arguments
    with open(os.path.join(args.save_folder, 'args.txt'), 'w') as f:
        f.write(str(args))

    # Save metadata
    meta_file = os.path.join(args.save_folder, 'metadata.pkl')
    pickle.dump({'args': args}, open(meta_file, 'wb'))


def load_data(args):
    """Load and prepare data."""
    # Load data files
    data_info = np.load(os.path.join(args.data_path, "datainfo.npy"), allow_pickle=True).item()
    dists = np.load(os.path.join(args.data_path, "dist.npy"))
    inds = np.load(os.path.join(args.data_path, "inds.npy"))

    # Process trajectories
    traj_length = data_info['length']
    dist_sp = [r for r in unflatten(dists, traj_length)]
    inds_sp = [r for r in unflatten(inds, traj_length)]

    # Combine data
    data = []
    for i in range(len(dist_sp)):
        mydists = torch.from_numpy(dist_sp[i])
        myinds = torch.from_numpy(inds_sp[i])
        data.append(torch.cat((mydists, myinds), axis=-1))

    return data


def prepare_dataloaders(data, args):
    """Prepare data loaders for training and validation."""
    dataset = TrajectoryDataset.from_trajectories(lagtime=args.tau, data=data)

    # Split into train and validation
    n_val = int(len(dataset) * args.val_frac)
    train_data, val_data = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])

    # Create data loaders
    loader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    loader_train_all = DataLoader(train_data, batch_size=len(dataset), shuffle=True)

    return loader_train, loader_val, loader_train_all


def initialize_model(args, device):
    """Initialize the RevVAMPNet model."""
    lobe = GraphVampNet()
    lobe_timelagged = deepcopy(lobe).to(device=device)
    lobe = lobe.to(device)

    vlu = VAMPU(args.num_classes, activation=torch.exp)
    vls = VAMPS(args.num_classes, activation=torch.exp, renorm=True)

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
    # Parse arguments
    args = buildParser().parse_args()

    # Setup
    device = setup_device()
    setup_directories(args)

    # Load and prepare data
    data = load_data(args)
    loader_train, loader_val, loader_train_all = prepare_dataloaders(data, args)

    # Initialize model
    vampnet = initialize_model(args, device)

    if args.train:
        # Train model
        from training import train
        model, all_train_epoch = train(
            vampnet=vampnet,
            train_loader=loader_train,
            validation_loader=loader_val,
            loader_train_all=loader_train_all,
            args=args
        )

        # Save training results
        save_training_results(vampnet, args, all_train_epoch)
    else:
        # Load pretrained model
        load_pretrained_model(vampnet, args)

    # Perform analysis
    perform_analysis(vampnet, data, args)


if __name__ == "__main__":
    main()
