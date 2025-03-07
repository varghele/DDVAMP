# args/preparation_args.py

import argparse
import os


def buildParser():
    """Create and return an argument parser for trajectory preparation parameters."""
    parser = argparse.ArgumentParser(description='Trajectory preparation parameters')

    # Input/Output Configuration
    io_group = parser.add_argument_group('Input/Output Configuration')
    io_group.add_argument('--topology', type=str, required=True,
                         help='Path to topology file (.pdb or .gro)')
    io_group.add_argument('--traj-folder', type=str, required=True,
                         help='Path to trajectory folder')

    # Processing Parameters
    proc_group = parser.add_argument_group('Processing Parameters')
    proc_group.add_argument('--num-neighbors', type=int, default=5,
                           help='Number of neighbors for trajectory processing')
    proc_group.add_argument('--stride', type=int, default=40,
                           help='Stride for trajectory processing')
    proc_group.add_argument('--chunk-size', type=int, default=5000,
                           help='Chunk size for processing')

    return parser


def validate_args(args):
    """Validate the parsed arguments."""
    # Check if topology file exists and has valid extension
    if not os.path.exists(args.topology):
        raise ValueError(f"Topology file not found: {args.topology}")
    if not args.topology.endswith(('.pdb', '.gro')):
        raise ValueError("Topology file must be either .pdb or .gro")

    # Check if trajectory folder exists
    if not os.path.exists(args.traj_folder):
        raise ValueError(f"Trajectory folder not found: {args.traj_folder}")

    # Validate numerical parameters
    if args.num_neighbors < 1:
        raise ValueError("num_neighbors must be positive")
    if args.stride < 1:
        raise ValueError("stride must be positive")
    if args.chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    return args


if __name__ == "__main__":
    parser = buildParser()
    args = parser.parse_args()
    args = validate_args(args)
