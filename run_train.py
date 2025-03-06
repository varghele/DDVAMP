#!/usr/bin/env python3
import os
import sys
from datetime import datetime
import uuid
import torch
import argparse

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from args.pipeline_args import buildParser as buildPipelineParser
from args.training_args import buildParser as buildTrainingParser
from args.preparation_args import buildParser as buildPreparationParser
from train.train_main import run_training


def parse_args():
    """Parse command line arguments and combine with defaults"""
    # Create base parsers
    pipeline_parser = buildPipelineParser()
    train_parser = buildTrainingParser()
    preparation_parser = buildPreparationParser()

    # Combine parsers
    parser = argparse.ArgumentParser(
        parents=[pipeline_parser, train_parser, preparation_parser],
        conflict_handler='resolve',
        description='Training script with command line arguments'
    )

    # Parse arguments
    args = parser.parse_args()

    # Set fixed data path
    args.data_path = os.path.abspath(os.path.join(project_root, "data", "processed"))

    return args


def main():
    # Get arguments
    args = parse_args()

    # Print configuration
    print("\nConfiguration:")
    print(f"Protein name: {args.protein_name}")
    print(f"Topology file: {args.topology}")
    print(f"Trajectory folder: {args.traj_folder}")
    print(f"Number of neighbors: {args.num_neighbors}")
    print(f"Nanoseconds: {args.ns}")
    print(f"Data path: {args.data_path}")

    # Create data directory if it doesn't exist
    if not os.path.exists(args.data_path):
        print(f"\nCreating data directory: {args.data_path}")
        os.makedirs(args.data_path, exist_ok=True)

    # Set device
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {args.device}")

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        # Run training
        results = run_training(args)
        print("\nTraining completed successfully!")
        print(f"Model saved in: {args.save_folder}")
        print(f"Total epochs trained: {results['epochs_trained']}")

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
