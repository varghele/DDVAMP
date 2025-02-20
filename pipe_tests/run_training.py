import os
import sys
from datetime import datetime
import uuid

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from args.pipeline_args import buildParser as buildPipelineParser
from args.training_args import buildParser as buildTrainingParser
from train.train_main import run_training
import argparse
import torch


def infer_params_from_path(data_path):
    """Infer parameters from the data directory path."""
    try:
        dir_name = os.path.basename(os.path.normpath(data_path))
        parts = dir_name.split('_')

        return {
            'protein_name': parts[0],
            'num_neighbors': int(parts[1].replace('nbrs', '')),
            'ns': int(parts[2].replace('ns', ''))
        }
    except Exception as e:
        raise ValueError(f"Failed to infer parameters from path: {data_path}. "
                         f"Expected format: data/proteinname/interim/proteinname_numbernbrs_numberns/")


def get_hardcoded_args():
    # Create parsers
    pipeline_parser = buildPipelineParser()
    train_parser = buildTrainingParser()

    # Combine parsers
    all_args = argparse.ArgumentParser(parents=[pipeline_parser, train_parser],
                                       conflict_handler='resolve')

    # Define the protein name and parameters
    protein_name = "ab42"
    num_neighbors = 10
    ns = 1  # nanoseconds

    # Construct absolute data path using project_root
    data_path = os.path.abspath(os.path.join(
        project_root,  # Use project_root (DDVAMP directory)
        "data",
        protein_name,
        "interim",
        f"{protein_name}_{num_neighbors}nbrs_{ns}ns"
    ))

    print(f"\nConstructed absolute data path: {data_path}")

    # Set your hardcoded values here
    hardcoded_args = [
        # Pipeline args
        "--protein-name", protein_name,
        "--steps", "training",

        # Model Architecture
        "--num_classes", "6",
        "--n_conv", "4",
        "--h_a", "16",
        "--h_g", "8",
        "--hidden", "16",
        "--dropout", "0.4",

        # Distance Parameters
        "--dmin", "0.0",
        "--dmax", "3.0",
        "--step", "0.2",

        # Model Configuration
        "--conv_type", "SchNet",
        "--num_heads", "2",
        "--residual",
        "--attention_pool",
        "--atom_init", "normal",

        # Training Parameters
        "--lr", "0.01",
        "--tau", "1",
        "--batch_size", "32",
        "--val_frac", "0.2",
        "--epochs", "50",
        "--pre-train-epoch", "50",
        "--seed", "42",
        "--score_method", "VAMPCE",

        # System Configuration
        "--save_checkpoints",

        # Data Configuration
        "--data-path", data_path  # Using the absolute path
    ]

    # Parse arguments
    args = all_args.parse_args(hardcoded_args)

    # Print path information for debugging
    print(f"Project root: {project_root}")
    print(f"Data path: {args.data_path}")
    print(f"Path exists: {os.path.exists(args.data_path)}")

    return args


def flush_cuda_cache():
    """Clear CUDA cache if GPU is being used"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def setup_save_directory(args, inferred_params):
    """Set up the save directory structure and return the path."""
    # Generate unique model ID
    model_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

    # Create model directory path
    save_folder = os.path.join(
        project_root,
        'data',
        inferred_params['protein_name'],
        'models',
        f"{inferred_params['protein_name']}_"
        f"{inferred_params['num_neighbors']}nbrs_"
        f"{inferred_params['ns']}ns_"
        f"{args.num_classes}classes",
        model_id
    )

    # Create directories
    os.makedirs(save_folder, exist_ok=True)

    # Save arguments
    with open(os.path.join(save_folder, 'args.txt'), 'w') as f:
        f.write(str(args))
        f.write("\n\nInferred parameters:\n")
        f.write(str(inferred_params))

    return save_folder


def main():
    # Get arguments
    args = get_hardcoded_args()

    # Check if data directory exists
    if not os.path.exists(args.data_path):
        print(f"\nWARNING: Data directory not found: {args.data_path}")
        print("Available directories in parent folder:")
        parent_dir = os.path.dirname(args.data_path)
        if os.path.exists(parent_dir):
            print("\n".join(os.listdir(parent_dir)))
        else:
            print(f"Parent directory does not exist: {parent_dir}")
            print("Creating required directories...")
            os.makedirs(args.data_path, exist_ok=True)
            print(f"Created directory: {args.data_path}")
            print("Please ensure the required data files are present before running training.")
            return

    # Infer parameters from data path
    inferred_params = infer_params_from_path(args.data_path)

    # Set up save directory
    args.save_folder = setup_save_directory(args, inferred_params)

    # Set device
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    print("Starting training pipeline with arguments:")
    print("\nInferred Parameters:")
    print(f"Protein name: {inferred_params['protein_name']}")
    print(f"Number of neighbors: {inferred_params['num_neighbors']}")
    print(f"Nanoseconds: {inferred_params['ns']}")

    print("\nModel Architecture:")
    print(f"Number of classes: {args.num_classes}")
    print(f"Convolution layers: {args.n_conv}")
    print(f"Hidden dimensions: h_a={args.h_a}, h_g={args.h_g}, hidden={args.hidden}")

    print("\nModel Configuration:")
    print(f"Convolution type: {args.conv_type}")
    print(f"Using residual connections: {args.residual}")
    print(f"Using attention pooling: {args.attention_pool}")

    print("\nTraining Parameters:")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"Score method: {args.score_method}")

    print("\nSave Configuration:")
    print(f"Save folder: {args.save_folder}")
    print(f"Data path: {args.data_path}")

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    try:
        # Run training
        results = run_training(args)
        model = results['model']
        epochs_trained = results['epochs_trained']

        print("\nTraining completed successfully!")
        print(f"Model saved in: {args.save_folder}")
        print(f"Total epochs trained: {epochs_trained}")

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
