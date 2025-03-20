#!/usr/bin/env python3
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.args.pipeline_args import buildParser as buildPipelineParser
from src.args.preparation_args import buildParser as buildPrepParser
from src.utils.preparation import run_pipeline
import argparse


def parse_args():
    """Parse command line arguments if provided, otherwise use hardcoded defaults."""
    # Create parsers
    pipeline_parser = buildPipelineParser()
    prep_parser = buildPrepParser()

    # Combine parsers
    parser = argparse.ArgumentParser(
        parents=[pipeline_parser, prep_parser],
        conflict_handler='resolve',
        description='Data preparation script with command line arguments'
    )

    # First check if any args were provided on the command line
    if len(sys.argv) > 1:
        # Use command line arguments
        args = parser.parse_args()
        args.using_hardcoded = False
        print("Using command line arguments")
    else:
        # Define hardcoded defaults
        print("No command line arguments provided. Using hardcoded defaults.")

        # Set your hardcoded values here
        hardcoded_args = [
            # Pipeline args
            "--protein-name", "ATR",
            "--steps", "preparation",

            # Preparation args
            #"--topology", "datasets/ab42/trajectories/trajectories/red/topol.gro",
            #"--traj-folder", "datasets/ab42/trajectories/trajectories/red/",
            #"--topology", "datasets/traj_revgraphvamp_org/trajectories/red/topol.gro",
            #"--traj-folder", "datasets/traj_revgraphvamp_org/trajectories/red/",
            #"--topology", "datasets/ab42/trajectories/trajectories/red/topol.gro",
            #"--traj-folder", "datasets/ab42/trajectories/trajectories/red/",
            #"--traj-folder", "datasets/TRP/DESRES-Trajectory_2JOF-0-protein/2JOF-0-protein/",
            #"--topology", "datasets/TRP/DESRES-Trajectory_2JOF-0-protein/2JOF-0-protein/2JOF-0-protein.mae",
            #"--traj-folder", "datasets/alanine_dipeptide",
            #"--topology", "datasets/alanine_dipeptide/ala-dipeptide-handmade.pdb",
            "--traj-folder", "datasets/ATR/",
            "--topology", "datasets/ATR/prot.gro",
            "--num-neighbors", "10",
            "--stride", "10",
            "--chunk-size", "1000"
        ]

        # Parse hardcoded arguments
        args = parser.parse_args(hardcoded_args)
        args.using_hardcoded = True

    return args


def infer_timestep(traj_folder, topology):
    """Infer the timestep from the trajectory files (in nanoseconds)."""
    # This is a placeholder. In a real implementation, you would examine the trajectory files
    # to determine the timestep or calculate it based on stride and total frames
    return 0.0005  # Default value (0.5 ps = 0.0005 ns)


def main():
    # Get arguments
    args = parse_args()

    print("\n" + "=" * 50)
    print("PREPARATION CONFIGURATION")
    print("=" * 50)
    print(f"Protein name: {args.protein_name}")
    print(f"Topology file: {args.topology}")
    print(f"Trajectory folder: {args.traj_folder}")
    print(f"Number of neighbors: {args.num_neighbors}")
    print(f"Stride: {args.stride}")
    print(f"Chunk size: {args.chunk_size}")
    print("=" * 50 + "\n")

    # Check if trajectory folder exists
    if not os.path.exists(args.traj_folder):
        print(f"ERROR: Trajectory folder not found: {args.traj_folder}")
        return

    # Check if topology file exists
    if not os.path.exists(args.topology):
        print(f"ERROR: Topology file not found: {args.topology}")
        return

    # Calculate approximate simulation time in nanoseconds
    try:
        timestep = infer_timestep(args.traj_folder, args.topology)
        ns = args.stride * timestep
        print(f"Estimated simulation time: {ns} ns (with stride {args.stride})")
    except Exception as e:
        print(f"Warning: Could not infer timestep: {str(e)}")
        ns = 1.0  # Default value
        print(f"Using default simulation time: {ns} ns")

    try:
        # Run preparation
        run_pipeline(args)

        # Construct output directory path
        subdir_name = f"{args.protein_name}_{args.num_neighbors}nbrs_{ns}ns"
        output_dir = os.path.join(
            project_root,
            "data",
            args.protein_name,
            "interim",
            subdir_name
        )

        print("\nPreparation completed successfully!")
        print(f"Results saved in: {output_dir}")
        print(f"Files created:")
        print(f"- {os.path.join(output_dir, 'datainfo_min.npy')}")
        print(f"- {os.path.join(output_dir, 'dist_min.npy')}")
        print(f"- {os.path.join(output_dir, 'inds_min.npy')}")

        # Print next steps
        print("\nNext steps:")
        print(f"Run training using: python run_train.py --data-path {output_dir} ...")

    except Exception as e:
        print(f"\nError during preparation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
