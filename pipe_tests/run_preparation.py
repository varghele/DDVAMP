import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from args.pipeline_args import buildParser as buildPipelineParser
from args.preparation_args import buildParser as buildPrepParser
from utils.preparation import run_pipeline
import argparse
from utils.traj_utils import infer_timestep

def get_hardcoded_args():
    # Create parsers
    pipeline_parser = buildPipelineParser()
    prep_parser = buildPrepParser()

    # Combine parsers
    all_args = argparse.ArgumentParser(parents=[pipeline_parser, prep_parser],
                                       conflict_handler='resolve')

    # Set your hardcoded values here
    hardcoded_args = [
        # Pipeline args
        #"--protein-name", "ab42",
        "--protein-name", "ATR",
        "--steps", "preparation",

        # Preparation args
        #"--topology", "../forked/RevGraphVAMP/trajectories/red/topol.gro",
        #"--traj-folder", "../forked/RevGraphVAMP/trajectories/red/",
        #"--topology", "../datasets/ab42/trajectories/trajectories/red/topol.gro",
        #"--traj-folder", "../datasets/ab42/trajectories/trajectories/red/",
        "--topology", "../datasets/ATR/prot.gro",
        "--traj-folder", "../datasets/ATR/",
        "--num-neighbors", "20",
        "--stride", "10",
        "--chunk-size", "5000"
    ]

    # Parse arguments
    args = all_args.parse_args(hardcoded_args)
    return args


def main():
    # Get arguments
    args = get_hardcoded_args()

    print("Starting preparation pipeline with arguments:")
    print(f"Protein name: {args.protein_name}")
    print(f"Topology file: {args.topology}")
    print(f"Trajectory folder: {args.traj_folder}")
    print(f"Number of neighbors: {args.num_neighbors}")
    print(f"Stride: {args.stride}")
    print(f"Chunk size: {args.chunk_size}")

    # Infer timestep from trajectories
    #timestep = infer_timestep(args.traj_folder, args.topology)

    # Run preparation
    run_pipeline(args)

    # Get the output directory from results
    #ns = int(args.stride * timestep)
    subdir_name = f"{args.protein_name}_{args.num_neighbors}nbrs_{0}ns"
    output_dir = os.path.join(args.interim_dir, subdir_name)

    print("\nPreparation completed successfully!")
    print(f"Results saved in: {output_dir}")
    print(f"Files created:")
    print(f"- {os.path.join(output_dir, 'datainfo_min.npy')}")
    print(f"- {os.path.join(output_dir, 'dist_min.npy')}")
    print(f"- {os.path.join(output_dir, 'inds_min.npy')}")


if __name__ == "__main__":
    main()

