import torch
import numpy as np
from tqdm import tqdm
import pyemma as pe
import mdtraj as md
from typing import List, Tuple, Union, Dict
from glob import glob
import os


class TrajectoryProcessor:
    def __init__(self, args):
        self.args = args
        self.num_residues = None
        self.data = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def prepare_topology(self) -> str:
        """Convert topology to PDB if needed and count residues"""
        if self.args.topology.endswith('.pdb'):
            pdb_file = self.args.topology
        elif self.args.topology.endswith('.gro'):
            pdb_file = self.args.topology.replace('.gro', '.pdb')
            if not os.path.exists(pdb_file):
                try:
                    traj = md.load(self.args.topology)
                    traj.save_pdb(pdb_file)
                except Exception as e:
                    raise RuntimeError(f"Failed to convert GRO to PDB: {str(e)}")
        else:
            raise ValueError("Topology file must be either .pdb or .gro")

        structure = md.load(pdb_file)
        self.num_residues = structure.topology.n_residues
        print(f"Number of residues detected: {self.num_residues}")
        return pdb_file

    def load_trajectories(self) -> Dict:
        """Load and process trajectory files"""
        pdb_file = self.prepare_topology()

        # Check if trajectory folder exists
        if not os.path.exists(self.args.traj_folder):
            raise ValueError(f"Trajectory folder not found: {self.args.traj_folder}")

        traj_pattern = os.path.join(self.args.traj_folder, "r?", "traj*")
        traj_files = sorted(glob(traj_pattern))

        if not traj_files:
            raise ValueError(f"No trajectory files found matching pattern: {traj_pattern}")

        k = self.args.protein_name
        residue, pair = self._process_single_simulation(k, pdb_file, traj_files)
        inpcon = self._load_trajectory(traj_files, pdb_file)

        self.data = {
            'residue_name': {k: residue},
            'pair_list': {k: pair},
            'inpcon': {k: inpcon}
        }
        return self.data

    def _process_single_simulation(self, sim_name: str, pdb_file: str, traj_files: List[str]) -> Tuple[Dict, List]:
        """Process a single simulation"""
        residue = {}
        pair = []
        feat = pe.coordinates.featurizer(pdb_file)
        feat.add_residue_mindist()

        for key in feat.describe():
            name = key.split(' ')
            if len(name) >= 5:  # Ensure we have enough elements
                ri, rj = name[2], name[4]
                try:
                    i, j = int(ri[3:]), int(rj[3:])
                    residue[i] = ri
                    residue[j] = rj
                    pair.append((i, j))
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping invalid residue pair in {key}: {str(e)}")

        return residue, pair

    def _load_trajectory(self, traj_files: List[str], pdb_file: str):
        """Load trajectory files"""
        valid_trajs = [f for f in traj_files if f.endswith(('.xtc', '.trr', '.dcd', '.nc', '.h5'))]
        if not valid_trajs:
            raise ValueError("No valid trajectory files found")

        feat = pe.coordinates.featurizer(pdb_file)
        feat.add_residue_mindist()
        return pe.coordinates.source(valid_trajs, feat)

    def get_neighbors(self, coords: Union[List[np.ndarray], np.ndarray], pair_list: List[Tuple[int, int]]) -> Tuple[
        np.ndarray, np.ndarray]:
        """Calculate nearest neighbors"""
        if isinstance(coords, (list, tuple)):
            return self._process_list_trajectories(coords, pair_list)
        return self._process_single_trajectory(coords, pair_list)

    def _process_single_trajectory(self, coords, pair_list):
        """Process a single trajectory with correct shape and distance handling"""
        n_frames = len(coords)
        n_residues = self.num_residues
        n_neighbors = self.args.num_neighbors

        # Initialize arrays with correct shape
        dists = []
        inds = []

        for frame in tqdm(coords, desc="Processing frames"):
            # Initialize distance matrix with large values (like reference)
            mut_dist = np.ones((n_residues, n_residues)) * 100.0  # Changed from 300.0 to 100.0

            # Fill the distance matrix (note the -1 in indices like reference)
            for idx, d in enumerate(frame):
                if idx >= len(pair_list):
                    break
                res_i, res_j = pair_list[idx]
                mut_dist[res_i - 1][res_j - 1] = d  # Added -1 to match reference
                mut_dist[res_j - 1][res_i - 1] = d  # Added -1 to match reference

            frame_dists = []
            frame_inds = []

            # Process each residue
            for dd in mut_dist:
                states_order = np.argsort(dd)
                neighbors = states_order[:n_neighbors]
                frame_dists.append(list(dd[neighbors]))  # Convert to list like reference
                frame_inds.append(list(neighbors))  # Convert to list like reference

            dists.append(frame_dists)
            inds.append(frame_inds)

        return np.array(dists), np.array(inds)

    def _process_list_trajectories(self, coords_list, pair_list):
        """Process a list of trajectories"""
        all_dists, all_inds = [], []
        for coords in coords_list:
            dists, inds = self._process_single_trajectory(coords, pair_list)
            all_dists.append(dists)
            all_inds.append(inds)
        return np.concatenate(all_dists), np.concatenate(all_inds)

    def process_and_save(self):
        """Process trajectories and save results"""
        k = self.args.protein_name

        try:
            # Get trajectory information
            lengths = [self.data['inpcon'][k].trajectory_lengths()]
            nframes = self.data['inpcon'][k].trajectory_lengths().sum()

            # Process coordinates
            coords = self.data['inpcon'][k].get_output()
            dists, inds = self.get_neighbors(coords, self.data['pair_list'][k])

            # Prepare output path
            ns = int(self.args.stride * 0.2)
            output_prefix = os.path.join(self.args.interim_dir, f"{k}_{self.args.num_neighbors}nbrs_{ns}ns_")

            # Save results
            os.makedirs(self.args.interim_dir, exist_ok=True)

            # Save data info
            data_info = {'length': lengths, '_nframes': nframes}
            np.save(f"{output_prefix}datainfo_min.npy", data_info)

            # Save distances and indices without vstack
            np.save(f"{output_prefix}dist_min.npy", dists)  # Remove vstack
            np.save(f"{output_prefix}inds_min.npy", inds)  # Remove vstack

            print(f"Results saved in: {self.args.interim_dir}")

        except Exception as e:
            print(f"Error during processing and saving: {str(e)}")
            raise


def run_pipeline(args) -> Dict:
    """Main function to run the preparation pipeline"""
    # Get the project root directory (one level up from pipe_tests)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Set interim_dir relative to project root
    if not hasattr(args, 'interim_dir'):
        args.interim_dir = os.path.join(project_root, 'data', args.protein_name, 'interim')

    # Create directories if they don't exist
    os.makedirs(args.interim_dir, exist_ok=True)

    processor = TrajectoryProcessor(args)
    processor.load_trajectories()
    processor.process_and_save()
    return processor.data


