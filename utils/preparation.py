import torch
import numpy as np
from tqdm import tqdm
import pyemma as pe
import mdtraj as md
from typing import List, Tuple, Union, Dict
from glob import glob
import os
import shutil
import psutil


class TrajectoryProcessor:
    def __init__(self, args):
        self.args = args
        self.num_residues = None
        self.data = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def prepare_topology(self) -> str:
        """Convert topology to PDB if needed and count residues"""
        # Generate source PDB path
        if self.args.topology.endswith('.pdb'):
            source_pdb = self.args.topology
        elif self.args.topology.endswith('.gro'):
            source_pdb = self.args.topology.replace('.gro', '.pdb')
            if not os.path.exists(source_pdb):
                try:
                    traj = md.load(self.args.topology)
                    traj.save_pdb(source_pdb)
                except Exception as e:
                    raise RuntimeError(f"Failed to convert GRO to PDB: {str(e)}")
        else:
            raise ValueError("Topology file must be either .pdb or .gro")

        # Copy PDB to interim directory
        dest_pdb = os.path.join(self.args.interim_dir, f"{self.args.protein_name}.pdb")
        try:
            shutil.copy2(source_pdb, dest_pdb)
            print(f"Copied topology PDB to: {dest_pdb}")
        except Exception as e:
            raise RuntimeError(f"Failed to copy PDB to interim directory: {str(e)}")

        # Load structure and count residues
        structure = md.load(dest_pdb)
        self.num_residues = structure.topology.n_residues
        print(f"Number of residues detected: {self.num_residues}")
        return dest_pdb

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

    def _infer_timestep(self) -> float:
        """
        Infer timestep from first trajectory file.

        Returns
        -------
        float
            Timestep in picoseconds
        """
        try:
            # Find first trajectory file
            first_traj_file = None
            for r_dir in sorted(os.listdir(self.args.traj_folder)):
                if r_dir.startswith('r'):
                    r_path = os.path.join(self.args.traj_folder, r_dir)
                    for f in os.listdir(r_path):
                        if f.startswith('traj'):
                            first_traj_file = os.path.join(r_path, f)
                            break
                if first_traj_file:
                    break

            if not first_traj_file:
                raise ValueError(f"No trajectory files found in {self.args.traj_folder}")

            # Load just the first trajectory to check timestep
            first_traj = md.load(first_traj_file, top=self.args.topology)

            if hasattr(first_traj, 'timestep'):
                timestep = first_traj.timestep
            elif hasattr(first_traj, 'time') and len(first_traj.time) > 1:
                timestep = first_traj.time[1] - first_traj.time[0]
            else:
                raise ValueError("Could not infer timestep from trajectory")

            print(f"Inferred timestep: {timestep} ps")
            return timestep

        except Exception as e:
            raise ValueError(f"Failed to infer timestep: {str(e)}")

    def _load_trajectory(self, traj_files: List[str], pdb_file: str):
        """Load trajectory files with memory-efficient processing"""
        valid_trajs = [f for f in traj_files if f.endswith(('.xtc', '.trr', '.dcd', '.nc', '.h5'))]
        if not valid_trajs:
            raise ValueError("No valid trajectory files found")

        # Create featurizer with memory-efficient settings
        feat = pe.coordinates.featurizer(pdb_file)
        feat.add_residue_mindist()

        # Use memory-efficient parameters in source creation
        return pe.coordinates.source(
            valid_trajs,
            feat,
            stride=self.args.stride,
            chunksize=1  # Process in smaller chunks
        )

    def get_neighbors(self, coords: Union[List[np.ndarray], np.ndarray], pair_list: List[Tuple[int, int]]) -> Tuple[
        np.ndarray, np.ndarray]:
        """Calculate nearest neighbors"""
        if isinstance(coords, (list, tuple)):
            return self._process_list_trajectories(coords, pair_list)
        return self._process_single_trajectory(coords, pair_list)

    def _process_single_trajectory(self, coords, pair_list):
        """Memory-efficient single trajectory processing"""
        n_frames = len(coords)
        n_residues = self.num_residues
        n_neighbors = self.args.num_neighbors

        # Pre-allocate arrays with float32 for memory efficiency
        dists = np.zeros((n_frames, n_residues, n_neighbors), dtype=np.float32)
        inds = np.zeros((n_frames, n_residues, n_neighbors), dtype=np.int32)

        # Process each frame
        for i in range(n_frames):
            # Create distance matrix efficiently
            mut_dist = np.full((n_residues, n_residues), 100.0, dtype=np.float32)

            # Fill distance matrix
            for idx, d in enumerate(coords[i]):
                if idx >= len(pair_list):
                    break
                res_i, res_j = pair_list[idx]
                mut_dist[res_i - 1, res_j - 1] = d
                mut_dist[res_j - 1, res_i - 1] = d

            # Find neighbors efficiently
            for j in range(n_residues):
                idx = np.argpartition(mut_dist[j], n_neighbors)[:n_neighbors]
                dists[i, j] = mut_dist[j, idx]
                inds[i, j] = idx

        return dists, inds

    def _process_list_trajectories(self, coords_list, pair_list):
        """Process a list of trajectories"""
        all_dists, all_inds = [], []
        for coords in coords_list:
            dists, inds = self._process_single_trajectory(coords, pair_list)
            all_dists.append(dists)
            all_inds.append(inds)
        return np.concatenate(all_dists), np.concatenate(all_inds)

    def process_and_save(self):
        """Process trajectories with memory-efficient streaming"""
        k = self.args.protein_name

        try:
            # Get trajectory information
            lengths = [self.data['inpcon'][k].trajectory_lengths()]
            nframes = self.data['inpcon'][k].trajectory_lengths().sum()
            timestep = self._infer_timestep()

            # Create output directory
            ns = int(self.args.stride * timestep) / 1000
            subdir_name = f"{k}_{self.args.num_neighbors}nbrs_{ns}ns"
            output_dir = os.path.join(self.args.interim_dir, subdir_name)
            os.makedirs(output_dir, exist_ok=True)

            # Process in chunks
            chunk_size = 1
            all_dists = []
            all_inds = []

            # Get data in chunks
            n_chunks = (nframes + chunk_size - 1) // chunk_size
            for i in tqdm(range(n_chunks), desc="Processing trajectory chunks"):
                # Get chunk
                chunk = self.data['inpcon'][k].get_output(stride=1, chunk=chunk_size)

                # Process chunk
                chunk_dists, chunk_inds = self.get_neighbors(chunk, self.data['pair_list'][k])

                # Store results
                all_dists.append(chunk_dists)
                all_inds.append(chunk_inds)

                # Clear memory
                del chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Combine results
            dists = np.concatenate(all_dists, axis=0)
            inds = np.concatenate(all_inds, axis=0)

            # Save results
            data_info = {
                'length': lengths,
                '_nframes': nframes,
                'timestep': timestep
            }
            np.save(os.path.join(output_dir, "datainfo_min.npy"), data_info)
            np.save(os.path.join(output_dir, "dist_min.npy"), dists)
            np.save(os.path.join(output_dir, "inds_min.npy"), inds)

            print(f"Results saved in: {output_dir}")
            return output_dir

        except Exception as e:
            print(f"Error during processing and saving: {str(e)}")
            raise

    def _save_large_array(self, arr, filename):
        """Save large arrays efficiently"""
        if arr.nbytes > 1e9:  # If array is larger than 1GB
            with open(filename, 'wb') as f:
                np.save(f, arr.shape)
                for chunk in np.array_split(arr, max(1, arr.shape[0] // 1000)):
                    np.save(f, chunk)
        else:
            np.save(filename, arr)


def run_pipeline(args) -> Dict:
    """Main function to run the preparation pipeline"""
    # Get the project root directory (one level up from pipe_tests)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Set interim_dir relative to project root
    if not hasattr(args, 'interim_dir'):
        args.interim_dir = os.path.join(project_root, 'data', args.protein_name, 'interim')

    # Create base interim directory if it doesn't exist
    os.makedirs(args.interim_dir, exist_ok=True)

    # Initialize processor and run pipeline
    processor = TrajectoryProcessor(args)
    processor.load_trajectories()
    output_dir = processor.process_and_save()  # This will now return the output directory path

    # Add output directory to the returned data
    processor.data['output_dir'] = output_dir
    return processor.data



