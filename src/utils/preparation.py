import torch
import numpy as np
from tqdm import tqdm
import pyemma as pe
import mdtraj as md
from typing import List, Tuple, Union, Dict
from glob import glob
import os
import shutil


class TrajectoryProcessor:
    def __init__(self, args):
        self.args = args
        self.num_residues = None
        self.data = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def prepare_topology(self) -> str:
        """Convert topology to PDB if needed and re-index residues"""
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

        # Load the topology
        topology = md.load_topology(source_pdb)

        # Create new topology with re-indexed residues
        new_top = md.Topology()

        # Copy chains and re-index residues
        new_residue_index = 0
        for chain in topology.chains:
            new_chain = new_top.add_chain()
            for residue in chain.residues:
                new_residue = new_top.add_residue(
                    name=residue.name,
                    chain=new_chain,
                    resSeq=new_residue_index + 1  # Start from 1
                )
                # Copy atoms
                for atom in residue.atoms:
                    new_top.add_atom(
                        name=atom.name,
                        element=atom.element,
                        residue=new_residue
                    )
                new_residue_index += 1

        # Load coordinates and create new trajectory
        traj = md.load(source_pdb)
        new_traj = md.Trajectory(
            xyz=traj.xyz,
            topology=new_top,
            time=traj.time,
            unitcell_lengths=traj.unitcell_lengths,
            unitcell_angles=traj.unitcell_angles
        )

        # Save re-indexed PDB to interim directory
        dest_pdb = os.path.join(self.args.interim_dir, f"{self.args.protein_name}.pdb")
        try:
            new_traj.save_pdb(dest_pdb)
            print(f"Saved re-indexed topology PDB to: {dest_pdb}")
        except Exception as e:
            raise RuntimeError(f"Failed to save re-indexed PDB: {str(e)}")

        # Store number of residues
        self.num_residues = new_top.n_residues
        print(f"Number of residues after re-indexing: {self.num_residues}")

        # Print residue mapping for verification
        print("\nResidue mapping (old → new):")
        for old_res, new_res in zip(topology.residues, new_top.residues):
            print(f"Residue {old_res.name} {old_res.resSeq} → {new_res.resSeq}")

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

    def _infer_timestep(self, traj_files: List[str], pdb_file: str) -> float:
        """Infer timestep from trajectories by loading multiple frames"""
        # Try to load first few frames to ensure we can calculate timestep
        traj_iterator = md.iterload(traj_files[0], top=pdb_file, chunk=2)
        first_chunk = next(traj_iterator)

        if len(first_chunk.time) < 2:
            raise ValueError("Trajectory must have at least 2 frames to infer timestep")

        # Calculate timestep from time differences
        timestep = first_chunk.time[1] - first_chunk.time[0]

        print(f"Inferred timestep: {timestep} ps")
        return timestep

    def _load_trajectory(self, traj_files: List[str], pdb_file: str):
        """Load trajectory files"""
        valid_trajs = [f for f in traj_files if f.endswith(('.xtc', '.trr', '.dcd', '.nc', '.h5'))]
        if not valid_trajs:
            raise ValueError("No valid trajectory files found")

        feat = pe.coordinates.featurizer(pdb_file)
        feat.add_residue_mindist()
        return pe.coordinates.source(valid_trajs, feat, chunksize=self.args.chunk_size)

    def get_neighbors(self, coords: Union[List[np.ndarray], np.ndarray], pair_list: List[Tuple[int, int]]) -> Tuple[
        np.ndarray, np.ndarray]:
        """Calculate nearest neighbors"""
        if isinstance(coords, (list, tuple)):
            return self._process_list_trajectories(coords, pair_list)
        return self._process_single_trajectory(coords, pair_list)

    def _process_single_trajectory(self, coords: np.ndarray, pair_list: List[Tuple[int, int]]) -> Tuple[
        np.ndarray, np.ndarray]:
        """Process a single trajectory"""
        # Get number of residues from pair_list
        num_residues = max(max(i, j) for i, j in pair_list)

        # Initialize arrays for all frames in the chunk
        all_frame_dists = []
        all_frame_inds = []

        try:
            # Process each frame in the chunk
            for frame in coords:  # frame shape will be (49455,)
                # Create distance matrix with padding for this frame
                mut_dist = np.ones((num_residues, num_residues)) * 300.0

                # Fill distance matrix using pair list
                for idx, d in enumerate(frame):
                    if idx >= len(pair_list):
                        print(f"Warning: More coordinates than pair list entries")
                        break

                    res_i, res_j = pair_list[idx]
                    # Adjust indices to be 0-based
                    res_i_idx = res_i - 1
                    res_j_idx = res_j - 1

                    # Verify indices are within bounds
                    if res_i_idx >= num_residues or res_j_idx >= num_residues:
                        print(
                            f"Warning: Index out of bounds - res_i: {res_i}, res_j: {res_j}, num_residues: {num_residues}")
                        continue

                    mut_dist[res_i_idx, res_j_idx] = d
                    mut_dist[res_j_idx, res_i_idx] = d  # Symmetric matrix

                # Find nearest neighbors for this frame
                frame_dists = []
                frame_inds = []

                for distances in mut_dist:
                    order = np.argsort(distances)
                    neighbors = order[:self.args.num_neighbors]
                    frame_dists.append(distances[neighbors])
                    frame_inds.append(neighbors)

                all_frame_dists.append(frame_dists)
                all_frame_inds.append(frame_inds)

            return np.array(all_frame_dists), np.array(all_frame_inds)

        except Exception as e:
            print(f"Error processing trajectory: {str(e)}")
            print(f"Coords shape: {coords.shape}")
            print(f"Number of residues: {num_residues}")
            print(f"Pair list length: {len(pair_list)}")
            raise

    def _process_single_trajectory_old(self, coords, pair_list):
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
            print(f"Total frames to process: {nframes}")

            # Infer number of residues from pair_list
            num_residues = max(max(i, j) for i, j in self.data['pair_list'][k])
            print(f"Inferred number of residues: {num_residues}")

            # Infer timestep from trajectories
            timestep = self._infer_timestep(
                glob(os.path.join(self.args.traj_folder, "r?", "traj*")),
                os.path.join(self.args.interim_dir, f"{self.args.protein_name}.pdb")
            )

            # Create subdirectory name and path
            ns = int(self.args.stride * timestep) / 1000
            subdir_name = f"{k}_{self.args.num_neighbors}nbrs_{ns}ns"
            output_dir = os.path.join(self.args.interim_dir, subdir_name)
            os.makedirs(output_dir, exist_ok=True)

            # Save data_info first
            data_info = {'length': lengths, '_nframes': nframes}
            np.save(os.path.join(output_dir, "datainfo_min.npy"), data_info)

            # Initialize arrays to store all chunks
            all_dists = []
            all_inds = []
            current_frame = 0

            # Create iterator with stride
            chunk_iterator = self.data['inpcon'][k].iterator(
                chunk=self.args.chunk_size,
                stride=self.args.stride,
                return_trajindex=True
            )

            with tqdm(total=nframes, desc="Processing trajectories") as pbar:
                for traj_idx, chunk in chunk_iterator:
                    chunk_dists, chunk_inds = self.get_neighbors(chunk, self.data['pair_list'][k])

                    # Append chunks to lists
                    all_dists.append(chunk_dists)
                    all_inds.append(chunk_inds)

                    # Update progress
                    chunk_size = chunk_dists.shape[0]
                    current_frame += chunk_size * self.args.stride
                    pbar.update(chunk_size * self.args.stride)

                # Verify final count
                if current_frame != nframes:
                    print(f"Warning: Processed {current_frame} frames out of expected {nframes}")

            # Concatenate all chunks and save
            final_dists = np.concatenate(all_dists, axis=0)
            final_inds = np.concatenate(all_inds, axis=0)

            # Save using np.save
            np.save(os.path.join(output_dir, "dist_min.npy"), final_dists)
            np.save(os.path.join(output_dir, "inds_min.npy"), final_inds)

            print(f"Results saved in: {output_dir}")
            return output_dir

        except Exception as e:
            print(f"Error during processing and saving: {str(e)}")
            raise


def run_pipeline(args) -> Dict:
    """Main function to run the preparation pipeline"""
    # Get the project root directory (one level up from pipe_tests)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


