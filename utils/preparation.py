import torch
import numpy as np
from tqdm import tqdm
import pyemma as pe
from typing import List, Tuple, Union, Optional, Dict
from dataclasses import dataclass
from glob import glob
import os


@dataclass
class TrajectoryConfig:
    """Configuration class for trajectory processing"""
    num_neighbors: int = 5
    stride: int = 5
    chunk_size: int = 5000
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_atoms: int = 42
    output_dir: str = "../intermediate"


class TrajectoryProcessor:
    def __init__(self, config: TrajectoryConfig):
        self.config = config
        print(f"Using device: {self.config.device}")

    def load_trajectories(self,
                          sim_names: List[str],
                          traj_pattern: str,
                          top_pattern: str) -> Dict:
        """
        Load trajectory files using PyEMMA

        Args:
            sim_names: List of simulation names
            traj_pattern: Pattern for trajectory files
            top_pattern: Pattern for topology files

        Returns:
            Dictionary containing trajectory data
        """
        trajs = {k: sorted(glob(traj_pattern.format(k))) for k in sim_names}
        top = {k: top_pattern.format(k) for k in sim_names}

        residue_name, pair_list, inpcon = {}, {}, {}

        for k in sim_names:
            residue = {}
            pair = []
            feat = pe.coordinates.featurizer(top[k])
            feat.add_residue_mindist()

            for key in feat.describe():
                name = key.split(' ')
                ri, rj = name[2], name[4]
                i, j = int(ri[3:]), int(rj[3:])
                residue[i] = ri
                residue[j] = rj
                pair.append((i, j))

            residue_name[k] = residue
            pair_list[k] = pair
            inpcon[k] = pe.coordinates.source(trajs[k], feat)

        return {'residue_name': residue_name,
                'pair_list': pair_list,
                'inpcon': inpcon}

    def get_neighbors(self,
                      all_coords: Union[List[np.ndarray], np.ndarray],
                      pair_list: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate nearest neighbors using distance matrix
        """
        if isinstance(all_coords, list):
            return self._process_list_trajectories(all_coords, pair_list)
        else:
            return self._process_single_trajectory(all_coords, pair_list)

    def _process_list_trajectories(self, coords_list, pair_list):
        all_dists, all_inds = [], []
        for coords in coords_list:
            dists, inds = self._process_single_trajectory(coords, pair_list)
            all_dists.append(dists)
            all_inds.append(inds)
        return np.array(all_dists), np.array(all_inds)

    def _process_single_trajectory(self, coords, pair_list):
        dists, inds = [], []
        for frame in tqdm(coords):
            mut_dist = np.ones((self.config.num_atoms, self.config.num_atoms)) * 300.0

            for idx, d in enumerate(frame):
                res_i, res_j = pair_list[idx]
                mut_dist[res_i][res_j] = d
                mut_dist[res_j][res_i] = d

            frame_dists, frame_inds = [], []
            for dd in mut_dist:
                states_order = np.argsort(dd)
                neighbors = states_order[:self.config.num_neighbors]
                frame_dists.append(dd[neighbors])
                frame_inds.append(neighbors)

            dists.append(frame_dists)
            inds.append(frame_inds)

        return np.array(dists), np.array(inds)

    def process_and_save(self, sim_names: List[str], data: Dict):
        """Process trajectories and save results"""
        for k in sim_names:
            lengths = [data['inpcon'][k].trajectory_lengths()]
            nframes = data['inpcon'][k].trajectory_lengths().sum()

            mindist_file = f"{self.config.output_dir}/mindist-780-{k}.npy"
            if not os.path.exists(mindist_file):
                print(f"Computing mindist for {k}...")
                con = np.vstack(data['inpcon'][k].get_output())
                np.save(mindist_file, con)

            ns = int(self.config.stride * 0.2)
            output_prefix = f"{self.config.output_dir}/{k}_{self.config.num_neighbors}nbrs_{ns}ns_"

            dists, inds = self.get_neighbors(data[k], data['pair_list'][k])

            data_info = {'length': lengths, '_nframes': nframes}
            np.save(f"{output_prefix}datainfo_min.npy", data_info)
            np.save(f"{output_prefix}dist_min.npy", np.vstack(dists))
            np.save(f"{output_prefix}inds_min.npy", np.vstack(inds))

    def chunks(self, data: Union[List[np.ndarray], np.ndarray]):
        """Split trajectory into chunks"""
        if isinstance(data, list):
            for data_tmp in data:
                yield from self._chunk_single_trajectory(data_tmp)
        else:
            yield from self._chunk_single_trajectory(data)

    def _chunk_single_trajectory(self, data: np.ndarray):
        for j in range(0, len(data), self.config.chunk_size):
            chunk = data[j:j + self.config.chunk_size, ...]
            print(f"Chunk shape: {chunk.shape}")
            yield chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-neighbors', type=int, default=5)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--traj-folder', type=str, required=True)
    parser.add_argument('--num-atoms', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default="../intermediate")

    args = parser.parse_args()

    config = TrajectoryConfig(
        num_neighbors=args.num_neighbors,
        stride=args.stride,
        num_atoms=args.num_atoms,
        output_dir=args.output_dir
    )

    processor = TrajectoryProcessor(config)

    sim_names = ("red",)
    traj_pattern = f"{args.traj_folder}/{{0}}/r?/traj*.xtc"
    top_pattern = f"{args.traj_folder}/{{0}}/topol.gro"

    data = processor.load_trajectories(sim_names, traj_pattern, top_pattern)
    processor.process_and_save(sim_names, data)


if __name__ == "__main__":
    main()
