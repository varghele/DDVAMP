import os
import sys
import mdtraj as md
from glob import glob


def infer_timestep(traj_folder: str, topology: str) -> float:
    """
    Infer timestep from trajectory files.

    Parameters
    ----------
    traj_folder : str
        Path to folder containing trajectories
    topology : str
        Path to topology file

    Returns
    -------
    float
        Timestep in picoseconds
    """
    # Find first trajectory file
    traj_pattern = os.path.join(traj_folder, "r?", "traj*")
    traj_files = sorted(glob(traj_pattern))

    if not traj_files:
        raise ValueError(f"No trajectory files found in {traj_folder}")

    # Load first trajectory to check timestep
    first_traj = md.load(traj_files[0], top=topology)

    if hasattr(first_traj, 'timestep'):
        timestep = first_traj.timestep
    elif hasattr(first_traj, 'time') and len(first_traj.time) > 1:
        timestep = first_traj.time[1] - first_traj.time[0]
    else:
        raise ValueError("Could not infer timestep from trajectory")

    print(f"Inferred timestep: {timestep} ps")
    return timestep
