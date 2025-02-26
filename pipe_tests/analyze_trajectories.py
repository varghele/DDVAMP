import mdtraj as md
import os
from glob import glob


def analyze_trajectories(traj_folder: str, topology_file: str):
    """
    Load all trajectories in a folder and print their lengths and timestep info.

    Parameters
    ----------
    traj_folder : str
        Path to folder containing trajectory files
    topology_file : str
        Path to topology file
    """
    # Get all trajectory files (assuming common extensions)
    traj_pattern = os.path.join(traj_folder, "**", "*.xtc")  # Adjust pattern as needed
    traj_files = sorted(glob(traj_pattern, recursive=True))

    if not traj_files:
        print(f"No trajectory files found in {traj_folder}")
        return

    print(f"Found {len(traj_files)} trajectory files")

    total_frames = 0
    for traj_file in traj_files:
        try:
            traj = md.load(traj_file, top=topology_file)

            # Get basic info
            n_frames = traj.n_frames
            total_frames += n_frames

            print(f"\nTrajectory: {os.path.basename(traj_file)}")
            print(f"Number of frames: {n_frames}")
            print(f"Duration (ps): {traj.time.max() if hasattr(traj, 'time') else 'Not available'}")

            # Try to infer timestep if time data is available
            if hasattr(traj, 'time') and len(traj.time) > 1:
                timestep = traj.time[1] - traj.time[0]
                print(f"Timestep (ps): {timestep}")
            else:
                print("Timestep: Not available in trajectory file")

        except Exception as e:
            print(f"\nError loading {os.path.basename(traj_file)}: {str(e)}")

    print(f"\nTotal frames across all trajectories: {total_frames}")


# Example usage:
if __name__ == "__main__":
    traj_folder = "../forked/RevGraphVAMP/trajectories/red/"
    topology_file = "../forked/RevGraphVAMP/trajectories/red/topol.gro"
    analyze_trajectories(traj_folder, topology_file)
