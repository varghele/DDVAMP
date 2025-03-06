import mdtraj as md
import os
from glob import glob


def count_residues_pdb(pdb_file: str) -> int:
    """
    Count the number of residues in a PDB file.

    Parameters
    ----------
    pdb_file : str
        Path to PDB file

    Returns
    -------
    int
        Number of residues in the PDB file
    """
    # Check if PDB file exists
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")

    try:
        # Load PDB topology
        topology = md.load_topology(pdb_file)
        n_residues = topology.n_residues

        # Print information
        print(f"PDB file: {os.path.basename(pdb_file)}")
        print(f"Number of residues: {n_residues}")
        print(f"Residue names: {[residue.name for residue in topology.residues]}")

        return n_residues

    except Exception as e:
        raise ValueError(f"Error reading PDB file: {str(e)}")


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
    # First load topology and print number of residues
    try:
        top = md.load_topology(topology_file)
        print(f"\nTopology file: {os.path.basename(topology_file)}")
        print(f"Number of residues: {top.n_residues}")
        print(f"Number of atoms: {top.n_atoms}")
        print("-" * 50)
    except Exception as e:
        print(f"Error loading topology file: {str(e)}")
        return

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
    #traj_folder = "../forked/RevGraphVAMP/trajectories/red/"
    #topology_file = "../forked/RevGraphVAMP/trajectories/red/topol.gro"
    traj_folder = "../datasets/ATR/r0/"
    topology_file = "../datasets/ATR/prot.gro"
    topology_pdb = "../datasets/ATR/prot.pdb"
    count_residues_pdb(topology_pdb)
    analyze_trajectories(traj_folder, topology_file)
