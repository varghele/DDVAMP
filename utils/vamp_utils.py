import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Tuple
import os
import mdtraj as md
from glob import glob
from tqdm import tqdm

def chunks(data, chunk_size=5000):
    '''
    splitting the trajectory into chunks for passing into analysis part
    data: list of trajectories
    chunk_size: the size of each chunk
    '''
    if type(data) == list:

        for data_tmp in data:
            for j in range(0, len(data_tmp), chunk_size):
                print(data_tmp[j:j + chunk_size, ...].shape)
                yield data_tmp[j:j + chunk_size, ...]

    else:

        for j in range(0, len(data), chunk_size):
            yield data[j:j + chunk_size, ...]

def scale(score):
    """Scale scores to [0,1] range"""
    score_min = score.min()
    score_max = score.max()
    if score_max == score_min:
        print(f"Warning: constant score value {score_max}")
        return np.zeros_like(score)
    return (score - score_min) / (score_max - score_min)


def estimate_koopman_op(trajs, tau):
    """
    Estimate the Koopman operator from trajectory data at a given lag time.

    Parameters
    ----------
    trajs : Union[np.ndarray, List[np.ndarray]]
        Single trajectory array of shape (timesteps, features) or
        list of trajectory arrays
    tau : int
        Lag time for estimation

    Returns
    -------
    np.ndarray
        Estimated Koopman operator matrix
    """
    if isinstance(trajs, list):
        # Ensure all trajectories have same number of features
        n_features = trajs[0].shape[1]
        filtered_trajs = []

        for t in trajs:
            if t.shape[0] > tau:  # Only include trajectories longer than tau
                filtered_trajs.append(t)

        if not filtered_trajs:
            raise ValueError(f"No trajectories longer than tau={tau} found")

        # Concatenate time-lagged pairs
        traj = np.concatenate([t[:-tau] for t in filtered_trajs], axis=0)
        traj_lag = np.concatenate([t[tau:] for t in filtered_trajs], axis=0)
    else:
        if trajs.shape[0] <= tau:
            raise ValueError(f"Trajectory length {trajs.shape[0]} must be greater than tau={tau}")
        traj = trajs[:-tau]
        traj_lag = trajs[tau:]

    # Compute correlation matrices
    c_0 = np.transpose(traj) @ traj
    c_tau = np.transpose(traj) @ traj_lag

    # Handle numerical stability
    eigv, eigvec = np.linalg.eig(c_0)
    include = eigv > 1e-7
    eigv = eigv[include]
    eigvec = eigvec[:, include]
    c0_inv = eigvec @ np.diag(1 / eigv) @ np.transpose(eigvec)

    koopman_op = c0_inv @ c_tau
    return koopman_op


def get_ck_test(traj, steps, tau):
    """
    Perform Chapman-Kolmogorov test comparing predicted vs estimated dynamics.

    Parameters
    ----------
    traj : Union[np.ndarray, List[np.ndarray]]
        Trajectory data
    steps : int
        Number of prediction steps
    tau : int
        Lag time between steps

    Returns
    -------
    List[np.ndarray]
        [predicted, estimated] arrays of shape (n_states, n_states, steps)
    """
    if type(traj) == list:
        n_states = traj[0].shape[1]
    else:
        n_states = traj.shape[1]

    predicted = np.zeros((n_states, n_states, steps))
    estimated = np.zeros((n_states, n_states, steps))

    predicted[:, :, 0] = np.identity(n_states)
    estimated[:, :, 0] = np.identity(n_states)

    for vector, i in zip(np.identity(n_states), range(n_states)):
        for n in range(1, steps):
            koop = estimate_koopman_op(traj, tau)
            koop_pred = np.linalg.matrix_power(koop, n)
            koop_est = estimate_koopman_op(traj, tau * n)

            predicted[i, :, n] = vector @ koop_pred
            estimated[i, :, n] = vector @ koop_est

    return [predicted, estimated]


def plot_ck_test(pred, est, n_states, steps, tau, save_folder, filename='ck_test.png'):
    """
    Plot Chapman-Kolmogorov test results.

    Parameters
    ----------
    pred : np.ndarray
        Predicted dynamics array
    est : np.ndarray
        Estimated dynamics array
    n_states : int
        Number of states
    steps : int
        Number of prediction steps
    tau : int
        Lag time between steps
    save_folder : str
        Directory to save plot
    filename : str, optional
        Name of output file, by default 'ck_test.png'
    """
    fig, ax = plt.subplots(n_states, n_states, sharex=True, sharey=True)
    for index_i in range(n_states):
        for index_j in range(n_states):
            ax[index_i][index_j].plot(range(0, steps * tau, tau), pred[index_i, index_j], color='b')
            ax[index_i][index_j].plot(range(0, steps * tau, tau), est[index_i, index_j], color='r', linestyle='--')
            ax[index_i][index_j].set_title(str(index_i + 1) + '->' + str(index_j + 1), fontsize='small')

    ax[0][0].set_ylim((-0.1, 1.1))
    ax[0][0].set_xlim((0, steps * tau))
    ax[0][0].axes.get_xaxis().set_ticks(np.round(np.linspace(0, steps * tau, 3)))
    plt.tight_layout()
    plt.savefig(save_folder + '/' + filename)


def get_its(traj, lags):
    """
    Calculate implied timescales from trajectory data at multiple lag times.

    Parameters
    ----------
    traj : Union[np.ndarray, List[np.ndarray]]
        Trajectory data array(s)
    lags : np.ndarray
        Array of lag times to analyze

    Returns
    -------
    np.ndarray
        Implied timescales array of shape (n_states-1, n_lags)
    """
    if type(traj) == list:
        outputsize = traj[0].shape[1]
    else:
        outputsize = traj.shape[1]
    its = np.zeros((outputsize - 1, len(lags)))

    for t, tau_lag in enumerate(lags):
        koopman_op = estimate_koopman_op(traj, tau_lag)
        k_eigvals, k_eigvec = np.linalg.eig(np.real(koopman_op))
        k_eigvals = np.sort(np.absolute(k_eigvals))
        k_eigvals = k_eigvals[:-1]
        its[:, t] = (-tau_lag / np.log(k_eigvals))

    return its


def plot_its(its: np.ndarray,
             lags: Union[List[int], np.ndarray],
             save_path: str,
             ylog: bool = False) -> None:
    """
    Plot implied timescales (ITS) as a function of lag time.

    Parameters
    ----------
    its : np.ndarray
        Array of shape (n_states-1, n_lags) containing implied timescales
    lags : Union[List[int], np.ndarray]
        Array or list of lag times used for ITS calculation
    save_path : str
        Path where the plot should be saved
    ylog : bool, optional
        If True, use log-log plot; if False, use semi-log plot. Default: False
    """
    plt.figure(figsize=(8, 6))

    # Convert lags to array if needed
    lags = np.array(lags)

    if ylog:
        # Log-log plot
        plt.loglog(lags, its.T[:, ::-1], linewidth=2)
        plt.loglog(lags, lags, 'k', label='y=x', linewidth=1)
        plt.fill_between(lags, lags, 0.99, alpha=0.2, color='k')
    else:
        # Semi-log plot
        plt.semilogy(lags, its.T[:, ::-1], linewidth=2)
        plt.semilogy(lags, lags, 'k', label='y=x', linewidth=1)
        plt.fill_between(lags, lags, 0.99, alpha=0.2, color='k')

    # Add labels and legend
    plt.xlabel('Lag Time', fontsize=12)
    plt.ylabel('Implied Timescale', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)

    # Save and close
    plt.savefig(os.path.join(save_path, 'its.png'), dpi=300, bbox_inches='tight')
    plt.close()


def analyze_model_outputs(
        model,
        data_np: List[np.ndarray],
        save_folder: str,
        batch_size: int = 1000,
        num_classes: int = 5,
        h_g: int = 64,
        num_atoms: int = 42,
        num_neighbors: int = 10
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Analyze model outputs and save embeddings, attention, and state probabilities.

    Parameters
    ----------
    model : VAMPNet
        Trained VAMP model
    data_np : List[np.ndarray]
        List of trajectory data arrays
    save_folder : str
        Path to save the analysis results
    batch_size : int, optional
        Size of batches for processing, default=1000
    num_classes : int, optional
        Number of classes/states, default=5
    h_g : int, optional
        Dimension of graph embeddings, default=64
    num_atoms : int, optional
        Number of atoms in the system, default=42
    num_neighbors : int, optional
        Number of neighbors per atom, default=10

    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]
        Tuple containing (probs, embeddings, attentions)
    """
    # Initialize lists for storing results
    probs = []
    embeddings = []
    attentions = []

    # Process each trajectory
    for data_tmp in data_np:
        mydata = chunks(data_tmp, chunk_size=batch_size)

        # Initialize arrays for current trajectory
        state_probs = np.zeros((data_tmp.shape[0], num_classes))
        emb_tmp = np.zeros((data_tmp.shape[0], h_g))
        attn_tmp = np.zeros((data_tmp.shape[0], num_atoms, num_neighbors))

        n_iter = 0
        for batch in mydata:
            batch_size_current = len(batch)

            # Get state probabilities
            state_probs[n_iter:n_iter + batch_size_current] = model.transform(batch)

            # Get embeddings and attention
            batch_tensor = torch.tensor(batch)
            emb_1, attn_1 = model.lobe(batch_tensor, return_emb=True, return_attn=True)

            # Store results
            emb_tmp[n_iter:n_iter + batch_size_current] = emb_1.cpu().detach().numpy()
            attn_tmp[n_iter:n_iter + batch_size_current] = attn_1.cpu().detach().numpy()

            n_iter += batch_size_current

        # Append results for current trajectory
        probs.append(state_probs)
        embeddings.append(emb_tmp)
        attentions.append(attn_tmp)

    # Save all results
    np.savez(os.path.join(save_folder, 'transformed.npz'), probs)
    np.savez(os.path.join(save_folder, 'embeddings.npz'), embeddings)
    np.savez(os.path.join(save_folder, 'attention.npz'), attentions)

    # Save first trajectory results separately
    np.savez(os.path.join(save_folder, 'transformed_0.npz'), probs[0])
    np.savez(os.path.join(save_folder, 'embeddings_0.npz'), embeddings[0])
    np.savez(os.path.join(save_folder, 'attention_0.npz'), attentions[0])

    print(f"Analysis complete. Results saved to {save_folder}")
    print(f"State probabilities shape: {probs[0].shape}")
    print(f"Number of trajectories: {len(probs)}")

    return probs, embeddings, attentions


def calculate_state_attention_maps(attentions: List[np.ndarray],
                                   neighbor_indices: List[np.ndarray],
                                   state_assignments: List[np.ndarray],
                                   num_classes: int,
                                   num_atoms: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate attention maps for each state from neighbor attention values.

    Parameters
    ----------
    attentions : List[np.ndarray]
        List of attention values for each trajectory
    neighbor_indices : List[np.ndarray]
        List of neighbor indices for each trajectory
    state_assignments : List[np.ndarray]
        List of state assignments for each trajectory
    num_classes : int
        Number of states
    num_atoms : int
        Number of atoms in the system

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        state_attention_maps: Average attention maps for each state
        state_populations: Population of each state
    """
    # Calculate state populations
    state_populations = np.zeros(num_classes)
    for states in state_assignments:
        unique, counts = np.unique(states, return_counts=True)
        state_populations[unique] += counts
    state_populations = state_populations / np.sum(state_populations)

    # Initialize state attention maps
    state_attention_maps = np.zeros((num_classes, num_atoms, num_atoms))

    # Process each state
    for state in range(num_classes):
        state_masks = [states == state for states in state_assignments]
        state_attentions = []

        for traj_idx, mask in enumerate(state_masks):
            if np.any(mask):
                # Get attention and indices for frames in this state
                att = attentions[traj_idx][mask]
                inds = neighbor_indices[traj_idx][mask]

                # Average over frames
                avg_att = att.mean(axis=0)
                avg_inds = inds[0]  # Use first frame's indices as they should be constant

                # Convert to full attention matrix
                full_att = convert_neighbor_attention(avg_att, avg_inds, num_atoms)
                state_attentions.append(full_att)

        if state_attentions:
            state_attention_maps[state] = np.mean(state_attentions, axis=0)

    return state_attention_maps, state_populations


def convert_neighbor_attention(neighbor_attn: np.ndarray, neighbor_indices: np.ndarray, num_atoms: int) -> np.ndarray:
    """
    Convert neighbor attention to full attention matrix considering all residue interactions.

    Parameters
    ----------
    neighbor_attn : np.ndarray
        Attention weights for neighbors (num_atoms x num_neighbors)
    neighbor_indices : np.ndarray
        Indices of neighbors for each atom (num_atoms x num_neighbors)
    num_atoms : int
        Total number of atoms/residues

    Returns
    -------
    np.ndarray
        Full attention matrix (num_atoms x num_atoms)
    """
    full_att = np.zeros((num_atoms, num_atoms))

    for i in range(num_atoms):
        # Get neighbors and their attention weights for atom i
        neighbors = neighbor_indices[i]
        weights = neighbor_attn[i]

        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)

        # Distribute attention to all atoms
        for j, (neighbor, weight) in enumerate(zip(neighbors, weights)):
            if neighbor < num_atoms:  # ensure valid neighbor index
                full_att[i, neighbor] = weight
                full_att[i, i] = 1.0 - np.sum(weights)  # self-attention

    # Symmetrize the attention matrix
    full_att = 0.5 * (full_att + full_att.T)

    return full_att


def plot_state_attention_maps(adjs, states_order, n_states, state_populations, save_path=None):
    """
    Plot attention maps for each state individually and in a combined figure.

    Parameters
    ----------
    adjs : np.ndarray
        Attention matrices for each state [n_states, n_atom, n_atom]
    states_order : np.ndarray
        Order of states by population
    n_states : int
        Number of states
    state_populations : np.ndarray
        Population of each state
    save_path : str, optional
        Base path to save the figures
    """
    plt.style.use('default')
    plt.rcParams['axes.linewidth'] = 1.0
    plt.set_cmap('RdYlBu_r')

    n_atoms = adjs[0].shape[0]
    x_range = np.arange(0, n_atoms, 2)
    x_label = np.arange(1, n_atoms + 1, 2)

    vmin = np.min([adj.min() for adj in adjs])
    vmax = np.max([adj.max() for adj in adjs])

    # Create individual plots for each state
    for i in range(n_states):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

        im = ax.imshow(adjs[states_order[i]], vmin=vmin, vmax=vmax)

        ax.hlines(np.arange(0, n_atoms) - 0.5, -0.5, n_atoms - 0.5,
                  color='k', linewidth=0.5, alpha=0.3)
        ax.vlines(np.arange(0, n_atoms) - 0.5, -0.5, n_atoms - 0.5,
                  color='k', linewidth=0.5, alpha=0.3)

        ax.set_xticks(x_range)
        ax.set_xticklabels(x_label, fontsize=8)
        ax.set_yticks(x_range)
        ax.set_yticklabels(x_label, fontsize=8)

        ax.set_title(f'State {i + 1}\nPopulation: {state_populations[states_order[i]]:.1%}',
                     fontsize=12, pad=10)

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Attention Weight', fontsize=10)

        if save_path:
            state_save_path = save_path.replace('.png', f'_state_{i + 1}.png')
            plt.savefig(state_save_path, bbox_inches='tight')
            print(f"Saved state {i + 1} plot to: {state_save_path}")
        plt.close()

    # Create combined plot
    # Calculate number of rows and columns needed
    n_cols = int(np.ceil(np.sqrt(n_states)))
    n_rows = int(np.ceil(n_states / n_cols))

    # Create combined figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), dpi=150)
    if n_states > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i in range(n_states):
        ax = axes[i]
        im = ax.imshow(adjs[states_order[i]], vmin=vmin, vmax=vmax)

        ax.hlines(np.arange(0, n_atoms) - 0.5, -0.5, n_atoms - 0.5,
                  color='k', linewidth=0.5, alpha=0.3)
        ax.vlines(np.arange(0, n_atoms) - 0.5, -0.5, n_atoms - 0.5,
                  color='k', linewidth=0.5, alpha=0.3)

        ax.set_xticks(x_range)
        ax.set_xticklabels(x_label, fontsize=8)
        ax.set_yticks(x_range)
        ax.set_yticklabels(x_label, fontsize=8)

        ax.set_title(f'State {i + 1}\nPopulation: {state_populations[states_order[i]]:.1%}',
                     fontsize=10, pad=10)

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Remove empty subplots
    for i in range(n_states, len(axes)):
        axes[i].remove()

    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Attention Weight', fontsize=10)

    fig.suptitle('Attention Maps by State', fontsize=14, y=0.95)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved combined plot to: {save_path}")

    plt.close()


def plot_state_attention_weights(state_attention_maps: np.ndarray,
                                 topology_file: str,
                                 n_states: int,
                                 save_path: str = None):
    """
    Plot average attention weights for residues across different states.

    Parameters
    ----------
    state_attention_maps : np.ndarray
        Attention maps for each state [n_states, n_atoms, n_atoms]
    topology_file : str
        Path to topology file (PDB or similar) to get residue names
    n_states : int
        Number of states
    save_path : str, optional
        Path to save the figure
    """

    # Get residue names from topology
    top = md.load(topology_file).topology
    residues = [f"{res.name}{res.resSeq}" for res in top.residues]
    n_atoms = len(residues)

    # Calculate scaled scores
    scores_p = np.stack([scale(state_map.sum(axis=0))
                         for state_map in state_attention_maps])

    # Create plot
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=200)

    # Plot heatmap
    h = ax.imshow(scores_p)

    # Set ticks and labels
    ax.set_xticks(np.arange(n_atoms))
    ax.set_yticks(np.arange(n_states))  # Set proper number of y-ticks
    ax.set_yticklabels([str(i) for i in range(1, n_states+1)])  # Label states from 1 to n_states
    ax.set_xticklabels(residues, fontweight='bold', rotation=90)

    # Add y-axis label
    ax.set_ylabel('States', fontweight='bold')

    # Add grid lines
    ax.hlines(np.arange(0, n_states) - 0.5, -0.5, n_atoms - 0.5,
              color='k', linewidth=1, alpha=0.5)
    ax.vlines(np.arange(0, n_atoms) - 0.5, -0.5, n_states - 0.5,
              color='k', linewidth=0.5, alpha=0.5)

    # Add colorbar
    cbar = plt.colorbar(h, ax=ax, fraction=0.01)

    # Set font properties
    fontsize = 10
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig, ax


def generate_state_structures(traj_folder: str,
                              topology_file: str,
                              state_assignments: List[np.ndarray],
                              save_dir: str,
                              protein_name: str,
                              stride: int = 10) -> List[str]:
    """
    Generate representative PDB structures for each conformational state from multiple trajectories.

    Parameters
    ----------
    traj_folder : str
        Path to the folder containing trajectory files
    topology_file : str
        Path to the topology file
    state_assignments : List[np.ndarray]
        List of state assignments for each frame
    save_dir : str
        Directory to save the output PDB files
    protein_name : str
        Name of the protein for file naming
    stride : int, optional
        Load every nth frame to reduce memory usage (default: 10)

    Returns
    -------
    List[str]
        Paths to the generated PDB files
    """
    # Get trajectory files
    traj_pattern = os.path.join(traj_folder, "r?", "traj*")
    traj_files = sorted(glob(traj_pattern))

    if not traj_files:
        raise ValueError(f"No trajectory files found in {traj_folder}")

    # Load trajectories with stride
    trajs = []
    frame_counts = []
    print(f"Loading trajectories with stride {stride}...")
    for traj_file in tqdm(traj_files, desc="Loading trajectories"):
        traj = md.load(traj_file, top=topology_file, stride=stride)
        trajs.append(traj)
        frame_counts.append(len(traj))

    print("Combining trajectories...")
    combined_traj = md.join(trajs)
    print(f"Total frames after stride: {len(combined_traj)}")

    # Adjust state assignments for stride
    strided_assignments = []
    current_idx = 0
    for count in frame_counts:
        strided_count = (count + stride - 1) // stride  # ceiling division
        if current_idx + strided_count <= len(state_assignments[0]):
            strided_assignments.append(state_assignments[0][current_idx:current_idx + strided_count])
        current_idx += strided_count

    # Flatten state assignments
    all_states = np.concatenate(strided_assignments)
    unique_states = np.unique(all_states)
    output_files = []

    print("Processing states...")
    for state in tqdm(unique_states, desc="Generating state structures"):
        # Get frames belonging to this state
        state_frames = np.where(all_states == state)[0]

        if len(state_frames) == 0:
            continue

        # Extract trajectory for this state
        state_traj = combined_traj[state_frames]

        # Calculate average structure
        average_structure = state_traj.superpose(state_traj[0])

        # Find frame closest to average
        average_xyz = average_structure.xyz.mean(axis=0)
        distances = np.sqrt(np.sum((state_traj.xyz - average_xyz) ** 2, axis=(1, 2)))
        representative_frame = state_frames[np.argmin(distances)]

        # Save representative structure
        output_file = os.path.join(save_dir, f"{protein_name}_state_{state + 1}.pdb")
        combined_traj[representative_frame].save_pdb(output_file)
        output_files.append(output_file)

        print(f"State {state + 1}: {len(state_frames)} frames, saved to {output_file}")

    return output_files
