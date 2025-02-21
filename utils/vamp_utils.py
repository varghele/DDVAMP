import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Tuple
import os


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

def calculate_state_attention_maps(attentions, state_assignments, num_classes, num_atoms):
    """
    Calculate attention maps for each state from neighbor attention values.

    Parameters
    ----------
    attentions : List[np.ndarray]
        List of attention values for each trajectory
    state_assignments : List[np.ndarray]
        List of state assignments for each trajectory
    num_classes : int
        Number of states
    num_atoms : int
        Number of atoms in the system

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        state_attention_maps: Average attention maps for each state (num_states, num_atoms, num_atoms)
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
        state_attentions = [att[mask] for att, mask in zip(attentions, state_masks)]

        if any(len(att) > 0 for att in state_attentions):
            # Convert and average attention maps
            converted_attentions = []
            for att_group in state_attentions:
                if len(att_group) > 0:
                    # Average over frames first
                    avg_att = att_group.mean(axis=0)
                    # Convert to full attention matrix
                    conv_att = convert_neighbor_attention(avg_att, num_atoms)
                    converted_attentions.append(conv_att)

            if converted_attentions:
                state_attention_maps[state] = np.mean(converted_attentions, axis=0)

    return state_attention_maps, state_populations


def convert_neighbor_attention(att, num_atoms):
    """
    Convert neighbor attention to full attention matrix.

    Parameters
    ----------
    att : np.ndarray
        Neighbor attention matrix (num_atoms x num_neighbors)
    num_atoms : int
        Number of atoms in the system

    Returns
    -------
    np.ndarray
        Full attention matrix (num_atoms x num_atoms)
    """
    full_att = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        weights = att[i]
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        for j, w in enumerate(weights):
            if j < len(weights):
                full_att[i, j] = w
    return full_att


def plot_state_attention_maps(adjs, states_order, n_states, state_populations, save_path=None):
    """
    Plot attention maps for each state.

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
        Path to save the figure
    """
    plt.style.use('default')
    plt.rcParams['axes.linewidth'] = 1.0
    plt.set_cmap('RdYlBu_r')

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=150)
    axes = axes.flatten()

    n_atoms = adjs[0].shape[0]
    x_range = np.arange(0, n_atoms, 2)
    x_label = np.arange(1, n_atoms + 1, 2)

    vmin = np.min([adj.min() for adj in adjs])
    vmax = np.max([adj.max() for adj in adjs])

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

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Attention Weight', fontsize=10)

    fig.suptitle('Attention Maps by State', fontsize=14, y=0.95)

    #plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
