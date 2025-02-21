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
