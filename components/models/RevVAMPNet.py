import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Callable, Tuple, List
from deeptime.decomposition.deep import VAMPNet
from deeptime.util.torch import disable_TF32, multi_dot
from tqdm import tqdm
from args.args import buildParser

if torch.cuda.is_available():
	device = torch.device('cuda')
	print('cuda is is available')
else:
	print('Using CPU')
	device = torch.device('cpu')

LAG_EPOCH = 1000
args = buildParser().parse_args()

class RevVAMPNet(VAMPNet):
    """
    Reversible VAMPNet implementation that extends VAMPNet with additional VAMP-specific functionality.

    Parameters
    ----------
    lobe : nn.Module
        Neural network module for processing instantaneous data
    lobe_timelagged : Optional[nn.Module]
        Neural network module for processing time-lagged data (if None, uses lobe)
    vampu : Optional[nn.Module]
        VAMP-U network module
    vamps : Optional[nn.Module]
        VAMP-S network module
    device : Optional[torch.device]
        Device to run computations on
    optimizer : Union[str, Callable]
        Optimizer to use for training
    learning_rate : float
        Learning rate for optimization
    score_method : str
        Method for computing VAMP score
    score_mode : str
        Mode for score computation
    epsilon : float
        Small constant for numerical stability
    dtype : np.dtype
        Data type for computations
    n_output : int
        Number of output dimensions
    """

    def __init__(self,
                 lobe: nn.Module,
                 lobe_timelagged: Optional[nn.Module] = None,
                 vampu: Optional[nn.Module] = None,
                 vamps: Optional[nn.Module] = None,
                 device: Optional[torch.device] = None,
                 optimizer: Union[str, Callable] = 'Adam',
                 learning_rate: float = 5e-4,
                 score_method: str = 'VAMP2',
                 score_mode: str = 'regularize',
                 epsilon: float = 1e-6,
                 dtype: np.dtype = np.float32,
                 n_output: int = args.num_classes):

        super().__init__(lobe, lobe_timelagged, device, optimizer, learning_rate,
                         score_method, score_mode, epsilon, dtype)

        # Initialize additional attributes
        self.n_output = n_output
        self._vampu = vampu
        self._vamps = vamps
        self._k_cache = {}
        self.network_lag = args.tau
        self._lag = args.tau
        self._K = None
        self.data = None

        # Validate VAMPCE configuration
        if score_method == 'VAMPCE':
            if vampu is None or vamps is None:
                raise ValueError("vampu and vamps modules must be defined for VAMPCE score method")

            # Setup optimizer with all parameters
            all_params = (
                    list(self.lobe.parameters()) +
                    list(self.lobe_timelagged.parameters()) +
                    list(self._vampu.parameters()) +
                    list(self._vamps.parameters())
            )
            self.setup_optimizer(optimizer, all_params)

    @property
    def K(self) -> np.ndarray:
        """
        The estimated Koopman operator.

        Returns
        -------
        np.ndarray
            Current Koopman operator estimate. Returns a 1x1 unit matrix if not estimated yet.
        """
        if self._K is None or self._reestimated:
            self._K = np.ones((1, 1))
        return self._K

    @property
    def vampu(self) -> nn.Module:
        """
        The VAMP-U network module.

        Returns
        -------
        nn.Module
            VAMP-U neural network module
        """
        return self._vampu

    @property
    def vamps(self) -> nn.Module:
        """
        The VAMP-S network module.

        Returns
        -------
        nn.Module
            VAMP-S neural network module
        """
        return self._vamps

    @property
    def lag(self) -> int:
        """
        The model lag time.

        Returns
        -------
        int
            Current lag time
        """
        return self._lag

    @lag.setter
    def lag(self, lag: int):
        """
        Update the model lag time for ITS calculation.

        Parameters
        ----------
        lag : int
            New lag time to update the model to
        """
        # Reset VAMP-S weights
        self._vamps.reset_weights()

        # Update auxiliary weights with current data
        data = self.data
        self.update_auxiliary_weights(data, optimize_u=False, optimize_S=True, reset_weights=False)

        # Train VAMP networks
        self.train_US(data, train_u=False, out_log=True)

        # First training phase - fixed U
        for _ in tqdm(range(LAG_EPOCH)):
            self.train_US(data, train_u=False)

        # Second training phase - update both U and S
        for _ in tqdm(range(LAG_EPOCH)):
            self.train_US(data)

        # Final training step with logging
        self.train_US(data, out_log=True)
        print(f"new lag {lag} ok")

        # Update internal state
        self._lag = lag
        self._reestimated = True

    def partial_fit(self,
                    data: Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]],
                    train_score_callback: Optional[Callable[[int, torch.Tensor], None]] = None) -> 'RevVAMPNet':
        """
        Perform a partial fit on the data.

        Parameters
        ----------
        data : tuple of (batch_0, batch_t)
            Tuple containing instantaneous and time-lagged data batches
        train_score_callback : callable, optional
            Callback function called after each partial fit with current step and score

        Returns
        -------
        self : RevVAMPNet
            Reference to self
        """
        # Set model precision
        self._set_model_precision()

        # Set models to training mode
        self.lobe.train()
        self.lobe_timelagged.train()

        # Validate input
        if not isinstance(data, (list, tuple)) or len(data) != 2:
            raise ValueError("Data must be a list/tuple of instantaneous and time-lagged batches")

        # Prepare data
        batch_0, batch_t = self._prepare_batch_data(data)

        # Forward pass and loss computation
        loss_value = self._forward_and_compute_loss(batch_0, batch_t)

        # Backward pass and optimization
        self._backward_and_optimize(loss_value)

        # Handle callbacks and scoring
        self._handle_training_step(loss_value, train_score_callback)

        return self

    def validate(self, validation_data: Tuple[torch.Tensor]) -> torch.Tensor:
        """
        Evaluate current model on validation data.

        Parameters
        ----------
        validation_data : tuple of torch.Tensor
            Tuple containing instantaneous and time-lagged validation data

        Returns
        -------
        torch.Tensor
            Validation score
        """
        with disable_TF32():
            # Set models to eval mode
            self._set_eval_mode()

            with torch.no_grad():
                # Forward pass
                val, val_t = self._forward_validation(validation_data)

                # Compute score
                score_value = self._compute_validation_score(val, val_t)

                return score_value

    def _set_model_precision(self):
        """Set model precision based on dtype."""
        if self.dtype == np.float32:
            self._lobe = self._lobe.float()
            self._lobe_timelagged = self._lobe_timelagged.float()
        elif self.dtype == np.float64:
            self._lobe = self._lobe.double()
            self._lobe_timelagged = self._lobe_timelagged.double()

    def _prepare_batch_data(self, data: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch data for training."""
        batch_0, batch_t = data[0], data[1]

        if isinstance(batch_0, np.ndarray):
            batch_0 = torch.from_numpy(batch_0.astype(self.dtype)).to(device=self.device)
        if isinstance(batch_t, np.ndarray):
            batch_t = torch.from_numpy(batch_t.astype(self.dtype)).to(device=self.device)

        return batch_0, batch_t

    def _forward_and_compute_loss(self, batch_0: torch.Tensor, batch_t: torch.Tensor) -> torch.Tensor:
        """Perform forward pass and compute loss."""
        self.optimizer.zero_grad()
        x_0 = self.lobe(batch_0)
        x_t = self.lobe_timelagged(batch_t)

        if self.score_method == 'VAMPCE':
            return self._vampce_forward(x_0, x_t)
        else:
            return vampnet_loss(x_0, x_t, method=self.score_method,
                                epsilon=self.epsilon, mode=self.score_mode)

    def _vampce_forward(self, x_0: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """Perform VAMPCE-specific forward pass."""
        self._vampu.train()
        self._vamps.train()
        self._vampu.u_kernel.retain_grad()
        self._vamps.s_kernel.retain_grad()

        vampu_out = self._vampu([x_0, x_t])
        vamps_out = self._vamps([x_0, x_t] + list(vampu_out))

        self._K = vamps_out[1][0]
        return vampnet_loss(vamps_out[0], vamps_out[0], method=self.score_method,
                            epsilon=self.epsilon, mode=self.score_mode)

    def estimate_koopman_op(self,
                            trajs: Union[List[np.ndarray], np.ndarray],
                            tau: int) -> np.ndarray:
        """
        Estimate Koopman operator directly from trajectories.

        Parameters
        ----------
        trajs : Union[List[np.ndarray], np.ndarray]
            Input trajectory or list of trajectories
        tau : int
            Lag time for estimation

        Returns
        -------
        np.ndarray
            Estimated Koopman operator
        """
        # Prepare trajectories
        if isinstance(trajs, list):
            traj = np.concatenate([t[:-tau] for t in trajs], axis=0)
            traj_lag = np.concatenate([t[tau:] for t in trajs], axis=0)
        else:
            traj = trajs[:-tau]
            traj_lag = trajs[tau:]

        # Convert to tensors
        traj = torch.as_tensor(traj, device=device)
        traj_lag = torch.as_tensor(traj_lag, device=device)

        # Compute through VAMP networks
        vampu_outputs = self._vampu([traj, traj_lag])
        vamps_outputs = self._vamps([traj, traj_lag] + list(vampu_outputs))

        # Extract and return Koopman operator
        k = vamps_outputs[1][0].detach().cpu().numpy()
        return k

    def estimate_koopman(self, lag: int) -> np.ndarray:
        """
        Estimate Koopman operator using cached results.

        Parameters
        ----------
        lag : int
            Lag time to estimate at

        Returns
        -------
        np.ndarray
            Koopman operator at specified lag time

        Notes
        -----
        Uses caching to avoid recomputing previously estimated operators.
        Updates internal model state when computing new estimates.
        """
        # Return cached result if available
        if lag in self._k_cache:
            return self._k_cache[lag]

        # Compute new estimate
        self.lag = lag
        K = np.array(self._K.detach().cpu())
        self._k_cache[lag] = K

        return K

    def reset_lag(self):
        """
        Reset the model to the original lag time.
        """
        self.lag = self.network_lag

    def get_its(self,
                traj: Union[List[np.ndarray], np.ndarray],
                lags: Union[List[int], np.ndarray],
                dt: float = 1.0) -> np.ndarray:
        """
        Calculate implied timescales (ITS) for a sequence of lag times.

        Parameters
        ----------
        traj : Union[List[np.ndarray], np.ndarray]
            Input trajectory or list of trajectories
        lags : Union[List[int], np.ndarray]
            Sequence of lag times to analyze
        dt : float, default=1.0
            Time step between frames

        Returns
        -------
        np.ndarray
            Array of shape (n_eigenvalues-1, n_lags) containing the implied timescales
        """
        try:
            its = np.empty((self.n_output - 1, len(lags)))

            for i, lag in enumerate(lags):
                # Estimate Koopman operator and compute eigenvalues
                K = self.estimate_koopman_op(traj, lag)
                k_eigvals = np.linalg.eigvals(np.real(K))
                k_eigvals = np.sort(np.abs(np.real(k_eigvals)))[:-1]  # Exclude stationary eigenvalue

                # Convert to implied timescales
                its[:, i] = -lag * dt / np.log(k_eigvals)

            return its
        finally:
            self.reset_lag()

    def ck_test(self,
                steps: int,
                tau: int,
                n_states: int = args.num_classes) -> List[np.ndarray]:
        """
        Perform Chapman-Kolmogorov test comparing predicted vs estimated dynamics.

        Parameters
        ----------
        steps : int
            Number of prediction steps
        tau : int
            Base lag time for predictions
        n_states : int, default=args.num_classes
            Number of states in the system

        Returns
        -------
        List[np.ndarray]
            List containing [predicted, estimated] arrays of shape (n_states, n_states, steps)
        """
        try:
            # Initialize arrays
            predicted = np.zeros((n_states, n_states, steps))
            estimated = np.zeros((n_states, n_states, steps))

            # Set initial conditions (identity matrix)
            predicted[:, :, 0] = np.identity(n_states)
            estimated[:, :, 0] = np.identity(n_states)

            # Set lag time and compute operators
            self.lag = tau
            temp_est = self._compute_koopman_operators(steps, tau, n_states)
            K = temp_est[1]  # Base operator

            # Compute predictions for each basis vector
            for i in range(n_states):
                vec = np.eye(n_states)[i]
                for nn in range(1, steps):
                    estimated[i, :, nn] = vec @ temp_est[nn]
                    predicted[i, :, nn] = vec @ np.linalg.matrix_power(K, nn)

            return [predicted, estimated]

        finally:
            self.reset_lag()

    def _compute_koopman_operators(self,
                                   steps: int,
                                   tau: int,
                                   n_states: int) -> np.ndarray:
        """
        Compute Koopman operators for different lag times.

        Parameters
        ----------
        steps : int
            Number of steps
        tau : int
            Base lag time
        n_states : int
            Number of states

        Returns
        -------
        np.ndarray
            Array containing Koopman operators at different lag times
        """
        temp_est = np.empty((steps, n_states, n_states))
        for nn in range(1, steps):
            temp_est[nn] = self.estimate_koopman(tau * nn)
        return temp_est
