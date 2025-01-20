import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Callable, Tuple, List
from deeptime.decomposition.deep import VAMPNet
from deeptime.util.torch import disable_TF32, multi_dot, map_data
from tqdm import tqdm
from args.args import buildParser
from components.scores.vamp_score import vamp_score
from components.losses.vampnet_loss import vampnet_loss
from components.computations.computations import covariances_E, matrix_inverse, _compute_pi


# Device configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('cuda is available')
else:
    print('Using CPU')
    device = torch.device('cpu')

LAG_EPOCH = 1000


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
                 n_output: Optional[int] = None,
                 args=None):

        # Initialize parser for default values if args not provided
        if args is None:
            parser = buildParser()
            args = parser.parse_args([])

        # Store args for later use
        self.args = args

        # Set up valid score methods (to extend the deeptime vampnet)
        self.valid_score_methods = ('VAMP1', 'VAMP2', 'VAMPE', 'VAMPCE')

        # Initialize step counter
        self._step = 0

        super().__init__(lobe, lobe_timelagged, device, optimizer, learning_rate,
                         score_method, score_mode, epsilon, dtype)

        # Use provided n_output or fall back to args.num_classes
        self.n_output = n_output if n_output is not None else args.num_classes

        # Initialize additional attributes
        self._vampu = vampu
        self._vamps = vamps
        self._k_cache = {}
        self.network_lag = args.tau
        self._lag = args.tau
        self._K = None
        self.data = None
        self._device = device
        self._dtype = dtype

        # Initialize optimizer
        #if isinstance(optimizer, str):
        #    optimizer = getattr(torch.optim, optimizer)

        # Collect all parameters
        #all_params = list(self.lobe.parameters())
        #if self.lobe_timelagged is not None:
        #    all_params.extend(list(self.lobe_timelagged.parameters()))
        #if self._vampu is not None:
        #    all_params.extend(list(self._vampu.parameters()))
        #if self._vamps is not None:
        #    all_params.extend(list(self._vamps.parameters()))

        # Create optimizer
        # self.optimizer = optimizer(all_params, lr=learning_rate)

        # Validate VAMPCE configuration
        if score_method == 'VAMPCE':
            assert vampu is not None and vamps is not None, f"vampu and vamps module must be defined "
            self.setup_optimizer(optimizer, list(self.lobe.parameters()) + list(self.lobe_timelagged.parameters()) +
                                 list(self._vampu.parameters()) + list(self._vamps.parameters()))

    def score_method(self, value: str):
        """
        Set the scoring method if it's valid. (this has to be done to extend the deeptime vampnet)

        Args:
            value (str): The scoring method to be set

        Raises:
            AssertionError: If the provided value is not in valid_score_methods
        """
        if value not in self.valid_score_methods:
            raise ValueError(
                f"Invalid scoring method '{value}'. "
                f"Available methods: {self.valid_score_methods}"
            )
        self._score_method = value

    def transform(self, data, instantaneous: bool = True, **kwargs):
        """Transforms data through the instantaneous or time-shifted network lobe.

        Parameters
        ----------
        data : numpy array or torch tensor
            The data to transform.
        instantaneous : bool, default=True
            Whether to use the instantaneous lobe or the time-shifted lobe for transformation.
        **kwargs
            Ignored kwargs for api compatibility.

        Returns
        -------
        transform : array_like
            List of numpy array or numpy array containing transformed data.
        """
        # Select appropriate network without calling eval()
        net = self._lobe if instantaneous else self._lobe_timelagged

        # Process data through network
        out = []
        for data_tensor in map_data(data, device=self._device, dtype=self._dtype):
            with torch.no_grad():
                out.append(net(data_tensor).cpu().numpy())

        return out if len(out) > 1 else out[0]

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
        try:
            # Set model precision
            self._set_model_precision()

            # Set models to training mode
            self.lobe.train()
            self.lobe_timelagged.train()

            # Set VAMP networks to training mode if they exist
            if hasattr(self, '_vampu') and self._vampu is not None:
                self._vampu.train()
            if hasattr(self, '_vamps') and self._vamps is not None:
                self._vamps.train()

            # Store data for later use
            self.data = data

            # Validate input
            if not isinstance(data, (list, tuple)) or len(data) != 2:
                raise ValueError("Data must be a list/tuple of instantaneous and time-lagged batches")

            # Prepare data
            batch_0, batch_t = self._prepare_batch_data(data)

            # Forward pass and loss computation
            loss_value = self._forward_and_compute_loss(batch_0, batch_t)

            # Backward pass and optimization
            if self.optimizer is not None:
                self.optimizer.zero_grad()
                loss_value.backward()
                self.optimizer.step()

            # Handle callbacks and scoring
            if train_score_callback is not None:
                with torch.no_grad():
                    train_score_callback(self._step, loss_value)

            self._step += 1
            return self

        except Exception as e:
            print(f"Partial fit failed with error: {str(e)}")
            raise

    def validate(self, validation_data: Tuple[torch.Tensor]) -> torch.Tensor:
        """
        Evaluate the current model on validation data and return the configured score.

        Parameters
        ----------
        validation_data : Tuple[torch.Tensor]
            Tuple containing (x_t, x_t+τ) validation data tensors, where:
            - x_t is the instantaneous data
            - x_t+τ is the time-lagged data

        Returns
        -------
        torch.Tensor
            Computed validation score based on the configured scoring method
        """
        with disable_TF32():
            # Set all networks to evaluation mode
            self.lobe.eval()
            self.lobe_timelagged.eval()
            if self.vamps is not None:
                self.vamps.eval()
                self.vampu.eval()

            with torch.no_grad():
                # Forward pass through the networks
                val_t0 = self.lobe(validation_data[0])
                val_tlag = self.lobe_timelagged(validation_data[1])

                # Compute score based on method
                if self.score_method == 'VAMPCE':
                    # Compute VAMP-U outputs
                    (u_out, v_out, C00_out, C11_out,
                     C01_out, sigma_out, mu_out) = self._vampu([val_t0, val_tlag])

                    # Compute VAMP-S outputs and final score
                    Ve_out, K_out, p_out, S_out = self._vamps([
                        val_t0, val_tlag, u_out, v_out, C00_out,
                        C11_out, C01_out, sigma_out])

                    score_value = vamp_score(
                        Ve_out, Ve_out,
                        method=self.score_method,
                        mode=self.score_mode,
                        epsilon=self.epsilon
                    )
                else:
                    score_value = vamp_score(
                        val_t0, val_tlag,
                        method=self.score_method,
                        mode=self.score_mode,
                        epsilon=self.epsilon
                    )

                return score_value

    def update_auxiliary_weights(self, data, optimize_u: bool = True, optimize_S: bool = False,
                                 reset_weights: bool = True):
        """
        Update the weights for the auxiliary model and return new output.

        Parameters
        ----------
        data : tuple
            Tuple containing (chi_0, chi_t) time-lagged data
        optimize_u : bool, optional
            Whether to optimize the u vector, by default True
        optimize_S : bool, optional
            Whether to optimize the S matrix, by default False
        reset_weights : bool, optional
            Whether to reset the weights for the vanilla VAMPNet model, by default True

        Returns
        -------
        tuple
            Updated K matrix and optimized parameters
        """
        # Unpack and prepare data
        batch_0, batch_t = data[0], data[1]
        chi_0 = torch.Tensor(batch_0).to(device)
        chi_t = torch.Tensor(batch_t).to(device)

        # Compute covariances
        C0inv, Ctau = covariances_E(chi_0, chi_t)

        # Get initial VAMP parameters
        (u_outd, v_outd, C00_outd, C11_outd,
         C01_outd, sigma_outd, mu_outd) = self._vampu([chi_0, chi_t])

        # Get initial VAMPS parameters
        Ve_out, K_out, p_out, S_out = self._vamps([
            chi_0, chi_t, u_outd, v_outd, C00_outd,
            C11_outd, C01_outd, sigma_outd])

        # Compute Koopman operator
        K = torch.Tensor(C0inv) @ Ctau.to('cpu')
        self._K = K_out[0]

        # Update u vector if requested
        if optimize_u:
            pi = _compute_pi(K)
            u_kernel = np.log(np.abs(C0inv @ pi))
            for param in self.vampu.parameters():
                with torch.no_grad():
                    param[:] = torch.Tensor(u_kernel)

        # Update S matrix if requested
        if optimize_S:
            # Compute updated VAMP parameters with new u vector
            (u_out, v_out, C00_out, C11_out,
             C01_out, sigma, mu_out) = self.vampu([chi_0, chi_t])

            # Compute S matrix
            sigma_inv = matrix_inverse(sigma[0])
            S_nonrev = K @ sigma_inv
            S_rev = 0.5 * (S_nonrev + S_nonrev.t())
            s_kernel = np.log(np.abs(0.5 * S_rev))

            # Update VAMPS parameters
            for param in self.vamps.parameters():
                with torch.no_grad():
                    param[:] = torch.Tensor(s_kernel)

        return self._K

    def train_US(self, data, lr_rate=1e-3, train_u=True, out_log=False):
        """Train the U and S networks of the VAMPNet model.

        Args:
            data: Tuple of (x_0, x_t) containing instantaneous and time-lagged data
            lr_rate: Learning rate for optimization
            train_u: Whether to train the U network
            out_log: Whether to print loss values
        """
        # Freeze main network
        self.lobe.requires_grad_(False)
        self.lobe_timelagged.requires_grad_(False)

        # Configure U network training
        if train_u:
            self._vampu.train()
            self._vampu.requires_grad_(True)
            self._vampu.u_kernel.retain_grad()
        else:
            self._vampu.requires_grad_(False)

        # Configure S network training
        self._vamps.train()
        self._vamps.s_kernel.retain_grad()

        # Prepare data
        self.optimizer.zero_grad()
        x_0 = torch.Tensor(data[0]).to(device)
        x_t = torch.Tensor(data[1]).to(device)

        # Forward pass through U network
        u_out, v_out, C00_out, C11_out, C01_out, sigma_out, mu_out = self._vampu([x_0, x_t])

        # Forward pass through S network
        Ve_out, K_out, p_out, S_out = self._vamps([
            x_0, x_t, u_out, v_out, C00_out,
            C11_out, C01_out, sigma_out
        ])

        # Store Koopman matrix
        self._K = K_out[0]

        # Calculate loss and update weights
        loss_value = vampnet_loss(
            Ve_out,
            Ve_out,
            method=self.score_method,
            epsilon=self.epsilon,
            mode=self.score_mode
        )
        loss_value.backward()
        self.optimizer.step()

        # Log loss if requested
        if out_log:
            print(f"Loss: {loss_value.item():.4f}")

        # Restore network states
        self.lobe.requires_grad_(True)
        self.lobe_timelagged.requires_grad_(True)
        self.lobe.train()
        self.lobe_timelagged.train()

        if not train_u:
            self._vampu.requires_grad_(True)
            self._vampu.train()

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
                n_states: Optional[int] = None) -> List[np.ndarray]:
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
        if n_states is None:
            n_states = self.n_output
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
