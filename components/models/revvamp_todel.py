class RevVAMPNet(VAMPNet):
    """
    Reversible VAMPNet implementation that extends the original VAMPNet with reversibility constraints.

    This implementation adds:
    - VAMP score computation with reversibility
    - Auxiliary weight optimization
    - Koopman operator estimation
    - Chapman-Kolmogorov test
    - Implied timescale calculation
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
                 dtype=np.float32,
                 n_output: int = None): #TODO: num classes have to be defined?
        """
        Initialize RevVAMPNet.

        Parameters
        ----------
        lobe : nn.Module
            Neural network module for instantaneous data
        lobe_timelagged : Optional[nn.Module]
            Neural network module for time-lagged data
        vampu : Optional[nn.Module]
            VAMP-U network module
        vamps : Optional[nn.Module]
            VAMP-S network module
        device : Optional[torch.device]
            Computation device (CPU/GPU)
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
        super().__init__(lobe, lobe_timelagged, device, optimizer, learning_rate,
                         score_method, score_mode, epsilon, dtype)

        self.n_output = n_output
        self._vampu = vampu
        self._vamps = vamps
        self._k_cache = {}
        self.network_lag = 1  # Default lag time
        self._lag = 1
        self._K = None
        self.data = None

        # Initialize optimizer if using VAMPCE scoring
        if score_method == 'VAMPCE':
            assert vampu is not None and vamps is not None, "vampu and vamps modules must be defined for VAMPCE"
            self.setup_optimizer(optimizer,
                                 list(self.lobe.parameters()) +
                                 list(self.lobe_timelagged.parameters()) +
                                 list(self._vampu.parameters()) +
                                 list(self._vamps.parameters()))

    @property
    def K(self) -> np.ndarray:
        """The estimated Koopman operator."""
        if self._K is None or self._reestimated:
            self._K = np.ones((1, 1))
        return self._K

    @property
    def vampu(self) -> nn.Module:
        """The VAMP-U module.

        Returns
        -------
        lobe : nn.Module
            The VAMP-U neural network module
        """
        return self._vampu

    @property
    def vamps(self) -> nn.Module:
        """The VAMP-S module.

        Returns
        -------
        lobe : nn.Module
            The VAMP-S neural network module
        """
        return self._vamps

    def set_data(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """
        Set the data for the model.

        Parameters
        ----------
        data : Tuple[torch.Tensor, torch.Tensor]
            Tuple containing (instantaneous, time-lagged) data
        """
        self.data = data

    @property
    def lag(self) -> int:
        """
        Get the current lag time of the model.

        Returns
        -------
        int
            Current lag time
        """
        return self._lag

    @lag.setter
    def lag(self, new_lag: int) -> None:
        """
        Update the model lag time and retrain auxiliary networks.

        Parameters
        ----------
        new_lag : int
            New lag time to set

        Notes
        -----
        This method:
        1. Resets VAMP-S weights
        2. Updates auxiliary weights
        3. Retrains the model with the new lag time
        """
        if new_lag == self._lag:
            return

        # Validate input
        if not isinstance(new_lag, int) or new_lag <= 0:
            raise ValueError(f"Lag time must be a positive integer, got {new_lag}")

        # Reset and update weights
        self._reset_and_update_weights()

        # Retrain model with new lag
        self._retrain_with_new_lag(new_lag)

        # Update internal state
        self._update_lag_state(new_lag)

    def _reset_and_update_weights(self) -> None:
        """Reset and update model weights."""
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("No data set for lag time update. Call set_data() first.")

        self._vamps.reset_weights()
        self.update_auxiliary_weights(
            data=self.data,
            optimize_u=False,
            optimize_S=True,
            reset_weights=False
        )

    def _retrain_with_new_lag(self, new_lag: int) -> None:
        """
        Retrain the model with the new lag time.

        Parameters
        ----------
        new_lag : int
            New lag time value
        """
        # Initial training step with logging
        self.train_US(self.data, train_u=False, out_log=True)

        # First training phase: VAMP-S only
        for _ in tqdm(range(LAG_EPOCH), desc="Training VAMP-S"):
            self.train_US(self.data, train_u=False)

        # Second training phase: Both VAMP-U and VAMP-S
        for _ in tqdm(range(LAG_EPOCH), desc="Training VAMP-U and VAMP-S"):
            self.train_US(self.data)

        # Final training step with logging
        self.train_US(self.data, out_log=True)

        logger.info(f"Model successfully updated to lag time {new_lag}")

    def _update_lag_state(self, new_lag: int) -> None:
        """
        Update internal lag state.

        Parameters
        ----------
        new_lag : int
            New lag time value
        """
        self._lag = new_lag
        self._reestimated = True

    def partial_fit(self, data,
                    train_score_callback: Optional[Callable[[int, torch.Tensor], None]] = None) -> 'RevVAMPNet':
        """
        Perform partial fitting of the model on a batch of data.

        Parameters
        ----------
        data : Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]
            Tuple containing instantaneous and time-lagged data
        train_score_callback : Optional[Callable[[int, torch.Tensor], None]]
            Optional callback function for training progress monitoring

        Returns
        -------
        self : RevVAMPNet
            Reference to self for method chaining
        """
        # Set proper dtype and training mode
        self._set_dtype()
        self._set_training_mode()

        # Validate and prepare input data
        batch_0, batch_t = self._prepare_data(data)

        # Forward pass
        self.optimizer.zero_grad()
        x_0 = self.lobe(batch_0)
        x_t = self.lobe_timelagged(batch_t)

        # Compute loss based on scoring method
        loss_value = self._compute_loss(x_0, x_t)

        # Backward pass and optimization
        loss_value.backward()
        self.optimizer.step()

        # Handle callbacks and logging
        self._handle_callbacks(loss_value, train_score_callback)

        return self

    def _set_dtype(self):
        """Set proper dtype for the model."""
        if self.dtype == np.float32:
            self._lobe = self._lobe.float()
            self._lobe_timelagged = self._lobe_timelagged.float()
        elif self.dtype == np.float64:
            self._lobe = self._lobe.double()
            self._lobe_timelagged = self._lobe_timelagged.double()

    def _set_training_mode(self):
        """Set models to training mode."""
        self.lobe.train()
        self.lobe_timelagged.train()
        if self.score_method == 'VAMPCE':
            self._vampu.train()
            self._vamps.train()
            self._vampu.u_kernel.retain_grad()
            self._vamps.s_kernel.retain_grad()

    def _prepare_data(self, data: Tuple[Union[torch.Tensor, np.ndarray], ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Validate and prepare input data.

        Parameters
        ----------
        data : Tuple[Union[torch.Tensor, np.ndarray], ...]
            Input data tuple

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Prepared data tensors
        """
        assert isinstance(data, (list, tuple)) and len(data) == 2, \
            "Data must be a list or tuple of batches for instantaneous and time-lagged data."

        batch_0, batch_t = data[0], data[1]

        if isinstance(batch_0, np.ndarray):
            batch_0 = torch.from_numpy(batch_0.astype(self.dtype)).to(device=self.device)
        if isinstance(batch_t, np.ndarray):
            batch_t = torch.from_numpy(batch_t.astype(self.dtype)).to(device=self.device)

        return batch_0, batch_t

    def _compute_loss(self, x_0: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """
        Compute loss based on scoring method.

        Parameters
        ----------
        x_0 : torch.Tensor
            Instantaneous data embeddings
        x_t : torch.Tensor
            Time-lagged data embeddings

        Returns
        -------
        torch.Tensor
            Computed loss value
        """
        if self.score_method == 'VAMPCE':
            outputs = self._vampu([x_0, x_t])
            u_out, v_out, C00_out, C11_out, C01_out, sigma_out, mu_out = outputs

            vamp_outputs = self._vamps([x_0, x_t, u_out, v_out, C00_out, C11_out, C01_out, sigma_out])
            Ve_out, K_out, p_out, S_out = vamp_outputs

            self._K = K_out[0]
            return vampnet_loss(Ve_out, Ve_out, method=self.score_method,
                                epsilon=self.epsilon, mode=self.score_mode)

        return vampnet_loss(x_0, x_t, method=self.score_method,
                            epsilon=self.epsilon, mode=self.score_mode)

    def _handle_callbacks(self, loss_value: torch.Tensor,
                          train_score_callback: Optional[Callable[[int, torch.Tensor], None]]):
        """
        Handle training callbacks and logging.

        Parameters
        ----------
        loss_value : torch.Tensor
            Current loss value
        train_score_callback : Optional[Callable[[int, torch.Tensor], None]]
            Callback function for training progress
        """
        if train_score_callback is not None:
            lval_detached = loss_value.detach()
            train_score_callback(self._step, -lval_detached)

        self._train_scores.append((self._step, (-loss_value).item()))
        self._step += 1

    def validate(self, validation_data: Tuple[torch.Tensor]) -> torch.Tensor:
        """
        Evaluate the model on validation data and return the configured score.

        Parameters
        ----------
        validation_data : Tuple[torch.Tensor]
            Tuple containing instantaneous and time-lagged validation data

        Returns
        -------
        torch.Tensor
            The validation score
        """
        with self._validation_context():
            # Forward pass through the model
            val, val_t = self._forward_validation(validation_data)

            # Compute score based on scoring method
            score_value = self._compute_validation_score(val, val_t)

            return score_value

    def _validation_context(self):
        """Context manager for validation mode."""

        class ValidationContext:
            def __init__(self, model):
                self.model = model

            def __enter__(self):
                # Disable TF32 for validation
                if hasattr(torch, 'set_float32_matmul_precision'):
                    self.previous_precision = torch.get_float32_matmul_precision()
                    torch.set_float32_matmul_precision('highest')

                # Set models to eval mode
                self.model.lobe.eval()
                self.model.lobe_timelagged.eval()
                if self.model.vamps is not None:
                    self.model.vamps.eval()
                    self.model.vampu.eval()

                torch.set_grad_enabled(False)

            def __exit__(self, exc_type, exc_val, exc_tb):
                # Restore previous state
                if hasattr(torch, 'set_float32_matmul_precision'):
                    torch.set_float32_matmul_precision(self.previous_precision)
                torch.set_grad_enabled(True)

        return ValidationContext(self)

    def _forward_validation(self, validation_data: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform forward pass on validation data.

        Parameters
        ----------
        validation_data : Tuple[torch.Tensor]
            Validation data tuple

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Processed validation data
        """
        val = self.lobe(validation_data[0])
        val_t = self.lobe_timelagged(validation_data[1])
        return val, val_t

    def _compute_validation_score(self, val: torch.Tensor, val_t: torch.Tensor) -> torch.Tensor:
        """
        Compute validation score based on scoring method.

        Parameters
        ----------
        val : torch.Tensor
            Processed instantaneous validation data
        val_t : torch.Tensor
            Processed time-lagged validation data

        Returns
        -------
        torch.Tensor
            Computed validation score
        """
        if self.score_method == 'VAMPCE':
            # Compute VAMPCE score
            u_out, v_out, C00_out, C11_out, C01_out, sigma_out, mu_out = self._vampu([val, val_t])
            Ve_out, K_out, p_out, S_out = self._vamps([
                val, val_t, u_out, v_out, C00_out, C11_out, C01_out, sigma_out
            ])
            return vamp_score(Ve_out, Ve_out,
                              method=self.score_method,
                              mode=self.score_mode,
                              epsilon=self.epsilon)
        else:
            # Compute regular VAMP score
            return vamp_score(val, val_t,
                              method=self.score_method,
                              mode=self.score_mode,
                              epsilon=self.epsilon)

    def update_auxiliary_weights(self,
                                 data: Tuple[Union[np.ndarray, torch.Tensor], ...],
                                 optimize_u: bool = True,
                                 optimize_S: bool = False,
                                 reset_weights: bool = True) -> None:
        """
        Update the weights for the auxiliary model (VAMP-U and VAMP-S).

        Parameters
        ----------
        data : Tuple[Union[np.ndarray, torch.Tensor], ...]
            Tuple containing (chi_0, chi_t) data
        optimize_u : bool, default=True
            Whether to optimize the u vector in VAMP-U
        optimize_S : bool, default=False
            Whether to optimize the S matrix in VAMP-S
        reset_weights : bool, default=True
            Whether to reset weights (currently unused)

        Returns
        -------
        None
        """
        # Prepare data
        chi_0, chi_t = self._prepare_auxiliary_data(data)

        # Compute initial estimates
        C0inv, Ctau = self._compute_covariances(chi_0, chi_t)
        K = self._compute_initial_koopman(C0inv, Ctau)

        # Update VAMP-U weights if requested
        if optimize_u:
            self._update_vampu_weights(C0inv, K)

        # Update VAMP-S weights if requested
        if optimize_S:
            self._update_vamps_weights(chi_0, chi_t, K)

    def _prepare_auxiliary_data(self,
                                data: Tuple[Union[np.ndarray, torch.Tensor], ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for auxiliary weight updates."""
        batch_0, batch_t = data[0], data[1]
        chi_0 = torch.as_tensor(batch_0, device=self.device)
        chi_t = torch.as_tensor(batch_t, device=self.device)
        return chi_0, chi_t

    def _compute_covariances(self,
                             chi_0: torch.Tensor,
                             chi_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute covariance matrices."""
        C0inv, Ctau = covariances_E(chi_0, chi_t)
        return C0inv, Ctau

    def _compute_initial_koopman(self,
                                 C0inv: torch.Tensor,
                                 Ctau: torch.Tensor) -> torch.Tensor:
        """Compute initial Koopman operator estimate."""
        # Get initial VAMP estimates
        vampu_outputs = self._vampu([chi_0, chi_t])
        vamps_outputs = self._vamps([chi_0, chi_t] + list(vampu_outputs))

        # Update Koopman operator
        K = torch.matmul(C0inv, Ctau.to('cpu'))
        self._K = vamps_outputs[1][0]  # K_out[0]

        return K

    def _update_vampu_weights(self,
                              C0inv: torch.Tensor,
                              K: torch.Tensor) -> None:
        """Update VAMP-U weights."""
        pi = _compute_pi(K)
        u_kernel = torch.log(torch.abs(torch.matmul(C0inv, pi)))

        with torch.no_grad():
            for param in self.vampu.parameters():
                param.copy_(u_kernel)

    def _update_vamps_weights(self,
                              chi_0: torch.Tensor,
                              chi_t: torch.Tensor,
                              K: torch.Tensor) -> None:
        """Update VAMP-S weights."""
        # Get VAMP-U outputs for S matrix computation
        vampu_outputs = self.vampu([chi_0, chi_t])
        sigma = vampu_outputs[5]  # Index 5 contains sigma

        # Compute S matrix
        sigma_inv = matrix_inverse(sigma[0])
        S_nonrev = torch.matmul(K, sigma_inv)
        S_rev = 0.5 * (S_nonrev + S_nonrev.t())
        s_kernel = torch.log(torch.abs(0.5 * S_rev))

        # Update VAMP-S weights
        with torch.no_grad():
            for param in self.vamps.parameters():
                param.copy_(s_kernel)

    def estimate_koopman(self, lag: int) -> np.ndarray:
        """
        Estimate the Koopman operator for a given lag time.

        Parameters
        ----------
        lag : int
            Lag time to estimate at

        Returns
        -------
        np.ndarray
            Koopman operator at specified lag time
        """
        # Return cached result if available
        if lag in self._k_cache:
            return self._k_cache[lag]

        # Compute new estimate
        self.lag = lag
        K = np.array(self._K.detach().cpu())
        self._k_cache[lag] = K

        return K

    def get_ck_test(self,
                    traj: Union[List[np.ndarray], np.ndarray],
                    steps: int,
                    tau: int) -> List[np.ndarray]:
        """
        Perform Chapman-Kolmogorov test comparing predicted vs estimated dynamics.

        Parameters
        ----------
        traj : Union[List[np.ndarray], np.ndarray]
            Input trajectory or list of trajectories
        steps : int
            Number of prediction steps
        tau : int
            Lag time for predictions

        Returns
        -------
        List[np.ndarray]
            List containing [predicted, estimated] arrays of shape (n_states, n_states, steps)
        """
        # Get number of states from trajectory data
        n_states = traj[0].shape[1] if isinstance(traj, list) else traj.shape[1]

        # Initialize arrays for predicted and estimated dynamics
        predicted = np.zeros((n_states, n_states, steps))
        estimated = np.zeros((n_states, n_states, steps))

        # Set initial condition (identity matrix)
        predicted[:, :, 0] = np.identity(n_states)
        estimated[:, :, 0] = np.identity(n_states)

        # Compute predictions for each basis vector
        for i, vector in enumerate(np.identity(n_states)):
            for n in range(1, steps):
                # Get Koopman operators
                koop = self.estimate_koopman_op(traj, tau)
                koop_pred = np.linalg.matrix_power(koop, n)
                koop_est = self.estimate_koopman_op(traj, tau * n)

                # Compute predictions
                predicted[i, :, n] = vector @ koop_pred
                estimated[i, :, n] = vector @ koop_est

        return [predicted, estimated]

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
                lambdas = np.linalg.eig(np.real(K))[0]  # Get eigenvalues only
                lambdas = np.sort(np.abs(np.real(lambdas)))[:LAST]

                # Convert to implied timescales
                its[:, i] = -lag * dt / np.log(lambdas)

            return its
        finally:
            self.reset_lag()







    def estimate_koopman(self, lag: int) -> np.ndarray:
        """
        Estimate Koopman operator for given lag time.

        Parameters
        ----------
        lag : int
            Lag time for estimation

        Returns
        -------
        K : np.ndarray
            Estimated Koopman operator
        """
        if lag in self._k_cache:
            return self._k_cache[lag]

        self.lag = lag
        K = np.array(self._K.detach().cpu())
        self._k_cache[lag] = K
        return K

    def get_its(self, traj, lags, dt: float = 1.0):
        """
        Calculate implied timescales for multiple lag times.

        Parameters
        ----------
        traj : Union[np.ndarray, List[np.ndarray]]
            Trajectory data
        lags : List[int]
            List of lag times
        dt : float
            Time step

        Returns
        -------
        its : np.ndarray
            Implied timescales
        """
        its = np.empty((self.n_output - 1, len(lags)))

        for i, lag in enumerate(lags):
            K = self.estimate_koopman_op(traj, lag)
            k_eigvals = np.linalg.eigvals(K)
            k_eigvals = np.sort(np.abs(np.real(k_eigvals)))[:-1]
            its[:, i] = -lag * dt / np.log(k_eigvals)

        self.reset_lag()
        return its

    # Add helper methods as needed
    def _set_dtype(self):
        """Set proper dtype for the model."""
        if self.dtype == np.float32:
            self._lobe = self._lobe.float()
            self._lobe_timelagged = self._lobe_timelagged.float()
        elif self.dtype == np.float64:
            self._lobe = self._lobe.double()
            self._lobe_timelagged = self._lobe_timelagged.double()

    def _prepare_data(self, data):
        """Prepare input data for training."""
        assert isinstance(data, (list, tuple)) and len(data) == 2, \
            "Data must be a list or tuple of batches for instantaneous and time-lagged data."

        batch_0, batch_t = data[0], data[1]

        if isinstance(batch_0, np.ndarray):
            batch_0 = torch.from_numpy(batch_0.astype(self.dtype)).to(device=self.device)
        if isinstance(batch_t, np.ndarray):
            batch_t = torch.from_numpy(batch_t.astype(self.dtype)).to(device=self.device)

        return batch_0, batch_t

    def _compute_loss(self, x_0, x_t):
        """Compute loss based on score method."""
        if self.score_method == 'VAMPCE':
            return self._compute_vampce_loss(x_0, x_t)
        return vampnet_loss(x_0, x_t, method=self.score_method,
                            epsilon=self.epsilon, mode=self.score_mode)

    def _compute_vampce_loss(self, x_0, x_t):
        """Compute VAMPCE-specific loss."""
        self._vampu.train()
        self._vamps.train()

        outputs = self._vampu([x_0, x_t])
        ve_out = self._vamps([x_0, x_t] + list(outputs))

        self._K = ve_out[1][0]  # Store Koopman operator
        return vampnet_loss(ve_out[0], ve_out[0], method=self.score_method,
                            epsilon=self.epsilon, mode=self.score_mode)
