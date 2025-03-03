import torch
import torch.nn as nn
from typing import Optional, Union, Callable, Tuple, List
import numpy as np
from deeptime.base import EstimatorTransformer
from deeptime.base_torch import DLEstimatorMixin
from deeptime.util.torch import disable_TF32
from tqdm import tqdm
from components.losses import vampnet_loss
from components.scores import vamp_score
from components.computations import covariances_E, _compute_pi, matrix_inverse
from components.models.VAMPNet import VAMPNetModel


class RevVAMPNet(EstimatorTransformer, DLEstimatorMixin, nn.Module):
    r""" Implementation of VAMPNets. :footcite:`mardt2018vampnets`
    These networks try to find an optimal featurization of data based on a VAMP score :footcite:`wu2020variational`
    by using neural networks as featurizing transforms which are equipped with a loss that is the negative VAMP score.
    This estimator is also a transformer and can be used to transform data into the optimized space.
    From there it can either be used to estimate Markov state models via making assignment probabilities
    crisp (in case of softmax output distributions) or to estimate the Koopman operator
    using the :class:`VAMP <deeptime.decomposition.VAMP>` estimator.

    Parameters
    ----------
    lobe : torch.nn.Module
        A neural network module which maps input data to some (potentially) lower-dimensional space.
    lobe_timelagged : torch.nn.Module, optional, default=None
        Neural network module for timelagged data, in case of None the lobes are shared (structure and weights).
    device : torch device, default=None
        The device on which the torch modules are executed.
    optimizer : str or Callable, default='Adam'
        An optimizer which can either be provided in terms of a class reference (like `torch.optim.Adam`) or
        a string (like `'Adam'`). Defaults to Adam.
    learning_rate : float, default=5e-4
        The learning rate of the optimizer.
    score_method : str, default='VAMP2'
        The scoring method which is used for optimization.
    score_mode : str, default='regularize'
        The mode under which inverses of positive semi-definite matrices are estimated. Per default, the matrices
        are perturbed by a small constant added to the diagonal. This makes sure that eigenvalues are not too
        small. For a complete list of modes, see :meth:`sym_inverse`.
    epsilon : float, default=1e-6
        The strength of the regularization under which matrices are inverted. Meaning depends on the score_mode,
        see :meth:`sym_inverse`.
    dtype : dtype, default=np.float32
        The data type of the modules and incoming data.

    See Also
    --------
    deeptime.decomposition.VAMP

    References
    ----------
    .. footbibliography::
    """
    _MUTABLE_INPUT_DATA = True

    def __init__(self, lobe: "torch.nn.Module",
                 lobe_timelagged: Optional["torch.nn.Module"] = None,
                 vampu: Optional[nn.Module] = None,
                 vamps: Optional[nn.Module] = None,
                 device=None, optimizer: Union[str, Callable] = 'Adam',
                 learning_rate: float = 5e-4,
                 activation_vampu: Optional["torch.nn.Module"] = None,
                 activation_vamps: Optional["torch.nn.Module"] = None,
                 num_classes: int = 1,
                 tau: int = 20,
                 score_method: str = 'VAMP2',
                 score_mode: str = 'regularize',
                 epsilon: float = 1e-6,
                 dtype=np.float32):

        # Initialize parent classes
        EstimatorTransformer.__init__(self)
        DLEstimatorMixin.__init__(self)
        nn.Module.__init__(self)

        # Register networks as modules
        self.lobe = lobe
        self.lobe_timelagged = lobe_timelagged or lobe
        self.add_module('lobe', self.lobe)
        self.add_module('lobe_timelagged', self.lobe_timelagged)

        # Set up configuration
        self.valid_score_methods = ('VAMP1', 'VAMP2', 'VAMPE', 'VAMPCE')
        self.score_method = score_method
        self.score_mode = score_mode
        self._step = 0
        self._epsilon = epsilon
        self.device = device
        self.learning_rate = learning_rate
        self.dtype = dtype
        self._train_scores = []
        self._validation_scores = []

        # Initialize VAMP networks
        #self._vampu = VAMPU(units=num_classes, activation=activation_vampu, device=device)
        #self._vamps = VAMPS(units=num_classes, activation=activation_vamps, device=device)
        self._vampu = vampu
        self._vamps = vamps
        self.add_module('_vampu', self._vampu)
        self.add_module('_vamps', self._vamps)

        # Initialize caches and state
        self._k_cache = {}
        self.network_lag = tau
        self._lag = tau
        self._K = None
        self.data = None
        self.LAG_EPOCH = 1000
        self.LAST = -1

        # Setup optimizer based on score method
        if score_method == "VAMPCE":
            assert self._vampu is not None and self._vamps is not None, "vampu and vamps module must be defined"
            all_params = (list(self.lobe.parameters()) +
                          list(self.lobe_timelagged.parameters()) +
                          list(self._vampu.parameters()) +
                          list(self._vamps.parameters()))
            self.setup_optimizer(optimizer, all_params)
        else:
            all_params = (list(self.lobe.parameters()) +
                          list(self.lobe_timelagged.parameters()))
            self.setup_optimizer(optimizer, all_params)

        # Use DLEstimatorMixin's method to set up optimizer
        if isinstance(optimizer, str):
            optimizer_cls = getattr(torch.optim, optimizer)
            self._optimizer = optimizer_cls(all_params, lr=learning_rate)
        else:
            self._optimizer = optimizer(all_params, lr=learning_rate)

    @property
    def optimizer(self):
        return self._optimizer

    def set_optimizer_lr(self, new_lr: float) -> None:
        """
        Set a new learning rate for all parameter groups in the optimizer.

        Args:
            new_lr (float): New learning rate value to set

        Example:
            vampnet.set_optimizer_lr(0.2)  # Sets learning rate to 0.2
        """
        if not hasattr(self, 'optimizer'):
            raise AttributeError("No optimizer found. Initialize optimizer before setting learning rate.")

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        # Optional: log the learning rate change
        if hasattr(self, '_step'):
            print(f"Step {self._step}: Learning rate set to {new_lr}")

    def reduce_optimizer_lr(self, multiplier: float) -> None:
        """
        Reduce the current learning rate by multiplying it with the given factor.

        Args:
            multiplier (float): Factor to multiply the current learning rate with (0 < multiplier < 1)

        Example:
            vampnet.reduce_optimizer_lr(0.2)  # Reduces current learning rate by factor of 0.2
        """
        if not hasattr(self, 'optimizer'):
            raise AttributeError("No optimizer found. Initialize optimizer before modifying learning rate.")

        if not 0 < multiplier < 1:
            raise ValueError("Multiplier must be between 0 and 1 to reduce learning rate.")

        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
            new_lr = current_lr * multiplier
            param_group['lr'] = new_lr

        # Optional: log the learning rate change
        if hasattr(self, '_step'):
            print(f"Step {self._step}: Learning rate reduced from {current_lr:.6f} to {new_lr:.6f}")

    def parameters(self, recurse: bool = True):
        """Return an iterator over module parameters.

        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Returns:
            Iterator over parameters
        """
        if self.score_method == "VAMPCE":
            params = (list(self.lobe.parameters(recurse)) +
                      list(self.lobe_timelagged.parameters(recurse)) +
                      list(self._vampu.parameters(recurse)) +
                      list(self._vamps.parameters(recurse)))
        else:
            params = (list(self.lobe.parameters(recurse)) +
                      list(self.lobe_timelagged.parameters(recurse)))

        for param in params:
            yield param

    @property
    def K(self) -> np.ndarray:
        """The estimated Koopman operator."""
        if self._K is None: # or self._reestimated: MARKER
            self._K = np.ones((1, 1))

        return self._K

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
        for _ in tqdm(range(self.LAG_EPOCH)):
            self.train_US(data, train_u=False)

        # Second training phase - update both U and S
        for _ in tqdm(range(self.LAG_EPOCH)):
            self.train_US(data)

        # Final training step with logging
        self.train_US(data, out_log=True)
        print(f"new lag {lag} ok")

        # Update internal state
        self._lag = lag
        self._reestimated = True

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
    def train_scores(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._train_scores)

    @property
    def validation_scores(self) -> np.ndarray:
        r""" The collected validation scores. First dimension contains the step, second dimension the score.
        Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._validation_scores)

    @property
    def epsilon(self) -> float:
        r""" Regularization parameter for matrix inverses.

        :getter: Gets the currently set parameter.
        :setter: Sets a new parameter. Must be non-negative.
        :type: float
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        assert value >= 0
        self._epsilon = value

    @property
    def score_method(self) -> str:
        r""" Property which steers the scoring behavior of this estimator.

        :getter: Gets the current score.
        :setter: Sets the score to use.
        :type: str
        """
        return self._score_method

    @score_method.setter
    def score_method(self, value: str):
        assert value in self.valid_score_methods, f"Tried setting an unsupported scoring method '{value}', " \
                                             f"available are {self.valid_score_methods}."
        self._score_method = value

    @property
    def lobe(self) -> "torch.nn.Module":
        r""" The instantaneous lobe of the VAMPNet.

        :getter: Gets the instantaneous lobe.
        :setter: Sets a new lobe.
        :type: torch.nn.Module
        """
        return self._lobe

    @lobe.setter
    def lobe(self, value: "torch.nn.Module"):
        self._lobe = value
        if self.dtype == np.float32:
            self._lobe = self._lobe.float()
        else:
            self._lobe = self._lobe.double()
        self._lobe = self._lobe.to(device=self.device)

    @property
    def lobe_timelagged(self) -> "torch.nn.Module":
        r""" The timelagged lobe of the VAMPNet.

        :getter: Gets the timelagged lobe. Can be the same a the instantaneous lobe.
        :setter: Sets a new lobe. Can be None, in which case the instantaneous lobe is shared.
        :type: torch.nn.Module
        """
        return self._lobe_timelagged

    @lobe_timelagged.setter
    def lobe_timelagged(self, value: Optional["torch.nn.Module"]):
        if value is None:
            value = self.lobe
        else:
            if self.dtype == np.float32:
                value = value.float()
            else:
                value = value.double()
        self._lobe_timelagged = value
        self._lobe_timelagged = self._lobe_timelagged.to(device=self.device)

    def check_gradients(self):
        """
        Check for NaN or infinity values in gradients of all parameters.
        Raises:
            ValueError: If any NaN or infinity values are detected in gradients.
        """
        gradient_issues = {}

        def check_grad(name, parameter):
            if parameter.grad is not None:
                if torch.isnan(parameter.grad).any():
                    gradient_issues[name] = {'issue': 'NaN', 'location': parameter.grad}
                if torch.isinf(parameter.grad).any():
                    gradient_issues[name] = {'issue': 'Inf', 'location': parameter.grad}

        # Check all network components
        for name, param in self.lobe.named_parameters():
            check_grad(f'lobe.{name}', param)

        for name, param in self.lobe_timelagged.named_parameters():
            check_grad(f'lobe_timelagged.{name}', param)

        if self.score_method == "VAMPCE":
            for name, param in self._vampu.named_parameters():
                check_grad(f'vampu.{name}', param)
            for name, param in self._vamps.named_parameters():
                check_grad(f'vamps.{name}', param)

        if gradient_issues:
            error_msg = "Gradient issues detected:\n"
            for name, issue in gradient_issues.items():
                error_msg += f"Parameter {name}: {issue['issue']} values detected\n"
            raise ValueError(error_msg)

    def stabilize_training(self, loss_value):
        """
        Stabilize training by handling various numerical issues including vanishing gradients.
        """
        diagnostic_info = {
            'loss_value': loss_value.item(),
            'u_out_stats': None,
            's_out_stats': None
        }
        # 1. Detailed loss value checks
        if torch.isnan(loss_value):
            raise ValueError(f"Loss value is NaN: {loss_value}")
        elif torch.isinf(loss_value):
            # Get the last layer outputs for diagnosis
            try:
                # Get VAMPU output stats if available
                if hasattr(self._vampu, 'last_output') and self._vampu.last_output is not None:
                    u_out = self._vampu.last_output
                    diagnostic_info['u_out_stats'] = {
                        'min': u_out.min().item(),
                        'max': u_out.max().item(),
                        'mean': u_out.mean().item(),
                        'has_nan': torch.isnan(u_out).any().item(),
                        'has_inf': torch.isinf(u_out).any().item()
                    }

                # Get VAMPS output stats if available
                if hasattr(self._vamps, 'last_output') and self._vamps.last_output is not None:
                    s_out = self._vamps.last_output
                    diagnostic_info['s_out_stats'] = {
                        'min': s_out.min().item(),
                        'max': s_out.max().item(),
                        'mean': s_out.mean().item(),
                        'has_nan': torch.isnan(s_out).any().item(),
                        'has_inf': torch.isinf(s_out).any().item()
                    }
            except:
                diagnostic_info = {'loss_value': loss_value.item()}

            raise ValueError(
                f"Loss value is infinite: {loss_value} "
                f"(positive infinite: {torch.isposinf(loss_value)}, "
                f"negative infinite: {torch.isneginf(loss_value)})\n"
                f"Diagnostic information:\n{diagnostic_info}"
            )

        # 2. Use gradient scaling for mixed precision and stability
        scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None

        try:
            # 3. Compute backward pass with scaled gradients
            if scaler:
                scaler.scale(loss_value).backward()
            else:
                loss_value.backward()

            # 4. Check for vanishing gradients
            grad_norm = 0.0
            num_params_with_grad = 0
            for name, param in self.named_parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item()
                    num_params_with_grad += 1

            if num_params_with_grad > 0:
                avg_grad_norm = grad_norm / num_params_with_grad

                # If gradients are too small, scale them up
                if avg_grad_norm < 1e-8:
                    scale_factor = 1e-4 / (avg_grad_norm + 1e-12)
                    for param in self.parameters():
                        if param.grad is not None:
                            param.grad.data.mul_(scale_factor)

            # 5. Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            # 6. Optimizer step with scaled gradients
            if scaler:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()

            # 7. Learning rate adjustment if gradients are consistently small
            if hasattr(self, 'grad_history'):
                self.grad_history.append(avg_grad_norm)
                if len(self.grad_history) > 10:
                    self.grad_history.pop(0)
                    if np.mean(self.grad_history) < 1e-6:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= 1.5
            else:
                self.grad_history = [avg_grad_norm]

        except RuntimeError as e:
            self.optimizer.zero_grad()
            raise RuntimeError(f"Backward pass failed with diagnostic info:\n{diagnostic_info}\nError: {str(e)}")

    def partial_fit(self, data, train_score_callback: Callable[[int, "torch.Tensor"], None] = None):
        r""" Performs a partial fit on data. This does not perform any batching.

        Parameters
        ----------
        data : tuple or list of length 2, containing instantaneous and timelagged data
            The data to train the lobe(s) on.
        train_score_callback : callable, optional, default=None
            An optional callback function which is evaluated after partial fit, containing the current step
            of the training (only meaningful during a :meth:`fit`) and the current score as torch Tensor.

        Returns
        -------
        self : VAMPNet
            Reference to self.
        """
        #self.check_gradients()
        if self.dtype == np.float32:
            self.lobe = self.lobe.float()
            self.lobe_timelagged = self.lobe_timelagged.float()
        elif self.dtype == np.float64:
            self.lobe = self.lobe.double()
            self.lobe_timelagged = self.lobe_timelagged.double()

        self.train()
        self.lobe.train()
        self.lobe_timelagged.train()

        assert isinstance(data, (list, tuple)) and len(data) == 2, \
            "Data must be a list or tuple of batches belonging to instantaneous " \
            "and respective time-lagged data."

        batch_0, batch_t = data[0], data[1]

        if isinstance(data[0], np.ndarray):
            batch_0 = torch.from_numpy(data[0].astype(self.dtype)).to(device=self.device)
        if isinstance(data[1], np.ndarray):
            batch_t = torch.from_numpy(data[1].astype(self.dtype)).to(device=self.device)

        # TODO: THIS IS NOT WORKING, THERE IS AN ERROR HERE!
        # If the tensors are already torch tensors but not float32, convert them
        if isinstance(batch_0, torch.Tensor) and batch_0.dtype != torch.float32:
            batch_0 = batch_0.float()
        if isinstance(batch_t, torch.Tensor) and batch_t.dtype != torch.float32:
            batch_t = batch_t.float()

        # Ensure inputs require gradients
        #batch_0, batch_t = batch_data[0].to(self.device), batch_data[1].to(self.device)
        batch_0.requires_grad_(True)
        batch_t.requires_grad_(True)

        self.optimizer.zero_grad()
        x_0 = self.lobe(batch_0)
        x_t = self.lobe_timelagged(batch_t)

        if self.score_method == 'VAMPCE':
            self._vampu.train()
            self._vamps.train()
            self._vampu.u_kernel.retain_grad()
            self._vamps.s_kernel.retain_grad()
            (u_out, v_out, C00_out, C11_out, C01_out, sigma_out, mu_out) = self._vampu([x_0, x_t])
            Ve_out, K_out, p_out, S_out = self._vamps([x_0, x_t, u_out, v_out, C00_out,C11_out, C01_out, sigma_out])
            self._K = K_out[0]
            loss_value = vampnet_loss(Ve_out, Ve_out, method=self.score_method, epsilon=self.epsilon,
                                      mode=self.score_mode)
        else:
            loss_value = vampnet_loss(x_0, x_t, method=self.score_method, epsilon=self.epsilon, mode=self.score_mode)
        # Stabilized Optimizer with Grad Scaling to prevent explosion or Vanish
        self.stabilize_training(loss_value)
        torch.cuda.synchronize()

        if train_score_callback is not None:
            lval_detached = loss_value.detach()
            train_score_callback(self._step, -lval_detached)
        self._train_scores.append((self._step, (-loss_value).item()))
        self._step += 1

        return self

    def validate(self, validation_data: Tuple["torch.Tensor"]) -> "torch.Tensor":
        r""" Evaluates the currently set lobe(s) on validation data and returns the value of the configured score.

        Parameters
        ----------
        validation_data : Tuple of torch Tensor containing instantaneous and timelagged data
            The validation data.

        Returns
        -------
        score : torch.Tensor
            The value of the score.
        """
        with disable_TF32():
            self.lobe.eval()
            self.lobe_timelagged.eval()
            if self.vamps is not None:
                self.vamps.eval()
                self.vampu.eval()

            with torch.no_grad():
                val = self.lobe(validation_data[0])
                val_t = self.lobe_timelagged(validation_data[1])
                if self.score_method == "VAMPCE":
                    (u_out, v_out, C00_out, C11_out, C01_out, sigma_out, mu_out) = self._vampu([val, val_t])
                    Ve_out, K_out, p_out, S_out = self._vamps([val, val_t, u_out, v_out,
                                                               C00_out, C11_out, C01_out, sigma_out])
                    score_value = vamp_score(Ve_out, Ve_out, method=self.score_method, mode=self.score_mode,
                                             epsilon=self.epsilon)
                else:
                    score_value = vamp_score(val, val_t, method=self.score_method, mode=self.score_mode,
                                             epsilon=self.epsilon)
                return score_value

    def update_auxiliary_weights(self, data, optimize_u: bool = True, optimize_S: bool = False,
                                 reset_weights: bool = True):
        """
        Update the weights for the auxiliary VAMP-U and VAMP-S networks.

        Parameters
        ----------
        data : tuple
            Tuple containing (chi_0, chi_t) data tensors
        optimize_u : bool, default=True
            Whether to optimize the VAMP-U weights
        optimize_S : bool, default=False
            Whether to optimize the VAMP-S weights
        reset_weights : bool, default=True
            Currently unused parameter for weight reset functionality

        Returns
        -------
        None
        """
        # Convert input data to tensors and move to device
        batch_0, batch_t = data[0], data[1]
        chi_0 = torch.Tensor(batch_0).to(self.device)
        chi_t = torch.Tensor(batch_t).to(self.device)

        # Calculate covariance matrices
        C0inv, Ctau = covariances_E(chi_0, chi_t)

        # Get current VAMP outputs
        (u_outd, v_outd, C00_outd, C11_outd,
         C01_outd, sigma_outd, mu_outd) = self._vampu([chi_0, chi_t])

        Ve_out, K_out, p_out, S_out = self._vamps([
            chi_0, chi_t, u_outd, v_outd, C00_outd,
            C11_outd, C01_outd, sigma_outd])

        # Calculate Koopman operator
        K = torch.Tensor(C0inv) @ Ctau.to('cpu')
        self._K = K_out[0]

        # Update VAMP-U weights if requested
        if optimize_u:
            pi = _compute_pi(K)
            u_kernel = np.log(np.abs(C0inv @ pi))
            for param in self.vampu.parameters():
                with torch.no_grad():
                    param[:] = torch.Tensor(u_kernel)

        # Update VAMP-S weights if requested
        if optimize_S:
            (u_out, v_out, C00_out, C11_out,
             C01_out, sigma, mu_out) = self.vampu([chi_0, chi_t])
            sigma_inv = matrix_inverse(sigma[0])
            S_nonrev = K @ sigma_inv
            S_rev = 0.5 * (S_nonrev + S_nonrev.t())
            s_kernel = np.log(np.abs(0.5 * S_rev))
            for param in self.vamps.parameters():
                with torch.no_grad():
                    param[:] = torch.Tensor(s_kernel)

    def train_US(self, data: Tuple[torch.Tensor, torch.Tensor],
                 lr_rate: float = 1e-3,
                 train_u: bool = True,
                 out_log: bool = False) -> None:
        """
        Train the VAMP-U and VAMP-S networks.

        Parameters
        ----------
        data : Tuple[torch.Tensor, torch.Tensor]
            Tuple of (instantaneous, time-lagged) data
        lr_rate : float, default=1e-3
            Learning rate for optimization
        train_u : bool, default=True
            Whether to train the VAMP-U network
        out_log : bool, default=False
            Whether to print loss values during training
        """
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # Monitor gradients
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN gradients detected in {name}")
        # Freeze main network parameters
        self.lobe.requires_grad_(False)
        self.lobe_timelagged.requires_grad_(False)

        # Configure VAMP-U training
        if train_u:
            self._vampu.train()
            self._vampu.requires_grad_(True)
            self._vampu.u_kernel.retain_grad()
        else:
            self._vampu.requires_grad_(False)

        # Configure VAMP-S training
        self._vamps.train()
        self._vamps.s_kernel.retain_grad()

        # Prepare data
        x_0, x_t = data[0], data[1]
        x_0 = torch.Tensor(x_0).to(self.device)
        x_t = torch.Tensor(x_t).to(self.device)

        # Forward pass
        self.optimizer.zero_grad()

        # VAMP-U forward pass
        u_out, v_out, C00_out, C11_out, C01_out, sigma_out, mu_out = self._vampu([x_0, x_t])

        # VAMP-S forward pass
        Ve_out, K_out, p_out, S_out = self._vamps([
            x_0, x_t, u_out, v_out, C00_out,
            C11_out, C01_out, sigma_out
        ])

        # Store Koopman operator
        self._K = K_out[0]

        # Compute and backpropagate loss
        loss_value = vampnet_loss(Ve_out, Ve_out,
                                  method=self.score_method,
                                  epsilon=self.epsilon,
                                  mode=self.score_mode)
        loss_value.backward()
        self.optimizer.step()

        # Optional loss logging
        if out_log:
            print(f"Loss: {loss_value.item():.6f}")

        # Restore network states
        self.lobe.requires_grad_(True)
        self.lobe_timelagged.requires_grad_(True)
        self.lobe.train()
        self.lobe_timelagged.train()

        if not train_u:
            self._vampu.requires_grad_(True)
            self._vampu.train()

    def estimate_koopman(self, lag: int) -> np.ndarray:
        """
        Estimate the Koopman operator for a given lag time.

        Uses cached results if available to avoid recomputation.

        Parameters
        ----------
        lag : int
            Lag time for the Koopman operator estimation

        Returns
        -------
        np.ndarray
            Estimated Koopman operator matrix for the specified lag time
        """
        # Return cached result if available
        if lag in self._k_cache:
            return self._k_cache[lag]

        # Update lag time and compute Koopman operator
        self.lag = lag
        koopman_op = self._K.detach().cpu().numpy()

        # Cache result for future use
        self._k_cache[lag] = koopman_op

        return koopman_op

    def estimate_koopman_op(self, trajectories: Union[List[np.ndarray], np.ndarray],
                            tau: int) -> np.ndarray:
        """
        Estimate the Koopman operator from trajectory data.

        Parameters
        ----------
        trajectories : Union[List[np.ndarray], np.ndarray]
            Either a list of trajectories or a single trajectory array
        tau : int
            Time lag for the Koopman operator estimation

        Returns
        -------
        np.ndarray
            Estimated Koopman operator matrix
        """
        # Process input trajectories
        if isinstance(trajectories, list):
            # Concatenate multiple trajectories
            instant_data = np.concatenate([t[:-tau] for t in trajectories], axis=0)
            lagged_data = np.concatenate([t[tau:] for t in trajectories], axis=0)
        else:
            # Single trajectory
            instant_data = trajectories[:-tau]
            lagged_data = trajectories[tau:]

        # Convert to tensors and move to device
        instant_data = torch.Tensor(instant_data).to(self.device)
        lagged_data = torch.Tensor(lagged_data).to(self.device)

        # VAMP-U forward pass
        u_out, v_out, C00_out, C11_out, C01_out, sigma_out, mu_out = self._vampu([
            instant_data, lagged_data
        ])

        # VAMP-S forward pass
        Ve_out, K_out, p_out, S_out = self._vamps([
            instant_data, lagged_data,
            u_out, v_out, C00_out, C11_out, C01_out, sigma_out
        ])

        # Extract and convert Koopman operator to numpy array
        koopman_op = K_out[0].detach().cpu().numpy()

        return koopman_op

    def its(self, lags: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Calculate implied timescales for a sequence of lag times.

        Parameters
        ----------
        lags : np.ndarray
            Array of lag times to analyze
        dt : float, default=1.0
            Time step between frames

        Returns
        -------
        np.ndarray
            Array of shape (n_output-1, n_lags) containing implied timescales
            for each lag time and eigenvalue
        """
        # Initialize output array (excluding stationary eigenvalue)
        n_timescales = self.n_output - 1
        implied_timescales = np.empty((n_timescales, len(lags)))

        # Calculate implied timescales for each lag time
        for i, lag in enumerate(lags):
            # Get Koopman operator for current lag
            koopman_op = self.estimate_koopman(lag)

            # Calculate eigenvalues of real part of Koopman operator
            eigenvals, _ = np.linalg.eig(np.real(koopman_op))

            # Sort eigenvalues by magnitude and exclude stationary eigenvalue
            sorted_eigenvals = np.sort(np.abs(eigenvals))[:-1]

            # Calculate implied timescales: -lag*dt/ln(λ)
            implied_timescales[:, i] = -lag * dt / np.log(sorted_eigenvals)

        # Reset lag time to default
        self.reset_lag()

        return implied_timescales

    def get_its(self, trajectory: np.ndarray, lags: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Calculate implied timescales (ITS) for a sequence of lag times.

        Parameters
        ----------
        trajectory : np.ndarray
            Input trajectory data
        lags : np.ndarray
            Array of lag times to analyze
        dt : float, default=1.0
            Time step between frames

        Returns
        -------
        np.ndarray
            Array of shape (n_output-1, n_lags) containing implied timescales
            for each lag time and eigenvalue
        """
        # Initialize output array (excluding stationary eigenvalue)
        n_timescales = self.n_output - 1
        implied_timescales = np.empty((n_timescales, len(lags)))

        # Calculate implied timescales for each lag time
        for i, lag in enumerate(lags):
            # Get Koopman operator for current lag
            koopman_op = self.estimate_koopman_op(trajectory, lag)

            # Calculate and sort eigenvalues
            eigenvals, _ = np.linalg.eig(np.real(koopman_op))
            sorted_eigenvals = np.sort(np.abs(np.real(eigenvals)))[:self.LAST]

            # Calculate implied timescales: -lag*dt/ln(λ)
            implied_timescales[:, i] = -lag * dt / np.log(sorted_eigenvals)

        # Reset lag time to default
        self.reset_lag()

        return implied_timescales

    def get_ck_test(self, trajectories: Union[List[np.ndarray], np.ndarray],
                    tau: int, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Chapman-Kolmogorov test by comparing predicted and estimated state transitions.

        Parameters
        ----------
        trajectories : Union[List[np.ndarray], np.ndarray]
            Either a list of trajectories or a single trajectory array
        tau : int
            Base lag time for the test
        steps : int
            Number of prediction steps to test

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            predicted : np.ndarray of shape (n_states, n_states, steps)
                State transitions predicted by repeated application of the base lag time operator
            estimated : np.ndarray of shape (n_states, n_states, steps)
                State transitions directly estimated at longer lag times
        """
        # Determine number of states from trajectory data
        if isinstance(trajectories, list):
            n_states = trajectories[0].shape[1]
        else:
            n_states = trajectories.shape[1]

        # Initialize arrays for predicted and estimated transitions
        predicted = np.zeros((n_states, n_states, steps))
        estimated = np.zeros((n_states, n_states, steps))

        # Set initial condition (identity matrix at t=0)
        predicted[:, :, 0] = np.identity(n_states)
        estimated[:, :, 0] = np.identity(n_states)

        # Compute predictions for each initial state and time step
        for i, initial_state in enumerate(np.identity(n_states)):
            for n in range(1, steps):
                # Get base Koopman operator
                koop_base = self.estimate_koopman_op(trajectories, tau)

                # Predict by repeated application of base operator
                koop_predicted = np.linalg.matrix_power(koop_base, n)

                # Directly estimate at n*tau
                koop_estimated = self.estimate_koopman_op(trajectories, tau * n)

                # Store results
                predicted[i, :, n] = initial_state @ koop_predicted
                estimated[i, :, n] = initial_state @ koop_estimated

        return predicted, estimated

    def reset_lag(self) -> None:
        """
        Reset the model's lag time to its original network lag value.

        This method restores the lag time parameter to the value that was
        initially set during network configuration.
        """
        self.lag = self.network_lag

    def fit(self, data_loader: "torch.utils.data.DataLoader", n_epochs=1, validation_loader=None,
            train_score_callback: Callable[[int, "torch.Tensor"], None] = None,
            validation_score_callback: Callable[[int, "torch.Tensor"], None] = None,
            progress=None, **kwargs):
        r""" Fits a VampNet on data.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The data to use for training. Should yield a tuple of batches representing
            instantaneous and time-lagged samples.
        n_epochs : int, default=1
            The number of epochs (i.e., passes through the training data) to use for training.
        validation_loader : torch.utils.data.DataLoader, optional, default=None
            Validation data, should also be yielded as a two-element tuple.
        train_score_callback : callable, optional, default=None
            Callback function which is invoked after each batch and gets as arguments the current training step
            as well as the score (as torch Tensor).
        validation_score_callback : callable, optional, default=None
            Callback function for validation data. Is invoked after each epoch if validation data is given
            and the callback function is not None. Same as the train callback, this gets the 'step' as well as
            the score.
        progress : context manager, optional, default=None
            Progress bar (eg tqdm), defaults to None.
        **kwargs
            Optional keyword arguments for scikit-learn compatibility

        Returns
        -------
        self : VAMPNet
            Reference to self.
        """
        from deeptime.util.platform import handle_progress_bar
        progress = handle_progress_bar(progress)
        self._step = 0

        # and train
        with disable_TF32():
            for _ in progress(range(n_epochs), desc="VAMPNet epoch", total=n_epochs, leave=False):
                for batch_0, batch_t in data_loader:
                    self.partial_fit((batch_0.to(device=self.device), batch_t.to(device=self.device)),
                                     train_score_callback=train_score_callback)
                if validation_loader is not None:
                    with torch.no_grad():
                        scores = []
                        for val_batch in validation_loader:
                            scores.append(
                                self.validate((val_batch[0].to(device=self.device), val_batch[1].to(device=self.device)))
                            )
                        mean_score = torch.mean(torch.stack(scores))
                        self._validation_scores.append((self._step, mean_score.item()))
                        if validation_score_callback is not None:
                            validation_score_callback(self._step, mean_score)
        return self

    def fetch_model(self) -> VAMPNetModel:
        r""" Yields the current model with training scores and VAMP components. """
        from copy import deepcopy

        # Create base model as before
        lobe = deepcopy(self.lobe)
        if self.lobe == self.lobe_timelagged:
            lobe_timelagged = lobe
        else:
            lobe_timelagged = deepcopy(self.lobe_timelagged)

        # Create VAMPNetModel
        model = VAMPNetModel(lobe, lobe_timelagged, dtype=self.dtype, device=self.device)

        # Copy train scores if they exist
        if hasattr(self, 'train_scores'):
            model._train_scores = deepcopy(self.train_scores)
        if hasattr(self, 'validation_scores'):
            model._validation_scores = deepcopy(self.validation_scores)

        # Add VAMP components as attributes (without using add_module to avoid recursion)
        if hasattr(self, '_vampu'):
            model._vampu = self._vampu
        if hasattr(self, '_vamps'):
            model._vamps = self._vamps

        return model

