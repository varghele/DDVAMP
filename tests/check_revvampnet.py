import torch
import numpy as np
import pytest
from torch import nn
from components.models.RevVAMPNet import RevVAMPNet


def test_revvampnet():
    # Test parameters
    batch_size = 32
    input_dim = 10
    output_dim = 5

    # Create simple test networks
    class TestLobe(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.linear(x)

    class TestVAMPU(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.u_kernel = nn.Parameter(torch.randn(dim, dim))

        def forward(self, inputs):
            x_0, x_t = inputs
            return [x_0 @ self.u_kernel, x_t @ self.u_kernel]

    class TestVAMPS(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.s_kernel = nn.Parameter(torch.randn(dim, dim))

        def forward(self, inputs):
            x_0, x_t, u_0, u_t = inputs
            k = self.s_kernel @ self.s_kernel.t()
            return [u_0 @ k, [k]]

        def reset_weights(self):
            nn.init.xavier_uniform_(self.s_kernel)

    # Create test data
    x_0 = torch.randn(batch_size, input_dim)
    x_t = torch.randn(batch_size, input_dim)

    # Initialize networks
    lobe = TestLobe(input_dim, output_dim)
    vampu = TestVAMPU(output_dim)
    vamps = TestVAMPS(output_dim)

    # Create RevVAMPNet instance
    model = RevVAMPNet(
        lobe=lobe,
        vampu=vampu,
        vamps=vamps,
        score_method='VAMP2',  # Change from 'VAMPCE' to 'VAMP2'
        num_classes=output_dim
    )

    # Test partial fit
    try:
        model.partial_fit((x_0, x_t))
        fit_success = True
    except Exception as e:
        fit_success = False
        print(f"Partial fit failed: {str(e)}")
    assert fit_success, "Partial fit should complete without errors"

    # Test validation
    try:
        score = model.validate((x_0, x_t))
        assert isinstance(score, torch.Tensor), "Validation should return a tensor"
        validation_success = True
    except Exception as e:
        validation_success = False
        print(f"Validation failed: {str(e)}")
    assert validation_success, "Validation should complete without errors"

    # Test Koopman operator estimation
    try:
        traj = np.random.randn(100, input_dim)
        K = model.estimate_koopman_op([traj], tau=2)
        assert K.shape == (output_dim, output_dim), "Koopman operator should have correct shape"
        koopman_success = True
    except Exception as e:
        koopman_success = False
        print(f"Koopman estimation failed: {str(e)}")
    assert koopman_success, "Koopman estimation should complete without errors"

    # Test ITS calculation
    try:
        lags = [1, 2, 3]
        its = model.get_its([traj], lags)
        assert its.shape == (output_dim - 1, len(lags)), "ITS should have correct shape"
        its_success = True
    except Exception as e:
        its_success = False
        print(f"ITS calculation failed: {str(e)}")
    assert its_success, "ITS calculation should complete without errors"

    # Test Chapman-Kolmogorov test
    try:
        steps = 3
        tau = 2
        predicted, estimated = model.get_ck_test(steps, tau)
        assert predicted.shape == (output_dim, output_dim, steps), "CK test results should have correct shape"
        ck_success = True
    except Exception as e:
        ck_success = False
        print(f"CK test failed: {str(e)}")
    assert ck_success, "CK test should complete without errors"


if __name__ == "__main__":
    test_revvampnet()
