import pytest
import torch
import torch.nn as nn
from components.models.vamps import VAMPS


class TestVAMPS:
    """Test suite for VAMPS module."""

    @pytest.fixture
    def device(self):
        """Fixture for device selection."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def vamps_model(self, device):
        """Fixture for VAMPS model initialization."""
        units = 4
        activation = nn.ReLU()
        return VAMPS(units=units, activation=activation, device=device)

    @pytest.fixture
    def sample_short_input(self, device):
        """Fixture for sample input with 5 tensors."""
        batch_size = 32
        feature_dim = 4
        v = torch.randn(batch_size, feature_dim, device=device)
        C00 = torch.randn(batch_size, feature_dim, feature_dim, device=device)
        C11 = torch.randn(batch_size, feature_dim, feature_dim, device=device)
        C01 = torch.randn(batch_size, feature_dim, feature_dim, device=device)
        sigma = torch.randn(batch_size, feature_dim, feature_dim, device=device)
        return [v, C00, C11, C01, sigma]

    @pytest.fixture
    def sample_full_input(self, device):
        """Fixture for sample input with 8 tensors."""
        batch_size = 32
        feature_dim = 4
        chi_t = torch.randn(batch_size, feature_dim, device=device)
        chi_tau = torch.randn(batch_size, feature_dim, device=device)
        u = torch.randn(batch_size, feature_dim, device=device)
        v = torch.randn(batch_size, feature_dim, device=device)
        C00 = torch.randn(batch_size, feature_dim, feature_dim, device=device)
        C11 = torch.randn(batch_size, feature_dim, feature_dim, device=device)
        C01 = torch.randn(batch_size, feature_dim, feature_dim, device=device)
        sigma = torch.randn(batch_size, feature_dim, feature_dim, device=device)
        return [chi_t, chi_tau, u, v, C00, C11, C01, sigma]

    def test_initialization(self, vamps_model):
        """Test proper initialization of VAMPS module."""
        assert isinstance(vamps_model, nn.Module)
        assert vamps_model.M == 4
        assert isinstance(vamps_model.activation, nn.ReLU)
        assert isinstance(vamps_model._s_kernel, nn.Parameter)
        assert vamps_model._s_kernel.shape == (4, 4)
        assert torch.allclose(vamps_model._s_kernel, 0.1 * torch.ones_like(vamps_model._s_kernel))
        assert vamps_model._init_weight is None

    def test_s_kernel_property(self, vamps_model):
        """Test s_kernel property accessor."""
        kernel = vamps_model.s_kernel
        assert isinstance(kernel, nn.Parameter)
        assert kernel.shape == (4, 4)
        assert torch.allclose(kernel, vamps_model._s_kernel)

    def test_reset_weights(self, vamps_model):
        """Test weight reset functionality."""
        # First reset should store initial weights
        initial_weights = vamps_model._s_kernel.clone()
        vamps_model.reset_weights()
        assert vamps_model._init_weight is not None
        assert torch.allclose(vamps_model._init_weight, initial_weights)

        # Modify weights
        with torch.no_grad():
            vamps_model._s_kernel.data = torch.randn_like(vamps_model._s_kernel)

        # Second reset should restore initial weights
        vamps_model.reset_weights()
        assert torch.allclose(vamps_model._s_kernel, initial_weights)

    def test_compute_output_shape(self, vamps_model):
        """Test output shape computation."""
        input_shape = [32, 4]
        output_shapes = vamps_model.compute_output_shape(input_shape)
        assert len(output_shapes) == 4
        assert output_shapes[:2] == [(4, 4), (4, 4)]
        assert output_shapes[2] == 4
        assert output_shapes[3] == (4, 4)

    def test_forward_short_input(self, vamps_model, sample_short_input):
        """Test forward pass with 5 input tensors."""
        outputs = vamps_model(sample_short_input)

        assert isinstance(outputs, list)
        assert len(outputs) == 4

        vamp_e_tile, K_tile, probs, S_tile = outputs
        batch_size = sample_short_input[0].shape[0]

        assert vamp_e_tile.shape == (batch_size, vamps_model.M, vamps_model.M)
        assert K_tile.shape == (batch_size, vamps_model.M, vamps_model.M)
        assert probs.shape == (batch_size, vamps_model.M)
        assert S_tile.shape == (batch_size, vamps_model.M, vamps_model.M)
        assert torch.all(torch.isfinite(probs))

    def test_forward_full_input(self, vamps_model, sample_full_input):
        """Test forward pass with 8 input tensors."""
        outputs = vamps_model(sample_full_input)

        assert isinstance(outputs, list)
        assert len(outputs) == 4

        vamp_e_tile, K_tile, probs, S_tile = outputs
        batch_size = sample_full_input[0].shape[0]

        assert vamp_e_tile.shape == (batch_size, vamps_model.M, vamps_model.M)
        assert K_tile.shape == (batch_size, vamps_model.M, vamps_model.M)
        assert probs.shape == (batch_size, vamps_model.M)
        assert S_tile.shape == (batch_size, vamps_model.M, vamps_model.M)
        assert torch.all(torch.isfinite(probs))

    def test_renormalization(self, device):
        """Test renormalization option."""
        vamps_renorm = VAMPS(units=4, activation=nn.ReLU(), renorm=True, device=device)
        vamps_no_renorm = VAMPS(units=4, activation=nn.ReLU(), renorm=False, device=device)

        sample_input = self.sample_short_input(device)

        output_renorm = vamps_renorm(sample_input)
        output_no_renorm = vamps_no_renorm(sample_input)

        # Outputs should be different when renormalization is applied
        assert not torch.allclose(output_renorm[0], output_no_renorm[0])

    def test_gradient_flow(self, vamps_model, sample_full_input):
        """Test gradient flow through the module."""
        outputs = vamps_model(sample_full_input)

        # Compute loss (sum of all outputs)
        loss = sum(output.sum() for output in outputs)
        loss.backward()

        # Check if gradients are computed
        assert vamps_model._s_kernel.grad is not None
        assert not torch.allclose(vamps_model._s_kernel.grad, torch.zeros_like(vamps_model._s_kernel))

    def test_numerical_stability(self, vamps_model, device):
        """Test numerical stability with extreme values."""
        batch_size = 32
        feature_dim = 4

        # Test with very small values
        small_input = [torch.rand(batch_size, feature_dim, device=device) * 1e-6 for _ in range(5)]
        outputs_small = vamps_model(small_input)

        # Test with very large values
        large_input = [torch.rand(batch_size, feature_dim, device=device) * 1e6 for _ in range(5)]
        outputs_large = vamps_model(large_input)

        # Check that outputs are finite
        for outputs in [outputs_small, outputs_large]:
            for output in outputs:
                assert torch.all(torch.isfinite(output))


if __name__ == "__main__":
    pytest.main([])
