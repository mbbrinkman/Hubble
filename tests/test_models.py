"""
test_models.py
--------------
Unit tests for model building and loading.
"""

import sys
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBuildFlow:
    """Tests for flow model construction."""

    def test_build_flow_default(self):
        """build_flow should create a valid flow with defaults."""
        from models import build_flow

        flow = build_flow()
        assert flow is not None

        # Check it has parameters
        n_params = sum(p.numel() for p in flow.parameters())
        assert n_params > 0

    def test_build_flow_custom_dims(self):
        """build_flow should respect custom dimensions."""
        from models import build_flow

        flow = build_flow(dim=3, hidden_dim=128, n_layers=4)
        assert flow is not None

    def test_flow_can_sample(self):
        """Flow should be able to generate samples."""
        from models import build_flow

        flow = build_flow(dim=5)
        flow.eval()

        with torch.no_grad():
            # Sample without context
            samples = flow.sample(10)
            if isinstance(samples, tuple):
                samples = samples[0]
            assert samples.shape == (10, 5)

    def test_flow_can_compute_log_prob(self):
        """Flow should compute log probabilities."""
        from config import DEVICE
        from models import build_flow

        flow = build_flow(dim=5)

        # Create dummy inputs
        theta = torch.randn(10, 5, device=DEVICE)
        context = torch.randn(10, 100, device=DEVICE)

        log_prob = flow.log_prob(inputs=theta, context=context)
        assert log_prob.shape == (10,)
        assert torch.isfinite(log_prob).all()

    def test_flow_on_correct_device(self):
        """Flow should be on the specified device."""
        from config import DEVICE
        from models import build_flow

        flow = build_flow()

        # Check first parameter is on correct device
        first_param = next(flow.parameters())
        assert first_param.device == DEVICE


class TestFlowGradients:
    """Tests for flow gradient computation."""

    def test_gradients_flow_during_training(self):
        """Gradients should flow through the model."""
        from config import DEVICE
        from models import build_flow

        flow = build_flow(dim=5)

        theta = torch.randn(10, 5, device=DEVICE, requires_grad=True)
        context = torch.randn(10, 100, device=DEVICE)

        loss = -flow.log_prob(inputs=theta, context=context).mean()
        loss.backward()

        # Check that model parameters have gradients
        for param in flow.parameters():
            assert param.grad is not None
