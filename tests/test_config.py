"""
test_config.py
--------------
Unit tests for configuration module.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPaths:
    """Tests for path configuration."""

    def test_root_exists(self):
        """ROOT should point to an existing directory."""
        from config import ROOT
        assert ROOT.exists()
        assert ROOT.is_dir()

    def test_paths_are_pathlib(self):
        """All paths should be pathlib.Path objects."""
        from config import paths
        assert isinstance(paths.pantheon_data, Path)
        assert isinstance(paths.obs_h5, Path)
        assert isinstance(paths.flow_weights, Path)

    def test_directories_created(self):
        """Data directories should be created on import."""
        from config import DATA_RAW, DATA_PROC, MODELS, RESULTS
        assert DATA_RAW.exists()
        assert DATA_PROC.exists()
        assert MODELS.exists()
        assert RESULTS.exists()


class TestConfig:
    """Tests for configuration dataclasses."""

    def test_flow_config_defaults(self):
        """FlowConfig should have sensible defaults."""
        from config import FlowConfig
        fc = FlowConfig()
        assert fc.dim == 5
        assert fc.hidden_dim == 256
        assert fc.n_layers == 8

    def test_training_config_defaults(self):
        """TrainingConfig should have sensible defaults."""
        from config import TrainingConfig
        tc = TrainingConfig()
        assert tc.n_train_samples == 300_000
        assert tc.batch_size == 1024
        assert tc.learning_rate == 1e-3
        assert tc.n_epochs == 10

    def test_physics_config_defaults(self):
        """PhysicsConfig should have correct parameter bounds."""
        from config import PhysicsConfig
        pc = PhysicsConfig()
        assert len(pc.theta_min) == 5
        assert len(pc.theta_max) == 5
        assert len(pc.param_names) == 5
        # H0 bounds
        assert pc.theta_min[0] < pc.theta_max[0]

    def test_config_container(self):
        """Main Config should contain all sub-configs."""
        from config import config
        assert hasattr(config, "flow")
        assert hasattr(config, "training")
        assert hasattr(config, "inference")
        assert hasattr(config, "physics")


class TestDevice:
    """Tests for device configuration."""

    def test_device_is_valid(self):
        """DEVICE should be a valid torch device."""
        import torch
        from config import DEVICE
        assert isinstance(DEVICE, torch.device)
        assert DEVICE.type in ["cpu", "cuda", "mps"]

    def test_get_device_function(self):
        """get_device should return a valid device."""
        from config import get_device
        device = get_device()
        assert device.type in ["cpu", "cuda", "mps"]


class TestSeed:
    """Tests for reproducibility."""

    def test_set_seed(self):
        """set_seed should make torch operations deterministic."""
        import torch
        from config import set_seed

        set_seed(42)
        a1 = torch.randn(10)

        set_seed(42)
        a2 = torch.randn(10)

        assert torch.allclose(a1, a2)

    def test_different_seeds_give_different_results(self):
        """Different seeds should give different random values."""
        import torch
        from config import set_seed

        set_seed(42)
        a1 = torch.randn(10)

        set_seed(123)
        a2 = torch.randn(10)

        assert not torch.allclose(a1, a2)
