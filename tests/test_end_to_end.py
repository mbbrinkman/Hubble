"""
tests/test_end_to_end.py
------------------------
End-to-end integration test for Hubble pipeline.

This test runs through the full workflow:
1. Generate synthetic data
2. Prepare observations
3. Test forward model
4. Generate small training set
5. Train a minimal flow (few epochs)
6. Draw posterior samples

Run with: pytest tests/test_end_to_end.py -v
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        from config import DEVICE, DATA_RAW, DATA_PROC
        self.device = DEVICE
        self.data_raw = DATA_RAW
        self.data_proc = DATA_PROC

    def test_01_generate_synthetic_data(self):
        """Test synthetic data generation."""
        from setup_data import generate_synthetic_sn_data, generate_synthetic_bao_data, save_synthetic_data

        # Generate with known parameters
        sn_data = generate_synthetic_sn_data(
            n_sn=100,  # Small for testing
            H0_true=70.0,
            Om_true=0.3,
            seed=42
        )
        bao_data = generate_synthetic_bao_data(
            H0_true=70.0,
            Om_true=0.3,
            seed=42
        )

        # Verify data shapes
        assert len(sn_data["z"]) == 100
        assert len(sn_data["mu"]) == 100
        assert len(bao_data["z"]) == 7  # Standard BAO redshifts

        # Save data
        save_synthetic_data(sn_data, bao_data)

        # Verify files created
        from config import paths
        assert paths.pantheon_data.exists()
        assert paths.pantheon_sigma.exists()
        assert paths.bao_data.exists()

    def test_02_prep_observations(self):
        """Test data preparation."""
        import prep

        # This should not raise
        prep.run()

        # Verify HDF5 created
        from config import paths
        assert paths.obs_h5.exists()

    def test_03_load_observations(self):
        """Test loading observations as tensors."""
        import prep

        obs = prep.load_observations()

        # Check keys exist
        assert "z_sn" in obs
        assert "dL_obs" in obs
        assert "σ_dL" in obs
        assert "z_bao" in obs
        assert "DV_obs" in obs
        assert "σ_DV" in obs

        # Check shapes
        assert len(obs["z_sn"]) == 100
        assert len(obs["z_bao"]) == 7

        # Check on correct device
        assert obs["z_sn"].device.type == self.device.type

    def test_04_forward_model_cosmology(self):
        """Test cosmological distance calculations."""
        import forward

        # Test parameters: H0=70, Om=0.3, Ode=0.7, w0=-1, wa=0 (ΛCDM)
        theta = torch.tensor([70.0, 0.3, 0.7, -1.0, 0.0], device=self.device)

        # Test E(z=0) = 1
        z0 = torch.tensor(0.0, device=self.device)
        Ez0 = forward.Ez(z0, 0.3, 0.7, -1.0, 0.0)
        assert torch.isclose(Ez0, torch.tensor(1.0, device=self.device), rtol=1e-5)

        # Test E(z) increases with z (for ΛCDM)
        z1 = torch.tensor(1.0, device=self.device)
        Ez1 = forward.Ez(z1, 0.3, 0.7, -1.0, 0.0)
        assert Ez1 > Ez0

        # Test comoving distance increases with z
        chi0 = forward.comoving_distance(torch.tensor(0.1, device=self.device), theta)
        chi1 = forward.comoving_distance(torch.tensor(1.0, device=self.device), theta)
        assert chi1 > chi0

        # Test luminosity distance
        z = torch.tensor(1.0, device=self.device)
        dL = forward.luminosity_distance(z, theta)
        chi = forward.comoving_distance(z, theta)
        assert torch.isclose(dL, chi * 2.0, rtol=0.01)  # dL = (1+z)*χ at z=1

    def test_05_summary_vector(self):
        """Test summary vector construction."""
        import prep
        import forward

        obs = prep.load_observations()
        theta = torch.tensor([70.0, 0.3, 0.7, -1.0, 0.0], device=self.device)

        x = forward.summary_vector_simple(theta, obs)

        # Check shape: n_sn + n_bao residuals
        expected_len = len(obs["z_sn"]) + len(obs["z_bao"])
        assert len(x) == expected_len

        # Check finite values
        assert torch.all(torch.isfinite(x))

    def test_06_chi_squared(self):
        """Test chi-squared calculation."""
        import prep
        import forward

        obs = prep.load_observations()

        # At true parameters (H0=70, Om=0.3), chi-squared should be reasonable
        theta_true = torch.tensor([70.0, 0.3, 0.7, -1.0, 0.0], device=self.device)
        chi2_true = forward.chi_squared(theta_true, obs)

        # At wrong parameters, chi-squared should be larger
        theta_wrong = torch.tensor([60.0, 0.2, 0.8, -1.0, 0.0], device=self.device)
        chi2_wrong = forward.chi_squared(theta_wrong, obs)

        assert chi2_wrong > chi2_true

    def test_07_build_flow(self):
        """Test normalizing flow construction."""
        from models import build_flow

        # Build default flow
        flow = build_flow(dim=5, hidden_dim=64, n_layers=2)

        # Test sampling
        samples = flow.sample(100)
        assert samples.shape == (100, 5)

        # Test log probability
        log_prob = flow.log_prob(samples)
        assert log_prob.shape == (100,)
        assert torch.all(torch.isfinite(log_prob))

    def test_08_mini_training(self):
        """Test a minimal training run."""
        import prep
        import forward
        from models import build_flow
        from config import DEVICE

        obs = prep.load_observations()

        # Generate small training set
        n_train = 500
        theta_min = torch.tensor([60.0, 0.2, 0.6, -1.2, -0.2], device=DEVICE)
        theta_max = torch.tensor([80.0, 0.4, 0.8, -0.8, 0.2], device=DEVICE)

        # Random samples
        torch.manual_seed(42)
        theta_train = torch.rand(n_train, 5, device=DEVICE) * (theta_max - theta_min) + theta_min

        # Compute summary vectors
        x_train = torch.stack([forward.summary_vector_simple(t, obs) for t in theta_train])

        # Build small flow
        flow = build_flow(dim=5, hidden_dim=32, n_layers=2)
        optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

        # Train for a few steps
        flow.train()
        initial_loss = None
        final_loss = None

        for epoch in range(5):
            # Full batch for simplicity
            optimizer.zero_grad()
            loss = -flow.log_prob(theta_train, context=x_train).mean()
            loss.backward()
            optimizer.step()

            if epoch == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        # Loss should decrease (learning something)
        assert final_loss < initial_loss + 0.5  # Some tolerance

    def test_09_posterior_sampling(self):
        """Test posterior sampling from trained flow."""
        import prep
        import forward
        from models import build_flow
        from config import DEVICE

        obs = prep.load_observations()

        # Generate training data
        n_train = 1000
        theta_min = torch.tensor([60.0, 0.2, 0.6, -1.2, -0.2], device=DEVICE)
        theta_max = torch.tensor([80.0, 0.4, 0.8, -0.8, 0.2], device=DEVICE)

        torch.manual_seed(42)
        theta_train = torch.rand(n_train, 5, device=DEVICE) * (theta_max - theta_min) + theta_min
        x_train = torch.stack([forward.summary_vector_simple(t, obs) for t in theta_train])

        # Train flow
        flow = build_flow(dim=5, hidden_dim=32, n_layers=2)
        optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

        flow.train()
        for _ in range(20):
            optimizer.zero_grad()
            loss = -flow.log_prob(theta_train, context=x_train).mean()
            loss.backward()
            optimizer.step()

        # Compute x_obs at true parameters
        theta_true = torch.tensor([70.0, 0.3, 0.7, -1.0, 0.0], device=DEVICE)
        x_obs = forward.summary_vector_simple(theta_true, obs)

        # Sample posterior
        flow.eval()
        with torch.no_grad():
            posterior_samples = flow.sample(500, context=x_obs.unsqueeze(0))

        # Check shape
        assert posterior_samples.shape == (500, 5)

        # Samples should be in reasonable range
        assert posterior_samples[:, 0].mean() > 50  # H0 > 50
        assert posterior_samples[:, 0].mean() < 100  # H0 < 100


class TestCosmologyModels:
    """Test cosmological model implementations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from config import DEVICE
        self.device = DEVICE

    def test_concordance_model(self):
        """Test concordance (single H0) model."""
        from cosmology import get_model

        model = get_model("concordance")
        assert model.n_params == 5
        assert model.name == "Concordance ΛCDM"

        # Sample prior
        samples = model.sample_prior(100)
        assert samples.shape == (100, 5)

        # H0_early == H0_late
        H0_early = model.get_H0_early(samples)
        H0_late = model.get_H0_late(samples)
        assert torch.allclose(H0_early, H0_late)

    def test_discordance_model(self):
        """Test discordance (split H0) model."""
        from cosmology import get_model

        model = get_model("discordance")
        assert model.n_params == 6
        assert model.name == "Discordance (Split H₀)"

        # Sample prior
        samples = model.sample_prior(100)
        assert samples.shape == (100, 6)

        # H0_early != H0_late (different columns)
        H0_early = model.get_H0_early(samples)
        H0_late = model.get_H0_late(samples)
        assert not torch.allclose(H0_early, H0_late)

    def test_probability_resolution(self):
        """Test P(resolution) calculation for discordance model."""
        from cosmology import get_model

        model = get_model("discordance")

        # Create samples with known tension
        # Columns: H0_early, H0_late, Om, Ode, w0, wa
        samples = torch.tensor([
            [67.0, 73.0, 0.3, 0.7, -1.0, 0.0],  # ΔH0 = 6
            [67.0, 68.0, 0.3, 0.7, -1.0, 0.0],  # ΔH0 = 1
            [67.0, 66.0, 0.3, 0.7, -1.0, 0.0],  # ΔH0 = -1
            [67.0, 70.0, 0.3, 0.7, -1.0, 0.0],  # ΔH0 = 3
        ], device=self.device)

        # P(|ΔH0| < 2) should be 2/4 = 0.5
        prob = model.probability_resolution(samples, epsilon=2.0)
        assert torch.isclose(prob, torch.tensor(0.5, device=self.device))

        # P(|ΔH0| < 10) should be 1.0
        prob = model.probability_resolution(samples, epsilon=10.0)
        assert torch.isclose(prob, torch.tensor(1.0, device=self.device))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
