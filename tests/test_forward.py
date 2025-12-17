"""
test_forward.py
---------------
Unit tests for the forward model (cosmological calculations).
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from forward import Ez, chi, DV


class TestEz:
    """Tests for the dimensionless Hubble factor E(z)."""

    def test_ez_at_z_zero(self):
        """E(z=0) should equal 1 in flat ΛCDM."""
        z = torch.tensor(0.0)
        # Standard ΛCDM: Ωm=0.3, Ωde=0.7, w0=-1, wa=0
        result = Ez(z, H0=70.0, Om=0.3, Ode=0.7, w0=-1.0, wa=0.0)
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-6)

    def test_ez_positive_for_all_z(self):
        """E(z) should be positive for all redshifts."""
        z = torch.linspace(0.0, 10.0, 100)
        result = Ez(z, H0=70.0, Om=0.3, Ode=0.7, w0=-1.0, wa=0.0)
        assert (result > 0).all()

    def test_ez_increases_with_z_matter_dominated(self):
        """E(z) should increase with z at high redshift (matter dominated)."""
        z = torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = Ez(z, H0=70.0, Om=0.3, Ode=0.7, w0=-1.0, wa=0.0)
        # Check that E(z) is monotonically increasing at high z
        for i in range(len(result) - 1):
            assert result[i + 1] > result[i]

    def test_ez_matter_only_universe(self):
        """E(z) = (1+z)^(3/2) for matter-only universe."""
        z = torch.tensor([0.0, 1.0, 2.0])
        result = Ez(z, H0=70.0, Om=1.0, Ode=0.0, w0=-1.0, wa=0.0)
        expected = (1.0 + z) ** 1.5
        assert torch.allclose(result, expected, atol=1e-5)


class TestChi:
    """Tests for comoving distance χ(z)."""

    def test_chi_at_z_zero(self):
        """χ(z=0) should be 0."""
        z = torch.tensor(0.0)
        result = chi(z, 70.0, 0.3, 0.7, -1.0, 0.0)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_chi_increases_with_z(self):
        """χ(z) should monotonically increase with z."""
        z_values = [0.1, 0.5, 1.0, 2.0]
        chi_values = []
        for z in z_values:
            chi_values.append(chi(torch.tensor(z), 70.0, 0.3, 0.7, -1.0, 0.0).item())

        for i in range(len(chi_values) - 1):
            assert chi_values[i + 1] > chi_values[i]

    def test_chi_positive_for_positive_z(self):
        """χ(z) should be positive for z > 0."""
        z = torch.tensor(1.0)
        result = chi(z, 70.0, 0.3, 0.7, -1.0, 0.0)
        assert result > 0


class TestDV:
    """Tests for volume distance D_V(z)."""

    def test_dv_positive(self):
        """D_V(z) should be positive for z > 0."""
        z = torch.tensor(0.5)
        result = DV(z, 70.0, 0.3, 0.7, -1.0, 0.0)
        assert result > 0

    def test_dv_increases_with_z(self):
        """D_V(z) should increase with z."""
        z_values = [0.1, 0.3, 0.5, 1.0]
        dv_values = []
        for z in z_values:
            dv_values.append(DV(torch.tensor(z), 70.0, 0.3, 0.7, -1.0, 0.0).item())

        for i in range(len(dv_values) - 1):
            assert dv_values[i + 1] > dv_values[i]


class TestPhysicalConsistency:
    """Tests for physical consistency of calculations."""

    def test_different_cosmologies_give_different_distances(self):
        """Different cosmological parameters should give different distances."""
        z = torch.tensor(1.0)

        # ΛCDM
        chi_lcdm = chi(z, 70.0, 0.3, 0.7, -1.0, 0.0)

        # Matter-only
        chi_matter = chi(z, 70.0, 1.0, 0.0, -1.0, 0.0)

        assert not torch.isclose(chi_lcdm, chi_matter, rtol=0.01)

    def test_h0_scaling(self):
        """Distance should scale inversely with H0 (in c/H0 units)."""
        z = torch.tensor(1.0)
        # Note: chi is in units of c/H0, so it shouldn't depend on H0
        # But E(z) doesn't depend on H0 either, so chi shouldn't change
        chi_70 = chi(z, 70.0, 0.3, 0.7, -1.0, 0.0)
        chi_100 = chi(z, 100.0, 0.3, 0.7, -1.0, 0.0)

        # In c/H0 units, the comoving distance is independent of H0
        assert torch.isclose(chi_70, chi_100, rtol=1e-5)
