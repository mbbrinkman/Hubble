"""
cosmology/wcdm.py
-----------------
Dynamic dark energy model (w₀-wa parameterization).

This is similar to concordance but with stronger emphasis on
the dark energy parameters as potential resolution mechanisms
for the Hubble tension.

The idea: if w₀ ≠ -1 or wa ≠ 0, the expansion history differs
from ΛCDM, which could reconcile early and late H₀ measurements.

Parameters: θ = (H₀, Ωm, Ωde, w₀, wa)
"""

import torch
from typing import Dict

from cosmology.base import CosmologicalModel, Parameter


class WCDMModel(CosmologicalModel):
    """
    Dynamic dark energy (w₀-wa) model.

    Same parameterization as concordance but with wider priors
    on dark energy parameters to explore tension resolution.
    """

    @property
    def name(self) -> str:
        return "Dynamic Dark Energy (w₀-wa)"

    @property
    def short_name(self) -> str:
        return "wcdm"

    @property
    def description(self) -> str:
        return (
            "Cosmology with dynamic dark energy equation of state w(a) = w₀ + wa(1-a). "
            "Deviations from w₀=-1, wa=0 (cosmological constant) could potentially "
            "resolve the Hubble tension by modifying the expansion history."
        )

    def _setup_parameters(self) -> None:
        """Define parameters with wider DE priors."""
        self.add_parameter(Parameter(
            name="H0",
            symbol="H₀",
            min_val=50.0,
            max_val=90.0,
            prior_type="uniform",
            description="Hubble constant [km/s/Mpc]"
        ))

        self.add_parameter(Parameter(
            name="Om",
            symbol="Ωm",
            min_val=0.15,
            max_val=0.50,
            prior_type="uniform",
            description="Matter density parameter"
        ))

        self.add_parameter(Parameter(
            name="Ode",
            symbol="Ωde",
            min_val=0.50,
            max_val=0.85,
            prior_type="uniform",
            description="Dark energy density parameter"
        ))

        # Wider priors on DE parameters
        self.add_parameter(Parameter(
            name="w0",
            symbol="w₀",
            min_val=-3.0,
            max_val=0.0,
            prior_type="uniform",  # Flat prior to explore full range
            description="Dark energy equation of state (today)"
        ))

        self.add_parameter(Parameter(
            name="wa",
            symbol="wa",
            min_val=-2.0,
            max_val=2.0,
            prior_type="uniform",  # Flat prior to explore full range
            description="Dark energy equation of state evolution"
        ))

    def get_H0_early(self, theta: torch.Tensor) -> torch.Tensor:
        """Early universe H₀."""
        if theta.dim() == 1:
            return theta[0]
        return theta[:, 0]

    def get_H0_late(self, theta: torch.Tensor) -> torch.Tensor:
        """Late universe H₀ (same as early in this model)."""
        if theta.dim() == 1:
            return theta[0]
        return theta[:, 0]

    def get_cosmology_params(self, theta: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract cosmological parameters."""
        if theta.dim() == 1:
            return {
                "H0_early": theta[0],
                "H0_late": theta[0],
                "Om": theta[1],
                "Ode": theta[2],
                "w0": theta[3],
                "wa": theta[4],
            }
        else:
            return {
                "H0_early": theta[:, 0],
                "H0_late": theta[:, 0],
                "Om": theta[:, 1],
                "Ode": theta[:, 2],
                "w0": theta[:, 3],
                "wa": theta[:, 4],
            }

    def is_phantom(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Check if dark energy is phantom (w < -1).

        Phantom dark energy has exotic properties (negative kinetic energy)
        but could help resolve Hubble tension.
        """
        if theta.dim() == 1:
            w0 = theta[3]
        else:
            w0 = theta[:, 3]
        return w0 < -1.0

    def compute_w_at_z(self, theta: torch.Tensor, z: float) -> torch.Tensor:
        """
        Compute equation of state at redshift z.

        w(z) = w₀ + wa * z / (1 + z)
        """
        if theta.dim() == 1:
            w0, wa = theta[3], theta[4]
        else:
            w0, wa = theta[:, 3], theta[:, 4]

        a = 1.0 / (1.0 + z)
        return w0 + wa * (1.0 - a)
