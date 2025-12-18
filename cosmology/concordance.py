"""
cosmology/concordance.py
------------------------
Standard ΛCDM concordance model.

This is the baseline model where a single H₀ applies to both
early and late universe measurements. Any tension between
datasets must be explained by statistical fluctuation or
systematic errors, not new physics.

Parameters: θ = (H₀, Ωm, Ωde, w₀, wa)
"""

import torch
from typing import Dict

from cosmology.base import CosmologicalModel, Parameter


class ConcordanceModel(CosmologicalModel):
    """
    Standard ΛCDM/wCDM concordance cosmology.

    Single Hubble constant for all epochs. This is the null hypothesis
    against which tension is measured.
    """

    @property
    def name(self) -> str:
        return "ΛCDM Concordance"

    @property
    def short_name(self) -> str:
        return "concordance"

    @property
    def description(self) -> str:
        return (
            "Standard cosmological model with a single Hubble constant. "
            "Early and late universe measurements should agree if this model is correct. "
            "Includes dynamic dark energy parameters (w₀, wa) but assumes no H₀ split."
        )

    def _setup_parameters(self) -> None:
        """Define the 5 concordance parameters."""
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
            min_val=0.20,
            max_val=0.45,
            prior_type="uniform",
            description="Matter density parameter"
        ))

        self.add_parameter(Parameter(
            name="Ode",
            symbol="Ωde",
            min_val=0.55,
            max_val=0.80,
            prior_type="uniform",
            description="Dark energy density parameter"
        ))

        self.add_parameter(Parameter(
            name="w0",
            symbol="w₀",
            min_val=-2.0,
            max_val=-0.3,
            prior_type="gaussian",
            prior_mean=-1.0,
            prior_std=0.3,
            description="Dark energy equation of state (today)"
        ))

        self.add_parameter(Parameter(
            name="wa",
            symbol="wa",
            min_val=-1.0,
            max_val=1.0,
            prior_type="gaussian",
            prior_mean=0.0,
            prior_std=0.5,
            description="Dark energy equation of state evolution"
        ))

    def get_H0_early(self, theta: torch.Tensor) -> torch.Tensor:
        """Early universe H₀ (same as late in concordance)."""
        if theta.dim() == 1:
            return theta[0]
        return theta[:, 0]

    def get_H0_late(self, theta: torch.Tensor) -> torch.Tensor:
        """Late universe H₀ (same as early in concordance)."""
        if theta.dim() == 1:
            return theta[0]
        return theta[:, 0]

    def get_cosmology_params(self, theta: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract cosmological parameters from theta vector."""
        if theta.dim() == 1:
            return {
                "H0_early": theta[0],
                "H0_late": theta[0],  # Same!
                "Om": theta[1],
                "Ode": theta[2],
                "w0": theta[3],
                "wa": theta[4],
            }
        else:
            return {
                "H0_early": theta[:, 0],
                "H0_late": theta[:, 0],  # Same!
                "Om": theta[:, 1],
                "Ode": theta[:, 2],
                "w0": theta[:, 3],
                "wa": theta[:, 4],
            }
