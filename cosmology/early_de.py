"""
cosmology/early_de.py
---------------------
Early Dark Energy (EDE) model.

This model adds a component of dark energy that was significant
at early times (around recombination) but dilutes away by today.
EDE is one of the leading theoretical proposals for resolving
the Hubble tension.

The mechanism: EDE reduces the sound horizon at recombination,
which means a higher H₀ is needed to match CMB observations.
This can bring early-universe H₀ into agreement with local measurements.

Parameters: θ = (H₀, Ωm, Ωde, w₀, wa, f_ede, z_c)
"""

import torch
from typing import Dict

from cosmology.base import CosmologicalModel, Parameter


class EarlyDarkEnergyModel(CosmologicalModel):
    """
    Early Dark Energy (EDE) model.

    Adds a transient dark energy component at early times that
    could resolve the Hubble tension by reducing the sound horizon.
    """

    @property
    def name(self) -> str:
        return "Early Dark Energy"

    @property
    def short_name(self) -> str:
        return "early_de"

    @property
    def description(self) -> str:
        return (
            "Cosmology with an early dark energy component that contributes "
            "significantly around recombination (z ~ 3000-5000) but dilutes away "
            "by today. This reduces the sound horizon, increasing the H₀ inferred "
            "from CMB data. A leading candidate for resolving the Hubble tension."
        )

    def _setup_parameters(self) -> None:
        """Define the 7 EDE parameters."""
        self.add_parameter(Parameter(
            name="H0",
            symbol="H₀",
            min_val=60.0,
            max_val=85.0,
            prior_type="uniform",
            description="Hubble constant [km/s/Mpc]"
        ))

        self.add_parameter(Parameter(
            name="Om",
            symbol="Ωm",
            min_val=0.25,
            max_val=0.40,
            prior_type="uniform",
            description="Matter density parameter (today)"
        ))

        self.add_parameter(Parameter(
            name="Ode",
            symbol="Ωde",
            min_val=0.60,
            max_val=0.75,
            prior_type="uniform",
            description="Late dark energy density parameter"
        ))

        self.add_parameter(Parameter(
            name="w0",
            symbol="w₀",
            min_val=-1.5,
            max_val=-0.5,
            prior_type="gaussian",
            prior_mean=-1.0,
            prior_std=0.2,
            description="Late DE equation of state"
        ))

        self.add_parameter(Parameter(
            name="wa",
            symbol="wa",
            min_val=-0.5,
            max_val=0.5,
            prior_type="gaussian",
            prior_mean=0.0,
            prior_std=0.2,
            description="Late DE evolution"
        ))

        # EDE-specific parameters
        self.add_parameter(Parameter(
            name="f_ede",
            symbol="fₑᵈₑ",
            min_val=0.0,
            max_val=0.2,
            prior_type="uniform",
            description="Maximum EDE fraction (at z_c)"
        ))

        self.add_parameter(Parameter(
            name="log10_z_c",
            symbol="log₁₀(zc)",
            min_val=3.0,
            max_val=4.5,
            prior_type="uniform",
            description="Log10 of critical redshift for EDE"
        ))

    def get_H0_early(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Effective early universe H₀.

        In EDE, the "early" H₀ is modified by the EDE contribution.
        The sound horizon is smaller, so the inferred H₀ is higher.
        """
        if theta.dim() == 1:
            H0 = theta[0]
            f_ede = theta[5]
        else:
            H0 = theta[:, 0]
            f_ede = theta[:, 5]

        # Approximate correction: H0_eff ≈ H0 * (1 + α * f_ede)
        # where α ~ 3-4 captures the sound horizon reduction
        # This is a simplified approximation; full calculation requires
        # solving the Friedmann equations with EDE component
        alpha = 3.5
        return H0 * (1 + alpha * f_ede)

    def get_H0_late(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Late universe H₀.

        By today, EDE has diluted away, so local H₀ is the bare value.
        """
        if theta.dim() == 1:
            return theta[0]
        return theta[:, 0]

    def get_cosmology_params(self, theta: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract all cosmological parameters."""
        if theta.dim() == 1:
            return {
                "H0_early": self.get_H0_early(theta),
                "H0_late": self.get_H0_late(theta),
                "Om": theta[1],
                "Ode": theta[2],
                "w0": theta[3],
                "wa": theta[4],
                "f_ede": theta[5],
                "z_c": 10 ** theta[6],  # Convert from log10
            }
        else:
            return {
                "H0_early": self.get_H0_early(theta),
                "H0_late": self.get_H0_late(theta),
                "Om": theta[:, 1],
                "Ode": theta[:, 2],
                "w0": theta[:, 3],
                "wa": theta[:, 4],
                "f_ede": theta[:, 5],
                "z_c": 10 ** theta[:, 6],
            }

    def ede_fraction(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute EDE energy density fraction at redshift z.

        Uses a simplified model where EDE peaks at z_c and falls off
        as a power law on either side.

        Parameters
        ----------
        z : torch.Tensor
            Redshift values
        theta : torch.Tensor
            Model parameters

        Returns
        -------
        torch.Tensor
            EDE fraction Ω_ede(z)
        """
        if theta.dim() == 1:
            f_ede = theta[5]
            z_c = 10 ** theta[6]
        else:
            f_ede = theta[:, 5].unsqueeze(-1)
            z_c = (10 ** theta[:, 6]).unsqueeze(-1)

        # Simplified EDE profile: peaks at z_c, width ~ z_c
        # Full model would use axion-like potential
        x = z / z_c
        profile = 1.0 / (1.0 + (x - 1) ** 2)

        return f_ede * profile

    def resolves_tension(self, theta: torch.Tensor, tolerance: float = 2.0) -> torch.Tensor:
        """
        Check if this parameter configuration resolves the tension.

        Tension is "resolved" if early and late H₀ agree within tolerance.

        Parameters
        ----------
        theta : torch.Tensor
            Model parameters
        tolerance : float
            Allowed difference in km/s/Mpc

        Returns
        -------
        torch.Tensor
            Boolean tensor indicating resolution
        """
        H0_early = self.get_H0_early(theta)
        H0_late = self.get_H0_late(theta)
        return torch.abs(H0_early - H0_late) < tolerance
