"""
cosmology/discordance.py
------------------------
Discordance model with split Hubble constant.

This model allows early-universe (CMB/BAO) and late-universe (SNe/Cepheids)
measurements to have different effective Hubble constants. This could arise from:

1. Unmodeled systematic errors in one or both measurement types
2. New physics that modifies the distance-redshift relation differently
   at early vs late times
3. A fundamental breakdown of the standard cosmological model

If this model is strongly preferred over concordance, the Hubble tension
is "real" in a Bayesian sense.

Parameters: θ = (H₀_early, H₀_late, Ωm, Ωde, w₀, wa)
"""

import torch
from typing import Dict

from cosmology.base import CosmologicalModel, Parameter


class DiscordanceModel(CosmologicalModel):
    """
    Split-H₀ discordance model.

    Allows different Hubble constants for early and late universe.
    This is the alternative hypothesis for testing Hubble tension.
    """

    @property
    def name(self) -> str:
        return "Split-H₀ Discordance"

    @property
    def short_name(self) -> str:
        return "discordance"

    @property
    def description(self) -> str:
        return (
            "Extended model with separate Hubble constants for early (CMB/BAO) "
            "and late (Cepheids/SNe) universe measurements. If the posterior "
            "strongly prefers H₀_early ≠ H₀_late, the Hubble tension is real. "
            "The probability P(|ΔH₀| < ε) quantifies resolution likelihood."
        )

    def _setup_parameters(self) -> None:
        """Define the 6 discordance parameters."""
        self.add_parameter(Parameter(
            name="H0_early",
            symbol="H₀ᵉᵃʳˡʸ",
            min_val=50.0,
            max_val=90.0,
            prior_type="uniform",
            description="Hubble constant from early universe (CMB/BAO) [km/s/Mpc]"
        ))

        self.add_parameter(Parameter(
            name="H0_late",
            symbol="H₀ˡᵃᵗᵉ",
            min_val=50.0,
            max_val=90.0,
            prior_type="uniform",
            description="Hubble constant from late universe (SNe/Cepheids) [km/s/Mpc]"
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
        """Early universe H₀ from CMB/BAO."""
        if theta.dim() == 1:
            return theta[0]
        return theta[:, 0]

    def get_H0_late(self, theta: torch.Tensor) -> torch.Tensor:
        """Late universe H₀ from local distance ladder."""
        if theta.dim() == 1:
            return theta[1]
        return theta[:, 1]

    def get_cosmology_params(self, theta: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract cosmological parameters from theta vector."""
        if theta.dim() == 1:
            return {
                "H0_early": theta[0],
                "H0_late": theta[1],  # Different!
                "Om": theta[2],
                "Ode": theta[3],
                "w0": theta[4],
                "wa": theta[5],
            }
        else:
            return {
                "H0_early": theta[:, 0],
                "H0_late": theta[:, 1],  # Different!
                "Om": theta[:, 2],
                "Ode": theta[:, 3],
                "w0": theta[:, 4],
                "wa": theta[:, 5],
            }

    def compute_delta_H0(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute the H₀ difference (tension magnitude).

        Parameters
        ----------
        theta : torch.Tensor
            Parameter samples

        Returns
        -------
        torch.Tensor
            |H₀_late - H₀_early| for each sample
        """
        H0_early = self.get_H0_early(theta)
        H0_late = self.get_H0_late(theta)
        return torch.abs(H0_late - H0_early)

    def probability_resolution(
        self,
        theta_samples: torch.Tensor,
        epsilon: float = 1.0
    ) -> tuple:
        """
        Compute the probability that the tension is resolvable.

        P(|H₀_late - H₀_early| < ε)

        A low probability means the data strongly prefer different H₀ values,
        indicating the tension is "real" and not a statistical fluctuation.

        Parameters
        ----------
        theta_samples : torch.Tensor
            Posterior samples from this model
        epsilon : float
            Tolerance in km/s/Mpc

        Returns
        -------
        tuple
            (probability, standard_error)
        """
        import numpy as np

        delta_H0 = self.compute_delta_H0(theta_samples)
        resolved = (delta_H0 < epsilon).float()

        prob = resolved.mean().item()
        n = len(resolved)
        std_err = np.sqrt(prob * (1 - prob) / n)

        return prob, std_err
