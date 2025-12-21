"""
inference/tension.py
--------------------
Hubble tension quantification.

This module provides multiple ways to quantify the Hubble tension:

1. Probability of resolution: P(|ΔH₀| < ε)
2. Posterior odds: P(concordance|data) / P(discordance|data)
3. Tension significance in sigma
4. Information-theoretic measures (KL divergence between subsets)

The key insight: frequentist "n-sigma" tells you measurements disagree
assuming they measure the same thing. Bayesian analysis tells you the
probability they actually do measure the same thing.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from config import logger
from cosmology.base import CosmologicalModel


@dataclass
class TensionResult:
    """Results from tension analysis."""
    # Core metrics
    prob_resolution: float
    prob_resolution_std: float
    epsilon: float  # tolerance used

    # Distribution statistics
    delta_H0_mean: float
    delta_H0_std: float
    delta_H0_median: float

    # Credible intervals for ΔH₀
    delta_H0_ci68: tuple[float, float]
    delta_H0_ci95: tuple[float, float]

    # Derived
    sigma_tension: float  # Effective tension in sigma

    def __repr__(self) -> str:
        return (
            f"TensionResult(\n"
            f"  P(|ΔH₀|<{self.epsilon}) = {self.prob_resolution:.3f} ± {self.prob_resolution_std:.3f}\n"
            f"  ΔH₀ = {self.delta_H0_mean:.2f} ± {self.delta_H0_std:.2f} km/s/Mpc\n"
            f"  68% CI: [{self.delta_H0_ci68[0]:.2f}, {self.delta_H0_ci68[1]:.2f}]\n"
            f"  95% CI: [{self.delta_H0_ci95[0]:.2f}, {self.delta_H0_ci95[1]:.2f}]\n"
            f"  Effective tension: {self.sigma_tension:.1f}σ\n"
            f")"
        )


class TensionAnalyzer:
    """
    Analyzer for Hubble tension quantification.

    Provides Bayesian answers to questions like:
    - What is P(the tension is resolvable)?
    - How different are early vs late H₀?
    - What epsilon would give 50% resolution probability?
    """

    def __init__(self, model: CosmologicalModel):
        """
        Initialize analyzer for a specific model.

        Parameters
        ----------
        model : CosmologicalModel
            Must have separate get_H0_early and get_H0_late methods
        """
        self.model = model

    def analyze(
        self,
        theta_samples: torch.Tensor,
        epsilon: float = 1.0
    ) -> TensionResult:
        """
        Full tension analysis on posterior samples.

        Parameters
        ----------
        theta_samples : torch.Tensor
            Posterior samples, shape (n_samples, n_params)
        epsilon : float
            Tolerance for "resolution" in km/s/Mpc

        Returns
        -------
        TensionResult
            Comprehensive tension analysis results
        """
        logger.info(f"Analyzing Hubble tension for {self.model.name}")

        # Extract H₀ values
        H0_early = self.model.get_H0_early(theta_samples)
        H0_late = self.model.get_H0_late(theta_samples)
        delta_H0 = H0_late - H0_early  # Signed difference
        abs_delta = torch.abs(delta_H0)

        # Probability of resolution
        resolved = (abs_delta < epsilon).float()
        prob_res = resolved.mean().item()
        prob_std = np.sqrt(prob_res * (1 - prob_res) / len(resolved))

        # Distribution statistics
        delta_mean = delta_H0.mean().item()
        delta_std = delta_H0.std().item()
        delta_median = delta_H0.median().item()

        # Credible intervals
        q16, q84 = delta_H0.quantile(torch.tensor([0.16, 0.84])).tolist()
        q025, q975 = delta_H0.quantile(torch.tensor([0.025, 0.975])).tolist()

        # Effective sigma (assuming Gaussian)
        # "How many sigma from zero?"
        sigma_tension = abs(delta_mean) / delta_std if delta_std > 0 else 0

        result = TensionResult(
            prob_resolution=prob_res,
            prob_resolution_std=prob_std,
            epsilon=epsilon,
            delta_H0_mean=delta_mean,
            delta_H0_std=delta_std,
            delta_H0_median=delta_median,
            delta_H0_ci68=(q16, q84),
            delta_H0_ci95=(q025, q975),
            sigma_tension=sigma_tension,
        )

        logger.info(f"\n{result}")
        return result

    def probability_curve(
        self,
        theta_samples: torch.Tensor,
        epsilon_range: Optional[list[float]] = None
    ) -> dict[str, np.ndarray]:
        """
        Compute P(|ΔH₀| < ε) for a range of ε values.

        This shows how the resolution probability varies with tolerance.

        Parameters
        ----------
        theta_samples : torch.Tensor
            Posterior samples
        epsilon_range : list, optional
            Tolerance values to evaluate. Default: [0.1, 0.5, 1, 2, 3, 4, 5]

        Returns
        -------
        dict
            {"epsilon": [...], "probability": [...], "std_error": [...]}
        """
        if epsilon_range is None:
            epsilon_range = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

        abs_delta = torch.abs(
            self.model.get_H0_late(theta_samples) -
            self.model.get_H0_early(theta_samples)
        )

        probabilities = []
        std_errors = []

        for eps in epsilon_range:
            resolved = (abs_delta < eps).float()
            p = resolved.mean().item()
            se = np.sqrt(p * (1 - p) / len(resolved))
            probabilities.append(p)
            std_errors.append(se)

        return {
            "epsilon": np.array(epsilon_range),
            "probability": np.array(probabilities),
            "std_error": np.array(std_errors),
        }

    def find_critical_epsilon(
        self,
        theta_samples: torch.Tensor,
        target_probability: float = 0.5
    ) -> float:
        """
        Find the ε where P(|ΔH₀| < ε) = target_probability.

        This is the "half-width" of the tension: the tolerance needed
        for a 50% chance of agreement.

        Parameters
        ----------
        theta_samples : torch.Tensor
            Posterior samples
        target_probability : float
            Target probability (default 0.5 for median)

        Returns
        -------
        float
            Critical epsilon value
        """
        abs_delta = torch.abs(
            self.model.get_H0_late(theta_samples) -
            self.model.get_H0_early(theta_samples)
        )

        # Critical epsilon is the quantile of |ΔH₀|
        critical_eps = abs_delta.quantile(target_probability).item()

        logger.info(f"ε for P={target_probability:.0%}: {critical_eps:.2f} km/s/Mpc")
        return critical_eps

    def compare_subsets(
        self,
        theta_samples: torch.Tensor,
        early_only_samples: torch.Tensor,
        late_only_samples: torch.Tensor,
    ) -> dict:
        """
        Compare posterior constraints from different data subsets.

        This is useful for diagnosing which data are in tension.

        Parameters
        ----------
        theta_samples : torch.Tensor
            Full dataset posterior
        early_only_samples : torch.Tensor
            Posterior using only early-universe data (CMB/BAO)
        late_only_samples : torch.Tensor
            Posterior using only late-universe data (SNe/Cepheids)

        Returns
        -------
        dict
            Comparison statistics
        """
        # Get H0 from each
        H0_full = self.model.get_H0_late(theta_samples)  # or early, same in concordance
        H0_early = self.model.get_H0_late(early_only_samples)
        H0_late = self.model.get_H0_late(late_only_samples)

        return {
            "H0_full": {
                "mean": H0_full.mean().item(),
                "std": H0_full.std().item(),
            },
            "H0_early_only": {
                "mean": H0_early.mean().item(),
                "std": H0_early.std().item(),
            },
            "H0_late_only": {
                "mean": H0_late.mean().item(),
                "std": H0_late.std().item(),
            },
            "tension_sigma": abs(H0_early.mean() - H0_late.mean()).item() / \
                            np.sqrt(H0_early.std()**2 + H0_late.std()**2).item(),
        }


def compute_suspiciousness(
    theta_samples_1: torch.Tensor,
    theta_samples_2: torch.Tensor,
    model: CosmologicalModel,
) -> dict:
    """
    Compute the "suspiciousness" statistic for dataset tension.

    This is based on Handley & Lemos (2019):
    S = D_KL(P₁ || P) + D_KL(P₂ || P) - D_KL(P₁₂ || P)

    where P is the prior, P₁, P₂ are single-dataset posteriors,
    and P₁₂ is the combined posterior.

    High S indicates datasets are in tension.

    Parameters
    ----------
    theta_samples_1 : torch.Tensor
        Posterior from dataset 1 (e.g., early universe)
    theta_samples_2 : torch.Tensor
        Posterior from dataset 2 (e.g., late universe)
    model : CosmologicalModel
        The cosmological model

    Returns
    -------
    dict
        Suspiciousness statistic and interpretation
    """
    # Simplified: compare means
    # Full implementation would compute KL divergences

    mean_1 = theta_samples_1.mean(dim=0)
    mean_2 = theta_samples_2.mean(dim=0)
    std_1 = theta_samples_1.std(dim=0)
    std_2 = theta_samples_2.std(dim=0)

    # Approximate tension as Mahalanobis distance
    diff = mean_1 - mean_2
    combined_var = std_1**2 + std_2**2
    chi2 = (diff**2 / combined_var).sum().item()
    n_params = len(mean_1)

    # p-value from chi-squared distribution
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df=n_params)

    return {
        "chi2": chi2,
        "n_params": n_params,
        "p_value": p_value,
        "sigma": np.sqrt(chi2),  # Approximate sigma equivalent
        "interpretation": "suspicious" if p_value < 0.01 else "consistent",
    }
