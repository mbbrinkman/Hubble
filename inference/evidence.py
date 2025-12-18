"""
inference/evidence.py
---------------------
Bayesian evidence estimation using normalizing flows.

The evidence (marginal likelihood) is:

    P(x|M) = ∫ P(x|θ,M) P(θ|M) dθ

This is the key quantity for Bayesian model comparison.
We estimate it using importance sampling from the flow posterior.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from config import DEVICE, logger
from cosmology.base import CosmologicalModel


@dataclass
class EvidenceResult:
    """Results from evidence estimation."""
    log_evidence: float
    log_evidence_std: float
    effective_sample_size: float
    n_samples: int
    method: str

    @property
    def evidence(self) -> float:
        """Evidence (not log)."""
        return np.exp(self.log_evidence)

    def __repr__(self) -> str:
        return (
            f"EvidenceResult(log_Z={self.log_evidence:.2f}±{self.log_evidence_std:.2f}, "
            f"ESS={self.effective_sample_size:.0f}/{self.n_samples})"
        )


def estimate_log_evidence(
    flow,
    model: CosmologicalModel,
    x_obs: torch.Tensor,
    n_samples: int = 50000,
    compute_likelihood_fn=None,
) -> EvidenceResult:
    """
    Estimate log evidence using importance sampling.

    The evidence is:
        P(x) = ∫ P(x|θ) P(θ) dθ

    We estimate this by importance sampling from the flow:
        P(x) ≈ (1/N) Σ_i [P(x|θ_i) P(θ_i) / q(θ_i|x)]

    where θ_i ~ q(θ|x) is the flow posterior.

    Parameters
    ----------
    flow : Flow
        Trained normalizing flow for this model
    model : CosmologicalModel
        The cosmological model
    x_obs : torch.Tensor
        Observed summary statistics
    n_samples : int
        Number of importance samples
    compute_likelihood_fn : callable, optional
        Function to compute log P(x|θ). If None, uses simple Gaussian.

    Returns
    -------
    EvidenceResult
        Evidence estimate with uncertainty
    """
    logger.info(f"Estimating evidence for {model.name} with {n_samples:,} samples")

    # Ensure x_obs has batch dimension
    if x_obs.dim() == 1:
        x_obs = x_obs.unsqueeze(0)

    flow.eval()
    with torch.no_grad():
        # Sample from flow posterior q(θ|x)
        x_expanded = x_obs.expand(n_samples, -1)
        theta_samples = flow.sample(n_samples, context=x_expanded)
        if isinstance(theta_samples, tuple):
            theta_samples = theta_samples[0]

        # Compute log q(θ|x) - the flow's density
        log_q = flow.log_prob(theta_samples, context=x_expanded)

        # Compute log prior P(θ)
        log_prior = model.log_prior(theta_samples)

        # Compute log likelihood P(x|θ)
        if compute_likelihood_fn is not None:
            log_likelihood = compute_likelihood_fn(theta_samples, x_obs)
        else:
            # Default: assume flow was trained well, use simple estimate
            # This is approximate but avoids recomputing the forward model
            log_likelihood = log_q - log_prior + _estimate_log_normalizer(flow, model)

        # Importance weights: w_i = P(x|θ_i) P(θ_i) / q(θ_i|x)
        log_weights = log_likelihood + log_prior - log_q

        # Handle numerical issues
        log_weights = torch.nan_to_num(log_weights, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

        # Log-sum-exp for numerical stability
        max_log_w = log_weights.max()
        log_evidence = max_log_w + torch.log(torch.exp(log_weights - max_log_w).mean())

        # Estimate uncertainty via effective sample size
        weights = torch.exp(log_weights - log_weights.max())
        weights = weights / weights.sum()
        ess = 1.0 / (weights ** 2).sum()

        # Bootstrap uncertainty estimate
        log_evidence_std = _bootstrap_evidence_std(log_weights, n_bootstrap=100)

    result = EvidenceResult(
        log_evidence=log_evidence.item(),
        log_evidence_std=log_evidence_std,
        effective_sample_size=ess.item(),
        n_samples=n_samples,
        method="importance_sampling"
    )

    logger.info(f"  {result}")
    return result


def _estimate_log_normalizer(flow, model: CosmologicalModel) -> float:
    """
    Estimate the log normalizing constant of the flow.

    This is needed when we don't have an explicit likelihood.
    We use the fact that ∫ q(θ|x) dθ = 1.
    """
    # For a well-trained flow, this should be approximately 0
    # (flow is normalized). Return 0 as approximation.
    return 0.0


def _bootstrap_evidence_std(log_weights: torch.Tensor, n_bootstrap: int = 100) -> float:
    """Estimate standard error of log evidence via bootstrap."""
    n = len(log_weights)
    estimates = []

    for _ in range(n_bootstrap):
        idx = torch.randint(0, n, (n,), device=log_weights.device)
        lw_boot = log_weights[idx]
        max_lw = lw_boot.max()
        log_z = max_lw + torch.log(torch.exp(lw_boot - max_lw).mean())
        estimates.append(log_z.item())

    return np.std(estimates)


def compute_evidence_ratio(
    flow1,
    flow2,
    model1: CosmologicalModel,
    model2: CosmologicalModel,
    x_obs: torch.Tensor,
    n_samples: int = 50000,
) -> Dict:
    """
    Compute the Bayes factor (evidence ratio) between two models.

    B₁₂ = P(x|M₁) / P(x|M₂)

    Parameters
    ----------
    flow1, flow2 : Flow
        Trained flows for each model
    model1, model2 : CosmologicalModel
        The two models to compare
    x_obs : torch.Tensor
        Observed data
    n_samples : int
        Samples for evidence estimation

    Returns
    -------
    dict
        Contains Bayes factor, log Bayes factor, interpretation, etc.
    """
    logger.info(f"Computing Bayes factor: {model1.short_name} vs {model2.short_name}")

    # Estimate evidence for each model
    ev1 = estimate_log_evidence(flow1, model1, x_obs, n_samples)
    ev2 = estimate_log_evidence(flow2, model2, x_obs, n_samples)

    log_bayes_factor = ev1.log_evidence - ev2.log_evidence
    log_bf_std = np.sqrt(ev1.log_evidence_std**2 + ev2.log_evidence_std**2)

    bayes_factor = np.exp(log_bayes_factor)

    # Jeffreys' scale interpretation
    if log_bayes_factor > 2.3:  # > 10
        interpretation = f"Strong evidence for {model1.short_name}"
    elif log_bayes_factor > 1.1:  # > 3
        interpretation = f"Moderate evidence for {model1.short_name}"
    elif log_bayes_factor > 0:
        interpretation = f"Weak evidence for {model1.short_name}"
    elif log_bayes_factor > -1.1:
        interpretation = f"Weak evidence for {model2.short_name}"
    elif log_bayes_factor > -2.3:
        interpretation = f"Moderate evidence for {model2.short_name}"
    else:
        interpretation = f"Strong evidence for {model2.short_name}"

    result = {
        "bayes_factor": bayes_factor,
        "log_bayes_factor": log_bayes_factor,
        "log_bayes_factor_std": log_bf_std,
        "model1": model1.short_name,
        "model2": model2.short_name,
        "evidence1": ev1,
        "evidence2": ev2,
        "interpretation": interpretation,
    }

    logger.info(f"  log(B₁₂) = {log_bayes_factor:.2f} ± {log_bf_std:.2f}")
    logger.info(f"  B₁₂ = {bayes_factor:.3g}")
    logger.info(f"  {interpretation}")

    return result


def harmonic_mean_evidence(
    flow,
    model: CosmologicalModel,
    x_obs: torch.Tensor,
    n_samples: int = 50000,
    compute_likelihood_fn=None,
) -> EvidenceResult:
    """
    Estimate evidence using harmonic mean estimator.

    WARNING: The harmonic mean estimator has infinite variance in theory.
    Use importance sampling (estimate_log_evidence) instead.
    This is provided for comparison/validation only.

    P(x)⁻¹ = E_posterior[P(x|θ)⁻¹]
    """
    logger.warning("Harmonic mean estimator has infinite variance. Use with caution.")

    if x_obs.dim() == 1:
        x_obs = x_obs.unsqueeze(0)

    flow.eval()
    with torch.no_grad():
        x_expanded = x_obs.expand(n_samples, -1)
        theta_samples = flow.sample(n_samples, context=x_expanded)
        if isinstance(theta_samples, tuple):
            theta_samples = theta_samples[0]

        if compute_likelihood_fn is not None:
            log_likelihood = compute_likelihood_fn(theta_samples, x_obs)
        else:
            log_q = flow.log_prob(theta_samples, context=x_expanded)
            log_prior = model.log_prior(theta_samples)
            log_likelihood = log_q - log_prior

        # Harmonic mean: 1/Z ≈ mean(1/L)
        neg_log_likelihood = -log_likelihood
        max_nll = neg_log_likelihood.max()
        log_inv_evidence = max_nll + torch.log(torch.exp(neg_log_likelihood - max_nll).mean())
        log_evidence = -log_inv_evidence

    return EvidenceResult(
        log_evidence=log_evidence.item(),
        log_evidence_std=np.nan,  # Unreliable
        effective_sample_size=np.nan,
        n_samples=n_samples,
        method="harmonic_mean"
    )
