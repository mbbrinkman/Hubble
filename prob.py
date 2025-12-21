"""
prob.py
-------
Post-hoc analysis of posterior samples.

Computes the probability that local and early-universe H0 measurements
agree within a tolerance, addressing the "Hubble tension" question.

In standard ΛCDM, H0_local = H0_early by construction (both are θ[0]).
This framework is ready for extended models where they may differ.

Usage:
    python prob.py
    # or via CLI: hubble analyze
"""

import numpy as np
import torch

from config import config, logger, paths


def load_posterior() -> torch.Tensor:
    """
    Load the posterior samples.

    Returns
    -------
    torch.Tensor
        Posterior samples, shape (n_samples, 5).
    """
    if not paths.posterior.exists():
        raise FileNotFoundError(
            f"Posterior samples not found at {paths.posterior}. "
            "Run 'python posterior.py' first."
        )

    logger.info(f"Loading posterior samples from {paths.posterior}")
    θ = torch.load(paths.posterior)
    logger.info(f"  Loaded {len(θ):,} samples")

    return θ


def compute_h0_tension_probability(
    θ: torch.Tensor,
    epsilon: float = 1.0
) -> tuple[float, float]:
    """
    Compute the probability that |H0_local - H0_early| < epsilon.

    Parameters
    ----------
    θ : torch.Tensor
        Posterior samples, shape (n_samples, 5).
    epsilon : float
        Tolerance in km/s/Mpc. Default is 1.0.

    Returns
    -------
    tuple[float, float]
        (probability, standard_error)

    Notes
    -----
    In ΛCDM, H0_local = H0_early = θ[:, 0], so this probability is trivially 1.
    The framework is designed for extended models where they may differ.
    """
    # In current ΛCDM implementation, both are the same
    # For extended models, modify these to compute H0 from different physics
    H0_local = θ[:, 0]
    H0_early = θ[:, 0]  # Same in ΛCDM; modify for extended models

    # Check if difference is within tolerance
    agreement = (H0_local - H0_early).abs() < epsilon

    # Compute probability and binomial standard error
    P = agreement.float().mean().item()
    n = agreement.numel()
    σ = np.sqrt(P * (1 - P) / n)

    return P, σ


def compute_posterior_statistics(θ: torch.Tensor) -> dict:
    """
    Compute summary statistics for the posterior.

    Parameters
    ----------
    θ : torch.Tensor
        Posterior samples, shape (n_samples, 5).

    Returns
    -------
    dict
        Dictionary with mean, std, median, and credible intervals.
    """
    stats = {}
    param_names = config.physics.param_names

    for i, name in enumerate(param_names):
        samples = θ[:, i]
        stats[name] = {
            "mean": samples.mean().item(),
            "std": samples.std().item(),
            "median": samples.median().item(),
            "q16": samples.quantile(0.16).item(),  # ~1σ lower
            "q84": samples.quantile(0.84).item(),  # ~1σ upper
            "q025": samples.quantile(0.025).item(),  # 95% CI lower
            "q975": samples.quantile(0.975).item(),  # 95% CI upper
        }

    return stats


def run(epsilon: float = 1.0) -> dict:
    """
    Run the full analysis pipeline.

    Parameters
    ----------
    epsilon : float
        H0 agreement tolerance in km/s/Mpc.
    """
    logger.info("=" * 60)
    logger.info("Starting posterior analysis")
    logger.info("=" * 60)

    θ = load_posterior()

    # Compute H0 tension probability
    P, σ = compute_h0_tension_probability(θ, epsilon)
    logger.info(f"H0 tension analysis (ε = {epsilon} km/s/Mpc):")
    logger.info(f"  P(|ΔH₀| < {epsilon}) = {P:.4f} ± {σ:.4f}")

    # Compute full posterior statistics
    stats = compute_posterior_statistics(θ)

    logger.info("Posterior parameter constraints:")
    for name, s in stats.items():
        logger.info(f"  {name}:")
        logger.info(f"    Mean ± Std: {s['mean']:.3f} ± {s['std']:.3f}")
        logger.info(f"    68% CI: [{s['q16']:.3f}, {s['q84']:.3f}]")
        logger.info(f"    95% CI: [{s['q025']:.3f}, {s['q975']:.3f}]")

    logger.info("Analysis complete!")

    # Return results for programmatic use
    return {"tension_prob": P, "tension_err": σ, "stats": stats}


if __name__ == "__main__":
    run()
