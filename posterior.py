"""
posterior.py
------------
Condition the trained flow on observed data and draw posterior samples.

The flow p(θ|x) is conditioned on x_obs to obtain samples from p(θ|x_obs),
the posterior distribution of cosmological parameters given observations.

Usage:
    python posterior.py
    # or via CLI: hubble sample
"""

import torch

from config import DEVICE, config, logger, paths, set_seed
from models import load_flow


def load_observed_summary() -> torch.Tensor:
    """
    Load the observed summary vector.

    Returns
    -------
    torch.Tensor
        Observed summary vector on the configured device.
    """
    if not paths.x_obs.exists():
        raise FileNotFoundError(
            f"Observed summary not found at {paths.x_obs}. "
            "Run 'python x_obs.py' first."
        )

    logger.info(f"Loading observed summary from {paths.x_obs}")
    x_obs = torch.load(paths.x_obs, map_location=DEVICE)

    # Ensure correct shape for conditioning (add batch dimension if needed)
    if x_obs.dim() == 1:
        x_obs = x_obs.unsqueeze(0)

    logger.info(f"  Summary vector shape: {x_obs.shape}")

    return x_obs


def sample_posterior(n_samples: int = None) -> torch.Tensor:
    """
    Draw samples from the posterior distribution p(θ|x_obs).

    Parameters
    ----------
    n_samples : int, optional
        Number of posterior samples to draw. Default from config.

    Returns
    -------
    torch.Tensor
        Posterior samples, shape (n_samples, 5).
    """
    n_samples = n_samples or config.inference.n_posterior_samples

    # Load trained flow
    flow = load_flow()

    # Load observed summary
    x_obs = load_observed_summary()

    logger.info(f"Drawing {n_samples:,} posterior samples...")

    # Condition flow on observations and sample
    with torch.no_grad():
        # Expand x_obs to match sample count
        x_expanded = x_obs.expand(n_samples, -1)
        θ_post = flow.sample(n_samples, context=x_expanded)

        # If sample returns tuple, take first element
        if isinstance(θ_post, tuple):
            θ_post = θ_post[0]

    logger.info(f"  Posterior samples shape: {θ_post.shape}")

    return θ_post


def run(n_samples: int = None) -> None:
    """
    Run the full posterior sampling pipeline.

    Parameters
    ----------
    n_samples : int, optional
        Number of posterior samples. Default from config.
    """
    set_seed()

    logger.info("=" * 60)
    logger.info("Starting posterior sampling")
    logger.info("=" * 60)

    θ_post = sample_posterior(n_samples)

    torch.save(θ_post, paths.posterior)
    logger.info(f"Saved posterior samples to {paths.posterior}")

    # Print summary statistics
    logger.info("Posterior summary statistics:")
    param_names = config.physics.param_names
    for i, name in enumerate(param_names):
        mean = θ_post[:, i].mean().item()
        std = θ_post[:, i].std().item()
        logger.info(f"  {name}: {mean:.3f} ± {std:.3f}")

    logger.info("Posterior sampling complete!")


if __name__ == "__main__":
    run()
