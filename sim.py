"""
sim.py
------
Training data generation for the normalizing flow.

Generates Sobol quasi-random samples in parameter space and computes
their summary vectors using the forward model.

Usage:
    python sim.py
    # or via CLI: hubble simulate
"""

import torch
import sobol_seq

import forward
import prep
from config import config, paths, DEVICE, logger, set_seed


def generate_sobol_samples(n_samples: int) -> torch.Tensor:
    """
    Generate Sobol quasi-random samples in the parameter space.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.

    Returns
    -------
    torch.Tensor
        Parameter samples, shape (n_samples, 5).
    """
    θ_min = torch.tensor(config.physics.theta_min, device=DEVICE)
    θ_max = torch.tensor(config.physics.theta_max, device=DEVICE)

    # Generate Sobol sequence in [0, 1]^5
    sob = torch.tensor(
        sobol_seq.i4_sobol_generate(5, n_samples),
        dtype=torch.float32,
        device=DEVICE
    )

    # Scale to parameter bounds
    θ_all = sob * (θ_max - θ_min) + θ_min

    logger.info(f"Generated {n_samples:,} Sobol samples in parameter space")
    logger.info(f"  θ_min = {config.physics.theta_min}")
    logger.info(f"  θ_max = {config.physics.theta_max}")

    return θ_all


def compute_summary_vectors(θ_all: torch.Tensor, obs: dict, batch_size: int = 1000) -> torch.Tensor:
    """
    Compute summary vectors for all parameter samples.

    Parameters
    ----------
    θ_all : torch.Tensor
        All parameter samples, shape (N, 5).
    obs : dict
        Observation data dictionary.
    batch_size : int
        Number of samples to process between progress updates.

    Returns
    -------
    torch.Tensor
        Summary vectors, shape (N, D_x).
    """
    n_samples = len(θ_all)
    logger.info(f"Computing {n_samples:,} summary vectors...")

    x_list = []
    for i, θ in enumerate(θ_all):
        if i % batch_size == 0:
            pct = 100 * i / n_samples
            logger.info(f"  Progress: {i:,}/{n_samples:,} ({pct:.1f}%)")
        x_list.append(forward.summary_vector_simple(θ, obs))

    logger.info(f"  Progress: {n_samples:,}/{n_samples:,} (100.0%)")

    return torch.stack(x_list)


def run(n_samples: int = None) -> None:
    """
    Run the full simulation pipeline.

    Parameters
    ----------
    n_samples : int, optional
        Number of training samples. Default from config.
    """
    set_seed()

    n_samples = n_samples or config.training.n_train_samples

    logger.info("=" * 60)
    logger.info("Starting training data generation")
    logger.info("=" * 60)

    # Load observations
    obs = prep.load_observations()

    # Generate parameter samples
    θ_all = generate_sobol_samples(n_samples)

    # Compute summary vectors
    x_all = compute_summary_vectors(θ_all, obs)

    # Save training data
    torch.save({"θ": θ_all, "x": x_all}, paths.train_data)
    logger.info(f"Saved training data to {paths.train_data}")
    logger.info(f"  θ shape: {θ_all.shape}")
    logger.info(f"  x shape: {x_all.shape}")

    logger.info("Training data generation complete!")


if __name__ == "__main__":
    run()
