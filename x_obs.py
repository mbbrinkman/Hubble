"""
x_obs.py
--------
Build the observed summary vector using the forward model.

IMPORTANT: The summary vector construction requires cosmological parameters
because the forward model computes model predictions (dL, DV) that depend on θ.
For real inference, you should either:

1. Use fiducial/best-fit parameters from external analysis
2. Marginalize over parameter uncertainty (more sophisticated)

The current implementation uses configurable fiducial parameters with
explicit warnings about this assumption.

Usage:
    python x_obs.py
    # or via CLI: hubble observe
"""

import warnings

import torch

import forward
import prep
from config import DEVICE, logger, paths

# Fiducial cosmological parameters for summary vector construction
# These should be updated based on external constraints (e.g., Planck best-fit)
FIDUCIAL_PARAMS = {
    "H0": 67.4,      # km/s/Mpc (Planck 2018)
    "Om": 0.315,     # Matter density
    "Ode": 0.685,    # Dark energy density (flat universe)
    "w0": -1.0,      # Dark energy equation of state
    "wa": 0.0,       # Dark energy evolution parameter
}


def get_fiducial_theta(device: torch.device = None) -> torch.Tensor:
    """
    Get the fiducial parameter vector for summary vector construction.

    Returns
    -------
    torch.Tensor
        Fiducial parameters [H0, Om, Ode, w0, wa], shape (5,).
    """
    device = device or DEVICE
    theta = torch.tensor(
        [FIDUCIAL_PARAMS["H0"], FIDUCIAL_PARAMS["Om"], FIDUCIAL_PARAMS["Ode"],
         FIDUCIAL_PARAMS["w0"], FIDUCIAL_PARAMS["wa"]],
        device=device,
        dtype=torch.float32
    )
    return theta


def compute_observed_summary(theta: torch.Tensor = None) -> torch.Tensor:
    """
    Compute the summary vector for the observed data.

    Parameters
    ----------
    theta : torch.Tensor, optional
        Cosmological parameters to use for model predictions.
        If None, uses fiducial parameters.

    Returns
    -------
    torch.Tensor
        Observed summary vector.

    Notes
    -----
    The summary vector depends on θ through the model predictions (dL, DV).
    For simulation-based inference, the choice of θ affects the residuals:
      - SN residuals: (dL_model(θ) - dL_obs) / σ
      - BAO residuals: (DV_model(θ) - DV_obs) / σ

    Using fiducial parameters is standard practice, but the choice should
    match the parameters used in training data generation for consistency.
    """
    if theta is None:
        theta = get_fiducial_theta()
        warnings.warn(
            f"Using fiducial cosmological parameters for x_obs: {FIDUCIAL_PARAMS}. "
            "Update FIDUCIAL_PARAMS if different values are needed.",
            UserWarning,
            stacklevel=2
        )

    obs = prep.load_observations()

    logger.info("Computing observed summary vector...")
    logger.info(f"  Using θ = {theta.tolist()}")

    x_obs = forward.summary_vector_simple(theta, obs)

    logger.info(f"  Summary vector length: {len(x_obs)}")

    return x_obs


def run(theta: torch.Tensor = None) -> None:
    """
    Run the observed summary computation and save to disk.

    Parameters
    ----------
    theta : torch.Tensor, optional
        Cosmological parameters. If None, uses fiducial values.
    """
    logger.info("=" * 60)
    logger.info("Computing observed summary vector")
    logger.info("=" * 60)

    x_obs = compute_observed_summary(theta)

    torch.save(x_obs, paths.x_obs)
    logger.info(f"Saved observed summary to {paths.x_obs}")

    logger.info("Observed summary computation complete!")


if __name__ == "__main__":
    run()
