"""
prep.py
-------
Data preparation module for Hubble.

Loads raw observational data (Pantheon+, BAO, JWST placeholders),
converts to consistent units, and stores:
  * Numerical arrays -> HDF5 (obs.h5)
  * Residual callables -> Torch pickle (residual_fns.pt)

Usage:
    python prep.py
    # or via CLI: hubble prep
"""

import warnings
import numpy as np
import h5py
import torch

from config import paths, DEVICE, logger


def validate_input_files() -> None:
    """Check that all required raw data files exist."""
    missing = []
    for name, path in [
        ("Pantheon+ SN Ia data", paths.pantheon_data),
        ("Pantheon sigma", paths.pantheon_sigma),
        ("BAO data", paths.bao_data),
    ]:
        if not path.exists():
            missing.append(f"  - {name}: {path}")

    if missing:
        raise FileNotFoundError(
            "Missing required raw data files:\n" + "\n".join(missing) + "\n"
            "Please download the data files to the data/raw/ directory."
        )


def load_and_process_observations() -> None:
    """
    Load raw observation files and create processed HDF5 file.

    Reads:
        - Pantheon+_SH0ES.dat: Type Ia supernova data
        - pantheon_sigma.npy: Uncertainty estimates
        - BAO_DV.dat: BAO measurements

    Writes:
        - obs.h5: Processed HDF5 with all observation arrays
    """
    validate_input_files()

    logger.info("Loading raw observation data...")

    with h5py.File(paths.obs_h5, "w") as f:
        # ---- Pantheon+ SN Ia ----
        logger.info(f"  Loading Pantheon+ data from {paths.pantheon_data}")
        sn = np.loadtxt(paths.pantheon_data)
        z_sn = sn[:, 0].astype(np.float64)
        mu_sn = sn[:, 1].astype(np.float64)

        # Convert distance modulus μ to luminosity distance dL [Mpc]
        # μ = 5 * log10(dL / 10pc) = 5 * log10(dL) + 25
        dL_sn = 10.0 ** ((mu_sn - 25.0) / 5.0)

        σ_sn = np.load(paths.pantheon_sigma)

        f["z_sn"] = z_sn
        f["dL_obs"] = dL_sn
        f["σ_dL"] = σ_sn
        logger.info(f"  Loaded {len(z_sn)} SN Ia measurements")

        # ---- BAO measurements ----
        logger.info(f"  Loading BAO data from {paths.bao_data}")
        bao = np.loadtxt(paths.bao_data)
        f["z_bao"] = bao[:, 0]
        f["DV_obs"] = bao[:, 1]
        f["σ_DV"] = bao[:, 2]
        logger.info(f"  Loaded {len(bao)} BAO measurements")

    logger.info(f"Wrote processed observations to {paths.obs_h5}")


# ---------------------------------------------------------------------------
# Placeholder residual functions
# ---------------------------------------------------------------------------
# These return zeros of the correct length so the pipeline runs.
# Replace with real photometry→distance calculations when available.


def cep_res(theta: torch.Tensor) -> torch.Tensor:
    """
    Cepheid variable residuals (placeholder).

    TODO: Replace with real Cepheid period-luminosity calculations.

    Parameters
    ----------
    theta : torch.Tensor
        Cosmological parameters.

    Returns
    -------
    torch.Tensor
        Residual vector of length 42.
    """
    warnings.warn(
        "Using placeholder Cepheid residuals (zeros). "
        "Replace cep_res() with real calculations.",
        UserWarning,
        stacklevel=2
    )
    return torch.zeros(42, device=theta.device)


def trgb_res(theta: torch.Tensor) -> torch.Tensor:
    """
    Tip of the Red Giant Branch (TRGB) residuals (placeholder).

    TODO: Replace with real TRGB distance calculations.

    Parameters
    ----------
    theta : torch.Tensor
        Cosmological parameters.

    Returns
    -------
    torch.Tensor
        Residual vector of length 8.
    """
    warnings.warn(
        "Using placeholder TRGB residuals (zeros). "
        "Replace trgb_res() with real calculations.",
        UserWarning,
        stacklevel=2
    )
    return torch.zeros(8, device=theta.device)


def lens_res(theta: torch.Tensor) -> torch.Tensor:
    """
    Gravitational lens time-delay residuals (placeholder).

    TODO: Replace with real lens system calculations.

    Parameters
    ----------
    theta : torch.Tensor
        Cosmological parameters.

    Returns
    -------
    torch.Tensor
        Residual vector of length 8.
    """
    warnings.warn(
        "Using placeholder lens residuals (zeros). "
        "Replace lens_res() with real calculations.",
        UserWarning,
        stacklevel=2
    )
    return torch.zeros(8, device=theta.device)


def save_residual_functions() -> None:
    """Save residual function callables for use by other modules."""
    torch.save(
        {
            "cep_res": cep_res,
            "trgb_res": trgb_res,
            "lens_res": lens_res,
        },
        paths.residual_fns
    )
    logger.info(f"Wrote residual functions to {paths.residual_fns}")


def run() -> None:
    """Run the full data preparation pipeline."""
    logger.info("=" * 60)
    logger.info("Starting data preparation")
    logger.info("=" * 60)

    load_and_process_observations()
    save_residual_functions()

    logger.info("Data preparation complete!")


if __name__ == "__main__":
    run()
