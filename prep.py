"""
prep.py
-------
Data preparation module for Hubble.

Loads raw observational data (Pantheon+, BAO) and converts to PyTorch tensors
ready for training and inference.

Usage:
    python prep.py
    # or via CLI: hubble prep
"""

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
            "Run 'python setup_data.py --synthetic' to generate test data,\n"
            "or download real data to the data/raw/ directory."
        )


def load_and_process_observations() -> None:
    """
    Load raw observation files and create processed HDF5 file.

    Reads:
        - Pantheon+_SH0ES.dat: Type Ia supernova data (z, mu)
        - pantheon_sigma.npy: Uncertainty estimates (σ_dL)
        - BAO_DV.dat: BAO measurements (z, DV, σ_DV)

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


def load_observations(device: torch.device = None) -> dict:
    """
    Load processed observations as PyTorch tensors.

    Parameters
    ----------
    device : torch.device, optional
        Device to place tensors on. Default from config.

    Returns
    -------
    dict
        Dictionary with keys:
        - z_sn, dL_obs, σ_dL: SN Ia data tensors
        - z_bao, DV_obs, σ_DV: BAO data tensors
    """
    device = device or DEVICE

    if not paths.obs_h5.exists():
        raise FileNotFoundError(
            f"Processed observations not found at {paths.obs_h5}.\n"
            "Run 'python prep.py' or 'hubble prep' first."
        )

    with h5py.File(paths.obs_h5, "r") as f:
        obs = {
            "z_sn": torch.tensor(f["z_sn"][:], dtype=torch.float32, device=device),
            "dL_obs": torch.tensor(f["dL_obs"][:], dtype=torch.float32, device=device),
            "σ_dL": torch.tensor(f["σ_dL"][:], dtype=torch.float32, device=device),
            "z_bao": torch.tensor(f["z_bao"][:], dtype=torch.float32, device=device),
            "DV_obs": torch.tensor(f["DV_obs"][:], dtype=torch.float32, device=device),
            "σ_DV": torch.tensor(f["σ_DV"][:], dtype=torch.float32, device=device),
        }

    logger.info(f"Loaded observations: {len(obs['z_sn'])} SNe, {len(obs['z_bao'])} BAO")
    return obs


def run() -> None:
    """Run the full data preparation pipeline."""
    logger.info("=" * 60)
    logger.info("Starting data preparation")
    logger.info("=" * 60)

    load_and_process_observations()

    logger.info("Data preparation complete!")


if __name__ == "__main__":
    run()
