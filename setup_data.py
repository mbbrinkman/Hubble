"""
setup_data.py
-------------
Download and prepare data for Hubble.

This script handles:
1. Downloading real Pantheon+ SN Ia data (if available)
2. Downloading BAO measurements
3. Generating synthetic data for testing/development

Usage:
    python setup_data.py --real      # Download real data
    python setup_data.py --synthetic # Generate synthetic test data
    python setup_data.py --both      # Both
"""

import numpy as np
from pathlib import Path
import urllib.request
import ssl
import warnings

from config import DATA_RAW, DATA_PROC, logger


# =============================================================================
# Real Data URLs
# =============================================================================

# Pantheon+ data from GitHub
PANTHEON_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_டCOV/Pantheon%2BSH0ES.dat"

# Backup: simplified version we create
PANTHEON_BACKUP = None  # We'll generate if download fails


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_synthetic_sn_data(
    n_sn: int = 1701,
    z_min: float = 0.001,
    z_max: float = 2.3,
    H0_true: float = 70.0,
    Om_true: float = 0.3,
    sigma_mu: float = 0.15,
    seed: int = 42
) -> dict:
    """
    Generate synthetic Type Ia supernova data.

    Uses a flat ΛCDM cosmology to generate distance moduli,
    then adds Gaussian noise.

    Parameters
    ----------
    n_sn : int
        Number of supernovae
    z_min, z_max : float
        Redshift range
    H0_true : float
        True Hubble constant (km/s/Mpc)
    Om_true : float
        True matter density
    sigma_mu : float
        Distance modulus uncertainty (mag)
    seed : int
        Random seed

    Returns
    -------
    dict
        {z, mu, sigma_mu, dL, sigma_dL}
    """
    np.random.seed(seed)

    # Log-uniform redshift distribution (more realistic)
    log_z = np.random.uniform(np.log(z_min), np.log(z_max), n_sn)
    z = np.sort(np.exp(log_z))

    # Compute true luminosity distance using flat ΛCDM
    # dL = (1+z) * χ(z), where χ = ∫ c/H(z') dz'
    c = 299792.458  # km/s

    def E(z, Om):
        """Dimensionless Hubble parameter for flat ΛCDM."""
        Ode = 1 - Om
        return np.sqrt(Om * (1 + z)**3 + Ode)

    def comoving_distance(z, H0, Om, n_points=1000):
        """Comoving distance in Mpc."""
        z_grid = np.linspace(0, z, n_points)
        integrand = 1.0 / E(z_grid, Om)
        chi = np.trapz(integrand, z_grid) * c / H0
        return chi

    # Compute distances
    dL_true = np.array([
        comoving_distance(zi, H0_true, Om_true) * (1 + zi)
        for zi in z
    ])

    # Distance modulus: μ = 5 * log10(dL / 10pc) = 5 * log10(dL) + 25
    mu_true = 5 * np.log10(dL_true) + 25

    # Add noise
    mu_obs = mu_true + np.random.normal(0, sigma_mu, n_sn)
    sigma_mu_arr = np.full(n_sn, sigma_mu)

    # Convert back to luminosity distance
    dL_obs = 10 ** ((mu_obs - 25) / 5)
    sigma_dL = dL_obs * sigma_mu * np.log(10) / 5  # Error propagation

    logger.info(f"Generated {n_sn} synthetic SNe Ia")
    logger.info(f"  z range: [{z.min():.3f}, {z.max():.3f}]")
    logger.info(f"  True cosmology: H0={H0_true}, Om={Om_true}")

    return {
        "z": z,
        "mu": mu_obs,
        "sigma_mu": sigma_mu_arr,
        "dL": dL_obs,
        "sigma_dL": sigma_dL,
        "true_params": {"H0": H0_true, "Om": Om_true}
    }


def generate_synthetic_bao_data(
    H0_true: float = 70.0,
    Om_true: float = 0.3,
    sigma_frac: float = 0.02,
    seed: int = 42
) -> dict:
    """
    Generate synthetic BAO measurements.

    Uses standard BAO redshifts and adds noise to D_V(z).

    Parameters
    ----------
    H0_true : float
        True Hubble constant
    Om_true : float
        True matter density
    sigma_frac : float
        Fractional uncertainty on D_V
    seed : int
        Random seed

    Returns
    -------
    dict
        {z, DV, sigma_DV}
    """
    np.random.seed(seed + 1)  # Different seed from SN

    # Standard BAO redshifts from various surveys
    z_bao = np.array([0.106, 0.15, 0.32, 0.57, 0.70, 1.48, 2.33])

    c = 299792.458  # km/s

    def E(z, Om):
        Ode = 1 - Om
        return np.sqrt(Om * (1 + z)**3 + Ode)

    def comoving_distance(z, H0, Om, n_points=1000):
        z_grid = np.linspace(0, z, n_points)
        integrand = 1.0 / E(z_grid, Om)
        chi = np.trapz(integrand, z_grid) * c / H0
        return chi

    def D_V(z, H0, Om):
        """Volume-averaged distance."""
        chi = comoving_distance(z, H0, Om)
        return (chi**2 * c * z / (H0 * E(z, Om))) ** (1/3)

    # Compute true D_V
    DV_true = np.array([D_V(z, H0_true, Om_true) for z in z_bao])

    # Add noise
    sigma_DV = DV_true * sigma_frac
    DV_obs = DV_true + np.random.normal(0, sigma_DV)

    logger.info(f"Generated {len(z_bao)} synthetic BAO measurements")

    return {
        "z": z_bao,
        "DV": DV_obs,
        "sigma_DV": sigma_DV,
        "true_params": {"H0": H0_true, "Om": Om_true}
    }


def save_synthetic_data(sn_data: dict, bao_data: dict) -> None:
    """Save synthetic data to files."""

    # Save SN data
    sn_path = DATA_RAW / "Pantheon+_SH0ES.dat"
    sn_arr = np.column_stack([sn_data["z"], sn_data["mu"]])
    np.savetxt(sn_path, sn_arr, header="z mu_obs", comments="# ")
    logger.info(f"Saved SN data to {sn_path}")

    # Save SN uncertainties (converted to dL space)
    sigma_path = DATA_RAW / "pantheon_sigma.npy"
    np.save(sigma_path, sn_data["sigma_dL"])
    logger.info(f"Saved SN uncertainties to {sigma_path}")

    # Save BAO data
    bao_path = DATA_RAW / "BAO_DV.dat"
    bao_arr = np.column_stack([bao_data["z"], bao_data["DV"], bao_data["sigma_DV"]])
    np.savetxt(bao_path, bao_arr, header="z DV sigma_DV", comments="# ")
    logger.info(f"Saved BAO data to {bao_path}")

    # Save true parameters for validation
    true_params = {
        "H0": sn_data["true_params"]["H0"],
        "Om": sn_data["true_params"]["Om"],
        "Ode": 1 - sn_data["true_params"]["Om"],
        "w0": -1.0,
        "wa": 0.0,
    }
    np.save(DATA_RAW / "true_params.npy", true_params)
    logger.info(f"Saved true parameters for validation")


# =============================================================================
# Real Data Download
# =============================================================================

def download_file(url: str, dest: Path) -> bool:
    """Download a file from URL."""
    try:
        # Create SSL context that doesn't verify (for GitHub raw)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        logger.info(f"Downloading {url}...")
        with urllib.request.urlopen(url, context=ctx, timeout=30) as response:
            content = response.read()

        with open(dest, 'wb') as f:
            f.write(content)

        logger.info(f"Saved to {dest}")
        return True

    except Exception as e:
        logger.warning(f"Download failed: {e}")
        return False


def download_real_data() -> bool:
    """
    Attempt to download real Pantheon+ and BAO data.

    Returns True if successful, False otherwise.
    """
    logger.info("Attempting to download real cosmological data...")

    # For now, we'll generate synthetic data that mimics real data
    # Real Pantheon+ data requires more complex parsing

    logger.warning(
        "Real data download not yet implemented. "
        "Use --synthetic to generate test data."
    )
    return False


# =============================================================================
# Main
# =============================================================================

def setup_synthetic(
    H0: float = 70.0,
    Om: float = 0.3,
    n_sn: int = 1701,
    seed: int = 42
) -> None:
    """Generate and save synthetic data."""

    logger.info("=" * 60)
    logger.info("Generating Synthetic Data")
    logger.info("=" * 60)

    # Generate
    sn_data = generate_synthetic_sn_data(
        n_sn=n_sn, H0_true=H0, Om_true=Om, seed=seed
    )
    bao_data = generate_synthetic_bao_data(
        H0_true=H0, Om_true=Om, seed=seed
    )

    # Save
    save_synthetic_data(sn_data, bao_data)

    logger.info("")
    logger.info("Synthetic data generated with TRUE parameters:")
    logger.info(f"  H0 = {H0} km/s/Mpc")
    logger.info(f"  Om = {Om}")
    logger.info(f"  Ode = {1-Om}")
    logger.info(f"  w0 = -1.0 (ΛCDM)")
    logger.info(f"  wa = 0.0")
    logger.info("")
    logger.info("Use these to validate posterior recovery!")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Set up data for Hubble")
    parser.add_argument("--real", action="store_true", help="Download real data")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data")
    parser.add_argument("--both", action="store_true", help="Both real and synthetic")
    parser.add_argument("--H0", type=float, default=70.0, help="True H0 for synthetic")
    parser.add_argument("--Om", type=float, default=0.3, help="True Om for synthetic")
    parser.add_argument("--n-sn", type=int, default=1701, help="Number of SNe")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Default to synthetic if nothing specified
    if not (args.real or args.synthetic or args.both):
        args.synthetic = True

    if args.real or args.both:
        success = download_real_data()
        if not success and not args.synthetic:
            logger.info("Falling back to synthetic data...")
            args.synthetic = True

    if args.synthetic or args.both:
        setup_synthetic(
            H0=args.H0, Om=args.Om, n_sn=args.n_sn, seed=args.seed
        )


if __name__ == "__main__":
    main()
