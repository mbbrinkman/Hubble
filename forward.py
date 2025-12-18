"""
forward.py
----------
Forward model that transforms cosmological parameters θ into summary statistics x.

Two modes are supported:
1. Simple mode: Uses SN Ia + BAO data only (no external dependencies)
2. Full mode: Adds CMB power spectra via CosmoPower emulator (optional)

θ  = (H0, Ωm, Ωde, w0, wa)     shape (5,) or (batch, 5)
"""

import torch
import numpy as np
from typing import Optional

from config import config, paths, DEVICE, logger

# Speed of light in km/s
C_LIGHT = 299792.458


# ---------------------------------------------------------------------------
# 1. CMB Emulator (optional, lazy loading)
# ---------------------------------------------------------------------------
_emu = None
_cmb_available = None


def cmb_available() -> bool:
    """Check if CMB emulator is available."""
    global _cmb_available
    if _cmb_available is None:
        try:
            import cosmopower as cp
            _cmb_available = paths.cmb_emulator.exists()
        except ImportError:
            _cmb_available = False
    return _cmb_available


def get_emulator():
    """Lazy-load the CMB power spectrum emulator."""
    global _emu
    if _emu is None:
        if not cmb_available():
            raise RuntimeError(
                "CMB emulator not available. Either:\n"
                "  1. Install cosmopower: pip install cosmopower\n"
                f"  2. Download model to: {paths.cmb_emulator}\n"
                "Or use simple mode (SN+BAO only) which doesn't require CMB."
            )
        import cosmopower as cp
        _emu = cp.cosmopower_Pk()
        _emu.load(str(paths.cmb_emulator))
        _emu.to(DEVICE)
        logger.info(f"Loaded CMB emulator from {paths.cmb_emulator}")
    return _emu


# ---------------------------------------------------------------------------
# 2. Cosmological distance functions
# ---------------------------------------------------------------------------
def Ez(z: torch.Tensor, Om: float, Ode: float, w0: float, wa: float) -> torch.Tensor:
    """
    Dimensionless Hubble factor E(z) = H(z)/H0 for flat w0-wa dark energy.

    Parameters
    ----------
    z : torch.Tensor
        Redshift values.
    Om, Ode, w0, wa : float
        Cosmological parameters.

    Returns
    -------
    torch.Tensor
        E(z) values.
    """
    a = 1.0 / (1.0 + z)
    # Dark energy density with CPL parameterization: w(a) = w0 + wa * (1 - a)
    # ρ_de ∝ a^(-3(1+w_eff)) where integral gives the exponential form below
    de_term = Ode * torch.exp(-3.0 * (1.0 + w0 + wa) * torch.log(a) + 3.0 * wa * (a - 1.0))
    return torch.sqrt(Om * (1.0 + z) ** 3 + de_term)


def Ez_batch(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Batch computation of E(z) for multiple parameter sets.

    Parameters
    ----------
    z : torch.Tensor
        Redshift values, shape (n_z,)
    theta : torch.Tensor
        Parameters, shape (batch, 5) where columns are [H0, Om, Ode, w0, wa]

    Returns
    -------
    torch.Tensor
        E(z) values, shape (batch, n_z)
    """
    # Extract parameters
    Om = theta[:, 1:2]    # (batch, 1)
    Ode = theta[:, 2:3]   # (batch, 1)
    w0 = theta[:, 3:4]    # (batch, 1)
    wa = theta[:, 4:5]    # (batch, 1)

    z = z.unsqueeze(0)    # (1, n_z)
    a = 1.0 / (1.0 + z)   # (1, n_z)

    # Dark energy term with CPL parameterization: w(a) = w0 + wa * (1 - a)
    de_term = Ode * torch.exp(-3.0 * (1.0 + w0 + wa) * torch.log(a) + 3.0 * wa * (a - 1.0))

    return torch.sqrt(Om * (1.0 + z) ** 3 + de_term)  # (batch, n_z)


def comoving_distance(z: torch.Tensor, theta: torch.Tensor, n_points: int = None) -> torch.Tensor:
    """
    Comoving distance χ(z) = (c/H0) ∫₀ᶻ dz'/E(z').

    Parameters
    ----------
    z : torch.Tensor
        Redshift, shape (n_z,) or scalar
    theta : torch.Tensor
        Parameters, shape (5,) or (batch, 5)
    n_points : int, optional
        Number of integration points. Default from config.

    Returns
    -------
    torch.Tensor
        Comoving distance in Mpc, shape matches input batch dimension
    """
    n_points = n_points or config.physics.integration_points

    # Handle scalar z
    if z.dim() == 0:
        z = z.unsqueeze(0)

    # Handle single theta
    single_theta = theta.dim() == 1
    if single_theta:
        theta = theta.unsqueeze(0)

    H0 = theta[:, 0]  # (batch,)

    # Vectorized integration for all z values
    results = []
    for zi in z:
        z_grid = torch.linspace(0.0, zi.item(), n_points, device=DEVICE)
        integrand = 1.0 / Ez_batch(z_grid, theta)  # (batch, n_points)
        dz = zi / (n_points - 1)
        chi = torch.trapezoid(integrand, z_grid)  # (batch,)
        results.append(chi)

    chi = torch.stack(results, dim=1)  # (batch, n_z)
    chi = chi * C_LIGHT / H0.unsqueeze(1)  # Convert to Mpc

    if single_theta:
        chi = chi.squeeze(0)

    return chi


def luminosity_distance(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Luminosity distance dL(z) = (1+z) * χ(z).

    Parameters
    ----------
    z : torch.Tensor
        Redshift
    theta : torch.Tensor
        Cosmological parameters

    Returns
    -------
    torch.Tensor
        Luminosity distance in Mpc
    """
    chi = comoving_distance(z, theta)
    if z.dim() == 0:
        return chi * (1.0 + z)
    return chi * (1.0 + z.unsqueeze(0) if theta.dim() > 1 else 1.0 + z)


def volume_distance(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Volume distance D_V(z) used for BAO.

    D_V = [χ(z)² * c*z / H(z)]^(1/3)

    Parameters
    ----------
    z : torch.Tensor
        Redshift
    theta : torch.Tensor
        Cosmological parameters

    Returns
    -------
    torch.Tensor
        Volume distance in Mpc
    """
    single_theta = theta.dim() == 1
    if single_theta:
        theta = theta.unsqueeze(0)

    H0 = theta[:, 0]
    chi = comoving_distance(z, theta)

    if z.dim() == 0:
        z = z.unsqueeze(0)

    # E(z) at the BAO redshifts
    E_z = Ez_batch(z, theta)  # (batch, n_z)
    H_z = H0.unsqueeze(1) * E_z  # H(z) in km/s/Mpc

    # D_V = [χ² * c*z / H(z)]^(1/3)
    DV = (chi ** 2 * C_LIGHT * z.unsqueeze(0) / H_z) ** (1.0 / 3.0)

    if single_theta:
        DV = DV.squeeze(0)

    return DV


# ---------------------------------------------------------------------------
# 3. Summary vector construction (Simple mode: SN + BAO only)
# ---------------------------------------------------------------------------
def summary_vector_simple(theta: torch.Tensor, obs: dict) -> torch.Tensor:
    """
    Build summary statistics using SN Ia + BAO data only.

    This mode does NOT require the CMB emulator and works out of the box.

    Parameters
    ----------
    theta : torch.Tensor
        Cosmological parameters, shape (5,) = [H0, Om, Ode, w0, wa]
    obs : dict
        Observation data with keys:
        - z_sn, dL_obs, σ_dL: Supernova data
        - z_bao, DV_obs, σ_DV: BAO data

    Returns
    -------
    torch.Tensor
        Summary vector (normalized residuals)
    """
    # Supernova residuals: (dL_model - dL_obs) / σ_dL
    dL_model = luminosity_distance(obs["z_sn"], theta)
    sn_res = (dL_model - obs["dL_obs"]) / obs["σ_dL"]

    # BAO residuals: (DV_model - DV_obs) / σ_DV
    DV_model = volume_distance(obs["z_bao"], theta)
    bao_res = (DV_model - obs["DV_obs"]) / obs["σ_DV"]

    return torch.cat([sn_res, bao_res])


def summary_vector_full(theta: torch.Tensor, obs: dict) -> torch.Tensor:
    """
    Build summary statistics including CMB power spectra.

    Requires CosmoPower emulator to be installed and available.

    Parameters
    ----------
    theta : torch.Tensor
        Cosmological parameters, shape (5,)
    obs : dict
        Observation data

    Returns
    -------
    torch.Tensor
        Summary vector including CMB spectra
    """
    emu = get_emulator()

    # CMB power spectra from emulator
    # Note: CosmoPower expects different parameter ordering
    # This mapping may need adjustment for your specific emulator
    Cl = emu(theta)

    # Distance residuals
    simple = summary_vector_simple(theta, obs)

    return torch.cat([Cl, simple])


def summary_vector(theta: torch.Tensor, obs: dict, use_cmb: bool = False) -> torch.Tensor:
    """
    Build summary statistics vector.

    Parameters
    ----------
    theta : torch.Tensor
        Cosmological parameters
    obs : dict
        Observation data
    use_cmb : bool
        Whether to include CMB power spectra (requires CosmoPower)

    Returns
    -------
    torch.Tensor
        Summary statistics vector
    """
    if use_cmb:
        return summary_vector_full(theta, obs)
    return summary_vector_simple(theta, obs)


def summary_vector_batch(
    theta_batch: torch.Tensor,
    obs: dict,
    use_cmb: bool = False
) -> torch.Tensor:
    """
    Compute summary vectors for a batch of parameter vectors.

    Parameters
    ----------
    theta_batch : torch.Tensor
        Batch of cosmological parameters, shape (batch_size, 5).
    obs : dict
        Observation data dictionary.
    use_cmb : bool
        Whether to include CMB data.

    Returns
    -------
    torch.Tensor
        Batch of summary vectors, shape (batch_size, D_x).
    """
    return torch.stack([
        summary_vector(theta, obs, use_cmb=use_cmb)
        for theta in theta_batch
    ])


# ---------------------------------------------------------------------------
# 4. Chi-squared likelihood (for evidence estimation)
# ---------------------------------------------------------------------------
def chi_squared(theta: torch.Tensor, obs: dict) -> torch.Tensor:
    """
    Compute χ² goodness of fit.

    χ² = Σ [(model - obs) / σ]²

    Parameters
    ----------
    theta : torch.Tensor
        Cosmological parameters
    obs : dict
        Observation data

    Returns
    -------
    torch.Tensor
        χ² value (scalar)
    """
    residuals = summary_vector_simple(theta, obs)
    return torch.sum(residuals ** 2)


def log_likelihood(theta: torch.Tensor, obs: dict) -> torch.Tensor:
    """
    Compute log-likelihood assuming Gaussian errors.

    log L = -0.5 * χ² + const

    Parameters
    ----------
    theta : torch.Tensor
        Cosmological parameters
    obs : dict
        Observation data

    Returns
    -------
    torch.Tensor
        Log-likelihood value
    """
    return -0.5 * chi_squared(theta, obs)


def log_likelihood_batch(theta_batch: torch.Tensor, obs: dict) -> torch.Tensor:
    """
    Compute log-likelihood for a batch of parameters.

    Parameters
    ----------
    theta_batch : torch.Tensor
        Parameters, shape (batch, 5)
    obs : dict
        Observation data

    Returns
    -------
    torch.Tensor
        Log-likelihoods, shape (batch,)
    """
    return torch.stack([log_likelihood(theta, obs) for theta in theta_batch])
