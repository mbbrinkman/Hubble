"""
forward.py
----------
Forward model that transforms cosmological parameters θ into summary statistics x.

θ  = (H0, Ωm, Ωde, w0, wa)     shape (5,) or (batch, 5)
obs = dict of tensors created in prep.py

The summary vector combines:
- CMB power spectra from CosmoPower emulator
- Type Ia supernova distance residuals
- BAO volume distance residuals
- Cepheid/TRGB/lens calibrator residuals
"""

import torch
import numpy as np
import cosmopower as cp

from config import config, paths, DEVICE, logger

# ---------------------------------------------------------------------------
# 1. CMB Emulator (lazy loading to avoid import-time side effects)
# ---------------------------------------------------------------------------
_emu = None


def get_emulator():
    """Lazy-load the CMB power spectrum emulator."""
    global _emu
    if _emu is None:
        if not paths.cmb_emulator.exists():
            raise FileNotFoundError(
                f"CMB emulator not found at {paths.cmb_emulator}. "
                "Please download the CosmoPower model file."
            )
        _emu = cp.cosmopower_Pk()
        _emu.load(str(paths.cmb_emulator))
        _emu.to(DEVICE)
        logger.info(f"Loaded CMB emulator from {paths.cmb_emulator}")
    return _emu


# ---------------------------------------------------------------------------
# 2. Cosmological distance functions
# ---------------------------------------------------------------------------
def Ez(z: torch.Tensor, H0: float, Om: float, Ode: float, w0: float, wa: float) -> torch.Tensor:
    """
    Dimensionless Hubble factor E(z) = H(z)/H0 for flat w0-wa dark energy.

    Parameters
    ----------
    z : torch.Tensor
        Redshift values.
    H0, Om, Ode, w0, wa : float
        Cosmological parameters.

    Returns
    -------
    torch.Tensor
        E(z) values.
    """
    a = 1.0 / (1.0 + z)
    w = w0 + wa * (1.0 - a)
    return torch.sqrt(Om * (1.0 + z) ** 3 + Ode * a ** (-3.0 * (1.0 + w)))


def chi(z: torch.Tensor, *theta, n_points: int = None) -> torch.Tensor:
    """
    Comoving distance χ(z) = ∫₀ᶻ dz'/E(z').

    Uses trapezoidal integration with configurable number of points.

    Parameters
    ----------
    z : torch.Tensor
        Redshift (scalar tensor).
    theta : tuple
        Cosmological parameters (H0, Om, Ode, w0, wa).
    n_points : int, optional
        Number of integration points. Default from config.

    Returns
    -------
    torch.Tensor
        Comoving distance in units of c/H0.
    """
    n_points = n_points or config.physics.integration_points
    z_grid = torch.linspace(0.0, z.item() if z.dim() == 0 else z, n_points, device=DEVICE)
    return torch.trapz(1.0 / Ez(z_grid, *theta), z_grid)


def DV(z: torch.Tensor, *theta) -> torch.Tensor:
    """
    Volume distance D_V(z) used for BAO measurements.

    D_V = [χ(z)² * z / E(z)]^(1/3)

    Parameters
    ----------
    z : torch.Tensor
        Redshift.
    theta : tuple
        Cosmological parameters.

    Returns
    -------
    torch.Tensor
        Volume distance.
    """
    chi_z = chi(z, *theta)
    return (chi_z ** 2 * z / Ez(z, *theta)) ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# 3. Summary vector construction
# ---------------------------------------------------------------------------
def summary_vector(theta: torch.Tensor, obs: dict) -> torch.Tensor:
    """
    Build the summary statistics vector x(θ) for the normalizing flow.

    Parameters
    ----------
    theta : torch.Tensor
        Cosmological parameters, shape (5,).
    obs : dict
        Observation data with keys:
        - z_sn, dL_obs, σ_dL: Supernova data
        - z_bao, DV_obs, σ_DV: BAO data
        - cep_res, trgb_res, lens_res: Callable residual functions

    Returns
    -------
    torch.Tensor
        1-D summary vector (length ~ 7600+).
    """
    emu = get_emulator()

    # CMB power spectra from emulator
    Cl = emu(theta)  # shape (~7500,)

    # Supernova distance residuals
    # Model: luminosity distance dL = χ(z) * (1+z)
    dL_mod = chi(obs["z_sn"], *theta) * (1.0 + obs["z_sn"])
    sn_res = (dL_mod - obs["dL_obs"]) / obs["σ_dL"]

    # BAO volume distance residuals
    DV_mod = DV(obs["z_bao"], *theta)
    bao_res = (DV_mod - obs["DV_obs"]) / obs["σ_DV"]

    # Local distance calibrators (user-supplied callables)
    cep = obs["cep_res"](theta)
    trgb = obs["trgb_res"](theta)
    lens = obs["lens_res"](theta)

    return torch.cat([Cl, sn_res, bao_res, cep, trgb, lens])


def summary_vector_batch(theta_batch: torch.Tensor, obs: dict) -> torch.Tensor:
    """
    Compute summary vectors for a batch of parameter vectors.

    This is more efficient than calling summary_vector in a loop.

    Parameters
    ----------
    theta_batch : torch.Tensor
        Batch of cosmological parameters, shape (batch_size, 5).
    obs : dict
        Observation data dictionary.

    Returns
    -------
    torch.Tensor
        Batch of summary vectors, shape (batch_size, D_x).
    """
    # For now, loop over batch (CMB emulator may not support batching)
    # TODO: Vectorize CMB emulator calls if supported
    return torch.stack([summary_vector(theta, obs) for theta in theta_batch])
