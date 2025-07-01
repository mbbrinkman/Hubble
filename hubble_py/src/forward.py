"""
forward.py
-----------
Very small helper module that turns a parameter vector \u03b8 into the
summary-vector x used by the normalising flow.

\u03b8  = (H0, \u03a9m, \u03a9de, w0, wa)     shape (5,)
obs = dict of tensors created in prep.py
"""

import torch, numpy as np, cosmopower as cp

# -----------------------------------------------------------------------
# 1. CMB emulator   (loads ~40 MB pickle, a one-liner)
# -----------------------------------------------------------------------
emu = cp.cosmopower_Pk()
emu.load("../models/CosmoPower_CMB_TT_TE_EE_L_highacc.pkl")
emu.to("cuda")

# -----------------------------------------------------------------------
# 2. Simple scalar helpers
# -----------------------------------------------------------------------
def Ez(z, H0, Om, Ode, w0, wa):
    """Dimensionless Hubble factor E(z)=H(z)/H0 (flat w0-wa)."""
    a = 1.0 / (1.0 + z)
    w = w0 + wa * (1.0 - a)
    return torch.sqrt(Om * (1.0 + z) ** 3 + Ode * a ** (-3.0 * (1.0 + w)))

def chi(z, *\u03b8):
    """Comoving distance \u03c7(z)=\u222b0^z dz'/E(z'). 128-point trapezoid is enough."""
    z_grid = torch.linspace(0.0, z, 128, device="cuda")
    return torch.trapz(1.0 / Ez(z_grid, *\u03b8), z_grid)

def DV(z, *\u03b8):
    """Volume distance used for BAO."""
    return (chi(z, *\u03b8) ** 2 * z / Ez(z, *\u03b8)) ** (1.0 / 3.0)

# -----------------------------------------------------------------------
# 3. Build summary vector  x(\u03b8)
# -----------------------------------------------------------------------
def summary_vector(theta, obs):
    """
    theta : (5,)  torch tensor on cuda
    obs   : dict with z_sn, dL_obs, \u03c3_dL,  z_bao, DV_obs, \u03c3_DV,
                     cep_res, trgb_res, lens_res
    returns: 1-D tensor  (length ~ a few thousand)
    """
    # CMB spectra (already flattened by CosmoPower)
    Cl = emu(theta)                       # shape (~7500,)

    # SN residuals
    dL_mod = chi(obs["z_sn"], *theta) * (1.0 + obs["z_sn"])
    sn_res = (dL_mod - obs["dL_obs"]) / obs["\u03c3_dL"]

    # BAO residuals
    DV_mod = DV(obs["z_bao"], *theta)
    bao_res = (DV_mod - obs["DV_obs"]) / obs["\u03c3_DV"]

    # Cepheid / TRGB / lens blocks
    cep  = obs["cep_res"](theta)   # user-supplied callables
    trgb = obs["trgb_res"](theta)
    lens = obs["lens_res"](theta)

    return torch.cat([Cl, sn_res, bao_res, cep, trgb, lens])
