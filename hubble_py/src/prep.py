"""
prep.py
-------
Loads raw Pantheon+, BAO, and JWST placeholder files,
converts everything to consistent units, and stores:

  * numerical arrays  ->  HDF5  (obs.h5)
  * residual callables ->  Torch pickle  (residual_fns.pt)

Nothing fancy—feel free to replace the placeholder residual
functions with real ones when you have them.
"""

import numpy as np, h5py, torch, pickle

# ------------- create HDF5 -------------------------------------------------
with h5py.File("../data/proc/obs.h5", "w") as f:
    # ---- Pantheon+ SN Ia ----
    sn = np.loadtxt("../data/raw/Pantheon+_SH0ES.dat")
    z_sn  = sn[:, 0].astype(np.float64)
    mu_sn = sn[:, 1].astype(np.float64)
    dL_sn = 10.0 ** ((mu_sn - 25.0) / 5.0)          # convert μ→dL[Mpc]
    σ_sn  = np.load("../data/raw/pantheon_sigma.npy")

    f["z_sn"]   = z_sn
    f["dL_obs"] = dL_sn
    f["σ_dL"]   = σ_sn

    # ---- BAO (single ASCII table) ----
    bao = np.loadtxt("../data/raw/BAO_DV.dat")
    f["z_bao"]  = bao[:, 0]
    f["DV_obs"] = bao[:, 1]
    f["σ_DV"]   = bao[:, 2]

# ------------- very simple residual builders ------------------------------
# Here they just return zeros of the correct length so the code runs.
# Replace with real photometry→distance calculations when ready.

def cep_res(theta):
    return torch.zeros(42, device=theta.device)  # 42 Cepheid residuals

def trgb_res(theta):
    return torch.zeros(8, device=theta.device)   # 8 TRGB points

def lens_res(theta):
    return torch.zeros(8, device=theta.device)   # 8 lens systems

torch.save({"cep_res":  cep_res,
            "trgb_res": trgb_res,
            "lens_res": lens_res},
           "../data/proc/residual_fns.pt")
print("prep.py finished: wrote obs.h5 and residual_fns.pt")
