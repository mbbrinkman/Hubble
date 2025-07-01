"""
x_obs.py
--------
Build the observed summary vector using the same forward model.
"""

import torch, h5py, forward, pickle

device = torch.device("cuda")

# Load obs dict
with h5py.File("../data/proc/obs.h5", "r") as f:
    obs_arrays = {k: torch.tensor(f[k][...], device=device) for k in f}
callbacks = torch.load("../data/proc/residual_fns.pt")
obs = {**obs_arrays, **callbacks}

\u03b8_dummy = torch.tensor([70., 0.32, 0.68, -1.0, 0.0], device=device)  # anything
x_obs = forward.summary_vector(\u03b8_dummy, obs)
torch.save(x_obs, "../data/proc/x_obs.pt")
print("x_obs.py finished: wrote x_obs.pt")
