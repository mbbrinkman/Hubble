"""
sim.py
------
Generate 300 000 Sobol points in parameter space and compute their
summary vectors *one by one* (simplicity > speed). Saves train.pt
containing the full tensors \u03b8 and x.
"""

import torch, sobol_seq, h5py, pickle, forward

device = torch.device("cuda")

# Parameter bounds :  \u03b8 = (H0, \u03a9m, \u03a9de, w0, wa)
\u03b8_min = torch.tensor([50., 0.30, 0.70, -1.0,  0.0], device=device)
\u03b8_max = torch.tensor([90., 0.40, 0.60, -0.5,  0.5], device=device)

N = 300_000
sob = torch.tensor(sobol_seq.i4_sobol_generate(5, N), device=device)
\u03b8_all = sob * (\u03b8_max - \u03b8_min) + \u03b8_min             # shape (N,5)

# Load obs arrays + residual functions
with h5py.File("../data/proc/obs.h5", "r") as f:
    obs_arrays = {k: torch.tensor(f[k][...], device=device) for k in f}
callbacks = torch.load("../data/proc/residual_fns.pt")
obs = {**obs_arrays, **callbacks}

# Compute summary vectors one by one (simple but slow)
x_list = []
for i, \u03b8 in enumerate(\u03b8_all):
    if i % 5000 == 0:
        print(f"{i}/{N} done")
    x_list.append(forward.summary_vector(\u03b8, obs))
x_all = torch.stack(x_list)                      # shape (N, Dx)

torch.save({"\u03b8": \u03b8_all, "x": x_all}, "../data/proc/train.pt")
print("sim.py finished: wrote train.pt")
