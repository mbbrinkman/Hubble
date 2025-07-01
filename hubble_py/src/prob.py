"""
prob.py
-------
Compute the probability that |H0_local − H0_early| < ε.
In ΛCDM both are the same, but we keep the formula general.
"""

import torch, numpy as np

\u03b8 = torch.load("../results/posterior.pt")
H0_local = \u03b8[:, 0]
H0_early = \u03b8[:, 0]          # identical in ΛCDM

eps = 1.0   # km s⁻¹ Mpc⁻¹
good = (H0_local - H0_early).abs() < eps

P = good.float().mean().item()
σ = np.sqrt(P * (1 - P) / good.numel())

print(f"P(|ΔH₀|<{eps}) = {P:.4f} ± {σ:.4f}")
