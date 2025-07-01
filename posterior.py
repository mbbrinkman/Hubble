"""
posterior.py
------------
Condition the trained flow on x_obs and draw 100 000 samples.
"""

import torch, nflows

dim = 5
layers = [nflows.transforms.MaskedAffineAutoregressiveTransform(dim, 256)
          for _ in range(8)]
flow = nflows.flows.Flow(
    nflows.transforms.CompositeTransform(layers),
    nflows.distributions.StandardNormal(dim)
).cuda()
flow.load_state_dict(torch.load("../models/flow.pt"))
flow.eval()

x_obs = torch.load("../data/proc/x_obs.pt").cuda()
flow = flow.condition(x_obs)

\u03b8_post = flow.sample(100_000)
torch.save(\u03b8_post, "../results/posterior.pt")
print("posterior.py finished: wrote posterior.pt")
