"""
flow_train.py
-------------
Train an 8-layer Masked Autoregressive Flow (MAF) on the simulated data.
Everything in full precision; one GPU is plenty.
"""

import torch, nflows
from torch.utils.data import DataLoader, TensorDataset

data = torch.load("../data/proc/train.pt")
\u03b8, x = data["\u03b8"], data["x"]
dim = \u03b8.shape[1]

# Build flow
transforms = [nflows.transforms.MaskedAffineAutoregressiveTransform(dim, 256)
              for _ in range(8)]
flow = nflows.flows.Flow(
    nflows.transforms.CompositeTransform(transforms),
    nflows.distributions.StandardNormal(dim)
).cuda()

loader = DataLoader(TensorDataset(x, \u03b8), batch_size=1024, shuffle=True)
opt = torch.optim.Adam(flow.parameters(), lr=1e-3)

for epoch in range(10):                     # 10 epochs are fine for demo
    for xb, \u03b8b in loader:
        loss = -flow.log_prob(inputs=\u03b8b, context=xb).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"epoch {epoch:2d}  loss {loss.item():.3f}")

torch.save(flow.state_dict(), "../models/flow.pt")
print("flow_train.py finished: wrote flow.pt")
