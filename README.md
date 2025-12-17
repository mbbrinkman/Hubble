# Hubble

Cosmological parameter estimation using normalizing flows.

Hubble is a modern inference pipeline that estimates posterior distributions of cosmological parameters by combining observational data (Type Ia supernovae, Baryon Acoustic Oscillations) with neural density estimation.

## Overview

The pipeline estimates the posterior distribution of five cosmological parameters:

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Hubble constant | H₀ | Expansion rate today (km/s/Mpc) |
| Matter density | Ωₘ | Fraction of matter in the universe |
| Dark energy density | Ωde | Fraction of dark energy |
| DE equation of state | w₀ | Dark energy pressure/density ratio |
| DE evolution | wₐ | Time evolution of w |

Instead of traditional MCMC sampling, Hubble uses **Masked Autoregressive Flows (MAF)** — a type of normalizing flow that learns the conditional density p(θ|x) from simulated training data, enabling fast amortized inference.

## Installation

```bash
# Clone the repository
git clone https://github.com/mbbrinkman/Hubble.git
cd Hubble

# Install dependencies
pip install -e .

# Or install dependencies manually
pip install torch nflows numpy h5py cosmopower sobol_seq click
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA-capable GPU (recommended) or CPU

## Quick Start

### Using the CLI

```bash
# Run the full pipeline
hubble run

# Or run individual steps
hubble prep        # Prepare observational data
hubble simulate    # Generate training data (300K samples)
hubble train       # Train the normalizing flow
hubble observe     # Compute observed summary vector
hubble sample      # Draw posterior samples
hubble analyze     # Analyze results
```

### Using Python

```python
import prep
import sim
import flow_train
import x_obs
import posterior
import prob

# Run each step
prep.run()
sim.run()
flow_train.run()
x_obs.run()
posterior.run()
results = prob.run()

print(f"H0 tension probability: {results['tension_prob']:.4f}")
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA PREPARATION                         │
├─────────────────────────────────────────────────────────────────┤
│  prep.py                                                         │
│  ├── Load Pantheon+ SN Ia data                                  │
│  ├── Load BAO measurements                                       │
│  └── Save processed observations (obs.h5)                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING DATA GENERATION                    │
├─────────────────────────────────────────────────────────────────┤
│  sim.py                                                          │
│  ├── Generate 300K Sobol samples in parameter space             │
│  ├── Compute summary vectors x(θ) via forward model             │
│  │   ├── CMB power spectra (CosmoPower emulator)                │
│  │   ├── SN distance residuals                                  │
│  │   └── BAO volume distance residuals                          │
│  └── Save training pairs (θ, x)                                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         FLOW TRAINING                            │
├─────────────────────────────────────────────────────────────────┤
│  flow_train.py                                                   │
│  ├── Build 8-layer MAF (256 hidden units each)                  │
│  ├── Train on (x, θ) pairs for 10 epochs                        │
│  └── Save trained model weights                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      POSTERIOR INFERENCE                         │
├─────────────────────────────────────────────────────────────────┤
│  x_obs.py → posterior.py → prob.py                              │
│  ├── Compute x_obs from real observations                       │
│  ├── Condition flow on x_obs                                    │
│  ├── Draw 100K posterior samples                                │
│  └── Compute statistics and H0 tension probability              │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

All configuration is centralized in `config.py`:

```python
from config import config

# Flow architecture
config.flow.dim          # 5 (parameter dimensions)
config.flow.hidden_dim   # 256
config.flow.n_layers     # 8

# Training
config.training.n_train_samples  # 300,000
config.training.batch_size       # 1024
config.training.learning_rate    # 1e-3
config.training.n_epochs         # 10

# Inference
config.inference.n_posterior_samples  # 100,000

# Parameter bounds
config.physics.theta_min  # [50, 0.30, 0.60, -1.0, 0.0]
config.physics.theta_max  # [90, 0.40, 0.70, -0.5, 0.5]
```

View current configuration:
```bash
hubble info
```

## Project Structure

```
Hubble/
├── config.py          # Centralized configuration
├── models.py          # Flow architecture definition
├── forward.py         # Cosmological forward model
├── prep.py            # Data preparation
├── sim.py             # Training data generation
├── flow_train.py      # Flow training
├── x_obs.py           # Observed summary computation
├── posterior.py       # Posterior sampling
├── prob.py            # Statistical analysis
├── main.py            # CLI entry point
├── pyproject.toml     # Project metadata
├── tests/             # Unit tests
│   ├── test_forward.py
│   ├── test_config.py
│   └── test_models.py
├── data/
│   ├── raw/           # Raw observational data
│   └── proc/          # Processed data files
├── models/            # Trained model weights
└── results/           # Posterior samples and analysis
```

## Data Requirements

Place the following files in `data/raw/`:

| File | Description | Source |
|------|-------------|--------|
| `Pantheon+_SH0ES.dat` | SN Ia distance measurements | [Pantheon+](https://pantheonplussh0es.github.io/) |
| `pantheon_sigma.npy` | SN Ia uncertainties | Derived from covariance matrix |
| `BAO_DV.dat` | BAO volume distances | Various surveys |

Place the CMB emulator in `models/`:

| File | Description | Source |
|------|-------------|--------|
| `CosmoPower_CMB_TT_TE_EE_L_highacc.pkl` | CMB power spectrum emulator | [CosmoPower](https://github.com/alessiospuriomancini/cosmopower) |

## CLI Reference

```
Usage: hubble [OPTIONS] COMMAND [ARGS]...

Commands:
  prep       Prepare observational data (Pantheon+, BAO)
  simulate   Generate training data using Sobol sampling
  train      Train the normalizing flow model
  observe    Compute the observed summary vector
  sample     Draw samples from the posterior distribution
  analyze    Analyze posterior samples and compute statistics
  run        Run the full Hubble pipeline end-to-end
  info       Display configuration and system information

Options:
  --seed INTEGER  Random seed for reproducibility (default: 42)
  -v, --verbose   Enable verbose output
```

### Examples

```bash
# Quick test run with fewer samples
hubble simulate --n-samples 10000
hubble train --epochs 5
hubble sample --n-samples 10000

# Full production run
hubble run --n-train 300000 --epochs 10 --n-posterior 100000

# Analyze with custom H0 tolerance
hubble analyze --epsilon 2.0
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_forward.py -v
```

## Key Features

- **Fast inference**: Trained flow samples posteriors in seconds vs. hours for MCMC
- **Amortized**: Once trained, can condition on any observation instantly
- **Modular**: Each pipeline step can run independently
- **Configurable**: All hyperparameters centralized and overridable
- **Reproducible**: Seed management for deterministic results
- **GPU-accelerated**: Automatic CUDA detection with CPU fallback

## Known Limitations

1. **Placeholder residuals**: Cepheid, TRGB, and gravitational lens residuals are currently zeros. Replace functions in `prep.py` with real implementations.

2. **Fiducial parameters**: The observed summary vector `x_obs` requires fiducial cosmological parameters. Current defaults are Planck 2018 best-fit values.

3. **Sequential simulation**: Training data generation processes samples one-by-one. Future work could vectorize the forward model.

## References

- **Normalizing Flows**: [Papamakarios et al. (2019)](https://arxiv.org/abs/1912.02762)
- **CosmoPower**: [Spurio Mancini et al. (2021)](https://arxiv.org/abs/2106.03846)
- **Pantheon+**: [Scolnic et al. (2022)](https://arxiv.org/abs/2112.03863)

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

Priority areas:
- Implement real Cepheid/TRGB/lens residual functions
- Vectorize forward model for faster simulation
- Add validation against known cosmological results
- Expand test coverage
