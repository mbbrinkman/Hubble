# Hubble

**Bayesian cosmological inference with normalizing flows.**

Hubble is an inference pipeline for cosmological parameter estimation and **Hubble tension quantification**. It provides:

- **Bayesian model comparison** via Bayes factors
- **Direct probability of tension resolution**: P(|ΔH₀| < ε)
- **Multiple cosmological models**: ΛCDM, split-H₀, dynamic dark energy, early dark energy
- **Fast amortized inference** using normalizing flows

## Current Status

**Working features:**
- Synthetic data generation with known cosmology
- Forward model for Type Ia supernovae and BAO distances
- Normalizing flow training and posterior sampling
- Multiple cosmological model definitions
- Bayesian evidence estimation
- Tension probability calculations

**Not yet implemented:**
- Real Pantheon+ data download (use synthetic for now)
- CMB power spectra integration (optional, requires CosmoPower)
- Cepheid/TRGB/lens distance calibrators

## Quick Start

### 1. Install Dependencies

```bash
cd Hubble
pip install -e .
```

Or install requirements manually:

```bash
pip install torch numpy h5py nflows sobol_seq click
```

### 2. Generate Synthetic Data

```bash
python setup_data.py --synthetic

# Options:
python setup_data.py --H0 70.0 --Om 0.3 --n-sn 1000 --seed 42
```

This creates synthetic SN Ia and BAO data with known cosmology (default: H₀=70, Ωm=0.3).

### 3. Prepare Observations

```bash
python prep.py
# or via CLI: hubble prep
```

### 4. Generate Training Data

```bash
python sim.py
# or with custom size:
hubble simulate --n-samples 10000
```

### 5. Train Flow

```bash
python flow_train.py
# or via CLI:
hubble train --epochs 10
```

### 6. Sample Posterior

```bash
python posterior.py
# or via CLI:
hubble sample --n-samples 50000
```

### 7. Analyze Results

```bash
hubble analyze
```

## Full Workflow Example

```bash
# 1. Generate synthetic data with H0=70, Om=0.3
python setup_data.py --synthetic --H0 70 --Om 0.3

# 2. Prepare data
python prep.py

# 3. Generate small training set (quick test)
hubble simulate --n-samples 10000

# 4. Train flow
hubble train --epochs 5

# 5. Compute observed summary
hubble observe

# 6. Sample posterior
hubble sample --n-samples 10000

# 7. Analyze
hubble analyze
```

## The Hubble Tension Problem

The "Hubble tension" is a >5σ discrepancy between:
- **Early universe** (CMB/BAO): H₀ ≈ 67.4 km/s/Mpc
- **Late universe** (Cepheids/SNe): H₀ ≈ 73.0 km/s/Mpc

Traditional frequentist analysis gives "n-sigma tension" but doesn't answer:
> **"What is the probability the tension is real vs. a statistical fluctuation?"**

Hubble answers this directly with Bayesian model comparison.

## Cosmological Models

| Model | Parameters | Description |
|-------|------------|-------------|
| `concordance` | 5 | Standard ΛCDM with single H₀ |
| `discordance` | 6 | Split H₀ (early vs. late universe) |
| `wcdm` | 5 | Dynamic dark energy (w₀-wa) |
| `early_de` | 7 | Early dark energy component |

### Training Models

```bash
# Train concordance model
hubble train-model concordance --n-samples 50000 --epochs 10

# Train discordance model
hubble train-model discordance --n-samples 50000 --epochs 10
```

### Model Comparison

```bash
# Compare models (requires trained flows)
hubble compare -m concordance -m discordance
```

### Tension Analysis

```bash
# Quantify Hubble tension
hubble tension --model discordance --epsilon 1.0

# Output includes:
# - P(|ΔH₀| < ε) - probability of resolution
# - ΔH₀ mean and credible interval
# - Statistical interpretation
```

## Project Structure

```
Hubble/
├── config.py              # Configuration management
├── models.py              # Flow architecture
├── forward.py             # Cosmological forward model
├── setup_data.py          # Data generation
├── prep.py                # Data preparation
├── sim.py                 # Training data generation
├── flow_train.py          # Flow training
├── x_obs.py               # Observed summary vector
├── posterior.py           # Posterior sampling
├── main.py                # CLI entry point
│
├── cosmology/             # Cosmological models
│   ├── base.py           # Base model class
│   ├── concordance.py    # Standard ΛCDM
│   ├── discordance.py    # Split-H₀ model
│   ├── wcdm.py           # Dynamic dark energy
│   └── early_de.py       # Early dark energy
│
├── inference/             # Inference modules
│   ├── evidence.py       # Evidence estimation
│   ├── tension.py        # Tension quantification
│   └── comparison.py     # Model comparison
│
├── analysis/              # Visualization
│   └── visualize.py      # Plotting functions
│
├── tests/                 # Unit tests
├── data/                  # Data files (generated)
├── models/                # Trained flows (generated)
└── results/               # Output files (generated)
```

## How It Works

### 1. Forward Model

Computes cosmological distances from parameters θ = (H₀, Ωm, Ωde, w₀, wa):
- **Luminosity distance** dL(z) for Type Ia supernovae
- **Volume distance** DV(z) for BAO measurements

### 2. Training Phase

For each cosmological model M:
1. Sample parameters θ from prior: θ ~ P(θ|M)
2. Compute summary statistics: x = f(θ)
3. Train normalizing flow to learn P(θ|x, M)

### 3. Inference Phase

Given observed data x_obs:
1. Load trained flow for model M
2. Sample posterior: θ ~ P(θ|x_obs, M)
3. Compute evidence: P(x_obs|M)

### 4. Model Comparison

Compute Bayes factor:
```
B = P(x_obs|concordance) / P(x_obs|discordance)
```

- B >> 1: Evidence for concordance (no tension)
- B << 1: Evidence for discordance (tension is real)

### 5. Tension Probability

Under the discordance model:
```
P(resolution) = P(|H₀_late - H₀_early| < ε | x_obs)
```

This directly answers: "What's the probability the measurements agree?"

## CLI Reference

```bash
hubble --help              # Show all commands

# Basic pipeline
hubble prep                # Prepare data
hubble simulate            # Generate training data
hubble train               # Train flow
hubble observe             # Compute x_obs
hubble sample              # Draw posterior samples
hubble analyze             # Analyze results
hubble run                 # Full pipeline

# Model comparison
hubble models              # List available models
hubble train-model MODEL   # Train specific model
hubble compare             # Compare models
hubble tension             # Quantify tension
```

## Running Tests

```bash
# Install test dependencies
pip install pytest

# Run tests
pytest tests/ -v

# End-to-end test
pytest tests/test_end_to_end.py -v
```

## Validating Results

With synthetic data, you can validate posterior recovery:

```bash
# Generate data with known H0=70, Om=0.3
python setup_data.py --H0 70 --Om 0.3

# Run full pipeline
hubble run

# Analyze - posterior should peak near true values
hubble analyze
```

The saved true parameters are in `data/raw/true_params.npy`.

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- nflows
- h5py
- numpy
- sobol_seq
- click

Optional:
- cosmopower (for CMB power spectra)
- matplotlib, corner (for visualization)

## References

- **Normalizing Flows**: [Papamakarios et al. (2019)](https://arxiv.org/abs/1912.02762)
- **Simulation-Based Inference**: [Cranmer et al. (2020)](https://arxiv.org/abs/1911.01429)
- **Hubble Tension**: [Di Valentino et al. (2021)](https://arxiv.org/abs/2103.01183)

## License

MIT License

## Development Roadmap

### Recently Implemented ✅

- **Vectorized Simpson's rule integration** for comoving distance (faster, more accurate)
- **Training improvements**: LR scheduling (cosine annealing), early stopping, checkpointing
- **Progress bars** with tqdm for long operations
- **Model selection criteria**: AIC, BIC, DIC, WAIC (in `inference/model_selection.py`)
- **CI/CD pipeline** with GitHub Actions (`.github/workflows/ci.yml`)
- **LaTeX export** for publication tables (`analysis/visualize.py`)

### TODO: Scientific Accuracy (High Priority)

- [ ] **Pantheon+ covariance matrix** - Use full covariance, not diagonal errors
- [ ] **Configurable flatness** - Add Ωk parameter for non-flat cosmologies
- [ ] **BAO sound horizon** - Include r_d and Alcock-Paczynski corrections
- [ ] **Nuisance marginalization** - Marginalize over SN absolute magnitude M, BAO r_d
- [ ] **Proper EDE physics** - Solve Friedmann equations instead of approximate scaling
- [ ] **Real Pantheon+ download** - Implement `download_real_data()`

### TODO: Validation & Comparison

- [ ] **MCMC sampler** - Add emcee/PyMC for flow validation
- [ ] **Nested sampling** - Add dynesty/MultiNest for robust evidence
- [ ] **Convergence diagnostics** - Gelman-Rubin, ESS, autocorrelation
- [ ] **Posterior predictive checks** - Validate model fit quality

### TODO: Data Support

- [ ] **Planck CMB likelihood** - Add CMB constraint support
- [ ] **Systematic errors** - Model calibration uncertainties
- [ ] **Blinding** - Add capability for unbiased analysis
- [ ] **Cepheid/TRGB calibrators** - Local distance ladder

### TODO: Infrastructure

- [ ] **YAML configuration** - External config file support
- [ ] **Input validation** - Type checking across all functions
- [ ] **FITS file I/O** - Standard astronomy format
- [ ] **Experiment tracking** - TensorBoard/W&B integration
- [ ] **API documentation** - Sphinx docs
- [ ] **Tutorial notebooks** - Jupyter examples

### TODO: Performance

- [ ] **GPU memory management** - Gradient checkpointing for large models
- [ ] **torch.compile()** - JIT optimization for critical paths
- [ ] **Distributed training** - Multi-GPU support

## Contributing

Contributions welcome! Priority areas:
- Real Pantheon+ data parsing
- CMB emulator integration
- Cepheid/TRGB/lens distance calibrators
- MCMC baseline for validation
- Neural Spline Flows for improved expressiveness
