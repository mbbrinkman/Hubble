# Hubble

**Bayesian cosmological inference with normalizing flows.**

Hubble is an advanced inference pipeline for cosmological parameter estimation and **Hubble tension quantification**. It provides:

- **Bayesian model comparison** via Bayes factors
- **Direct probability of tension resolution**: P(|ΔH₀| < ε)
- **Multiple cosmological models**: ΛCDM, split-H₀, dynamic dark energy, early dark energy
- **Fast amortized inference** using normalizing flows

## The Hubble Tension Problem

The "Hubble tension" is a >5σ discrepancy between:
- **Early universe** (CMB/BAO): H₀ ≈ 67.4 km/s/Mpc
- **Late universe** (Cepheids/SNe): H₀ ≈ 73.0 km/s/Mpc

Traditional frequentist analysis gives "n-sigma tension" but doesn't answer:
> **"What is the probability the tension is real vs. a statistical fluctuation?"**

Hubble answers this directly with Bayesian model comparison.

## Key Features

| Feature | Description |
|---------|-------------|
| **P(resolution)** | Direct probability that |ΔH₀| < ε under the discordance model |
| **Bayes factors** | Compare concordance vs. discordance hypotheses |
| **Multiple models** | ΛCDM, split-H₀, wCDM, early dark energy |
| **Fast inference** | Posterior in seconds, not hours |
| **Amortized** | Train once, condition on any observation instantly |

## Installation

```bash
git clone https://github.com/mbbrinkman/Hubble.git
cd Hubble
pip install -e .
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA GPU (recommended) or CPU

## Quick Start

### Basic Pipeline

```bash
hubble prep           # Prepare data
hubble train-model concordance -n 100000   # Train concordance model
hubble train-model discordance -n 100000   # Train discordance model
hubble observe        # Compute observed summary
hubble compare        # Compare models!
```

### Tension Analysis

```bash
# Quantify Hubble tension
hubble tension --model discordance --epsilon 1.0

# Output:
# P(|ΔH₀| < 1.0) = 0.023 ± 0.005
# ΔH₀ = 5.4 ± 1.2 km/s/Mpc
# CONCLUSION: Strong evidence that Hubble tension is REAL
```

### Model Comparison

```bash
# Compare concordance vs. discordance
hubble compare -m concordance -m discordance

# Output:
# Bayes factor (concordance/discordance) = 0.03
# → Strong evidence for discordance (tension is real)
```

## Cosmological Models

| Model | Parameters | Description |
|-------|------------|-------------|
| `concordance` | 5 | Standard ΛCDM with single H₀ |
| `discordance` | 6 | Split H₀ (early vs. late universe) |
| `wcdm` | 5 | Dynamic dark energy (w₀-wa) |
| `early_de` | 7 | Early dark energy component |

```bash
# List all models
hubble models

# Train any model
hubble train-model early_de --n-samples 100000 --epochs 20
```

## CLI Reference

### Basic Commands

```bash
hubble prep          # Prepare observational data
hubble simulate      # Generate training data
hubble train         # Train default flow
hubble observe       # Compute observed summary
hubble sample        # Draw posterior samples
hubble analyze       # Basic posterior analysis
hubble run           # Full pipeline
```

### Model Comparison Commands

```bash
hubble models                    # List available models
hubble train-model MODEL         # Train flow for specific model
hubble compare -m M1 -m M2       # Compare models via Bayes factors
hubble tension -m MODEL          # Quantify H₀ tension
```

### Visualization Commands

```bash
hubble plot-corner -m MODEL      # Corner plot of posterior
hubble plot-tension -m MODEL     # Tension probability curve
```

### Examples

```bash
# Full tension analysis workflow
hubble prep
hubble train-model concordance -n 100000 -e 15
hubble train-model discordance -n 100000 -e 15
hubble observe
hubble compare -m concordance -m discordance --output results.json
hubble tension -m discordance -e 2.0
hubble plot-corner -m discordance
hubble plot-tension -m discordance
```

## Project Structure

```
Hubble/
├── config.py              # Configuration management
├── models.py              # Flow architecture
├── forward.py             # Cosmological forward model
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
├── data/                  # Data files
├── models/                # Trained flows
└── results/               # Output files
```

## How It Works

### 1. Training Phase

For each cosmological model M:
1. Sample parameters θ from prior: θ ~ P(θ|M)
2. Compute summary statistics: x = f(θ)
3. Train flow to learn P(θ|x, M)

### 2. Inference Phase

Given observed data x_obs:
1. Load trained flow for model M
2. Sample posterior: θ ~ P(θ|x_obs, M)
3. Compute evidence: P(x_obs|M)

### 3. Model Comparison

Compute Bayes factor:
```
B = P(x_obs|concordance) / P(x_obs|discordance)
```

- B >> 1: Evidence for concordance (no tension)
- B << 1: Evidence for discordance (tension is real)

### 4. Tension Probability

Under the discordance model:
```
P(resolution) = P(|H₀_late - H₀_early| < ε | x_obs)
```

This directly answers: "What's the probability the measurements agree?"

## Scientific Output

For a paper, you would report:

> Using simulation-based inference with normalizing flows, we compute
> the Bayesian evidence for concordance (single H₀) versus discordance
> (split H₀) models. We find a Bayes factor of B = 0.03 ± 0.01,
> corresponding to strong evidence for discordance.
>
> Under the discordance model, we find P(|ΔH₀| < 1 km/s/Mpc) = 0.02,
> indicating a 2% probability that the Hubble tension is a statistical
> fluctuation.

## Comparison with Other Tools

| Tool | Approach | Model Comparison | Tension Metric | Speed |
|------|----------|------------------|----------------|-------|
| CosmoMC | MCMC | Via nested sampling | Manual | Slow |
| Cobaya | MCMC+nested | PolyChord | Manual | Slow |
| MontePython | MCMC | Limited | Manual | Slow |
| sbi/pydelfi | Neural | Not native | Not native | Fast |
| **Hubble** | Flows | **Native Bayes factors** | **P(resolution)** | **Fast** |

## Data Requirements

Place in `data/raw/`:
- `Pantheon+_SH0ES.dat` - SN Ia distances
- `pantheon_sigma.npy` - Uncertainties
- `BAO_DV.dat` - BAO measurements

Place in `models/`:
- `CosmoPower_CMB_TT_TE_EE_L_highacc.pkl` - CMB emulator

## References

- **Normalizing Flows**: [Papamakarios et al. (2019)](https://arxiv.org/abs/1912.02762)
- **Simulation-Based Inference**: [Cranmer et al. (2020)](https://arxiv.org/abs/1911.01429)
- **Hubble Tension**: [Di Valentino et al. (2021)](https://arxiv.org/abs/2103.01183)
- **CosmoPower**: [Spurio Mancini et al. (2021)](https://arxiv.org/abs/2106.03846)
- **Early Dark Energy**: [Poulin et al. (2019)](https://arxiv.org/abs/1811.04083)

## License

MIT License

## Contributing

Contributions welcome! Priority areas:
- Implement real Cepheid/TRGB/lens residual functions
- Add MCMC baseline for validation
- Implement Neural Spline Flows for improved expressiveness
- Add more extended cosmological models
