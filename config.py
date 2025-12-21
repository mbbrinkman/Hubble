"""
config.py
---------
Centralized configuration for paths, hyperparameters, and device settings.
All other modules should import from here instead of hardcoding values.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "proc"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

# Ensure directories exist
for d in [DATA_RAW, DATA_PROC, MODELS, RESULTS]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Device Configuration
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """Get the best available device with fallback to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)

# ---------------------------------------------------------------------------
# Hyperparameter Configuration
# ---------------------------------------------------------------------------
@dataclass
class FlowConfig:
    """Normalizing flow architecture configuration."""
    dim: int = 5
    hidden_dim: int = 256
    n_layers: int = 8

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    n_train_samples: int = 300_000
    batch_size: int = 1024
    learning_rate: float = 1e-3
    n_epochs: int = 10

@dataclass
class InferenceConfig:
    """Inference configuration."""
    n_posterior_samples: int = 100_000

@dataclass
class PhysicsConfig:
    """Physics model configuration."""
    integration_points: int = 128
    # Parameter bounds: θ = (H0, Ωm, Ωde, w0, wa)
    theta_min: list[float] = field(default_factory=lambda: [50.0, 0.30, 0.60, -1.0, 0.0])
    theta_max: list[float] = field(default_factory=lambda: [90.0, 0.40, 0.70, -0.5, 0.5])
    # Parameter names for reference
    param_names: list[str] = field(default_factory=lambda: ["H0", "Ωm", "Ωde", "w0", "wa"])

@dataclass
class Config:
    """Main configuration container."""
    flow: FlowConfig = field(default_factory=FlowConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)

# Default configuration instance
config = Config()

# ---------------------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------------------
class Paths:
    """Centralized file path definitions."""
    # Raw data
    pantheon_data = DATA_RAW / "Pantheon+_SH0ES.dat"
    pantheon_sigma = DATA_RAW / "pantheon_sigma.npy"
    bao_data = DATA_RAW / "BAO_DV.dat"

    # Processed data
    obs_h5 = DATA_PROC / "obs.h5"
    residual_fns = DATA_PROC / "residual_fns.pt"
    train_data = DATA_PROC / "train.pt"
    x_obs = DATA_PROC / "x_obs.pt"

    # Models
    cmb_emulator = MODELS / "CosmoPower_CMB_TT_TE_EE_L_highacc.pkl"
    flow_weights = MODELS / "flow.pt"

    # Results
    posterior = RESULTS / "posterior.pt"

paths = Paths()

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure logging for the package."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("hubble")

logger = setup_logging()
