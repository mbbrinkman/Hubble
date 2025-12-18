"""
models.py
---------
Shared neural network model definitions.
This ensures flow architecture is defined in ONE place only.

Supports multiple cosmological models with different parameter dimensions.
"""

import torch
import nflows
from nflows.transforms import MaskedAffineAutoregressiveTransform, CompositeTransform
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from pathlib import Path
from typing import Optional, Union

from config import config, paths, DEVICE, MODELS, logger


def build_flow(
    dim: int = None,
    hidden_dim: int = None,
    n_layers: int = None,
    device: torch.device = None
) -> Flow:
    """
    Build the Masked Autoregressive Flow (MAF) for posterior estimation.

    Parameters
    ----------
    dim : int, optional
        Dimensionality of the parameter space. Default from config.
    hidden_dim : int, optional
        Hidden layer dimension for each MAF layer. Default from config.
    n_layers : int, optional
        Number of MAF layers. Default from config.
    device : torch.device, optional
        Device to place the model on. Default from config.

    Returns
    -------
    Flow
        The constructed normalizing flow model.
    """
    dim = dim or config.flow.dim
    hidden_dim = hidden_dim or config.flow.hidden_dim
    n_layers = n_layers or config.flow.n_layers
    device = device or DEVICE

    transforms = [
        MaskedAffineAutoregressiveTransform(dim, hidden_dim)
        for _ in range(n_layers)
    ]

    flow = Flow(
        transform=CompositeTransform(transforms),
        distribution=StandardNormal([dim])
    ).to(device)

    logger.debug(f"Built MAF: dim={dim}, hidden={hidden_dim}, layers={n_layers}, device={device}")
    return flow


def build_flow_for_model(model_name: str, hidden_dim: int = None, n_layers: int = None) -> Flow:
    """
    Build a flow for a specific cosmological model.

    Parameters
    ----------
    model_name : str
        Name of the cosmological model (e.g., "concordance", "discordance")
    hidden_dim : int, optional
        Hidden layer dimension. Default from config.
    n_layers : int, optional
        Number of MAF layers. Default from config.

    Returns
    -------
    Flow
        Flow with correct dimensionality for the model
    """
    from cosmology import get_model

    model = get_model(model_name)
    dim = model.n_params

    logger.info(f"Building flow for {model.name} (dim={dim})")
    return build_flow(dim=dim, hidden_dim=hidden_dim, n_layers=n_layers)


def get_flow_path(model_name: str) -> Path:
    """Get the path for a model's flow weights."""
    return MODELS / f"flow_{model_name}.pt"


def save_flow(
    flow: Flow,
    model_name: str = None,
    path: Path = None,
    metadata: dict = None
) -> None:
    """
    Save flow model with optional metadata for reproducibility.

    Parameters
    ----------
    flow : Flow
        The trained flow model.
    model_name : str, optional
        Name of the cosmological model. Used to determine save path.
    path : Path, optional
        Explicit path to save to. Overrides model_name-based path.
    metadata : dict, optional
        Additional metadata (hyperparameters, training info, etc.)
    """
    if path is None:
        if model_name:
            path = get_flow_path(model_name)
        else:
            path = paths.flow_weights

    # Infer dimension from flow
    dim = flow._distribution._shape[0]

    save_dict = {
        "state_dict": flow.state_dict(),
        "config": {
            "dim": dim,
            "hidden_dim": config.flow.hidden_dim,
            "n_layers": config.flow.n_layers,
        },
        "model_name": model_name,
    }
    if metadata:
        save_dict["metadata"] = metadata

    torch.save(save_dict, path)
    logger.info(f"Saved flow model to {path}")


def load_flow(
    path_or_model: Union[Path, str] = None,
    device: torch.device = None
) -> Flow:
    """
    Load a trained flow model from disk.

    Parameters
    ----------
    path_or_model : Path or str, optional
        Either a path to the weights file, or a model name (e.g., "concordance").
        If None, loads the default flow.
    device : torch.device, optional
        Device to load the model onto.

    Returns
    -------
    Flow
        The loaded flow model in eval mode.
    """
    device = device or DEVICE

    # Determine path
    if path_or_model is None:
        path = paths.flow_weights
    elif isinstance(path_or_model, str) and not Path(path_or_model).exists():
        # Assume it's a model name
        path = get_flow_path(path_or_model)
    else:
        path = Path(path_or_model)

    if not path.exists():
        raise FileNotFoundError(f"Flow weights not found at {path}")

    checkpoint = torch.load(path, map_location=device)

    # Handle both old format (just state_dict) and new format (dict with config)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        flow_config = checkpoint.get("config", {})
        dim = flow_config.get("dim", config.flow.dim)
        hidden_dim = flow_config.get("hidden_dim", config.flow.hidden_dim)
        n_layers = flow_config.get("n_layers", config.flow.n_layers)
    else:
        # Old format: checkpoint is the state_dict itself
        state_dict = checkpoint
        dim = config.flow.dim
        hidden_dim = config.flow.hidden_dim
        n_layers = config.flow.n_layers

    flow = build_flow(dim=dim, hidden_dim=hidden_dim, n_layers=n_layers, device=device)
    flow.load_state_dict(state_dict)
    flow.eval()

    logger.info(f"Loaded flow model from {path}")
    return flow


def check_flow_exists(model_name: str) -> bool:
    """Check if a trained flow exists for a model."""
    return get_flow_path(model_name).exists()


def list_available_flows() -> list:
    """List all available trained flows."""
    available = []
    for path in MODELS.glob("flow_*.pt"):
        model_name = path.stem.replace("flow_", "")
        available.append(model_name)
    return available
