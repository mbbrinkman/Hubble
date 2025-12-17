"""
models.py
---------
Shared neural network model definitions.
This ensures flow architecture is defined in ONE place only.
"""

import torch
import nflows
from nflows.transforms import MaskedAffineAutoregressiveTransform, CompositeTransform
from nflows.distributions import StandardNormal
from nflows.flows import Flow

from config import config, paths, DEVICE, logger


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


def save_flow(flow: Flow, metadata: dict = None) -> None:
    """
    Save flow model with optional metadata for reproducibility.

    Parameters
    ----------
    flow : Flow
        The trained flow model.
    metadata : dict, optional
        Additional metadata (hyperparameters, training info, etc.)
    """
    save_dict = {
        "state_dict": flow.state_dict(),
        "config": {
            "dim": config.flow.dim,
            "hidden_dim": config.flow.hidden_dim,
            "n_layers": config.flow.n_layers,
        }
    }
    if metadata:
        save_dict["metadata"] = metadata

    torch.save(save_dict, paths.flow_weights)
    logger.info(f"Saved flow model to {paths.flow_weights}")


def load_flow(device: torch.device = None) -> Flow:
    """
    Load a trained flow model from disk.

    Parameters
    ----------
    device : torch.device, optional
        Device to load the model onto.

    Returns
    -------
    Flow
        The loaded flow model in eval mode.
    """
    device = device or DEVICE

    if not paths.flow_weights.exists():
        raise FileNotFoundError(f"Flow weights not found at {paths.flow_weights}")

    checkpoint = torch.load(paths.flow_weights, map_location=device)

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

    logger.info(f"Loaded flow model from {paths.flow_weights}")
    return flow
