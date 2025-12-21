"""
cosmology/__init__.py
---------------------
Cosmological models for Hubble inference.
"""

from cosmology.base import CosmologicalModel
from cosmology.concordance import ConcordanceModel
from cosmology.discordance import DiscordanceModel
from cosmology.early_de import EarlyDarkEnergyModel
from cosmology.wcdm import WCDMModel

__all__ = [
    "CosmologicalModel",
    "ConcordanceModel",
    "DiscordanceModel",
    "WCDMModel",
    "EarlyDarkEnergyModel",
]

# Model registry for CLI access
MODELS = {
    "concordance": ConcordanceModel,
    "discordance": DiscordanceModel,
    "wcdm": WCDMModel,
    "early_de": EarlyDarkEnergyModel,
}


def get_model(name: str) -> CosmologicalModel:
    """Get a model instance by name."""
    if name not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODELS[name]()
