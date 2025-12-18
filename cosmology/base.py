"""
cosmology/base.py
-----------------
Base class for cosmological models.

Each model defines:
- Parameter space (names, bounds, priors)
- Forward model (θ → x)
- How H₀ is interpreted for early vs late universe
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Tuple
import torch
import numpy as np

from config import DEVICE, logger


@dataclass
class Parameter:
    """Definition of a single model parameter."""
    name: str
    symbol: str
    min_val: float
    max_val: float
    prior_type: str = "uniform"  # "uniform", "gaussian", "log_uniform"
    prior_mean: Optional[float] = None
    prior_std: Optional[float] = None
    description: str = ""

    def sample_prior(self, n_samples: int, device: torch.device = DEVICE) -> torch.Tensor:
        """Draw samples from the prior distribution."""
        if self.prior_type == "uniform":
            return torch.rand(n_samples, device=device) * (self.max_val - self.min_val) + self.min_val
        elif self.prior_type == "gaussian":
            samples = torch.randn(n_samples, device=device) * self.prior_std + self.prior_mean
            # Truncate to bounds
            samples = torch.clamp(samples, self.min_val, self.max_val)
            return samples
        elif self.prior_type == "log_uniform":
            log_min, log_max = np.log(self.min_val), np.log(self.max_val)
            log_samples = torch.rand(n_samples, device=device) * (log_max - log_min) + log_min
            return torch.exp(log_samples)
        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")

    def log_prior(self, values: torch.Tensor) -> torch.Tensor:
        """Compute log prior probability."""
        # Check bounds
        in_bounds = (values >= self.min_val) & (values <= self.max_val)

        if self.prior_type == "uniform":
            log_p = torch.where(
                in_bounds,
                torch.tensor(-np.log(self.max_val - self.min_val), device=values.device),
                torch.tensor(-np.inf, device=values.device)
            )
        elif self.prior_type == "gaussian":
            log_p = -0.5 * ((values - self.prior_mean) / self.prior_std) ** 2
            log_p = log_p - np.log(self.prior_std) - 0.5 * np.log(2 * np.pi)
            log_p = torch.where(in_bounds, log_p, torch.tensor(-np.inf, device=values.device))
        elif self.prior_type == "log_uniform":
            log_p = torch.where(
                in_bounds,
                -torch.log(values) - np.log(np.log(self.max_val / self.min_val)),
                torch.tensor(-np.inf, device=values.device)
            )
        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")

        return log_p


class CosmologicalModel(ABC):
    """
    Abstract base class for cosmological models.

    Each model defines a specific parameterization of the universe
    and how observational data maps to those parameters.
    """

    def __init__(self):
        self._parameters: List[Parameter] = []
        self._setup_parameters()

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        pass

    @property
    @abstractmethod
    def short_name(self) -> str:
        """Short identifier for files/CLI."""
        pass

    @property
    def description(self) -> str:
        """Detailed model description."""
        return ""

    @abstractmethod
    def _setup_parameters(self) -> None:
        """Define model parameters. Called in __init__."""
        pass

    @property
    def parameters(self) -> List[Parameter]:
        """List of model parameters."""
        return self._parameters

    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return len(self._parameters)

    @property
    def param_names(self) -> List[str]:
        """List of parameter names."""
        return [p.name for p in self._parameters]

    @property
    def param_symbols(self) -> List[str]:
        """List of parameter symbols (for plotting)."""
        return [p.symbol for p in self._parameters]

    @property
    def theta_min(self) -> torch.Tensor:
        """Minimum bounds for all parameters."""
        return torch.tensor([p.min_val for p in self._parameters], device=DEVICE)

    @property
    def theta_max(self) -> torch.Tensor:
        """Maximum bounds for all parameters."""
        return torch.tensor([p.max_val for p in self._parameters], device=DEVICE)

    def add_parameter(self, param: Parameter) -> None:
        """Add a parameter to the model."""
        self._parameters.append(param)

    def sample_prior(self, n_samples: int) -> torch.Tensor:
        """
        Draw samples from the joint prior distribution.

        Returns
        -------
        torch.Tensor
            Shape (n_samples, n_params)
        """
        samples = torch.stack([
            p.sample_prior(n_samples) for p in self._parameters
        ], dim=1)
        return samples

    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute log prior probability for parameter vectors.

        Parameters
        ----------
        theta : torch.Tensor
            Shape (n_samples, n_params) or (n_params,)

        Returns
        -------
        torch.Tensor
            Log prior probability for each sample
        """
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)

        log_p = torch.zeros(theta.shape[0], device=theta.device)
        for i, param in enumerate(self._parameters):
            log_p = log_p + param.log_prior(theta[:, i])

        return log_p

    @abstractmethod
    def get_H0_early(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Extract the early-universe Hubble constant from parameters.

        This is H₀ as inferred from CMB/BAO (sound horizon calibration).

        Parameters
        ----------
        theta : torch.Tensor
            Parameter vector(s)

        Returns
        -------
        torch.Tensor
            H₀ values for early universe
        """
        pass

    @abstractmethod
    def get_H0_late(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Extract the late-universe Hubble constant from parameters.

        This is H₀ as inferred from local distance ladder (Cepheids, SNe).

        Parameters
        ----------
        theta : torch.Tensor
            Parameter vector(s)

        Returns
        -------
        torch.Tensor
            H₀ values for late universe
        """
        pass

    @abstractmethod
    def get_cosmology_params(self, theta: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convert parameter vector to dictionary of cosmological parameters.

        Returns standardized parameters for forward model:
        - H0_early, H0_late (may be same or different)
        - Om (matter density)
        - Ode (dark energy density)
        - w0 (DE equation of state)
        - wa (DE evolution)
        - Any model-specific parameters

        Parameters
        ----------
        theta : torch.Tensor
            Parameter vector

        Returns
        -------
        dict
            Dictionary of cosmological parameters
        """
        pass

    def summary_vector(self, theta: torch.Tensor, obs: dict) -> torch.Tensor:
        """
        Compute summary statistics for given parameters.

        This wraps the forward model with model-specific parameter mapping.

        Parameters
        ----------
        theta : torch.Tensor
            Model parameters
        obs : dict
            Observation data

        Returns
        -------
        torch.Tensor
            Summary statistics vector
        """
        from forward import summary_vector as fwd_summary

        # Get cosmological parameters for this model
        cosmo = self.get_cosmology_params(theta)

        # Build standard theta vector for forward model
        # Forward model expects: (H0, Om, Ode, w0, wa)
        # Use H0_late for distance calculations (local calibration)
        theta_fwd = torch.stack([
            cosmo["H0_late"],
            cosmo["Om"],
            cosmo["Ode"],
            cosmo["w0"],
            cosmo["wa"]
        ])

        return fwd_summary(theta_fwd, obs)

    def __repr__(self) -> str:
        params_str = ", ".join([f"{p.symbol}" for p in self._parameters])
        return f"{self.name}({params_str})"
