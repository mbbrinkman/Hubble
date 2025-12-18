"""
inference/comparison.py
-----------------------
Bayesian model comparison for cosmological models.

This module orchestrates the comparison of multiple cosmological models
to determine which best explains the data. The key outputs are:

1. Bayes factors between model pairs
2. Posterior model probabilities
3. Ranking of models by evidence
4. Tension resolution probability under each model
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json

from config import DEVICE, paths, logger
from cosmology.base import CosmologicalModel
from inference.evidence import estimate_log_evidence, EvidenceResult
from inference.tension import TensionAnalyzer, TensionResult


@dataclass
class ModelResult:
    """Results for a single model."""
    model: CosmologicalModel
    evidence: EvidenceResult
    tension: Optional[TensionResult] = None
    posterior_samples: Optional[torch.Tensor] = None
    flow_path: Optional[Path] = None

    @property
    def log_evidence(self) -> float:
        return self.evidence.log_evidence

    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        d = {
            "model_name": self.model.name,
            "model_short_name": self.model.short_name,
            "n_params": self.model.n_params,
            "log_evidence": self.evidence.log_evidence,
            "log_evidence_std": self.evidence.log_evidence_std,
            "effective_sample_size": self.evidence.effective_sample_size,
        }
        if self.tension:
            d["tension"] = {
                "prob_resolution": self.tension.prob_resolution,
                "prob_resolution_std": self.tension.prob_resolution_std,
                "epsilon": self.tension.epsilon,
                "delta_H0_mean": self.tension.delta_H0_mean,
                "delta_H0_std": self.tension.delta_H0_std,
                "sigma_tension": self.tension.sigma_tension,
            }
        return d


@dataclass
class ComparisonResult:
    """Results from comparing multiple models."""
    models: List[ModelResult]
    bayes_factors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    model_probabilities: Dict[str, float] = field(default_factory=dict)
    ranking: List[str] = field(default_factory=list)
    best_model: str = ""

    def __repr__(self) -> str:
        lines = ["=" * 60, "Model Comparison Results", "=" * 60, ""]

        lines.append("Model Evidence:")
        for mr in sorted(self.models, key=lambda x: -x.log_evidence):
            lines.append(f"  {mr.model.short_name:15s}: log(Z) = {mr.log_evidence:8.2f}")

        lines.append("")
        lines.append("Model Probabilities (assuming equal priors):")
        for name, prob in sorted(self.model_probabilities.items(), key=lambda x: -x[1]):
            lines.append(f"  {name:15s}: {prob:6.1%}")

        lines.append("")
        lines.append(f"Best model: {self.best_model}")

        if self.bayes_factors:
            lines.append("")
            lines.append("Bayes Factors (row/column):")
            names = list(self.bayes_factors.keys())
            header = "                " + "  ".join(f"{n:12s}" for n in names)
            lines.append(header)
            for n1 in names:
                row = f"  {n1:12s}"
                for n2 in names:
                    if n1 == n2:
                        row += f"  {'---':>12s}"
                    else:
                        bf = self.bayes_factors.get(n1, {}).get(n2, np.nan)
                        if np.isfinite(bf):
                            row += f"  {bf:12.2e}"
                        else:
                            row += f"  {'---':>12s}"
                lines.append(row)

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        return {
            "models": [m.to_dict() for m in self.models],
            "bayes_factors": self.bayes_factors,
            "model_probabilities": self.model_probabilities,
            "ranking": self.ranking,
            "best_model": self.best_model,
        }

    def save(self, path: Path) -> None:
        """Save results to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved comparison results to {path}")


class ModelComparison:
    """
    Compare multiple cosmological models using Bayesian evidence.

    Example usage:
    ```
    comparison = ModelComparison()
    comparison.add_model("concordance", flow_concordance, model_concordance)
    comparison.add_model("discordance", flow_discordance, model_discordance)
    results = comparison.compare(x_obs)
    print(results)
    ```
    """

    def __init__(self):
        self.models: Dict[str, CosmologicalModel] = {}
        self.flows: Dict[str, Any] = {}
        self._results: Optional[ComparisonResult] = None

    def add_model(
        self,
        name: str,
        flow,
        model: CosmologicalModel,
    ) -> None:
        """
        Add a model to the comparison.

        Parameters
        ----------
        name : str
            Identifier for this model
        flow : Flow
            Trained normalizing flow
        model : CosmologicalModel
            Model definition
        """
        self.models[name] = model
        self.flows[name] = flow
        logger.info(f"Added model '{name}' ({model.name}, {model.n_params} params)")

    def compare(
        self,
        x_obs: torch.Tensor,
        n_samples: int = 50000,
        n_posterior_samples: int = 100000,
        epsilon: float = 1.0,
    ) -> ComparisonResult:
        """
        Compare all registered models.

        Parameters
        ----------
        x_obs : torch.Tensor
            Observed summary statistics
        n_samples : int
            Samples for evidence estimation
        n_posterior_samples : int
            Samples for posterior/tension analysis
        epsilon : float
            Tolerance for tension resolution

        Returns
        -------
        ComparisonResult
            Full comparison results
        """
        logger.info("=" * 60)
        logger.info("Starting Bayesian Model Comparison")
        logger.info("=" * 60)

        if len(self.models) < 2:
            raise ValueError("Need at least 2 models for comparison")

        model_results = []

        # Compute evidence and tension for each model
        for name, model in self.models.items():
            logger.info(f"\nAnalyzing: {name}")
            flow = self.flows[name]

            # Evidence
            evidence = estimate_log_evidence(flow, model, x_obs, n_samples)

            # Posterior samples
            flow.eval()
            with torch.no_grad():
                x_exp = x_obs.unsqueeze(0).expand(n_posterior_samples, -1) if x_obs.dim() == 1 \
                    else x_obs.expand(n_posterior_samples, -1)
                theta_post = flow.sample(n_posterior_samples, context=x_exp)
                if isinstance(theta_post, tuple):
                    theta_post = theta_post[0]

            # Tension analysis
            analyzer = TensionAnalyzer(model)
            tension = analyzer.analyze(theta_post, epsilon=epsilon)

            model_results.append(ModelResult(
                model=model,
                evidence=evidence,
                tension=tension,
                posterior_samples=theta_post,
            ))

        # Compute Bayes factors
        bayes_factors = {}
        for mr1 in model_results:
            name1 = mr1.model.short_name
            bayes_factors[name1] = {}
            for mr2 in model_results:
                name2 = mr2.model.short_name
                if name1 != name2:
                    log_bf = mr1.log_evidence - mr2.log_evidence
                    bayes_factors[name1][name2] = np.exp(log_bf)

        # Compute posterior model probabilities (equal prior)
        log_evidences = np.array([mr.log_evidence for mr in model_results])
        max_log_ev = log_evidences.max()
        probs = np.exp(log_evidences - max_log_ev)
        probs = probs / probs.sum()

        model_probs = {
            mr.model.short_name: p
            for mr, p in zip(model_results, probs)
        }

        # Ranking
        ranking = sorted(
            [mr.model.short_name for mr in model_results],
            key=lambda n: -model_probs[n]
        )

        self._results = ComparisonResult(
            models=model_results,
            bayes_factors=bayes_factors,
            model_probabilities=model_probs,
            ranking=ranking,
            best_model=ranking[0],
        )

        logger.info(f"\n{self._results}")
        return self._results

    def get_tension_summary(self) -> str:
        """
        Get a human-readable summary of tension analysis.

        Returns
        -------
        str
            Summary text suitable for papers/reports
        """
        if self._results is None:
            raise ValueError("Run compare() first")

        lines = []
        lines.append("Hubble Tension Analysis Summary")
        lines.append("=" * 40)

        for mr in self._results.models:
            if mr.tension:
                lines.append(f"\n{mr.model.name}:")
                lines.append(f"  P(|ΔH₀| < {mr.tension.epsilon}) = "
                           f"{mr.tension.prob_resolution:.3f} ± {mr.tension.prob_resolution_std:.3f}")
                lines.append(f"  ΔH₀ = {mr.tension.delta_H0_mean:.2f} ± {mr.tension.delta_H0_std:.2f} km/s/Mpc")
                lines.append(f"  Effective tension: {mr.tension.sigma_tension:.1f}σ")

        # Key conclusions
        best = self._results.best_model
        best_prob = self._results.model_probabilities[best]

        lines.append(f"\nConclusion:")
        lines.append(f"  Best model: {best} (P = {best_prob:.1%})")

        # Find discordance results if present
        discord_mr = next(
            (mr for mr in self._results.models if mr.model.short_name == "discordance"),
            None
        )
        if discord_mr and discord_mr.tension:
            prob_res = discord_mr.tension.prob_resolution
            eps = discord_mr.tension.epsilon
            lines.append(f"  Under split-H₀ model: {prob_res:.1%} probability of resolution "
                        f"within {eps} km/s/Mpc")
            if prob_res < 0.05:
                lines.append("  → Strong evidence that Hubble tension is REAL")
            elif prob_res < 0.32:
                lines.append("  → Moderate evidence that Hubble tension is real")
            else:
                lines.append("  → Tension may be resolvable")

        return "\n".join(lines)


def quick_compare(
    x_obs: torch.Tensor,
    model_names: List[str] = ["concordance", "discordance"],
    n_samples: int = 50000,
) -> ComparisonResult:
    """
    Quick comparison of models using pre-trained flows.

    Parameters
    ----------
    x_obs : torch.Tensor
        Observed summary statistics
    model_names : list
        Names of models to compare
    n_samples : int
        Samples for evidence estimation

    Returns
    -------
    ComparisonResult
        Comparison results
    """
    from cosmology import get_model
    from models import load_flow

    comparison = ModelComparison()

    for name in model_names:
        model = get_model(name)
        flow_path = paths.MODELS / f"flow_{name}.pt"

        if not flow_path.exists():
            raise FileNotFoundError(
                f"No trained flow for '{name}'. "
                f"Run 'hubble train --model {name}' first."
            )

        flow = load_flow(flow_path)
        comparison.add_model(name, flow, model)

    return comparison.compare(x_obs, n_samples=n_samples)
