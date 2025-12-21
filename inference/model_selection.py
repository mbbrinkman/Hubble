"""
inference/model_selection.py
----------------------------
Model selection criteria for comparing cosmological models.

Provides information-theoretic criteria as alternatives/complements
to Bayesian evidence:

- AIC: Akaike Information Criterion
- BIC: Bayesian Information Criterion
- DIC: Deviance Information Criterion
- WAIC: Widely Applicable Information Criterion

These are useful when evidence estimation is uncertain or computationally
expensive, and provide different perspectives on model complexity penalties.
"""

from dataclasses import dataclass

import numpy as np
import torch

from config import logger


@dataclass
class ModelSelectionResult:
    """Results from model selection criteria computation."""
    model_name: str
    n_params: int
    n_data: int
    max_log_likelihood: float
    mean_log_likelihood: float
    var_log_likelihood: float

    # Information criteria (lower is better)
    aic: float
    aic_c: float  # Corrected AIC for small samples
    bic: float
    dic: float
    waic: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "n_params": self.n_params,
            "n_data": self.n_data,
            "max_log_likelihood": self.max_log_likelihood,
            "mean_log_likelihood": self.mean_log_likelihood,
            "aic": self.aic,
            "aic_c": self.aic_c,
            "bic": self.bic,
            "dic": self.dic,
            "waic": self.waic,
        }


def compute_aic(log_likelihood_max: float, n_params: int) -> float:
    """
    Akaike Information Criterion.

    AIC = -2 * log(L_max) + 2 * k

    where L_max is the maximum likelihood and k is the number of parameters.
    Lower values indicate better model fit relative to complexity.

    Parameters
    ----------
    log_likelihood_max : float
        Maximum log-likelihood achieved
    n_params : int
        Number of free parameters

    Returns
    -------
    float
        AIC value
    """
    return -2 * log_likelihood_max + 2 * n_params


def compute_aic_corrected(log_likelihood_max: float, n_params: int, n_data: int) -> float:
    """
    Corrected AIC for small sample sizes.

    AIC_c = AIC + (2k² + 2k) / (n - k - 1)

    where n is the number of data points. This correction prevents
    overfitting when n/k is small (< 40 rule of thumb).

    Parameters
    ----------
    log_likelihood_max : float
        Maximum log-likelihood
    n_params : int
        Number of parameters
    n_data : int
        Number of data points

    Returns
    -------
    float
        Corrected AIC value
    """
    aic = compute_aic(log_likelihood_max, n_params)
    if n_data - n_params - 1 > 0:
        correction = (2 * n_params**2 + 2 * n_params) / (n_data - n_params - 1)
    else:
        correction = float('inf')
    return aic + correction


def compute_bic(log_likelihood_max: float, n_params: int, n_data: int) -> float:
    """
    Bayesian Information Criterion (Schwarz criterion).

    BIC = -2 * log(L_max) + k * log(n)

    BIC penalizes model complexity more strongly than AIC for large n.
    Approximates log marginal likelihood under certain conditions.

    Parameters
    ----------
    log_likelihood_max : float
        Maximum log-likelihood
    n_params : int
        Number of parameters
    n_data : int
        Number of data points

    Returns
    -------
    float
        BIC value
    """
    return -2 * log_likelihood_max + n_params * np.log(n_data)


def compute_dic(
    log_likelihood_mean: float,
    log_likelihood_at_mean: float,
) -> float:
    """
    Deviance Information Criterion.

    DIC = D_bar + p_D = -2 * E[log L] + 2 * p_D

    where p_D = D_bar - D(θ_bar) is the effective number of parameters.
    D_bar is the posterior expectation of the deviance.

    Parameters
    ----------
    log_likelihood_mean : float
        Mean log-likelihood over posterior samples E[log L(θ)]
    log_likelihood_at_mean : float
        Log-likelihood at posterior mean log L(E[θ])

    Returns
    -------
    float
        DIC value
    """
    D_bar = -2 * log_likelihood_mean
    D_at_mean = -2 * log_likelihood_at_mean
    p_D = D_bar - D_at_mean  # Effective number of parameters
    return D_bar + p_D


def compute_waic(
    log_likelihoods: np.ndarray,
) -> tuple[float, float]:
    """
    Widely Applicable Information Criterion (Watanabe-Akaike).

    WAIC = -2 * (lppd - p_WAIC)

    where lppd is the log pointwise predictive density and p_WAIC
    is the effective number of parameters.

    WAIC is fully Bayesian and works for singular models where
    BIC breaks down.

    Parameters
    ----------
    log_likelihoods : np.ndarray
        Log-likelihoods for each posterior sample, shape (n_samples,)

    Returns
    -------
    tuple[float, float]
        (WAIC value, effective number of parameters p_WAIC)
    """
    # Log pointwise predictive density
    # lppd = log(E[L(θ)]) ≈ log(mean(exp(log_likelihoods)))
    # Use log-sum-exp for numerical stability
    max_ll = np.max(log_likelihoods)
    lppd = max_ll + np.log(np.mean(np.exp(log_likelihoods - max_ll)))

    # Effective number of parameters (variance of log-likelihood)
    p_waic = np.var(log_likelihoods)

    waic = -2 * (lppd - p_waic)
    return waic, p_waic


def compute_all_criteria(
    posterior_samples: torch.Tensor,
    log_likelihood_fn,
    obs: dict,
    model_name: str = "model",
    n_data: int = None,
) -> ModelSelectionResult:
    """
    Compute all model selection criteria for a given model.

    Parameters
    ----------
    posterior_samples : torch.Tensor
        Posterior samples, shape (n_samples, n_params)
    log_likelihood_fn : callable
        Function that computes log-likelihood for a parameter vector
    obs : dict
        Observation data
    model_name : str
        Name of the model
    n_data : int, optional
        Number of data points. If None, estimated from obs.

    Returns
    -------
    ModelSelectionResult
        All computed criteria
    """
    n_samples, n_params = posterior_samples.shape

    # Estimate n_data from observations
    if n_data is None:
        n_data = len(obs.get("z_sn", [])) + len(obs.get("z_bao", []))

    logger.info(f"Computing model selection criteria for {model_name}...")
    logger.info(f"  n_params: {n_params}, n_data: {n_data}, n_samples: {n_samples}")

    # Compute log-likelihoods for all posterior samples
    log_likelihoods = []
    for theta in posterior_samples:
        ll = log_likelihood_fn(theta, obs)
        log_likelihoods.append(ll.item() if hasattr(ll, 'item') else ll)

    log_likelihoods = np.array(log_likelihoods)

    # Statistics
    max_ll = np.max(log_likelihoods)
    mean_ll = np.mean(log_likelihoods)
    var_ll = np.var(log_likelihoods)

    # Posterior mean parameters
    theta_mean = posterior_samples.mean(dim=0)
    ll_at_mean = log_likelihood_fn(theta_mean, obs)
    ll_at_mean = ll_at_mean.item() if hasattr(ll_at_mean, 'item') else ll_at_mean

    # Compute criteria
    aic = compute_aic(max_ll, n_params)
    aic_c = compute_aic_corrected(max_ll, n_params, n_data)
    bic = compute_bic(max_ll, n_params, n_data)
    dic = compute_dic(mean_ll, ll_at_mean)
    waic, p_waic = compute_waic(log_likelihoods)

    logger.info(f"  AIC: {aic:.2f}, BIC: {bic:.2f}, DIC: {dic:.2f}, WAIC: {waic:.2f}")

    return ModelSelectionResult(
        model_name=model_name,
        n_params=n_params,
        n_data=n_data,
        max_log_likelihood=max_ll,
        mean_log_likelihood=mean_ll,
        var_log_likelihood=var_ll,
        aic=aic,
        aic_c=aic_c,
        bic=bic,
        dic=dic,
        waic=waic,
    )


def compare_models(
    results: list[ModelSelectionResult],
) -> dict[str, dict[str, float]]:
    """
    Compare multiple models using all criteria.

    Parameters
    ----------
    results : list[ModelSelectionResult]
        Results for each model

    Returns
    -------
    dict
        Comparison table with delta values relative to best model
    """
    # Find best (minimum) for each criterion
    criteria = ['aic', 'aic_c', 'bic', 'dic', 'waic']

    best = {c: min(r.__getattribute__(c) for r in results) for c in criteria}

    comparison = {}
    for r in results:
        comparison[r.model_name] = {
            c: r.__getattribute__(c) - best[c]
            for c in criteria
        }
        comparison[r.model_name]['is_best'] = {
            c: r.__getattribute__(c) == best[c]
            for c in criteria
        }

    return comparison


def format_comparison_table(
    results: list[ModelSelectionResult],
    comparison: dict[str, dict[str, float]],
) -> str:
    """
    Format comparison results as a readable table.

    Parameters
    ----------
    results : list[ModelSelectionResult]
        Results for each model
    comparison : dict
        Output from compare_models()

    Returns
    -------
    str
        Formatted table
    """
    lines = []
    lines.append("=" * 80)
    lines.append("Model Selection Criteria Comparison")
    lines.append("=" * 80)
    lines.append("")

    # Header
    header = f"{'Model':<20} {'AIC':>10} {'BIC':>10} {'DIC':>10} {'WAIC':>10} {'k':>5}"
    lines.append(header)
    lines.append("-" * len(header))

    # Absolute values
    for r in sorted(results, key=lambda x: x.aic):
        line = f"{r.model_name:<20} {r.aic:>10.2f} {r.bic:>10.2f} {r.dic:>10.2f} {r.waic:>10.2f} {r.n_params:>5d}"
        lines.append(line)

    lines.append("")
    lines.append("Delta values (relative to best):")
    lines.append("-" * len(header))

    for r in sorted(results, key=lambda x: x.aic):
        c = comparison[r.model_name]
        # Mark best with *
        aic_str = f"{c['aic']:>10.2f}" if c['aic'] > 0 else f"{'0.00 *':>10}"
        bic_str = f"{c['bic']:>10.2f}" if c['bic'] > 0 else f"{'0.00 *':>10}"
        dic_str = f"{c['dic']:>10.2f}" if c['dic'] > 0 else f"{'0.00 *':>10}"
        waic_str = f"{c['waic']:>10.2f}" if c['waic'] > 0 else f"{'0.00 *':>10}"

        line = f"{r.model_name:<20} {aic_str} {bic_str} {dic_str} {waic_str}"
        lines.append(line)

    lines.append("")
    lines.append("Interpretation: Δ < 2: substantial support, Δ 4-7: less support, Δ > 10: no support")

    return "\n".join(lines)
