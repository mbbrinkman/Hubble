"""
analysis/visualize.py
---------------------
Visualization tools for Hubble results.

Generates publication-quality plots:
- Corner plots of posterior distributions
- H₀ tension visualization
- Model comparison plots
- Resolution probability curves
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

from config import paths, logger
from cosmology.base import CosmologicalModel

# Lazy import matplotlib to avoid issues if not installed
_plt = None
_corner = None


def _get_plt():
    global _plt
    if _plt is None:
        try:
            import matplotlib.pyplot as plt
            plt.style.use('seaborn-v0_8-whitegrid')
            _plt = plt
        except ImportError:
            raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib")
    return _plt


def _get_corner():
    global _corner
    if _corner is None:
        try:
            import corner
            _corner = corner
        except ImportError:
            warnings.warn("corner package not installed. Install with: pip install corner")
            _corner = None
    return _corner


def plot_corner(
    theta_samples,
    model: CosmologicalModel,
    truths: Optional[List[float]] = None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Create a corner plot of posterior samples.

    Parameters
    ----------
    theta_samples : torch.Tensor or np.ndarray
        Posterior samples, shape (n_samples, n_params)
    model : CosmologicalModel
        Model for parameter names/labels
    truths : list, optional
        True parameter values to mark
    output_path : Path, optional
        Where to save the figure
    title : str, optional
        Plot title
    **kwargs
        Additional arguments to corner.corner

    Returns
    -------
    matplotlib.figure.Figure
        The corner plot figure
    """
    plt = _get_plt()
    corner = _get_corner()

    if corner is None:
        warnings.warn("corner package not available, skipping corner plot")
        return None

    # Convert to numpy
    if hasattr(theta_samples, 'cpu'):
        samples = theta_samples.cpu().numpy()
    else:
        samples = np.asarray(theta_samples)

    # Default kwargs
    default_kwargs = {
        "labels": model.param_symbols,
        "quantiles": [0.16, 0.5, 0.84],
        "show_titles": True,
        "title_kwargs": {"fontsize": 12},
        "label_kwargs": {"fontsize": 14},
    }
    default_kwargs.update(kwargs)

    if truths is not None:
        default_kwargs["truths"] = truths

    fig = corner.corner(samples, **default_kwargs)

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved corner plot to {output_path}")

    return fig


def plot_tension_curve(
    tension_curve: Dict[str, np.ndarray],
    output_path: Optional[Path] = None,
    title: str = "Hubble Tension Resolution Probability",
) -> Any:
    """
    Plot P(|ΔH₀| < ε) as a function of ε.

    Parameters
    ----------
    tension_curve : dict
        Output from TensionAnalyzer.probability_curve()
        {"epsilon": [...], "probability": [...], "std_error": [...]}
    output_path : Path, optional
        Where to save the figure
    title : str
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_plt()

    fig, ax = plt.subplots(figsize=(8, 6))

    eps = tension_curve["epsilon"]
    prob = tension_curve["probability"]
    err = tension_curve["std_error"]

    ax.errorbar(eps, prob, yerr=err, fmt='o-', capsize=4, linewidth=2, markersize=8)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')

    ax.set_xlabel("ε (km/s/Mpc)", fontsize=14)
    ax.set_ylabel("P(|ΔH₀| < ε)", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max(eps) * 1.1)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add interpretation zones
    ax.axhspan(0, 0.05, alpha=0.1, color='red', label='Strong tension')
    ax.axhspan(0.05, 0.32, alpha=0.1, color='orange')
    ax.axhspan(0.32, 1.0, alpha=0.1, color='green')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved tension curve to {output_path}")

    return fig


def plot_model_comparison(
    comparison_result,
    output_path: Optional[Path] = None,
) -> Any:
    """
    Plot model comparison results.

    Parameters
    ----------
    comparison_result : ComparisonResult
        Output from ModelComparison.compare()
    output_path : Path, optional
        Where to save the figure

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_plt()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Log evidence
    ax1 = axes[0]
    names = [mr.model.short_name for mr in comparison_result.models]
    log_evs = [mr.log_evidence for mr in comparison_result.models]
    log_ev_stds = [mr.evidence.log_evidence_std for mr in comparison_result.models]

    y_pos = np.arange(len(names))
    ax1.barh(y_pos, log_evs, xerr=log_ev_stds, align='center', alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=12)
    ax1.set_xlabel("log Evidence", fontsize=14)
    ax1.set_title("Model Evidence", fontsize=16)

    # Right: Model probabilities
    ax2 = axes[1]
    probs = [comparison_result.model_probabilities[n] for n in names]

    bars = ax2.barh(y_pos, probs, align='center', alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=12)
    ax2.set_xlabel("Posterior Probability", fontsize=14)
    ax2.set_title("Model Probabilities (equal priors)", fontsize=16)
    ax2.set_xlim(0, 1)

    # Add percentage labels
    for bar, prob in zip(bars, probs):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{prob:.1%}', va='center', fontsize=12)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved model comparison to {output_path}")

    return fig


def plot_H0_posteriors(
    posteriors: Dict[str, Tuple[np.ndarray, CosmologicalModel]],
    output_path: Optional[Path] = None,
    reference_values: Optional[Dict[str, float]] = None,
) -> Any:
    """
    Plot H₀ posterior distributions from multiple models.

    Parameters
    ----------
    posteriors : dict
        {model_name: (theta_samples, model)} for each model
    output_path : Path, optional
        Where to save the figure
    reference_values : dict, optional
        Reference H₀ values to mark, e.g., {"Planck": 67.4, "SH0ES": 73.0}

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_plt()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(posteriors)))

    for (name, (samples, model)), color in zip(posteriors.items(), colors):
        # Get H0 (could be early or late depending on model)
        if hasattr(samples, 'cpu'):
            samples = samples.cpu()

        H0_early = model.get_H0_early(samples).numpy()
        H0_late = model.get_H0_late(samples).numpy()

        # Plot main H0 distribution
        ax.hist(H0_late, bins=50, density=True, alpha=0.5,
               color=color, label=f"{name} (late)")

        # If different, plot early too
        if not np.allclose(H0_early, H0_late):
            ax.hist(H0_early, bins=50, density=True, alpha=0.3,
                   color=color, linestyle='--', label=f"{name} (early)")

    # Add reference lines
    if reference_values:
        for ref_name, ref_val in reference_values.items():
            ax.axvline(ref_val, linestyle='--', linewidth=2,
                      label=f'{ref_name}: {ref_val}')

    ax.set_xlabel("H₀ (km/s/Mpc)", fontsize=14)
    ax.set_ylabel("Probability Density", fontsize=14)
    ax.set_title("H₀ Posterior Distributions", fontsize=16)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved H0 posteriors to {output_path}")

    return fig


def plot_delta_H0_distribution(
    theta_samples,
    model: CosmologicalModel,
    epsilon: float = 1.0,
    output_path: Optional[Path] = None,
) -> Any:
    """
    Plot the distribution of ΔH₀ = H₀_late - H₀_early.

    Parameters
    ----------
    theta_samples : torch.Tensor
        Posterior samples
    model : CosmologicalModel
        Model (must have different H0_early and H0_late)
    epsilon : float
        Resolution threshold to mark
    output_path : Path, optional
        Where to save

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_plt()

    if hasattr(theta_samples, 'cpu'):
        theta_samples = theta_samples.cpu()

    H0_early = model.get_H0_early(theta_samples).numpy()
    H0_late = model.get_H0_late(theta_samples).numpy()
    delta_H0 = H0_late - H0_early

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    n, bins, patches = ax.hist(delta_H0, bins=100, density=True, alpha=0.7)

    # Color by resolution
    for patch, left, right in zip(patches, bins[:-1], bins[1:]):
        if abs((left + right) / 2) < epsilon:
            patch.set_facecolor('green')
            patch.set_alpha(0.8)
        else:
            patch.set_facecolor('red')
            patch.set_alpha(0.5)

    # Resolution zone
    ax.axvspan(-epsilon, epsilon, alpha=0.2, color='green', label=f'|ΔH₀| < {epsilon}')
    ax.axvline(0, color='black', linestyle='-', linewidth=2)

    # Statistics
    mean_delta = np.mean(delta_H0)
    std_delta = np.std(delta_H0)
    prob_res = np.mean(np.abs(delta_H0) < epsilon)

    ax.axvline(mean_delta, color='blue', linestyle='--', label=f'Mean: {mean_delta:.2f}')

    textstr = f'ΔH₀ = {mean_delta:.2f} ± {std_delta:.2f}\nP(resolution) = {prob_res:.1%}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', horizontalalignment='right', bbox=props)

    ax.set_xlabel("ΔH₀ = H₀(late) - H₀(early) (km/s/Mpc)", fontsize=14)
    ax.set_ylabel("Probability Density", fontsize=14)
    ax.set_title("Hubble Tension: ΔH₀ Distribution", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved delta H0 distribution to {output_path}")

    return fig
