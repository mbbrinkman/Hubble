"""
main.py
-------
Command-line interface for the Hubble cosmological inference pipeline.

Basic Usage:
    hubble prep        # Prepare observational data
    hubble simulate    # Generate training data
    hubble train       # Train the normalizing flow
    hubble observe     # Compute observed summary vector
    hubble sample      # Draw posterior samples
    hubble analyze     # Analyze posterior samples
    hubble run         # Run the full pipeline

Model Comparison:
    hubble models              # List available cosmological models
    hubble train-model         # Train flow for a specific model
    hubble compare             # Compare models via Bayes factors
    hubble tension             # Quantify Hubble tension

Visualization:
    hubble plot-corner         # Generate corner plot
    hubble plot-tension        # Plot tension probability curve
"""

import click
from pathlib import Path

from config import logger, set_seed, RESULTS


@click.group()
@click.option("--seed", default=42, help="Random seed for reproducibility")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def cli(seed: int, verbose: bool):
    """Hubble: Cosmological parameter estimation with normalizing flows."""
    set_seed(seed)
    if verbose:
        import logging
        logger.setLevel(logging.DEBUG)


# ===========================================================================
# Basic Pipeline Commands
# ===========================================================================

@cli.command()
def prep():
    """Prepare observational data (Pantheon+, BAO)."""
    import prep as prep_module
    prep_module.run()


@cli.command()
@click.option("--n-samples", "-n", default=None, type=int,
              help="Number of training samples (default: 300,000)")
@click.option("--model", "-m", default="concordance",
              help="Cosmological model to simulate (default: concordance)")
def simulate(n_samples: int, model: str):
    """Generate training data using Sobol sampling."""
    import sim
    # For now, use default simulation (model-specific simulation TBD)
    sim.run(n_samples=n_samples)


@cli.command()
@click.option("--epochs", "-e", default=None, type=int,
              help="Number of training epochs (default: 10)")
@click.option("--batch-size", "-b", default=None, type=int,
              help="Batch size (default: 1024)")
@click.option("--lr", default=None, type=float,
              help="Learning rate (default: 1e-3)")
def train(epochs: int, batch_size: int, lr: float):
    """Train the normalizing flow model."""
    import flow_train
    from config import config

    if epochs:
        config.training.n_epochs = epochs
    if batch_size:
        config.training.batch_size = batch_size
    if lr:
        config.training.learning_rate = lr

    flow_train.run()


@cli.command()
def observe():
    """Compute the observed summary vector."""
    import x_obs
    x_obs.run()


@cli.command()
@click.option("--n-samples", "-n", default=None, type=int,
              help="Number of posterior samples (default: 100,000)")
def sample(n_samples: int):
    """Draw samples from the posterior distribution."""
    import posterior
    posterior.run(n_samples=n_samples)


@cli.command()
@click.option("--epsilon", "-e", default=1.0, type=float,
              help="H0 tolerance in km/s/Mpc (default: 1.0)")
def analyze(epsilon: float):
    """Analyze posterior samples and compute statistics."""
    import prob
    prob.run(epsilon=epsilon)


@cli.command()
@click.option("--n-train", default=None, type=int,
              help="Number of training samples")
@click.option("--epochs", "-e", default=None, type=int,
              help="Number of training epochs")
@click.option("--n-posterior", default=None, type=int,
              help="Number of posterior samples")
@click.pass_context
def run(ctx, n_train: int, epochs: int, n_posterior: int):
    """Run the full Hubble pipeline end-to-end."""
    logger.info("=" * 60)
    logger.info("Running full Hubble pipeline")
    logger.info("=" * 60)

    ctx.invoke(prep)
    ctx.invoke(simulate, n_samples=n_train)
    ctx.invoke(train, epochs=epochs)
    ctx.invoke(observe)
    ctx.invoke(sample, n_samples=n_posterior)
    ctx.invoke(analyze)

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)


# ===========================================================================
# Model Comparison Commands
# ===========================================================================

@cli.command("models")
def list_models():
    """List available cosmological models."""
    from cosmology import MODELS

    click.echo("\nAvailable Cosmological Models")
    click.echo("=" * 50)

    for name, model_cls in MODELS.items():
        model = model_cls()
        click.echo(f"\n{name}:")
        click.echo(f"  Name: {model.name}")
        click.echo(f"  Parameters: {model.n_params}")
        click.echo(f"  Symbols: {', '.join(model.param_symbols)}")
        if model.description:
            # Wrap description
            desc = model.description
            if len(desc) > 60:
                desc = desc[:57] + "..."
            click.echo(f"  Description: {desc}")

    # Check for trained flows
    from models import list_available_flows
    available = list_available_flows()
    if available:
        click.echo(f"\nTrained flows available: {', '.join(available)}")
    else:
        click.echo("\nNo trained flows found. Run 'hubble train-model' first.")


@cli.command("train-model")
@click.argument("model_name")
@click.option("--n-samples", "-n", default=100000, type=int,
              help="Number of training samples (default: 100,000)")
@click.option("--epochs", "-e", default=10, type=int,
              help="Number of training epochs (default: 10)")
@click.option("--batch-size", "-b", default=1024, type=int,
              help="Batch size (default: 1024)")
@click.option("--lr", default=1e-3, type=float,
              help="Learning rate (default: 1e-3)")
def train_model(model_name: str, n_samples: int, epochs: int, batch_size: int, lr: float):
    """
    Train a flow for a specific cosmological model.

    MODEL_NAME: One of 'concordance', 'discordance', 'wcdm', 'early_de'
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import h5py

    from cosmology import get_model
    from models import build_flow_for_model, save_flow
    from config import config, paths, DEVICE
    import forward

    logger.info("=" * 60)
    logger.info(f"Training flow for model: {model_name}")
    logger.info("=" * 60)

    # Get model
    model = get_model(model_name)
    logger.info(f"Model: {model.name} ({model.n_params} parameters)")

    # Load observations
    logger.info("Loading observations...")
    with h5py.File(paths.obs_h5, "r") as f:
        obs_arrays = {k: torch.tensor(f[k][...], device=DEVICE) for k in f}
    callbacks = torch.load(paths.residual_fns)
    obs = {**obs_arrays, **callbacks}

    # Generate training data for this model
    logger.info(f"Generating {n_samples:,} training samples...")
    theta_all = model.sample_prior(n_samples)

    x_list = []
    for i, theta in enumerate(theta_all):
        if i % 5000 == 0:
            logger.info(f"  Progress: {i:,}/{n_samples:,}")
        x_list.append(model.summary_vector(theta, obs))
    x_all = torch.stack(x_list)

    logger.info(f"  theta shape: {theta_all.shape}, x shape: {x_all.shape}")

    # Build and train flow
    flow = build_flow_for_model(model_name)
    n_params = sum(p.numel() for p in flow.parameters())
    logger.info(f"Flow has {n_params:,} parameters")

    loader = DataLoader(TensorDataset(x_all, theta_all), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

    logger.info(f"Training for {epochs} epochs...")
    best_loss = float("inf")

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for xb, tb in loader:
            loss = -flow.log_prob(inputs=tb, context=xb).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        marker = " *" if avg_loss < best_loss else ""
        if avg_loss < best_loss:
            best_loss = avg_loss
        logger.info(f"  Epoch {epoch+1:2d}/{epochs}  loss: {avg_loss:.4f}{marker}")

    # Save
    metadata = {
        "n_samples": n_samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "final_loss": avg_loss,
    }
    save_flow(flow, model_name=model_name, metadata=metadata)
    logger.info(f"Training complete for {model_name}!")


@cli.command("compare")
@click.option("--models", "-m", multiple=True, default=["concordance", "discordance"],
              help="Models to compare (can specify multiple)")
@click.option("--n-samples", "-n", default=50000, type=int,
              help="Samples for evidence estimation (default: 50,000)")
@click.option("--epsilon", "-e", default=1.0, type=float,
              help="H0 tolerance for tension analysis (default: 1.0)")
@click.option("--output", "-o", default=None, type=str,
              help="Output file for results (JSON)")
def compare_models(models: tuple, n_samples: int, epsilon: float, output: str):
    """
    Compare cosmological models using Bayes factors.

    Computes evidence for each model and reports:
    - Bayes factors between model pairs
    - Posterior model probabilities
    - Tension resolution probability under each model
    """
    import torch
    from cosmology import get_model
    from models import load_flow, check_flow_exists
    from config import paths, DEVICE
    from inference.comparison import ModelComparison

    logger.info("=" * 60)
    logger.info("Bayesian Model Comparison")
    logger.info("=" * 60)

    # Check all flows exist
    for model_name in models:
        if not check_flow_exists(model_name):
            raise click.ClickException(
                f"No trained flow for '{model_name}'. "
                f"Run 'hubble train-model {model_name}' first."
            )

    # Load observed summary
    if not paths.x_obs.exists():
        raise click.ClickException(
            "Observed summary not found. Run 'hubble observe' first."
        )
    x_obs = torch.load(paths.x_obs, map_location=DEVICE)

    # Set up comparison
    comparison = ModelComparison()
    for model_name in models:
        model = get_model(model_name)
        flow = load_flow(model_name)
        comparison.add_model(model_name, flow, model)

    # Run comparison
    results = comparison.compare(x_obs, n_samples=n_samples, epsilon=epsilon)

    # Print summary
    click.echo("\n" + str(results))
    click.echo("\n" + comparison.get_tension_summary())

    # Save if requested
    if output:
        output_path = Path(output)
        results.save(output_path)


@cli.command("tension")
@click.option("--model", "-m", default="discordance",
              help="Model to use for tension analysis (default: discordance)")
@click.option("--n-samples", "-n", default=100000, type=int,
              help="Number of posterior samples (default: 100,000)")
@click.option("--epsilon", "-e", default=1.0, type=float,
              help="H0 tolerance in km/s/Mpc (default: 1.0)")
def tension_analysis(model: str, n_samples: int, epsilon: float):
    """
    Quantify the Hubble tension under a specific model.

    Reports:
    - P(|ΔH₀| < ε): probability tension is resolvable
    - ΔH₀ distribution statistics
    - Resolution probability curve
    """
    import torch
    from cosmology import get_model
    from models import load_flow, check_flow_exists
    from config import paths, DEVICE
    from inference.tension import TensionAnalyzer

    logger.info("=" * 60)
    logger.info(f"Hubble Tension Analysis ({model})")
    logger.info("=" * 60)

    if not check_flow_exists(model):
        raise click.ClickException(
            f"No trained flow for '{model}'. "
            f"Run 'hubble train-model {model}' first."
        )

    # Load
    cosmo_model = get_model(model)
    flow = load_flow(model)
    x_obs = torch.load(paths.x_obs, map_location=DEVICE)

    # Sample posterior
    logger.info(f"Drawing {n_samples:,} posterior samples...")
    flow.eval()
    with torch.no_grad():
        x_exp = x_obs.unsqueeze(0).expand(n_samples, -1) if x_obs.dim() == 1 else x_obs.expand(n_samples, -1)
        theta_samples = flow.sample(n_samples, context=x_exp)
        if isinstance(theta_samples, tuple):
            theta_samples = theta_samples[0]

    # Analyze
    analyzer = TensionAnalyzer(cosmo_model)
    result = analyzer.analyze(theta_samples, epsilon=epsilon)

    # Probability curve
    curve = analyzer.probability_curve(theta_samples)
    click.echo("\nResolution Probability Curve:")
    click.echo("-" * 40)
    for eps, prob, err in zip(curve["epsilon"], curve["probability"], curve["std_error"]):
        bar = "█" * int(prob * 30)
        click.echo(f"  ε={eps:4.1f}: {prob:5.1%} ± {err:4.1%} {bar}")

    # Critical epsilon
    eps_50 = analyzer.find_critical_epsilon(theta_samples, 0.5)
    click.echo(f"\nCritical ε (50% resolution): {eps_50:.2f} km/s/Mpc")

    # Interpretation
    click.echo("\n" + "=" * 40)
    if result.prob_resolution < 0.05:
        click.echo("CONCLUSION: Strong evidence that Hubble tension is REAL")
        click.echo(f"  Only {result.prob_resolution:.1%} chance of resolution within {epsilon} km/s/Mpc")
    elif result.prob_resolution < 0.32:
        click.echo("CONCLUSION: Moderate evidence for Hubble tension")
    else:
        click.echo("CONCLUSION: Tension may be resolvable")


# ===========================================================================
# Visualization Commands
# ===========================================================================

@cli.command("plot-corner")
@click.option("--model", "-m", default="concordance",
              help="Model to plot (default: concordance)")
@click.option("--output", "-o", default=None, type=str,
              help="Output file path (default: results/corner_{model}.png)")
def plot_corner_cmd(model: str, output: str):
    """Generate a corner plot of posterior samples."""
    import torch
    from cosmology import get_model
    from models import load_flow, check_flow_exists
    from config import paths, DEVICE, RESULTS
    from analysis.visualize import plot_corner

    if not check_flow_exists(model):
        raise click.ClickException(f"No trained flow for '{model}'.")

    cosmo_model = get_model(model)
    flow = load_flow(model)
    x_obs = torch.load(paths.x_obs, map_location=DEVICE)

    logger.info(f"Generating corner plot for {model}...")

    flow.eval()
    with torch.no_grad():
        x_exp = x_obs.unsqueeze(0).expand(50000, -1) if x_obs.dim() == 1 else x_obs.expand(50000, -1)
        theta_samples = flow.sample(50000, context=x_exp)
        if isinstance(theta_samples, tuple):
            theta_samples = theta_samples[0]

    output_path = Path(output) if output else RESULTS / f"corner_{model}.png"
    plot_corner(theta_samples, cosmo_model, output_path=output_path,
                title=f"Posterior: {cosmo_model.name}")

    click.echo(f"Saved corner plot to {output_path}")


@cli.command("plot-tension")
@click.option("--model", "-m", default="discordance",
              help="Model to use (default: discordance)")
@click.option("--output", "-o", default=None, type=str,
              help="Output file path")
def plot_tension_cmd(model: str, output: str):
    """Plot the tension resolution probability curve."""
    import torch
    from cosmology import get_model
    from models import load_flow, check_flow_exists
    from config import paths, DEVICE, RESULTS
    from inference.tension import TensionAnalyzer
    from analysis.visualize import plot_tension_curve

    if not check_flow_exists(model):
        raise click.ClickException(f"No trained flow for '{model}'.")

    cosmo_model = get_model(model)
    flow = load_flow(model)
    x_obs = torch.load(paths.x_obs, map_location=DEVICE)

    logger.info(f"Generating tension curve for {model}...")

    flow.eval()
    with torch.no_grad():
        x_exp = x_obs.unsqueeze(0).expand(100000, -1) if x_obs.dim() == 1 else x_obs.expand(100000, -1)
        theta_samples = flow.sample(100000, context=x_exp)
        if isinstance(theta_samples, tuple):
            theta_samples = theta_samples[0]

    analyzer = TensionAnalyzer(cosmo_model)
    curve = analyzer.probability_curve(theta_samples,
                                       epsilon_range=[0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])

    output_path = Path(output) if output else RESULTS / f"tension_curve_{model}.png"
    plot_tension_curve(curve, output_path=output_path)

    click.echo(f"Saved tension curve to {output_path}")


# ===========================================================================
# Info Command
# ===========================================================================

@cli.command()
def info():
    """Display configuration and system information."""
    import torch
    from config import config, DEVICE, paths, ROOT

    click.echo("\nHubble Configuration")
    click.echo("=" * 50)

    click.echo(f"\nProject root: {ROOT}")
    click.echo(f"Device: {DEVICE}")
    click.echo(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        click.echo(f"CUDA device: {torch.cuda.get_device_name(0)}")

    click.echo(f"\nFlow architecture:")
    click.echo(f"  Hidden dim: {config.flow.hidden_dim}")
    click.echo(f"  Layers: {config.flow.n_layers}")

    click.echo(f"\nTraining config:")
    click.echo(f"  Default samples: {config.training.n_train_samples:,}")
    click.echo(f"  Batch size: {config.training.batch_size}")
    click.echo(f"  Learning rate: {config.training.learning_rate}")
    click.echo(f"  Epochs: {config.training.n_epochs}")

    click.echo(f"\nInference config:")
    click.echo(f"  Posterior samples: {config.inference.n_posterior_samples:,}")

    # Available models
    from cosmology import MODELS as COSMO_MODELS
    click.echo(f"\nCosmological models: {', '.join(COSMO_MODELS.keys())}")

    # Trained flows
    from models import list_available_flows
    available = list_available_flows()
    click.echo(f"Trained flows: {', '.join(available) if available else 'None'}")

    click.echo(f"\nData paths:")
    click.echo(f"  Raw: {paths.pantheon_data.parent}")
    click.echo(f"  Processed: {paths.obs_h5.parent}")
    click.echo(f"  Models: {paths.flow_weights.parent}")
    click.echo(f"  Results: {paths.posterior.parent}")


if __name__ == "__main__":
    cli()
