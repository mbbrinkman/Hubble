"""
main.py
-------
Command-line interface for the Hubble cosmological inference pipeline.

Usage:
    hubble prep        # Prepare observational data
    hubble simulate    # Generate training data
    hubble train       # Train the normalizing flow
    hubble observe     # Compute observed summary vector
    hubble sample      # Draw posterior samples
    hubble analyze     # Analyze posterior samples
    hubble run         # Run the full pipeline
"""

import click

from config import logger, set_seed


@click.group()
@click.option("--seed", default=42, help="Random seed for reproducibility")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def cli(seed: int, verbose: bool):
    """Hubble: Cosmological parameter estimation with normalizing flows."""
    set_seed(seed)
    if verbose:
        import logging
        logger.setLevel(logging.DEBUG)


@cli.command()
def prep():
    """Prepare observational data (Pantheon+, BAO)."""
    import prep as prep_module
    prep_module.run()


@cli.command()
@click.option("--n-samples", "-n", default=None, type=int,
              help="Number of training samples (default: 300,000)")
def simulate(n_samples: int):
    """Generate training data using Sobol sampling."""
    import sim
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

    # Override config if options provided
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

    # Step 1: Prep
    logger.info("\n[1/6] Preparing data...")
    ctx.invoke(prep)

    # Step 2: Simulate
    logger.info("\n[2/6] Generating training data...")
    ctx.invoke(simulate, n_samples=n_train)

    # Step 3: Train
    logger.info("\n[3/6] Training flow model...")
    ctx.invoke(train, epochs=epochs)

    # Step 4: Observe
    logger.info("\n[4/6] Computing observed summary...")
    ctx.invoke(observe)

    # Step 5: Sample
    logger.info("\n[5/6] Drawing posterior samples...")
    ctx.invoke(sample, n_samples=n_posterior)

    # Step 6: Analyze
    logger.info("\n[6/6] Analyzing results...")
    ctx.invoke(analyze)

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)


@cli.command()
def info():
    """Display configuration and system information."""
    import torch
    from config import config, DEVICE, paths, ROOT

    click.echo("Hubble Configuration")
    click.echo("=" * 40)

    click.echo(f"\nProject root: {ROOT}")
    click.echo(f"Device: {DEVICE}")
    click.echo(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        click.echo(f"CUDA device: {torch.cuda.get_device_name(0)}")

    click.echo(f"\nFlow architecture:")
    click.echo(f"  Dimensions: {config.flow.dim}")
    click.echo(f"  Hidden dim: {config.flow.hidden_dim}")
    click.echo(f"  Layers: {config.flow.n_layers}")

    click.echo(f"\nTraining config:")
    click.echo(f"  Train samples: {config.training.n_train_samples:,}")
    click.echo(f"  Batch size: {config.training.batch_size}")
    click.echo(f"  Learning rate: {config.training.learning_rate}")
    click.echo(f"  Epochs: {config.training.n_epochs}")

    click.echo(f"\nInference config:")
    click.echo(f"  Posterior samples: {config.inference.n_posterior_samples:,}")

    click.echo(f"\nParameter bounds:")
    for i, name in enumerate(config.physics.param_names):
        lo = config.physics.theta_min[i]
        hi = config.physics.theta_max[i]
        click.echo(f"  {name}: [{lo}, {hi}]")

    click.echo(f"\nData paths:")
    click.echo(f"  Raw data: {paths.pantheon_data.parent}")
    click.echo(f"  Processed: {paths.obs_h5.parent}")
    click.echo(f"  Models: {paths.flow_weights.parent}")
    click.echo(f"  Results: {paths.posterior.parent}")


if __name__ == "__main__":
    cli()
