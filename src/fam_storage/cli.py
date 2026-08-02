"""Command-line interface for fam-storage."""

from __future__ import annotations

import logging
import sys

import click

from fam_storage.config import load_config


@click.group()
def cli() -> None:
    """fam-storage: extract family photos from cloud and store them locally."""


@cli.command()
@click.option(
    "--config",
    "config_path",
    default="config/config.yaml",
    show_default=True,
    help="Path to the YAML configuration file.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Detect faces but skip the upload to Nextcloud.",
)
@click.option(
    "--rebuild-registry",
    is_flag=True,
    default=False,
    help="Force re-encoding of all reference images (ignore cache).",
)
def run(config_path: str, dry_run: bool, rebuild_registry: bool) -> None:
    """Run the full photo extraction pipeline."""
    from fam_storage.pipeline import Pipeline
    from fam_storage.registry import FaceRegistry

    try:
        config = load_config(config_path)
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        click.echo(
            f"Copy config/config.yaml.example to {config_path} and fill in your credentials.",
            err=True,
        )
        sys.exit(1)

    if dry_run:
        config.pipeline.dry_run = True

    _configure_logging(config.pipeline.log_level)

    if rebuild_registry:
        from pathlib import Path

        cache = Path(config.detection.encodings_cache)
        if cache.exists():
            cache.unlink()
            click.echo(f"Removed registry cache: {cache}")

    pipeline = Pipeline(config)
    try:
        pipeline.setup()
        pipeline.run()
    except RuntimeError as exc:
        click.echo(f"Pipeline error: {exc}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user.")
        sys.exit(0)


@cli.command()
@click.option(
    "--config",
    "config_path",
    default="config/config.yaml",
    show_default=True,
    help="Path to the YAML configuration file.",
)
def build_registry(config_path: str) -> None:
    """Build (or rebuild) the face encoding registry from reference images."""
    try:
        config = load_config(config_path)
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    _configure_logging(config.pipeline.log_level)

    from fam_storage.registry import FaceRegistry

    registry = FaceRegistry(
        references_dir=config.references_dir,
        cache_path=config.detection.encodings_cache,
    )
    registry.load(force_rebuild=True)

    if registry.is_empty():
        click.echo(
            f"No faces found. Add reference images to '{config.references_dir}/'.",
            err=True,
        )
        sys.exit(1)

    click.echo(
        f"Registry built with {len(registry.people)} person(s): "
        + ", ".join(registry.names())
    )


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
