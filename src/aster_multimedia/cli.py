"""Command line interface for ASTER."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import click
import yaml

from .config import CONFIG_MAP, BaseTaskConfig, load_config, merge_cli_overrides
from .registry import registry

# Import task modules so that registry decorators run on startup
from .tasks import image_to_image, image_to_video, text_to_text, video_to_image  # noqa: F401

_LOG_FORMAT = "[%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)


@click.group()
@click.option("--verbose", is_flag=True, help="Enable debug logging")
def app(verbose: bool) -> None:
    """ASTER multimodal task runner."""

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@app.command("list")
def list_tasks() -> None:
    """List registered tasks."""

    for name, entry in registry.list():
        click.echo(f"{name:15} - {entry.description}")


def _load_task_config(config_path: Path, identifier: str, task_type: str | None) -> BaseTaskConfig:
    if config_path.exists():
        configs = load_config(config_path)
        if identifier in configs:
            return configs[identifier]

    if task_type is None:
        raise click.UsageError(
            f"No configuration entry named '{identifier}'. Provide --task-type to create a default config."
        )

    if task_type not in CONFIG_MAP:
        raise click.BadParameter(
            f"Unsupported task type '{task_type}'. Available: {', '.join(sorted(CONFIG_MAP))}"
        )
    config_cls = CONFIG_MAP[task_type]
    return config_cls(task=task_type, name=identifier)


def _parse_overrides(overrides: tuple[str, ...]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise click.BadParameter("Overrides must use the format key=value")
        key, raw_value = item.split("=", 1)
        parsed[key] = yaml.safe_load(raw_value)
    return parsed


@app.command("run")
@click.argument("identifier")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=Path("configs/sample.yaml"),
    help="Path to a YAML configuration file.",
)
@click.option(
    "--task-type",
    type=click.Choice(sorted(CONFIG_MAP)),
    default=None,
    help="Task type to instantiate if the identifier is not found in the config file.",
)
@click.option(
    "--override",
    multiple=True,
    help="Override configuration values, e.g. --override prompt='Hello'",
)
def run(identifier: str, config_path: Path, task_type: str | None, override: tuple[str, ...]) -> None:
    """Execute a task defined in the configuration file."""

    overrides = _parse_overrides(override)
    config = _load_task_config(config_path, identifier, task_type)
    if overrides:
        config = merge_cli_overrides(config, overrides)

    click.echo(f"Running {config.task} ({config.name})...")
    task = registry.create(config.task, config=config)
    result = task.run()
    if isinstance(result, dict):
        for key, value in result.items():
            click.echo(f"{key}: {value}")
    else:
        click.echo(result)
