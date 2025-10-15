"""Base task definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..config import BaseTaskConfig


class Task(Protocol):
    """Protocol for all tasks."""

    config: BaseTaskConfig

    def run(self) -> Any:
        """Execute the task."""


@dataclass
class SimpleTask:
    """Basic dataclass implementation of a Task."""

    config: BaseTaskConfig

    def prepare_output_dir(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
