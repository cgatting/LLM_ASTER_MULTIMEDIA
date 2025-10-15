"""Task registry for ASTER."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, Iterable

from .tasks.base import Task


@dataclass
class RegistryEntry:
    """Stores factory callable metadata."""

    factory: Callable[..., Task]
    description: str


class TaskRegistry:
    """Simple in-memory registry mapping task names to factories."""

    def __init__(self) -> None:
        self._registry: Dict[str, RegistryEntry] = {}

    def register(self, name: str, description: str) -> Callable[[Callable[..., Task]], Callable[..., Task]]:
        normalized = name.strip().lower()

        def decorator(factory: Callable[..., Task]) -> Callable[..., Task]:
            self._registry[normalized] = RegistryEntry(factory=factory, description=description)
            return factory

        return decorator

    def create(self, name: str, **kwargs: Any) -> Task:
        normalized = name.strip().lower()
        if normalized not in self._registry:
            available = ", ".join(sorted(self._registry)) or "<empty>"
            raise KeyError(f"Task '{name}' is not registered. Available: {available}")
        entry = self._registry[normalized]
        return entry.factory(**kwargs)

    def list(self) -> Iterable[tuple[str, RegistryEntry]]:
        return sorted(self._registry.items(), key=lambda pair: pair[0])


registry = TaskRegistry()
