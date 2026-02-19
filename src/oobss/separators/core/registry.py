"""Registry utilities for separator and strategy factories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .base import BaseSeparator


class RegistryError(RuntimeError):
    """Raised for invalid registry operations."""


@dataclass
class SeparatorRegistry:
    """Simple name-to-factory mapping for separators."""

    _factories: dict[str, Callable[[dict[str, object]], BaseSeparator]] = field(
        default_factory=dict
    )

    def register(
        self,
        name: str,
        factory: Callable[[dict[str, object]], BaseSeparator],
        *,
        overwrite: bool = False,
    ) -> None:
        if not overwrite and name in self._factories:
            raise RegistryError(f"Separator '{name}' is already registered.")
        self._factories[name] = factory

    def create(
        self, name: str, params: dict[str, object] | None = None
    ) -> BaseSeparator:
        if name not in self._factories:
            available = ", ".join(sorted(self._factories)) or "<none>"
            raise RegistryError(
                f"Unknown separator '{name}'. Available separators: {available}"
            )
        return self._factories[name]({} if params is None else params)

    def available(self) -> list[str]:
        return sorted(self._factories)
