"""Helpers for loading benchmark configuration files.

Usage
-----
This module is used by benchmark scripts (for example
``examples/benchmark_dataset.py``) to load:

- common config: ``examples/benchmark/config/common.yaml``
- method configs: ``examples/benchmark/config/methods/*.yaml``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from oobss.configs import load_yaml, merge_overrides
from .config_schema import CommonExperimentConfig, parse_common_config
from .methods import validate_builtin_method_params


@dataclass(frozen=True)
class MethodConfig:
    """Container for method-specific configuration."""

    id: str
    label: str
    type: str
    enabled: bool
    params: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: Path) -> "MethodConfig":
        data = _load_yaml(path)
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "MethodConfig":
        return cls(
            id=str(data["id"]),
            label=str(data.get("label", data["id"])),
            type=str(data.get("type", data["id"])),
            enabled=bool(data.get("enabled", True)),
            params=dict(data.get("params", {})),
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    return load_yaml(path)


def load_common_config(
    path: Path,
    *,
    overrides: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Load the repository-level experiment configuration."""

    cfg = load_yaml(path, overrides=overrides)
    cfg.setdefault("dataset", {})
    cfg.setdefault("stft", {})
    cfg.setdefault("evaluation", {})
    cfg.setdefault("metrics", {})
    cfg.setdefault("runtime", {})
    return cfg


def load_common_config_schema(
    path: Path,
    *,
    overrides: Iterable[str] | None = None,
) -> CommonExperimentConfig:
    """Load and decode common benchmark config into a typed schema."""
    raw = load_common_config(path, overrides=overrides)
    return parse_common_config(raw)


def load_method_configs(
    directory: Path,
    *,
    selected: Iterable[str] | None = None,
    overrides: Iterable[str] | None = None,
) -> list[MethodConfig]:
    """Discover and load method YAML files."""

    candidates = sorted(directory.glob("*.yaml"))
    methods_map: dict[str, dict[str, Any]] = {}
    allowlist = {name.strip() for name in selected or [] if name}

    for path in candidates:
        method_data = _load_yaml(path)
        method = MethodConfig.from_mapping(method_data)
        methods_map[method.id] = method_data

    if overrides:
        _validate_method_override_keys(methods_map, overrides)
        methods_map = merge_overrides(methods_map, overrides)

    methods: list[MethodConfig] = []
    for method_id in sorted(methods_map):
        method = MethodConfig.from_mapping(methods_map[method_id])
        if allowlist and method.id not in allowlist:
            continue
        if not method.enabled:
            continue
        validate_builtin_method_params(method.type, method.params)
        methods.append(method)

    if allowlist:
        missing = allowlist - {m.id for m in methods}
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Requested methods not found or disabled: {missing_list}")

    return methods


def _validate_method_override_keys(
    methods_map: dict[str, dict[str, Any]],
    overrides: Iterable[str],
) -> None:
    known = set(methods_map.keys())
    for item in overrides:
        if not item or "=" not in item:
            continue
        key = item.split("=", 1)[0]
        method_id = key.split(".", 1)[0]
        if method_id and method_id not in known:
            known_list = ", ".join(sorted(known)) or "<none>"
            raise ValueError(
                f"Unknown method override target '{method_id}'. "
                f"Available methods: {known_list}"
            )
