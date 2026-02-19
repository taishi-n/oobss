"""Configuration helpers for oobss."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Iterable

import yaml

try:
    OmegaConf = importlib.import_module("omegaconf").OmegaConf
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "oobss requires 'omegaconf'. Install dependencies with `uv sync`."
    ) from exc


def _as_str_key_dict(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping in {context}, got {type(value)!r}")
    return {str(key): item for key, item in value.items()}


def load_yaml(
    path: str | Path,
    *,
    overrides: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Load a YAML file into a dictionary, with optional dotlist overrides."""
    override_list = [item for item in (overrides or []) if item]
    cfg = OmegaConf.load(Path(path))
    if override_list:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(override_list))
    loaded = OmegaConf.to_container(cfg, resolve=True)
    return _as_str_key_dict(loaded, context=str(path))


def merge_overrides(
    data: dict[str, Any],
    overrides: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Merge dotlist overrides into an existing mapping."""
    override_list = [item for item in (overrides or []) if item]
    if not override_list:
        return dict(data)
    merged = OmegaConf.merge(
        OmegaConf.create(data), OmegaConf.from_dotlist(override_list)
    )
    container = OmegaConf.to_container(merged, resolve=True)
    return _as_str_key_dict(container, context="merged overrides")


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """Write dictionary data to a YAML file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
