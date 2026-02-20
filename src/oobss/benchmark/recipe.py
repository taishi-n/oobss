"""Recipe models for reproducible benchmark execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .config_schema import CommonExperimentConfig, common_config_to_dict


@dataclass(frozen=True)
class ExperimentRecipe:
    """Resolved benchmark recipe used by runners and task builders."""

    dataset: dict[str, Any]
    stft: dict[str, Any]
    evaluation: dict[str, Any]
    metrics: dict[str, Any]
    runtime: dict[str, Any]
    track_subset: tuple[str, ...]
    sample_limit: int | None
    duration_limit: float | None


def recipe_from_common_config(
    common_cfg: Mapping[str, Any] | CommonExperimentConfig,
    *,
    track_subset: Iterable[str] | None = None,
    sample_limit: int | None = None,
) -> ExperimentRecipe:
    """Build an :class:`ExperimentRecipe` from common benchmark config."""
    if isinstance(common_cfg, CommonExperimentConfig):
        cfg_map = common_config_to_dict(common_cfg)
    else:
        cfg_map = dict(common_cfg)

    dataset_cfg = dict(cfg_map.get("dataset", {}))
    dataset_cfg.setdefault("type", "torchrir_dynamic")

    duration_limit_obj = dataset_cfg.get("duration_sec")
    duration_limit: float | None
    if duration_limit_obj is None:
        duration_limit = None
    else:
        duration_limit = float(duration_limit_obj)
        if duration_limit <= 0:
            duration_limit = None

    return ExperimentRecipe(
        dataset=dataset_cfg,
        stft=dict(cfg_map.get("stft", {})),
        evaluation=dict(cfg_map.get("evaluation", {})),
        metrics=dict(cfg_map.get("metrics", {})),
        runtime=dict(cfg_map.get("runtime", {})),
        track_subset=tuple(item.strip() for item in (track_subset or []) if item),
        sample_limit=sample_limit,
        duration_limit=duration_limit,
    )
