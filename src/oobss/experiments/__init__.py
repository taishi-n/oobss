"""Reusable benchmarking and experiment orchestration utilities."""

from __future__ import annotations

from .engine import ExperimentEngine
from .methods import (
    MethodRunnerRegistry,
    STFTPlan,
    default_method_runner_registry,
    validate_builtin_method_params,
)
from .metrics import normalize_framewise_metrics
from .postprocess import (
    ParameterEstimationResult,
    PerReferenceSeparationResult,
    gaussian_source_model_weight,
    mixing_matrix_from_demixing_for_reference,
    separate_with_reference,
)
from .torchrir_dynamic import (
    TorchrirDynamicDataset,
    build_torchrir_dynamic_dataloader,
)

__all__ = [
    "ExperimentEngine",
    "MethodRunnerRegistry",
    "STFTPlan",
    "default_method_runner_registry",
    "validate_builtin_method_params",
    "normalize_framewise_metrics",
    "ParameterEstimationResult",
    "PerReferenceSeparationResult",
    "gaussian_source_model_weight",
    "mixing_matrix_from_demixing_for_reference",
    "separate_with_reference",
    "TorchrirDynamicDataset",
    "build_torchrir_dynamic_dataloader",
    "config_loader",
    "config_schema",
    "dataset",
    "engine",
    "methods",
    "pipeline",
    "recipe",
    "reporting",
    "run_suite",
]
