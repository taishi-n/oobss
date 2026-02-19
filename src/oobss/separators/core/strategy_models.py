"""Typed request/response models for strategy interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class CovarianceRequest:
    """Input container for covariance update strategies."""

    observed: np.ndarray
    source_model: np.ndarray
    prev_cov: np.ndarray | None = None
    alpha: float | None = None


@dataclass(slots=True)
class InitializationRequest:
    """Input container for initialization strategies."""

    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SourceModelRequest:
    """Input container for source-model strategy updates."""

    estimated: np.ndarray | None = None
    n_freq: int | None = None
    basis: np.ndarray | None = None
    activ: np.ndarray | None = None
    y_power: np.ndarray | None = None


@dataclass(slots=True)
class SourceModelResult:
    """Output container for source-model strategy updates."""

    source_model: np.ndarray | None = None
    basis: np.ndarray | None = None
    activ: np.ndarray | None = None


@dataclass(slots=True)
class NormalizationRequest:
    """Input container for normalization strategies."""

    estimate: np.ndarray
    observations: np.ndarray | None = None
    demix_filter: np.ndarray | None = None


@dataclass(slots=True)
class ReconstructionRequest:
    """Input container for reconstruction strategies."""

    mixture: np.ndarray
    demix_filter: np.ndarray | None = None
    source_power: np.ndarray | None = None


@dataclass(slots=True)
class ReconstructionResult:
    """Output container for reconstruction strategies."""

    estimate: np.ndarray
    mask: np.ndarray | None = None


@dataclass(slots=True)
class PermutationRequest:
    """Input container for permutation strategies."""

    score: np.ndarray
    reference: np.ndarray | None = None
    estimate: np.ndarray | None = None
    filter_length: int = 512
    default_perm: np.ndarray | None = None


@dataclass(slots=True)
class LossRequest:
    """Input container for loss strategies."""

    observations: np.ndarray | None = None
    estimated: np.ndarray | None = None
    demix_filter: np.ndarray | None = None
    source_model_name: str | None = None
    basis: np.ndarray | None = None
    activ: np.ndarray | None = None
    axis: int | None = None
    eps: float = float(np.finfo(np.float64).eps)


@dataclass(slots=True)
class OnlineFrameRequest:
    """Per-frame runtime options for streaming separators."""

    n_sources: int | None = None
    component_to_source: np.ndarray | None = None
    return_mask: bool = False
    reference_mic: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NMFUpdateRequest:
    """Input container for one online NMF update step."""

    v: np.ndarray
    basis: np.ndarray
    stat_a: np.ndarray
    stat_b: np.ndarray
    inner_iter: int
    beta: int
    alpha: float
    batch_counter: int
    t: int
    eps: float
    h_prev: np.ndarray | None = None


@dataclass(slots=True)
class NMFUpdateResult:
    """Output container for one online NMF update step."""

    h: np.ndarray
    basis: np.ndarray
    stat_a: np.ndarray
    stat_b: np.ndarray
    batch_counter: int
    t: int


@dataclass(slots=True)
class ComponentAssignmentRequest:
    """Input container for component-to-source assignment strategies."""

    n_components: int
    n_sources: int
    component_to_source: np.ndarray | None = None
