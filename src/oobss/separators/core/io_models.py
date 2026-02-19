"""Typed data models shared by separator implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class BatchRequest:
    """Execution options for batch separators.

    Parameters
    ----------
    reference_mic:
        Reference microphone index for scale restoration/evaluation.
    sample_rate:
        Sampling rate in Hz when time-domain input is supplied.
    metadata:
        Additional method-specific options.
    """

    reference_mic: int = 0
    sample_rate: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StreamRequest:
    """Execution options for streaming separators.

    Parameters
    ----------
    frame_axis:
        Axis in input tensor that corresponds to frame/time index.
    reference_mic:
        Optional reference microphone index.
    n_sources:
        Optional source count override for methods that require it.
    component_to_source:
        Optional NMF component-to-source assignment.
    return_mask:
        If ``True``, separator may return masks instead of separated spectra.
    metadata:
        Additional method-specific options.
    """

    frame_axis: int = -1
    reference_mic: int | None = None
    n_sources: int | None = None
    component_to_source: np.ndarray | None = None
    return_mask: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SeparatorState:
    """Mutable runtime state used across iterative or online updates."""

    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StreamingSeparatorState:
    """Typed runtime state for streaming BSS-style separators.

    Parameters
    ----------
    source_model:
        Optional source model state snapshot at the current frame.
    demix_filter:
        Optional demixing filter snapshot.
    mix_filter:
        Optional mixing filter snapshot (typically inverse of ``demix_filter``).
    frame_index:
        Number of processed frames associated with this state.
    metadata:
        Additional algorithm-specific state values.
    """

    source_model: np.ndarray | None = None
    demix_filter: np.ndarray | None = None
    mix_filter: np.ndarray | None = None
    frame_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SeparationOutput:
    """Unified separation result container.

    Parameters
    ----------
    estimate_time:
        Optional separated signals in time domain ``(n_src, n_samples)``.
    estimate_tf:
        Optional separated signals in TF domain.
    mask:
        Optional source mask values.
    demix_filter:
        Optional demixing filter/matrix.
    permutation:
        Optional source permutation indices.
    state:
        Optional internal runtime state snapshot.
    metadata:
        Free-form method metadata for logging/benchmarking.
    """

    estimate_time: np.ndarray | None = None
    estimate_tf: np.ndarray | None = None
    mask: np.ndarray | None = None
    demix_filter: np.ndarray | None = None
    permutation: np.ndarray | None = None
    state: SeparatorState | StreamingSeparatorState | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            self.estimate_time is None
            and self.estimate_tf is None
            and self.mask is None
        ):
            raise ValueError(
                "SeparationOutput requires estimate_time, estimate_tf, or mask."
            )
