"""Protocol interfaces for unified separator execution."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .io_models import (
    BatchRequest,
    SeparationOutput,
    SeparatorState,
    StreamRequest,
    StreamingSeparatorState,
)
from .strategy_models import OnlineFrameRequest


class SeparatorProtocol(Protocol):
    """Common protocol for all separator implementations."""

    @property
    def n_sources(self) -> int:
        """Return number of separated sources."""

    def reset(self) -> None:
        """Reset internal runtime state."""


class BatchSeparatorProtocol(SeparatorProtocol, Protocol):
    """Protocol for TF-domain batch separation."""

    def fit_transform_tf(
        self,
        mixture_tf: np.ndarray,
        *,
        n_iter: int = 0,
        request: BatchRequest | None = None,
    ) -> SeparationOutput:
        """Run iterative updates for one TF-domain input tensor."""

    def forward(
        self,
        mixture: np.ndarray,
        *,
        n_iter: int = 0,
        request: BatchRequest | None = None,
        is_time_input: bool | None = None,
    ) -> SeparationOutput:
        """Torch-like entry point for batch separators."""


class StreamingSeparatorProtocol(SeparatorProtocol, Protocol):
    """Protocol for frame-wise streaming separation."""

    def process_stream_tf(
        self,
        stream_tf: np.ndarray,
        *,
        request: StreamRequest | None = None,
    ) -> SeparationOutput:
        """Process an input stream and return stacked frame outputs."""

    def forward_streaming(
        self,
        frame: np.ndarray,
        *,
        state: SeparatorState | StreamingSeparatorState | None = None,
        request: OnlineFrameRequest | None = None,
    ) -> tuple[np.ndarray, SeparatorState | StreamingSeparatorState]:
        """Process one frame and return output with updated state."""
