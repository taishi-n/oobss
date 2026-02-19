"""Base classes for unified separator execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .io_models import (
    BatchRequest,
    SeparationOutput,
    SeparatorState,
    StreamRequest,
    StreamingSeparatorState,
)
from .strategy_models import OnlineFrameRequest


class BaseSeparator(ABC):
    """Common top-level contract for all separators."""

    @property
    @abstractmethod
    def n_sources(self) -> int:
        """Return number of separated sources."""

    def reset(self) -> None:
        """Reset internal state (override in subclasses when needed)."""

    def __call__(self, *args: Any, **kwargs: Any) -> SeparationOutput:
        """Alias for :meth:`forward` to provide a torch-like call style."""
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> SeparationOutput:
        """Execute separation and return a typed output object."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement forward()."
        )


class BaseIterativeSeparator(BaseSeparator):
    """Base class for iterative batch-style separators."""

    @abstractmethod
    def step(self) -> None:
        """Run one update step."""

    @abstractmethod
    def get_estimate(self) -> np.ndarray:
        """Return current estimate in TF domain."""

    def fit_transform_tf(
        self,
        mixture_tf: np.ndarray,
        *,
        n_iter: int = 0,
        request: BatchRequest | None = None,
    ) -> SeparationOutput:
        """Bind TF input, run iterations, and return TF-domain estimate."""
        del request
        self.bind_mixture_tf(np.asarray(mixture_tf))
        if n_iter > 0:
            self.run(int(n_iter))
        return SeparationOutput(estimate_tf=self.get_estimate())

    def fit_transform_time(
        self,
        mixture_time: np.ndarray,
        *,
        n_iter: int = 0,
        request: BatchRequest | None = None,
    ) -> SeparationOutput:
        """Bind time-domain input if supported, then run iterations."""
        sample_rate = None if request is None else request.sample_rate
        self.bind_mixture_time(np.asarray(mixture_time), sample_rate)
        if n_iter > 0:
            self.run(int(n_iter))
        return SeparationOutput(estimate_tf=self.get_estimate())

    def bind_mixture_tf(self, mixture_tf: np.ndarray) -> None:
        """Bind TF-domain input before iterative updates.

        Subclasses should override this when they support external TF input via
        :meth:`separate`.
        """
        raise ValueError(f"{self.__class__.__name__} does not support TF-domain input.")

    def bind_mixture_time(
        self, mixture_time: np.ndarray, sample_rate: int | None
    ) -> None:
        """Bind time-domain input before iterative updates."""
        del sample_rate
        raise ValueError(
            f"{self.__class__.__name__} does not support time-domain input."
        )

    def run(self, n_iter: int) -> np.ndarray:
        """Execute ``n_iter`` update steps and return final estimate."""
        if n_iter < 0:
            raise ValueError("n_iter must be non-negative")
        for _ in range(n_iter):
            self.step()
        return self.get_estimate()

    def forward(
        self,
        mixture: np.ndarray,
        *,
        n_iter: int = 0,
        request: BatchRequest | None = None,
        is_time_input: bool | None = None,
    ) -> SeparationOutput:
        """Run batch separation from TF-domain or time-domain input."""
        mixture_arr = np.asarray(mixture)
        if is_time_input is None:
            is_time = not (np.iscomplexobj(mixture_arr) or mixture_arr.ndim >= 3)
        else:
            is_time = bool(is_time_input)
        if is_time:
            return self.fit_transform_time(
                mixture_arr,
                n_iter=int(n_iter),
                request=request,
            )
        return self.fit_transform_tf(
            mixture_arr,
            n_iter=int(n_iter),
            request=request,
        )


class BaseStreamingSeparator(BaseSeparator):
    """Base class for frame-wise streaming separators."""

    @abstractmethod
    def process_frame(
        self,
        frame: np.ndarray,
        request: OnlineFrameRequest | None = None,
    ) -> np.ndarray:
        """Process a single frame."""

    def get_state(self) -> SeparatorState | StreamingSeparatorState:
        """Return current separator state snapshot."""
        return SeparatorState()

    def set_state(self, state: SeparatorState | StreamingSeparatorState) -> None:
        """Restore separator state from a snapshot."""
        del state

    def forward_streaming(
        self,
        frame: np.ndarray,
        *,
        state: SeparatorState | StreamingSeparatorState | None = None,
        request: OnlineFrameRequest | None = None,
    ) -> tuple[np.ndarray, SeparatorState | StreamingSeparatorState]:
        """Process one frame and return ``(separated_frame, updated_state)``."""
        if state is not None:
            self.set_state(state)
        output = self.process_frame(frame, request=request)
        return output, self.get_state()

    def process_stream(
        self,
        stream: np.ndarray,
        *,
        frame_axis: int = -1,
        request: OnlineFrameRequest | None = None,
    ) -> np.ndarray:
        """Process all frames in ``stream`` and stack outputs on the last axis."""
        frames = np.moveaxis(stream, frame_axis, 0)
        if frames.shape[0] == 0:
            raise ValueError("stream must contain at least one frame")

        outputs = [self.process_frame(frames[0], request=request)]
        for idx in range(1, frames.shape[0]):
            outputs.append(self.process_frame(frames[idx], request=request))
        return np.stack(outputs, axis=-1)

    def process_stream_tf(
        self,
        stream_tf: np.ndarray,
        *,
        request: StreamRequest | None = None,
    ) -> SeparationOutput:
        """Process all frames in ``stream_tf`` using a typed stream request."""
        request_obj = StreamRequest() if request is None else request
        frame_request = OnlineFrameRequest(
            n_sources=request_obj.n_sources,
            component_to_source=request_obj.component_to_source,
            return_mask=request_obj.return_mask,
            reference_mic=request_obj.reference_mic,
            metadata=dict(request_obj.metadata),
        )
        output = self.process_stream(
            stream_tf,
            frame_axis=int(request_obj.frame_axis),
            request=frame_request,
        )
        return SeparationOutput(estimate_tf=output, state=self.get_state())

    def forward(
        self,
        stream_tf: np.ndarray,
        *,
        request: StreamRequest | None = None,
    ) -> SeparationOutput:
        """Torch-like forward alias for full streaming input."""
        return self.process_stream_tf(stream_tf, request=request)
