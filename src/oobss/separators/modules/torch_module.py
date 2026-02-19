"""PyTorch module wrapper for oobss separators."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from oobss.separators.core import (
    BaseStreamingSeparator,
    BatchRequest,
    OnlineFrameRequest,
    StreamRequest,
)

try:  # pragma: no cover - optional dependency
    _TORCH = importlib.import_module("torch")
    _TORCH_NN = importlib.import_module("torch.nn")
except ModuleNotFoundError as exc:  # pragma: no cover
    _TORCH = None
    _TORCH_NN = None
    _TORCH_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:  # pragma: no cover
    _TORCH_IMPORT_ERROR = None


_TorchModuleBase = _TORCH_NN.Module if _TORCH_NN is not None else object


class TorchSeparatorModule(_TorchModuleBase):
    """Wrap a separator object with a ``forward`` API compatible with PyTorch."""

    def __init__(self, separator: Any) -> None:
        if _TORCH_IMPORT_ERROR is not None:
            raise RuntimeError(
                "TorchSeparatorModule requires the optional 'torch'."
            ) from _TORCH_IMPORT_ERROR

        super().__init__()
        self.separator = separator

    def as_nn_module(self) -> Any:
        """Return ``self`` as an ``nn.Module`` instance."""
        return self

    def forward(
        self,
        mixture: Any,
        *,
        n_iter: int = 0,
        sample_rate: int | None = None,
        reference_mic: int = 0,
        frame_axis: int = -1,
        frame_request: OnlineFrameRequest | None = None,
    ) -> dict[str, Any]:
        """Run separation and return torch tensors keyed by output field names."""
        if _TORCH is None:
            raise RuntimeError("Torch is not available.")
        mixture_np = mixture.detach().cpu().numpy()
        is_tf_input = np.iscomplexobj(mixture_np) or mixture_np.ndim >= 3
        if isinstance(self.separator, BaseStreamingSeparator):
            request_obj = (
                StreamRequest(
                    frame_axis=frame_axis,
                    reference_mic=reference_mic,
                )
                if frame_request is None
                else StreamRequest(
                    frame_axis=frame_axis,
                    reference_mic=frame_request.reference_mic,
                    n_sources=frame_request.n_sources,
                    component_to_source=frame_request.component_to_source,
                    return_mask=frame_request.return_mask,
                    metadata=dict(frame_request.metadata),
                )
            )
            output = self.separator.process_stream_tf(
                mixture_np,
                request=request_obj,
            )
        elif hasattr(self.separator, "fit_transform_tf") and is_tf_input:
            output = self.separator.fit_transform_tf(
                mixture_np,
                n_iter=int(n_iter),
                request=BatchRequest(
                    reference_mic=reference_mic,
                    sample_rate=sample_rate,
                ),
            )
        elif hasattr(self.separator, "fit_transform_time") and not is_tf_input:
            output = self.separator.fit_transform_time(
                mixture_np,
                n_iter=int(n_iter),
                request=BatchRequest(
                    reference_mic=reference_mic,
                    sample_rate=sample_rate,
                ),
            )
        else:
            raise TypeError(
                "separator must implement BaseStreamingSeparator, "
                "fit_transform_tf(), or fit_transform_time()."
            )

        result: dict[str, Any] = {}
        mixture_device = getattr(mixture, "device", None)
        if output.estimate_time is not None:
            estimate_time = _TORCH.from_numpy(output.estimate_time)
            result["estimate_time"] = (
                estimate_time.to(mixture_device)
                if mixture_device is not None
                else estimate_time
            )
        if output.estimate_tf is not None:
            estimate_tf = _TORCH.from_numpy(output.estimate_tf)
            result["estimate_tf"] = (
                estimate_tf.to(mixture_device)
                if mixture_device is not None
                else estimate_tf
            )
        if output.mask is not None:
            mask = _TORCH.from_numpy(output.mask)
            result["mask"] = (
                mask.to(mixture_device) if mixture_device is not None else mask
            )
        return result
