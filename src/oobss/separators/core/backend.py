"""Backend abstraction for NumPy/Torch-compatible separator code."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Backend(ABC):
    """Minimal tensor backend protocol used by wrappers and future modules."""

    name: str

    @abstractmethod
    def asarray(self, value: Any) -> Any:
        """Convert value into backend tensor array."""

    @abstractmethod
    def abs_square(self, value: Any) -> Any:
        """Compute squared magnitude."""

    @abstractmethod
    def matmul(self, lhs: Any, rhs: Any) -> Any:
        """Matrix multiplication."""

    @abstractmethod
    def to_numpy(self, value: Any) -> np.ndarray:
        """Convert backend tensor to NumPy array."""


class NumpyBackend(Backend):
    """NumPy backend implementation."""

    name = "numpy"

    def asarray(self, value: Any) -> np.ndarray:
        return np.asarray(value)

    def abs_square(self, value: np.ndarray) -> np.ndarray:
        return np.abs(value) ** 2

    def matmul(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return lhs @ rhs

    def to_numpy(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value)


class TorchBackend(Backend):
    """Torch backend implementation loaded lazily.

    This backend keeps torch as an optional dependency and only imports it when
    instantiated.
    """

    name = "torch"

    def __init__(self) -> None:
        try:
            self.torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "TorchBackend requires the optional 'torch' dependency."
            ) from exc

    def asarray(self, value: Any) -> Any:
        return self.torch.as_tensor(value)

    def abs_square(self, value: Any) -> Any:
        return value.abs() ** 2

    def matmul(self, lhs: Any, rhs: Any) -> Any:
        return lhs @ rhs

    def to_numpy(self, value: Any) -> np.ndarray:
        return value.detach().cpu().numpy()
