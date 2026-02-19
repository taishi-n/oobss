"""Public dataset API.

This module re-exports dataset adapter primitives used by experiment pipelines.
"""

from __future__ import annotations

from .experiments.dataset import (
    AdapterFactory,
    BaseDatasetAdapter,
    DatasetLoader,
    TorchrirDynamicDatasetAdapter,
    TrackAudio,
    TrackHandle,
    build_torchrir_dynamic_adapter,
    create_loader,
    loader_registry,
)

__all__ = [
    "AdapterFactory",
    "BaseDatasetAdapter",
    "DatasetLoader",
    "TorchrirDynamicDatasetAdapter",
    "TrackAudio",
    "TrackHandle",
    "build_torchrir_dynamic_adapter",
    "create_loader",
    "loader_registry",
]
