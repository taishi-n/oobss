"""Dataset adapter and loader APIs."""

from .base import (
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
from .torchrir_dynamic import TorchrirDynamicDataset, build_torchrir_dynamic_dataloader

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
    "TorchrirDynamicDataset",
    "build_torchrir_dynamic_dataloader",
]
