"""Dataset adapters and loader factory for experiment pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TypeAlias

import numpy as np

from .torchrir_dynamic import (
    discover_torchrir_scene_paths,
    load_torchrir_dynamic_scene,
)


@dataclass(frozen=True)
class TrackAudio:
    """Container holding stems, mixture, and metadata for one track.

    Attributes
    ----------
    track_id:
        Unique identifier of the track in the dataset.
    path:
        Backing track path, if available.
    stems:
        Time-domain stem signals with shape ``(n_src, n_samples, n_mic)``.
    mix:
        Time-domain mixture with shape ``(n_samples, n_mic)``.
    sample_rate:
        Sampling rate in Hz.
    """

    track_id: str
    path: Path
    stems: np.ndarray  # shape: (n_src, n_samples, n_mic)
    mix: np.ndarray  # shape: (n_samples, n_mic)
    sample_rate: int

    @property
    def n_src(self) -> int:
        """Return number of sources."""
        return int(self.stems.shape[0])

    @property
    def n_mic(self) -> int:
        """Return number of microphones."""
        return int(self.stems.shape[2])

    @property
    def n_samples(self) -> int:
        """Return number of samples."""
        return int(self.stems.shape[1])

    @property
    def duration(self) -> float:
        """Return track duration in seconds."""
        return self.n_samples / float(self.sample_rate)


@dataclass(frozen=True)
class TrackHandle:
    """Opaque reference to a track discovered by a dataset loader.

    The payload is loader-specific metadata required to retrieve the actual
    audio for the track. It must remain pickle-safe because tasks can run in
    subprocess workers.
    """

    track_id: str
    payload: dict[str, Any]


class BaseDatasetAdapter(ABC):
    """Abstract dataset adapter used by experiment pipelines."""

    @abstractmethod
    def discover(
        self,
        *,
        include: Iterable[str] | None = None,
        sample_limit: int | None = None,
    ) -> list[TrackHandle]:
        """Discover available track handles."""

    @abstractmethod
    def load(
        self,
        handle: TrackHandle,
        *,
        duration_sec: float | None = None,
    ) -> TrackAudio:
        """Load one track represented by ``handle``."""

    @abstractmethod
    def stem_names(self) -> list[str]:
        """Return ordered stem labels used in reporting outputs."""


DatasetLoader = BaseDatasetAdapter
AdapterFactory: TypeAlias = Callable[[dict[str, Any]], BaseDatasetAdapter]


@dataclass(frozen=True)
class TorchrirDynamicDatasetAdapter(BaseDatasetAdapter):
    """Adapter for dynamic torchrir scene directories."""

    root: Path
    duration_sec: float | None = None

    def discover(
        self,
        *,
        include: Iterable[str] | None = None,
        sample_limit: int | None = None,
    ) -> list[TrackHandle]:
        """Return discovered dynamic-scene handles sorted by scene ID."""
        scenes = discover_torchrir_scene_paths(
            self.root,
            include=include,
            sample_limit=sample_limit,
        )
        return [
            TrackHandle(
                track_id=path.name,
                payload={"path": str(path)},
            )
            for path in scenes
        ]

    def load(
        self,
        handle: TrackHandle,
        *,
        duration_sec: float | None = None,
    ) -> TrackAudio:
        """Load one dynamic scene and return canonical track audio."""
        raw_path = handle.payload.get("path")
        if not isinstance(raw_path, str):
            raise ValueError(f"Invalid track handle payload for {handle.track_id}")
        scene = load_torchrir_dynamic_scene(
            Path(raw_path),
            duration_sec=self.duration_sec if duration_sec is None else duration_sec,
        )
        stems = np.asarray(scene["stems"], dtype=np.float64)
        mix = np.asarray(scene["mix"], dtype=np.float64)
        sample_rate = int(scene["sample_rate"])
        return TrackAudio(
            track_id=str(scene["scene_id"]),
            path=Path(str(scene["path"])),
            stems=stems,
            mix=mix,
            sample_rate=sample_rate,
        )

    def stem_names(self) -> list[str]:
        """Return ordered source names from the first discovered scene."""
        scenes = discover_torchrir_scene_paths(self.root, sample_limit=1)
        if not scenes:
            return []
        sample = load_torchrir_dynamic_scene(
            scenes[0],
            duration_sec=self.duration_sec,
        )
        source_names_obj = sample.get("source_names", [])
        if not isinstance(source_names_obj, list):
            return []
        return [str(name) for name in source_names_obj]


def build_torchrir_dynamic_adapter(
    dataset_cfg: dict[str, Any],
) -> BaseDatasetAdapter:
    """Create :class:`TorchrirDynamicDatasetAdapter` from dataset configuration."""
    root = (
        Path(dataset_cfg.get("root", "outputs/cmu_arctic_torchrir_dynamic_dataset"))
        .expanduser()
        .resolve()
    )
    duration_obj = dataset_cfg.get("duration_sec")
    duration_sec = None if duration_obj is None else float(duration_obj)
    return TorchrirDynamicDatasetAdapter(root=root, duration_sec=duration_sec)


def loader_registry(
    overrides: Mapping[str, AdapterFactory] | None = None,
) -> dict[str, AdapterFactory]:
    """Return registry mapping loader type names to factories."""
    registry: dict[str, AdapterFactory] = {
        "torchrir_dynamic": build_torchrir_dynamic_adapter,
    }
    if overrides:
        registry.update(overrides)
    return registry


def create_loader(
    dataset_cfg: dict[str, Any],
    *,
    registry_overrides: Mapping[str, AdapterFactory] | None = None,
) -> BaseDatasetAdapter:
    """Instantiate a dataset loader from ``dataset_cfg``.

    Parameters
    ----------
    dataset_cfg:
        Dataset configuration. Must include ``type`` when using non-default
        adapters. The default type is ``torchrir_dynamic``.
    registry_overrides:
        Optional injected adapter factories for tests or external integrations.
    """
    cfg = dict(dataset_cfg)
    loader_type = str(cfg.get("type", "torchrir_dynamic"))
    registry = loader_registry(overrides=registry_overrides)
    try:
        factory = registry[loader_type]
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown dataset loader type: {loader_type}. Available: {available}"
        ) from exc
    return factory(cfg)
