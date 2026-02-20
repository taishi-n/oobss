"""PyTorch-friendly dataset utilities for dynamic torchrir scene directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import soundfile as sf

torch: Any
DataLoader: Any

try:  # pragma: no cover - optional dependency
    import torch as _torch
    from torch.utils.data import DataLoader as _DataLoader
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None
    DataLoader = None
else:  # pragma: no cover - optional dependency
    torch = _torch
    DataLoader = _DataLoader


TorchrirSample = dict[str, Any]
CollateFn = Callable[[list[TorchrirSample]], Any]


def discover_torchrir_scene_paths(
    root: Path,
    *,
    include: Iterable[str] | None = None,
    sample_limit: int | None = None,
) -> list[Path]:
    """Discover ``scene_*`` directories under ``root``."""
    dataset_root = Path(root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    allow = {item.strip() for item in include or [] if item}
    scenes = [
        path
        for path in sorted(dataset_root.iterdir())
        if path.is_dir() and path.name.startswith("scene_")
    ]
    if allow:
        scenes = [path for path in scenes if path.name in allow]
        missing = allow - {path.name for path in scenes}
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Requested scenes missing: {missing_list}")
    if sample_limit is not None and sample_limit > 0:
        scenes = scenes[:sample_limit]
    return scenes


def load_torchrir_dynamic_scene(
    scene_path: Path,
    *,
    duration_sec: float | None = None,
    include_metadata: bool = False,
) -> TorchrirSample:
    """Load one dynamic torchrir scene into canonical numpy arrays."""
    path = Path(scene_path)
    if not path.exists():
        raise FileNotFoundError(f"Scene path not found: {path}")

    mix_path = path / "mixture.wav"
    if not mix_path.exists():
        raise FileNotFoundError(f"Missing mixture file: {mix_path}")
    mix, sample_rate = sf.read(mix_path, always_2d=True)
    mix_np = np.asarray(mix, dtype=np.float64)

    source_paths = sorted(path.glob("source_*.wav"))
    if not source_paths:
        raise ValueError(f"No source stems found in {path}")

    sources: list[np.ndarray] = []
    source_names: list[str] = []
    for source_path in source_paths:
        audio, stem_sr = sf.read(source_path, always_2d=True)
        if int(stem_sr) != int(sample_rate):
            raise ValueError(
                f"Sample-rate mismatch in {source_path}: {stem_sr} != {sample_rate}"
            )
        source = np.asarray(audio, dtype=np.float64)
        if source.shape[1] != mix_np.shape[1]:
            raise ValueError(
                f"Channel mismatch in {source_path}: {source.shape[1]} != {mix_np.shape[1]}"
            )
        sources.append(source)
        source_names.append(source_path.stem)

    min_samples = min(
        [int(mix_np.shape[0])] + [int(source.shape[0]) for source in sources],
    )
    if duration_sec is not None and duration_sec > 0:
        duration_samples = int(float(duration_sec) * int(sample_rate))
        if duration_samples <= 0:
            raise ValueError("duration_sec is too small for the sample rate.")
        min_samples = min(min_samples, duration_samples)

    if min_samples <= 0:
        raise ValueError(f"Scene {path} contains no audio samples.")

    mix_trimmed = mix_np[:min_samples, :]
    stems = np.stack([source[:min_samples, :] for source in sources], axis=0)

    payload: TorchrirSample = {
        "scene_id": path.name,
        "path": str(path),
        "mix": mix_trimmed,
        "stems": stems,
        "sample_rate": int(sample_rate),
        "source_names": source_names,
    }
    if include_metadata:
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as fh:
                payload["metadata"] = json.load(fh)
        else:
            payload["metadata"] = {}
    return payload


class TorchrirDynamicDataset:
    """Dataset with ``__len__`` / ``__getitem__`` for dynamic torchrir scenes."""

    def __init__(
        self,
        root: Path | str,
        *,
        return_type: str = "torch",
        include: Sequence[str] | None = None,
        sample_limit: int | None = None,
        duration_sec: float | None = None,
        dtype: Any | None = None,
        include_metadata: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.return_type = str(return_type).lower()
        if self.return_type not in {"torch", "numpy"}:
            raise ValueError("return_type must be either 'torch' or 'numpy'.")
        if self.return_type == "torch" and torch is None:
            raise RuntimeError(
                "TorchrirDynamicDataset(return_type='torch') requires torch."
            )

        self.duration_sec = duration_sec
        self.include_metadata = bool(include_metadata)
        default_dtype = None if torch is None else torch.float32
        self.dtype = default_dtype if dtype is None else dtype
        self._scene_paths = discover_torchrir_scene_paths(
            self.root,
            include=include,
            sample_limit=sample_limit,
        )

    def __len__(self) -> int:
        return len(self._scene_paths)

    def __getitem__(self, index: int) -> TorchrirSample:
        scene = load_torchrir_dynamic_scene(
            self._scene_paths[index],
            duration_sec=self.duration_sec,
            include_metadata=self.include_metadata,
        )
        if self.return_type == "numpy":
            return scene
        if torch is None:  # pragma: no cover - guarded in __init__
            raise RuntimeError("torch is required for return_type='torch'.")

        mix_np = np.asarray(scene["mix"])
        stems_np = np.asarray(scene["stems"])
        mix = torch.as_tensor(mix_np, dtype=self.dtype)
        stems = torch.as_tensor(stems_np, dtype=self.dtype)

        output: TorchrirSample = dict(scene)
        output["mix"] = mix
        output["stems"] = stems
        return output


def collate_scene_batch_as_list(batch: list[TorchrirSample]) -> list[TorchrirSample]:
    """Keep variable-length scene samples as a plain list."""
    return list(batch)


def build_torchrir_dynamic_dataloader(
    *,
    root: Path | str,
    return_type: str = "torch",
    include: Sequence[str] | None = None,
    sample_limit: int | None = None,
    duration_sec: float | None = None,
    dtype: Any | None = None,
    include_metadata: bool = False,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    collate_fn: CollateFn | None = None,
) -> Any:
    """Build a ``torch.utils.data.DataLoader`` for dynamic torchrir scenes."""
    if DataLoader is None:
        raise RuntimeError("build_torchrir_dynamic_dataloader requires torch.")

    dataset = TorchrirDynamicDataset(
        root=root,
        return_type=return_type,
        include=include,
        sample_limit=sample_limit,
        duration_sec=duration_sec,
        dtype=dtype,
        include_metadata=include_metadata,
    )
    resolved_collate = collate_scene_batch_as_list if collate_fn is None else collate_fn
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
        collate_fn=resolved_collate,
    )
