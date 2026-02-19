from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from oobss.dataset import TorchrirDynamicDatasetAdapter, create_loader
from oobss.experiments.torchrir_dynamic import (
    TorchrirDynamicDataset,
    build_torchrir_dynamic_dataloader,
    discover_torchrir_scene_paths,
)


def _write_wave(path: Path, data: np.ndarray, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, sample_rate)


def _make_scene(
    root: Path,
    scene_id: str,
    *,
    n_src: int = 2,
    n_mic: int = 2,
    mix_samples: int = 128,
    source_samples: list[int] | None = None,
    sample_rate: int = 8000,
    with_metadata: bool = False,
) -> None:
    scene_dir = root / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    time = np.linspace(0.0, 1.0, mix_samples, endpoint=False)
    mix = np.stack(
        [
            0.4 * np.sin(2.0 * np.pi * 2.0 * time),
            0.4 * np.cos(2.0 * np.pi * 2.0 * time),
        ],
        axis=1,
    )
    if n_mic != 2:
        mix = np.tile(mix[:, :1], (1, n_mic))
    _write_wave(scene_dir / "mixture.wav", mix, sample_rate)

    lengths = (
        source_samples
        if source_samples is not None
        else [mix_samples for _ in range(n_src)]
    )
    for src_idx in range(n_src):
        length = int(lengths[src_idx])
        t = np.linspace(0.0, 1.0, length, endpoint=False)
        source = np.stack(
            [
                0.2 * np.sin(2.0 * np.pi * (src_idx + 1) * t),
                0.2 * np.cos(2.0 * np.pi * (src_idx + 1) * t),
            ],
            axis=1,
        )
        if n_mic != 2:
            source = np.tile(source[:, :1], (1, n_mic))
        _write_wave(scene_dir / f"source_{src_idx:02d}.wav", source, sample_rate)

    if with_metadata:
        with (scene_dir / "metadata.json").open("w", encoding="utf-8") as fh:
            json.dump({"scene_id": scene_id, "kind": "test"}, fh)


def test_torchrir_dynamic_dataset_returns_tensors(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    _make_scene(tmp_path, "scene_0000", with_metadata=True)
    _make_scene(tmp_path, "scene_0001")

    dataset = TorchrirDynamicDataset(
        tmp_path,
        return_type="torch",
        include_metadata=True,
    )
    assert len(dataset) == 2

    sample = dataset[0]
    assert sample["scene_id"] == "scene_0000"
    assert isinstance(sample["mix"], torch.Tensor)
    assert isinstance(sample["stems"], torch.Tensor)
    assert sample["mix"].shape == (128, 2)
    assert sample["stems"].shape == (2, 128, 2)
    assert sample["mix"].dtype == torch.float32
    assert sample["metadata"]["kind"] == "test"


def test_build_torchrir_dynamic_dataloader_returns_list_batch(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    _make_scene(tmp_path, "scene_0000")
    _make_scene(tmp_path, "scene_0001")

    dataloader = build_torchrir_dynamic_dataloader(
        root=tmp_path,
        batch_size=2,
        num_workers=0,
    )
    batch = next(iter(dataloader))
    assert isinstance(batch, list)
    assert len(batch) == 2
    assert all("mix" in item and "stems" in item for item in batch)


def test_create_loader_builds_torchrir_dynamic_loader(tmp_path: Path) -> None:
    _make_scene(
        tmp_path,
        "scene_0000",
        source_samples=[96, 140],
        mix_samples=120,
    )
    loader = create_loader(
        {
            "type": "torchrir_dynamic",
            "root": str(tmp_path),
        }
    )
    assert isinstance(loader, TorchrirDynamicDatasetAdapter)

    handles = loader.discover()
    assert [handle.track_id for handle in handles] == ["scene_0000"]
    loaded = loader.load(handles[0])
    assert loaded.track_id == "scene_0000"
    assert loaded.stems.shape == (2, 96, 2)
    assert loaded.mix.shape == (96, 2)


def test_discover_torchrir_scene_paths_validates_include(tmp_path: Path) -> None:
    _make_scene(tmp_path, "scene_0000")
    with pytest.raises(ValueError):
        discover_torchrir_scene_paths(tmp_path, include=["scene_9999"])


def test_torchrir_builder_api_available_when_installed() -> None:
    datasets = pytest.importorskip("torchrir.datasets")
    if not hasattr(datasets, "build_dynamic_cmu_arctic_dataset"):
        pytest.skip(
            "torchrir.datasets.build_dynamic_cmu_arctic_dataset is unavailable in "
            "the installed torchrir version."
        )
