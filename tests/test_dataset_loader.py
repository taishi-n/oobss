from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from oobss.dataloaders import TorchrirDynamicDatasetAdapter, create_loader


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
    sample_rate: int = 8000,
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

    for src_idx in range(n_src):
        t = np.linspace(0.0, 1.0, mix_samples, endpoint=False)
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


def test_create_loader_builds_torchrir_dynamic_adapter(tmp_path: Path) -> None:
    loader = create_loader(
        {
            "type": "torchrir_dynamic",
            "root": str(tmp_path),
        }
    )
    assert isinstance(loader, TorchrirDynamicDatasetAdapter)


def test_torchrir_dynamic_adapter_discovers_and_loads_scene(tmp_path: Path) -> None:
    _make_scene(tmp_path, "scene_0000")

    loader = create_loader(
        {
            "type": "torchrir_dynamic",
            "root": str(tmp_path),
        }
    )
    handles = loader.discover()
    assert [handle.track_id for handle in handles] == ["scene_0000"]

    loaded = loader.load(handles[0])
    assert loaded.stems.shape == (2, 128, 2)
    assert loaded.mix.shape == (128, 2)


def test_create_loader_rejects_unknown_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        create_loader({"type": "unknown_loader", "root": str(tmp_path)})
