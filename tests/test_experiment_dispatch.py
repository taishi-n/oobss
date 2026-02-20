from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pytest

from oobss.benchmark.config_loader import MethodConfig, load_method_configs
from oobss.benchmark.config_schema import (
    EvaluationConfig,
    FrameEvalConfig,
    MetricsConfig,
)
from oobss.dataloaders import DatasetLoader, TrackAudio, TrackHandle
from oobss.benchmark.engine import (
    expand_method_grids,
    merge_method_grids,
    parse_grid_overrides,
)
from oobss.benchmark.methods import (
    MethodRunnerRegistry,
    default_method_runner_registry,
)
from oobss.evaluation.metrics import MetricsBundle
from oobss.benchmark.pipeline import ExperimentTask, run_task
from oobss.separators.core import SeparationOutput
from oobss.signal import STFTPlan


def test_method_runner_registry_accepts_overrides() -> None:
    def custom_runner(
        mix: np.ndarray,
        params: Mapping[str, object],
        stft: STFTPlan,
        sample_rate: int,
    ) -> SeparationOutput:
        del params, stft, sample_rate
        return SeparationOutput(
            estimate_time=np.real(mix.T),
            metadata={"reference_mic": 0},
        )

    registry = default_method_runner_registry()
    registry.register("batch_auxiva", custom_runner, overwrite=True)
    out = registry.resolve("batch_auxiva").run(
        np.ones((8, 2), dtype=np.float64),
        {},
        STFTPlan(fft_size=16, hop_size=8, window="hann"),
        8000,
    )
    assert out.estimate_time is not None
    assert out.estimate_time.shape == (2, 8)


def test_run_task_uses_injected_method_override(monkeypatch) -> None:
    import oobss.benchmark.pipeline as pipeline

    n_src = 2
    n_samples = 64
    n_mic = 2

    stems = np.ones((n_src, n_samples, n_mic), dtype=np.float64)
    mix = np.sum(stems, axis=0)
    track = TrackAudio(
        track_id="track_01",
        path=Path("/tmp/track_01"),
        stems=stems,
        mix=mix,
        sample_rate=8000,
    )

    class FakeLoader(DatasetLoader):
        def discover(
            self,
            *,
            include: Iterable[str] | None = None,
            sample_limit: int | None = None,
        ) -> list[TrackHandle]:
            del include, sample_limit
            return [TrackHandle(track_id="track_01", payload={})]

        def load(
            self,
            handle: TrackHandle,
            *,
            duration_sec: float | None = None,
        ) -> TrackAudio:
            del handle, duration_sec
            return track

        def stem_names(self) -> list[str]:
            return ["vocals", "drums"]

    def fake_compute_metrics(
        reference: np.ndarray,
        estimate: np.ndarray,
        mixture: np.ndarray,
        sample_rate: int,
        *,
        filter_length: int,
        frame_cfg: FrameEvalConfig | None,
        permutation_strategy,
    ) -> MetricsBundle:
        del (
            estimate,
            mixture,
            sample_rate,
            filter_length,
            frame_cfg,
            permutation_strategy,
        )
        n_local_src = reference.shape[0]
        zeros = np.zeros((n_local_src,), dtype=np.float64)
        return MetricsBundle(
            sdr_mix=zeros,
            sdr_est=zeros,
            sir_est=zeros,
            sar_est=zeros,
            permutation=np.arange(n_local_src, dtype=np.int64),
            framewise=None,
        )

    called = {"value": False}

    def custom_runner(
        mix_in: np.ndarray,
        params: Mapping[str, object],
        stft: STFTPlan,
        sample_rate: int,
    ) -> SeparationOutput:
        del params, stft, sample_rate
        called["value"] = True
        est = np.real(mix_in[:, :n_src].T)
        return SeparationOutput(
            estimate_time=est,
            metadata={"reference_mic": 0, "from_override": True},
        )

    monkeypatch.setattr(pipeline, "compute_metrics", fake_compute_metrics)

    task = ExperimentTask(
        track_id=track.track_id,
        track=TrackHandle(track_id=track.track_id, payload={}),
        loader=FakeLoader(),
        method=MethodConfig(
            id="custom_method",
            label="Custom",
            type="custom_type",
            enabled=True,
            params={},
        ),
        stem_names=["vocals", "drums"],
        stft_plan=STFTPlan(fft_size=16, hop_size=8, window="hann"),
        evaluation=EvaluationConfig(filter_length=1, frame=FrameEvalConfig()),
        metrics_cfg=MetricsConfig(),
        duration_limit=None,
    )

    registry = MethodRunnerRegistry()
    registry.register("custom_type", custom_runner)
    out = run_task(task, runner_registry=registry)

    assert called["value"] is True
    assert out.success is True
    assert out.method_metadata["from_override"] is True


def test_expand_method_grids_builds_cartesian_variants() -> None:
    methods = [
        MethodConfig(
            id="batch_ilrma",
            label="Batch ILRMA",
            type="batch_ilrma",
            enabled=True,
            params={"n_iter": 50, "n_basis": 2},
        )
    ]
    expanded = expand_method_grids(
        methods,
        {
            "batch_ilrma": {
                "n_basis": [2, 4],
                "n_iter": [10, 20],
            }
        },
    )
    assert len(expanded) == 4
    ids = {method.id for method in expanded}
    assert any("n_basis=4" in method_id for method_id in ids)
    assert any("n_iter=20" in method_id for method_id in ids)


def test_parse_and_merge_grid_overrides() -> None:
    cli_grid = parse_grid_overrides(
        ["batch_ilrma.n_basis=2,4", "online_ilrma.beta=1,2"]
    )
    merged = merge_method_grids(
        {"batch_ilrma": {"n_iter": [10]}},
        cli_grid,
    )
    assert merged["batch_ilrma"]["n_iter"] == [10]
    assert merged["batch_ilrma"]["n_basis"] == [2, 4]
    assert merged["online_ilrma"]["beta"] == [1, 2]


def test_load_method_configs_validates_builtin_method_params(tmp_path: Path) -> None:
    (tmp_path / "bad_method.yaml").write_text(
        "\n".join(
            [
                "id: bad_batch_ilrma",
                "label: Bad Batch ILRMA",
                "type: batch_ilrma",
                "enabled: true",
                "params:",
                "  n_basis: 2",
                "  unknown_key: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(Exception):
        load_method_configs(tmp_path)
