"""Execution pipeline for large-scale separation experiments."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Optional, cast

import numpy as np
import soundfile as sf

from oobss.dataset import DatasetLoader, TrackAudio, TrackHandle, create_loader
from oobss.separators.core import PermutationStrategy
from oobss.separators.strategies import (
    BssEvalPermutationStrategy,
    IdentityPermutationStrategy,
)

from .config_loader import MethodConfig
from .config_schema import (
    EvaluationConfig,
    MetricsConfig,
    parse_evaluation_config,
    parse_metrics_config,
    parse_stft_config,
)
from .methods import (
    MethodRunnerRegistry,
    STFTPlan,
    default_method_runner_registry,
)
from .metrics import MetricsBundle, compute_metrics
from .recipe import ExperimentRecipe

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentTask:
    track_id: str
    track: TrackHandle
    loader: DatasetLoader
    method: MethodConfig
    stem_names: list[str]
    stft_plan: STFTPlan
    evaluation: EvaluationConfig
    metrics_cfg: MetricsConfig
    duration_limit: Optional[float]


@dataclass
class ExperimentOutput:
    task: ExperimentTask
    duration: float
    elapsed: float
    real_time_factor: float
    success: bool
    metrics: Optional[MetricsBundle]
    separation: Optional[np.ndarray]
    method_metadata: dict[str, object]
    error: Optional[str]
    frame_hop_sec: Optional[float]
    sample_rate: int
    permutation: Optional[np.ndarray]
    stem_names: list[str]

    def to_summary_record(
        self,
        run_dir: Path,
        *,
        detail_path: Path,
        frame_path: Path | None,
        audio_path: Path | None,
    ) -> dict[str, object]:
        summary: dict[str, object] = {
            "track_id": self.task.track_id,
            "method_id": self.task.method.id,
            "method_label": self.task.method.label,
            "duration_sec": self.duration,
            "elapsed_sec": self.elapsed,
            "real_time_factor": self.real_time_factor,
            "success": self.success,
            "detail_path": str(detail_path.relative_to(run_dir)),
        }
        if self.error:
            summary["error"] = self.error
        if self.frame_hop_sec is not None:
            summary["frame_hop_sec"] = self.frame_hop_sec
        if self.metrics is not None:
            summary.update(self.metrics.to_summary())
        if frame_path is not None:
            summary["frame_metrics_path"] = str(frame_path.relative_to(run_dir))
        if audio_path is not None:
            summary["estimate_audio_path"] = str(audio_path.relative_to(run_dir))
        summary["stem_names"] = list(self.stem_names)
        return cast(dict[str, object], _clean_json(summary))


def build_tasks(
    *,
    loader: DatasetLoader,
    methods: Iterable[MethodConfig],
    stft_cfg: Mapping[str, object],
    evaluation: Mapping[str, object],
    metrics_cfg: Mapping[str, object],
    track_subset: Iterable[str] | None = None,
    sample_limit: int | None = None,
    duration_limit: Optional[float] = None,
) -> list[ExperimentTask]:
    tracks = loader.discover(include=track_subset, sample_limit=sample_limit)
    if sample_limit is not None and sample_limit > 0:
        LOGGER.info("Sample limit applied: using first %d tracks", len(tracks))
    stft_config = parse_stft_config(stft_cfg)
    evaluation_config = parse_evaluation_config(evaluation)
    metrics_config = parse_metrics_config(metrics_cfg)
    plan = STFTPlan(
        fft_size=int(stft_config.fft_size),
        hop_size=int(stft_config.hop_size),
        window=str(stft_config.window),
    )

    stem_labels = loader.stem_names()

    tasks: list[ExperimentTask] = []
    for track in tracks:
        for method in methods:
            tasks.append(
                ExperimentTask(
                    track_id=track.track_id,
                    track=track,
                    loader=loader,
                    method=method,
                    stem_names=list(stem_labels),
                    stft_plan=plan,
                    evaluation=evaluation_config,
                    metrics_cfg=metrics_config,
                    duration_limit=duration_limit,
                )
            )
    return tasks


def build_tasks_from_recipe(
    recipe: ExperimentRecipe,
    *,
    methods: Iterable[MethodConfig],
) -> list[ExperimentTask]:
    """Build tasks from a resolved :class:`ExperimentRecipe`."""
    loader = create_loader(recipe.dataset)
    return build_tasks(
        loader=loader,
        methods=methods,
        stft_cfg=recipe.stft,
        evaluation=recipe.evaluation,
        metrics_cfg=recipe.metrics,
        track_subset=recipe.track_subset or None,
        sample_limit=recipe.sample_limit,
        duration_limit=recipe.duration_limit,
    )


def run_task(
    task: ExperimentTask,
    *,
    runner_registry: MethodRunnerRegistry | None = None,
) -> ExperimentOutput:
    track: Optional[TrackAudio] = None
    separation = None
    start = time.perf_counter()

    resolved_registry = (
        default_method_runner_registry() if runner_registry is None else runner_registry
    )

    try:
        track = task.loader.load(task.track, duration_sec=task.duration_limit)
        LOGGER.info("Running %s on %s", task.method.id, task.track_id)

        method_runner = resolved_registry.resolve(task.method.type)
        method_params = dict(task.method.params)
        if task.method.type == "online_isnmf" and "n_sources" not in method_params:
            method_params["n_sources"] = int(track.stems.shape[0])
        separation = method_runner.run(
            track.mix,
            method_params,
            task.stft_plan,
            track.sample_rate,
        )
        if separation.estimate_time is None:
            raise ValueError(
                f"Method '{task.method.id}' did not return estimate_time in SeparationOutput."
            )
        elapsed = time.perf_counter() - start
        rtf = elapsed / max(track.duration, 1e-9)

        ref_mic_obj = task.method.params.get("ref_mic")
        ref_mic = (
            _coerce_int(ref_mic_obj, name="ref_mic")
            if ref_mic_obj is not None
            else int(task.evaluation.ref_mic)
        )
        reference = track.stems[:, :, ref_mic]
        mixture = track.mix.T

        metrics = compute_metrics(
            reference,
            separation.estimate_time,
            mixture,
            track.sample_rate,
            filter_length=int(task.evaluation.filter_length),
            frame_cfg=task.evaluation.frame,
            permutation_strategy=_resolve_permutation_strategy(
                task.method.params,
                default_filter_length=int(task.evaluation.filter_length),
            ),
        )

        frame_cfg = task.evaluation.frame
        hop_sec = frame_cfg.hop_sec
        if hop_sec is None:
            hop_sec = frame_cfg.window_sec
        hop_sec = float(hop_sec) if hop_sec is not None else None

        return ExperimentOutput(
            task=task,
            duration=track.duration,
            elapsed=elapsed,
            real_time_factor=rtf,
            success=True,
            metrics=metrics,
            separation=separation.estimate_time,
            method_metadata=dict(separation.metadata),
            frame_hop_sec=hop_sec,
            sample_rate=track.sample_rate,
            permutation=np.array(metrics.permutation, copy=True)
            if metrics.permutation is not None
            else None,
            stem_names=list(task.stem_names),
            error=None,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        elapsed = time.perf_counter() - start
        LOGGER.exception(
            "Task failed | method=%s track=%s error=%s",
            task.method.id,
            task.track_id,
            exc,
        )
        duration = track.duration if track is not None else 0.0
        method_metadata = dict(separation.metadata) if separation is not None else {}
        return ExperimentOutput(
            task=task,
            duration=duration,
            elapsed=elapsed,
            real_time_factor=math.nan,
            success=False,
            metrics=None,
            separation=None,
            method_metadata=method_metadata,
            frame_hop_sec=None,
            sample_rate=track.sample_rate if track is not None else 0,
            permutation=None,
            stem_names=list(task.stem_names),
            error=str(exc),
        )


def write_outputs(
    outputs: Iterable[ExperimentOutput],
    run_root: Path,
    *,
    save_framewise: bool,
    summary_precision: int,
    save_audio: bool,
) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    results_path = run_root / "results.jsonl"

    with results_path.open("w", encoding="utf-8") as jsonl:
        for output in outputs:
            method_dir = run_root / output.task.method.id / output.task.track_id
            method_dir.mkdir(parents=True, exist_ok=True)

            detail: dict[str, object] = {
                "success": output.success,
                "method_metadata": output.method_metadata,
            }
            if output.frame_hop_sec is not None:
                detail["frame_hop_sec"] = output.frame_hop_sec
            detail["stem_names"] = list(output.stem_names)

            frame_path: Path | None = None
            audio_path: Path | None = None
            if output.success and output.metrics is not None:
                detail.update(
                    {
                        "sdr_mix": _to_list(output.metrics.sdr_mix, summary_precision),
                        "sdr_est": _to_list(output.metrics.sdr_est, summary_precision),
                        "sir_est": _to_list(output.metrics.sir_est, summary_precision),
                        "sar_est": _to_list(output.metrics.sar_est, summary_precision),
                        "permutation": output.metrics.permutation.tolist(),
                    }
                )
                if save_framewise and output.metrics.framewise:
                    frame_path = method_dir / "frame_metrics.npz"
                    framewise = output.metrics.framewise
                    framewise_kwargs = {
                        name: np.asarray(values) for name, values in framewise.items()
                    }
                    np.savez_compressed(
                        frame_path,
                        allow_pickle=False,
                        **framewise_kwargs,
                    )
                    detail["frame_metrics_file"] = frame_path.name
            else:
                detail["error"] = output.error

            if save_audio and output.success and output.separation is not None:
                audio = np.array(output.separation, copy=False)
                perm = output.permutation
                if perm is not None and audio.shape[0] == perm.shape[0]:
                    audio = audio[perm, :]
                audio_path = method_dir / "estimate.wav"
                sf.write(audio_path, audio.T, output.sample_rate)
                detail["artifacts"] = {"estimate_audio": audio_path.name}

            detail_path = method_dir / "detail.json"
            with detail_path.open("w", encoding="utf-8") as fh:
                json.dump(_clean_json(detail), fh, indent=2)

            record = output.to_summary_record(
                run_root,
                detail_path=detail_path,
                frame_path=frame_path,
                audio_path=audio_path,
            )
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

    return results_path


def generate_run_directory(base: Path, *, overwrite: bool) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"run_{timestamp}"
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=overwrite)
    return run_dir


def _to_list(array: np.ndarray, precision: int) -> list[float | None]:
    values: list[float | None] = []
    for item in np.ravel(array):
        if math.isnan(item) or math.isinf(item):
            values.append(None)
        else:
            values.append(round(float(item), precision))
    return values


def _clean_json(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def _resolve_permutation_strategy(
    method_params: Mapping[str, object],
    *,
    default_filter_length: int,
) -> PermutationStrategy:
    strategy_obj = method_params.get("permutation_strategy")
    if isinstance(strategy_obj, PermutationStrategy):
        return strategy_obj

    name = str(method_params.get("permutation", "bss_eval")).lower()
    if name in {"bss_eval", "default"}:
        return BssEvalPermutationStrategy(filter_length=default_filter_length)
    if name in {"identity", "none"}:
        return IdentityPermutationStrategy()
    raise ValueError(f"Unknown permutation strategy: {name}")


def _coerce_int(value: object, *, name: str) -> int:
    if isinstance(value, (int, float, np.integer, np.floating, str)):
        return int(value)
    raise TypeError(f"{name} must be int-convertible, got {type(value).__name__}")
