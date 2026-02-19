"""Typed OmegaConf schemas for benchmark configurations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib
from typing import Any, Mapping, TypeVar, cast

try:
    OmegaConf = importlib.import_module("omegaconf").OmegaConf
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "oobss.experiments.config_schema requires 'omegaconf'. Install dependencies with `uv sync`."
    ) from exc


@dataclass
class DatasetConfig:
    """Dataset configuration schema."""

    type: str = "torchrir_dynamic"
    root: str = "outputs/cmu_arctic_torchrir_dynamic_dataset"
    ref_mic: int = 0
    duration_sec: float | None = None


@dataclass
class STFTConfig:
    """STFT configuration schema."""

    fft_size: int = 2048
    hop_size: int = 1024
    window: str = "hann"


@dataclass
class FrameEvalConfig:
    """Frame-wise evaluation options."""

    window_sec: float = 5.0
    hop_sec: float | None = None
    compute_permutation: bool = True
    scaling: bool = True


@dataclass
class RealtimeEvalConfig:
    """Realtime evaluation options."""

    track_offset_sec: float = 0.0


@dataclass
class EvaluationConfig:
    """Evaluation configuration schema."""

    filter_length: int = 512
    ref_mic: int = 0
    frame: FrameEvalConfig = field(default_factory=FrameEvalConfig)
    realtime: RealtimeEvalConfig = field(default_factory=RealtimeEvalConfig)


@dataclass
class MetricsConfig:
    """Metrics and persistence configuration schema."""

    save_framewise: bool = True
    frame_format: str = "npz"
    summary_precision: int = 6


@dataclass
class RuntimeConfig:
    """Runtime execution configuration schema."""

    output_dir: str = "outputs/dataset_benchmark"
    workers: int = 1
    log_level: str = "INFO"
    seed: int | None = None
    dry_run: bool = False
    overwrite: bool = False
    reports_dir: str | None = None


@dataclass
class ExperimentsConfig:
    """Experiment-wide grid configuration schema."""

    grid: dict[str, dict[str, list[Any]]] = field(default_factory=dict)


@dataclass
class CommonExperimentConfig:
    """Top-level benchmark configuration schema."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    stft: STFTConfig = field(default_factory=STFTConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    experiments: ExperimentsConfig = field(default_factory=ExperimentsConfig)


TSchema = TypeVar("TSchema")


def _decode_schema(
    data: Mapping[str, object],
    schema: type[TSchema],
) -> TSchema:
    base = OmegaConf.structured(schema)
    loaded = OmegaConf.create(dict(data))
    merged = OmegaConf.merge(base, loaded)
    decoded = OmegaConf.to_object(merged)
    if not isinstance(decoded, schema):
        raise TypeError(f"Failed to decode config as {schema.__name__}")
    return cast(TSchema, decoded)


def parse_common_config(data: Mapping[str, object]) -> CommonExperimentConfig:
    """Decode a mapping into :class:`CommonExperimentConfig`."""
    return _decode_schema(data, CommonExperimentConfig)


def parse_evaluation_config(data: Mapping[str, object]) -> EvaluationConfig:
    """Decode a mapping into :class:`EvaluationConfig`."""
    return _decode_schema(data, EvaluationConfig)


def parse_metrics_config(data: Mapping[str, object]) -> MetricsConfig:
    """Decode a mapping into :class:`MetricsConfig`."""
    return _decode_schema(data, MetricsConfig)


def parse_stft_config(data: Mapping[str, object]) -> STFTConfig:
    """Decode a mapping into :class:`STFTConfig`."""
    return _decode_schema(data, STFTConfig)


def common_config_to_dict(config: CommonExperimentConfig) -> dict[str, Any]:
    """Convert :class:`CommonExperimentConfig` to plain dictionary."""
    return asdict(config)
