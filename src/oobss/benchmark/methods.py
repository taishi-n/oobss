"""Separation method runners used by large-scale benchmarks.

Usage
-----
This module is typically used through CLI wrappers:

- ``uv run python examples/benchmark_dataset.py --dry-run --sample-limit 2``
- ``uv run python -m oobss.benchmark.cli run --dry-run``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import importlib
from typing import Callable, Mapping, Protocol, TypeVar, cast

import numpy as np
from scipy.signal import ShortTimeFFT

from oobss.signal import STFTPlan, build_stft
from oobss.separators.core import (
    ComponentAssignmentStrategy,
    CovarianceUpdateStrategy,
    NormalizationRequest,
    NormalizationStrategy,
    OnlineNMFUpdateStrategy,
    SeparationOutput,
    SourceModelStrategy,
    SpatialUpdateStrategy,
    StreamRequest,
)
from oobss.separators import AuxIVA, ILRMA, OnlineAuxIVA, OnlineILRMA, OnlineISNMF
from oobss.separators.strategies import (
    default_strategy_registry,
    IdentityNormalization,
    ProjectionBackNormalization,
    StrategyRegistry,
)
from oobss.separators.utils import demix

try:
    OmegaConf = importlib.import_module("omegaconf").OmegaConf
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "oobss.benchmark.methods requires 'omegaconf'. Install dependencies with `uv sync`."
    ) from exc

TParams = TypeVar("TParams")


@dataclass(frozen=True)
class BatchAuxIVAParams:
    n_iter: int = 30
    spatial: str = "ip1"
    source: str = "gauss"
    covariance: str = "batch"
    ref_mic: int = 0
    normalization: str = "projection_back"
    seed: int | None = None


@dataclass(frozen=True)
class BatchILRMAParams:
    n_iter: int = 100
    spatial: str = "ip1"
    source: str = "ilrma_nmf"
    covariance: str = "batch"
    n_basis: int = 10
    ref_mic: int = 0
    normalization: str = "projection_back"
    seed: int | None = None
    return_nmf_factors: bool = False


@dataclass(frozen=True)
class BlockBatchAuxIVAParams:
    block_size: int = 200
    overlap: int = 0
    n_iter: int = 50
    spatial: str = "ip1"
    source: str = "gauss"
    covariance: str = "batch"
    ref_mic: int = 0
    normalization: str = "projection_back"
    seed: int | None = None


@dataclass(frozen=True)
class BlockBatchILRMAParams:
    block_size: int = 200
    overlap: int = 0
    n_iter: int = 50
    spatial: str = "ip1"
    source: str = "ilrma_nmf"
    covariance: str = "batch"
    n_basis: int = 10
    ref_mic: int = 0
    normalization: str = "projection_back"
    seed: int | None = None
    return_nmf_factors: bool = False


@dataclass(frozen=True)
class OnlineAuxIVAParams:
    forget: float = 0.99
    inner_iter: int = 5
    cov_scale: float = 1.0e-6
    ref_mic: int = 0
    spatial: str = "ip1"
    source: str = "gauss"
    covariance: str = "ema"


@dataclass(frozen=True)
class OnlineILRMAParams:
    n_bases: int = 10
    beta: int = 1
    forget: float = 0.99
    inner_iter: int = 5
    cov_scale: float = 1.0e-6
    ref_mic: int = 0
    keep_h: bool = False
    random_state: int | None = None
    return_nmf_factors: bool = False
    spatial: str = "ip1"
    covariance: str = "ema"
    nmf: str = "multiplicative"


@dataclass(frozen=True)
class OnlineISNMFParams:
    n_components: int = 8
    ref_mic: int = 0
    analysis_channel: int = 0
    beta: int = 2
    forget: float = 0.99
    inner_iter: int = 20
    keep_h: bool | str = "auto"
    eps: float = 1.0e-12
    n_sources: int | None = None
    component_to_source: list[int] | None = None
    random_state: int | None = None
    return_nmf_factors: bool = False
    nmf: str = "multiplicative"
    assignment: str = "modulo"


MethodRunnerFn = Callable[
    [np.ndarray, Mapping[str, object], STFTPlan, int], SeparationOutput
]

_METHOD_PARAM_SCHEMAS: dict[str, type[object]] = {
    "batch_auxiva": BatchAuxIVAParams,
    "batch_ilrma": BatchILRMAParams,
    "blockbatch_auxiva": BlockBatchAuxIVAParams,
    "blockbatch_ilrma": BlockBatchILRMAParams,
    "online_auxiva": OnlineAuxIVAParams,
    "online_ilrma": OnlineILRMAParams,
    "online_isnmf": OnlineISNMFParams,
}


class MethodRunner(Protocol):
    """Protocol for benchmark method runners."""

    def run(
        self,
        mix: np.ndarray,
        params: Mapping[str, object],
        stft_cfg: STFTPlan,
        sample_rate: int,
    ) -> SeparationOutput:
        """Execute separation and return result."""


class BaseMethodRunner(ABC):
    """Base class for typed benchmark method runners."""

    method_type: str

    @abstractmethod
    def run(
        self,
        mix: np.ndarray,
        params: Mapping[str, object],
        stft_cfg: STFTPlan,
        sample_rate: int,
    ) -> SeparationOutput:
        """Execute separation and return result."""


class FunctionalMethodRunner(BaseMethodRunner):
    """Adapter that wraps a plain callable into ``MethodRunner``."""

    method_type = "callable"

    def __init__(self, fn: MethodRunnerFn) -> None:
        self._fn = fn

    def run(
        self,
        mix: np.ndarray,
        params: Mapping[str, object],
        stft_cfg: STFTPlan,
        sample_rate: int,
    ) -> SeparationOutput:
        return self._fn(mix, params, stft_cfg, sample_rate)


def _decode_params(raw_params: Mapping[str, object], schema: type[TParams]) -> TParams:
    base = OmegaConf.structured(schema)
    overrides = OmegaConf.create(dict(raw_params))
    merged = OmegaConf.merge(base, overrides)
    decoded = OmegaConf.to_object(merged)
    if not isinstance(decoded, schema):
        raise TypeError(f"Failed to decode params as {schema.__name__}")
    return cast(TParams, decoded)


def validate_builtin_method_params(
    method_type: str, params: Mapping[str, object]
) -> None:
    """Validate params for built-in method types.

    Unknown method types are ignored so external plugin types remain supported.
    """
    schema = _METHOD_PARAM_SCHEMAS.get(method_type)
    if schema is None:
        return
    _decode_params(params, schema)


def _channel_first_spectrogram(stft: ShortTimeFFT, mix: np.ndarray) -> np.ndarray:
    """Compute STFT returning array shaped ``(n_mic, n_freq, n_frame)``."""
    return stft.stft(mix.T)


def _frame_first_spectrogram(stft: ShortTimeFFT, mix: np.ndarray) -> np.ndarray:
    """Return STFT shaped ``(n_frame, n_freq, n_mic)``."""
    spec = _channel_first_spectrogram(stft, mix)
    return spec.transpose(2, 1, 0)


def _freq_first_spectrogram(stft: ShortTimeFFT, mix: np.ndarray) -> np.ndarray:
    """Return STFT shaped ``(n_freq, n_mic, n_frame)``."""
    spec = _channel_first_spectrogram(stft, mix)
    return spec.transpose(1, 0, 2)


def _resolve_normalization_strategy(
    normalization_name: str,
    *,
    ref_mic: int,
) -> NormalizationStrategy:
    strategy_name = normalization_name.lower()
    if strategy_name in {"projection_back", "projback", "pb"}:
        return ProjectionBackNormalization(ref_mic=ref_mic)
    if strategy_name in {"identity", "none"}:
        return IdentityNormalization()
    raise ValueError(f"Unknown normalization strategy: {normalization_name}")


def _validate_blockbatch_schedule(
    *,
    block_size: int,
    overlap: int,
    n_iter: int,
    method_name: str,
) -> None:
    """Validate block scheduling and iteration settings for block-batch runners."""
    if block_size <= 0:
        raise ValueError(f"{method_name}: block_size must be > 0, got {block_size}")
    if overlap < 0 or overlap >= block_size:
        raise ValueError(
            f"{method_name}: overlap must satisfy 0 <= overlap < block_size, "
            f"got overlap={overlap}, block_size={block_size}"
        )
    if n_iter < 0:
        raise ValueError(f"{method_name}: n_iter must be >= 0, got {n_iter}")


def _strategy_registry() -> StrategyRegistry:
    return default_strategy_registry()


def _create_spatial_strategy(
    registry: StrategyRegistry, name: str
) -> SpatialUpdateStrategy:
    return cast(SpatialUpdateStrategy, registry.create("spatial", name))


def _create_source_strategy(
    registry: StrategyRegistry,
    name: str,
) -> SourceModelStrategy:
    return cast(SourceModelStrategy, registry.create("source", name))


def _create_covariance_strategy(
    registry: StrategyRegistry,
    name: str,
    *,
    alpha: float | None = None,
) -> CovarianceUpdateStrategy:
    params = {} if alpha is None else {"alpha": alpha}
    return cast(
        CovarianceUpdateStrategy, registry.create("covariance", name, params=params)
    )


def _create_nmf_strategy(
    registry: StrategyRegistry, name: str
) -> OnlineNMFUpdateStrategy:
    return cast(OnlineNMFUpdateStrategy, registry.create("nmf", name))


def _create_assignment_strategy(
    registry: StrategyRegistry,
    name: str,
) -> ComponentAssignmentStrategy:
    return cast(ComponentAssignmentStrategy, registry.create("assignment", name))


class BatchAuxIVARunner(BaseMethodRunner):
    """Runner for batch AuxIVA."""

    method_type = "batch_auxiva"

    def run(
        self,
        mix: np.ndarray,
        params: Mapping[str, object],
        stft_cfg: STFTPlan,
        sample_rate: int,
    ) -> SeparationOutput:
        cfg = _decode_params(params, BatchAuxIVAParams)
        stft = build_stft(stft_cfg, sample_rate)
        obs = _frame_first_spectrogram(stft, mix)
        strategy_registry = _strategy_registry()

        separator = AuxIVA(
            obs,
            spatial=_create_spatial_strategy(strategy_registry, cfg.spatial),
            source=_create_source_strategy(strategy_registry, cfg.source),
            covariance=_create_covariance_strategy(strategy_registry, cfg.covariance),
        )
        output = separator.fit_transform_tf(obs, n_iter=cfg.n_iter)
        estimate_tf = output.estimate_tf
        if estimate_tf is None:
            raise ValueError("BatchAuxIVA runner expected TF estimate output.")

        normalization = _resolve_normalization_strategy(
            cfg.normalization,
            ref_mic=cfg.ref_mic,
        )
        projected = normalization.apply(
            NormalizationRequest(
                estimate=estimate_tf,
                observations=obs,
                demix_filter=separator.demix_filter,
            )
        )
        est = np.real(stft.istft(projected.transpose(2, 1, 0)))
        return SeparationOutput(
            estimate_time=est[:, : mix.shape[0]],
            metadata={
                "reference_mic": cfg.ref_mic,
                "iterations": cfg.n_iter,
            },
        )


class BatchILRMARunner(BaseMethodRunner):
    """Runner for batch ILRMA."""

    method_type = "batch_ilrma"

    def run(
        self,
        mix: np.ndarray,
        params: Mapping[str, object],
        stft_cfg: STFTPlan,
        sample_rate: int,
    ) -> SeparationOutput:
        cfg = _decode_params(params, BatchILRMAParams)
        stft = build_stft(stft_cfg, sample_rate)
        obs = _frame_first_spectrogram(stft, mix)
        strategy_registry = _strategy_registry()
        separator = ILRMA(
            obs,
            n_basis=cfg.n_basis,
            random_state=cfg.seed,
            spatial=_create_spatial_strategy(strategy_registry, cfg.spatial),
            source=_create_source_strategy(strategy_registry, cfg.source),
            covariance=_create_covariance_strategy(strategy_registry, cfg.covariance),
        )
        output = separator.fit_transform_tf(obs, n_iter=cfg.n_iter)
        estimate_tf = output.estimate_tf
        if estimate_tf is None:
            raise ValueError("BatchILRMA runner expected TF estimate output.")

        normalization = _resolve_normalization_strategy(
            cfg.normalization,
            ref_mic=cfg.ref_mic,
        )
        projected = normalization.apply(
            NormalizationRequest(
                estimate=estimate_tf,
                observations=obs,
                demix_filter=separator.demix_filter,
            )
        )
        est = np.real(stft.istft(projected.transpose(2, 1, 0)))
        extras: dict[str, object] = {"iterations": cfg.n_iter, "n_basis": cfg.n_basis}
        if cfg.return_nmf_factors:
            extras["nmf_factors"] = {
                "type": "ilrma",
                "basis": separator.basis.tolist(),
                "activations": separator.activ.tolist(),
            }

        metadata: dict[str, object] = {"reference_mic": cfg.ref_mic}
        metadata.update(extras)
        return SeparationOutput(
            estimate_time=est[:, : mix.shape[0]],
            metadata=metadata,
        )


class BlockBatchAuxIVARunner(BaseMethodRunner):
    """Runner for block-batch AuxIVA with overlap-aware TF aggregation.

    The mixture is transformed to frame-first STFT ``(n_frame, n_freq, n_mic)``,
    then processed with sliding blocks. For each block:

    - frames ``[start:end)`` are separated with AuxIVA.
    - the previous block's demixing matrix is reused as warm start.
    - projection-back (or configured normalization) is applied.
    - normalized TF estimates are accumulated into a running buffer.

    Unlike the previous implementation, trailing blocks shorter than
    ``block_size`` are still processed, so tail frames are never left as zeros.

    When ``overlap > 0``, overlapping frame estimates are merged by weighted
    averaging with unit weights per block contribution. This avoids hard
    overwrite behavior at block boundaries.

    Constraints
    -----------
    - ``block_size > 0``
    - ``0 <= overlap < block_size``
    - ``n_iter >= 0``
    """

    method_type = "blockbatch_auxiva"

    def run(
        self,
        mix: np.ndarray,
        params: Mapping[str, object],
        stft_cfg: STFTPlan,
        sample_rate: int,
    ) -> SeparationOutput:
        """Separate one mixture using block-batch AuxIVA.

        Parameters
        ----------
        mix:
            Time-domain mixture of shape ``(n_samples, n_mic)``.
        params:
            Decoded as :class:`BlockBatchAuxIVAParams`.
        stft_cfg:
            STFT analysis/synthesis plan.
        sample_rate:
            Sampling rate in Hz.

        Returns
        -------
        SeparationOutput
            Time-domain estimate in ``estimate_time`` with shape
            ``(n_src, n_samples)`` where ``n_src == n_mic`` for determined
            separation.
        """
        cfg = _decode_params(params, BlockBatchAuxIVAParams)
        _validate_blockbatch_schedule(
            block_size=cfg.block_size,
            overlap=cfg.overlap,
            n_iter=cfg.n_iter,
            method_name=self.method_type,
        )
        stft = build_stft(stft_cfg, sample_rate)
        obs = _frame_first_spectrogram(stft, mix)
        n_frame = obs.shape[0]
        result_accum = np.zeros_like(obs)
        weight_accum = np.zeros((n_frame, 1, 1), dtype=np.float64)

        normalization = _resolve_normalization_strategy(
            cfg.normalization,
            ref_mic=cfg.ref_mic,
        )
        strategy_registry = _strategy_registry()
        start = 0
        prev_demix: np.ndarray | None = None
        while start < n_frame:
            end = min(start + cfg.block_size, n_frame)
            block = obs[start:end]
            separator = AuxIVA(
                block,
                spatial=_create_spatial_strategy(strategy_registry, cfg.spatial),
                source=_create_source_strategy(strategy_registry, cfg.source),
                covariance=_create_covariance_strategy(
                    strategy_registry, cfg.covariance
                ),
            )
            if prev_demix is not None:
                separator.demix_filter[...] = prev_demix
                separator.estimated = demix(block, separator.demix_filter)
            separator.run(cfg.n_iter)
            prev_demix = separator.demix_filter.copy()
            projected = normalization.apply(
                NormalizationRequest(
                    estimate=separator.estimated,
                    observations=block,
                    demix_filter=separator.demix_filter,
                )
            )
            result_accum[start:end] += projected
            weight_accum[start:end] += 1.0
            if end == n_frame:
                break
            start = end - cfg.overlap if cfg.overlap else end

        result = result_accum / np.maximum(weight_accum, 1.0e-12)
        est = np.real(stft.istft(result.transpose(2, 1, 0)))
        return SeparationOutput(
            estimate_time=est[:, : mix.shape[0]],
            metadata={
                "reference_mic": cfg.ref_mic,
                "iterations": cfg.n_iter,
                "block_size": cfg.block_size,
                "overlap": cfg.overlap,
            },
        )


class BlockBatchILRMARunner(BaseMethodRunner):
    """Runner for block-batch ILRMA with state reuse and overlap averaging.

    Processing is done in frame-first STFT blocks. Each block runs ILRMA and
    contributes its normalized TF estimate to an accumulation buffer.

    State transfer across blocks:
    - demixing matrix ``W`` is reused as warm start.
    - NMF basis/activation are reused when available.
    - activation is truncated to the current block length when needed.

    Trailing short blocks (``len < block_size``) are processed instead of being
    skipped, preventing tail silence in the reconstructed waveform.

    Overlapping blocks are merged by weighted averaging with unit per-block
    weights at each frame.

    Constraints
    -----------
    - ``block_size > 0``
    - ``0 <= overlap < block_size``
    - ``n_iter >= 0``
    """

    method_type = "blockbatch_ilrma"

    def run(
        self,
        mix: np.ndarray,
        params: Mapping[str, object],
        stft_cfg: STFTPlan,
        sample_rate: int,
    ) -> SeparationOutput:
        """Separate one mixture using block-batch ILRMA.

        Parameters
        ----------
        mix:
            Time-domain mixture of shape ``(n_samples, n_mic)``.
        params:
            Decoded as :class:`BlockBatchILRMAParams`.
        stft_cfg:
            STFT analysis/synthesis plan.
        sample_rate:
            Sampling rate in Hz.

        Returns
        -------
        SeparationOutput
            Time-domain estimate in ``estimate_time`` with shape
            ``(n_src, n_samples)`` where ``n_src == n_mic`` for determined
            separation.
        """
        cfg = _decode_params(params, BlockBatchILRMAParams)
        _validate_blockbatch_schedule(
            block_size=cfg.block_size,
            overlap=cfg.overlap,
            n_iter=cfg.n_iter,
            method_name=self.method_type,
        )
        stft = build_stft(stft_cfg, sample_rate)
        obs = _frame_first_spectrogram(stft, mix)
        n_frame = obs.shape[0]
        result_accum = np.zeros_like(obs)
        weight_accum = np.zeros((n_frame, 1, 1), dtype=np.float64)

        rng = np.random.default_rng(cfg.seed)
        prev_demix: np.ndarray | None = None
        prev_basis: np.ndarray | None = None
        prev_activ: np.ndarray | None = None

        normalization = _resolve_normalization_strategy(
            cfg.normalization,
            ref_mic=cfg.ref_mic,
        )
        strategy_registry = _strategy_registry()
        start = 0
        while start < n_frame:
            end = min(start + cfg.block_size, n_frame)
            block = obs[start:end]
            separator = ILRMA(
                block,
                n_basis=cfg.n_basis,
                rng=rng,
                spatial=_create_spatial_strategy(strategy_registry, cfg.spatial),
                source=_create_source_strategy(strategy_registry, cfg.source),
                covariance=_create_covariance_strategy(
                    strategy_registry, cfg.covariance
                ),
            )
            reused_source_model = False
            if prev_demix is not None:
                separator.demix_filter[...] = prev_demix
                separator.estimated = demix(block, separator.demix_filter)
            if prev_basis is not None:
                separator.basis = prev_basis.copy()
                reused_source_model = True
            if prev_activ is not None:
                if prev_activ.shape[1] >= block.shape[0]:
                    separator.activ = prev_activ[:, : block.shape[0], :].copy()
                    reused_source_model = True
                else:
                    separator.activ = prev_activ.copy()
                    reused_source_model = True
            if reused_source_model:
                separator.source_model = separator.init_source_model()
                separator.loss = separator.calc_loss()
            separator.run(cfg.n_iter)
            prev_demix = separator.demix_filter.copy()
            prev_basis = separator.basis.copy()
            prev_activ = separator.activ.copy()
            projected = normalization.apply(
                NormalizationRequest(
                    estimate=separator.estimated,
                    observations=block,
                    demix_filter=separator.demix_filter,
                )
            )
            result_accum[start:end] += projected
            weight_accum[start:end] += 1.0
            if end == n_frame:
                break
            start = end - cfg.overlap if cfg.overlap else end

        result = result_accum / np.maximum(weight_accum, 1.0e-12)
        est = np.real(stft.istft(result.transpose(2, 1, 0)))
        extras: dict[str, object] = {
            "iterations": cfg.n_iter,
            "block_size": cfg.block_size,
            "overlap": cfg.overlap,
            "n_basis": cfg.n_basis,
        }
        if cfg.return_nmf_factors and prev_basis is not None and prev_activ is not None:
            extras["nmf_factors"] = {
                "type": "ilrma",
                "basis": prev_basis.tolist(),
                "activations": prev_activ.tolist(),
            }

        metadata = {"reference_mic": cfg.ref_mic}
        metadata.update(extras)
        return SeparationOutput(
            estimate_time=est[:, : mix.shape[0]],
            metadata=metadata,
        )


class OnlineAuxIVARunner(BaseMethodRunner):
    """Runner for OnlineAuxIVA."""

    method_type = "online_auxiva"

    def run(
        self,
        mix: np.ndarray,
        params: Mapping[str, object],
        stft_cfg: STFTPlan,
        sample_rate: int,
    ) -> SeparationOutput:
        cfg = _decode_params(params, OnlineAuxIVAParams)
        stft = build_stft(stft_cfg, sample_rate)
        spec = _freq_first_spectrogram(stft, mix)
        n_freq, n_mic, _ = spec.shape
        strategy_registry = _strategy_registry()

        model = OnlineAuxIVA(
            n_mic=n_mic,
            n_freq=n_freq,
            ref_mic=cfg.ref_mic,
            forget=cfg.forget,
            inner_iter=cfg.inner_iter,
            cov_scale=cfg.cov_scale,
            spatial=_create_spatial_strategy(strategy_registry, cfg.spatial),
            source=_create_source_strategy(strategy_registry, cfg.source),
            covariance=_create_covariance_strategy(
                strategy_registry,
                cfg.covariance,
                alpha=cfg.forget,
            ),
        )
        output = model.process_stream_tf(
            spec,
            request=StreamRequest(frame_axis=2, reference_mic=cfg.ref_mic),
        )
        separated = output.estimate_tf
        if separated is None:
            raise ValueError("OnlineAuxIVA runner expected TF estimate output.")

        est = np.real(stft.istft(separated, f_axis=0, t_axis=2))
        return SeparationOutput(
            estimate_time=est[: mix.shape[0], :].T,
            metadata={
                "reference_mic": cfg.ref_mic,
                "inner_iter": cfg.inner_iter,
                "forget": cfg.forget,
            },
        )


class OnlineILRMARunner(BaseMethodRunner):
    """Runner for OnlineILRMA."""

    method_type = "online_ilrma"

    def run(
        self,
        mix: np.ndarray,
        params: Mapping[str, object],
        stft_cfg: STFTPlan,
        sample_rate: int,
    ) -> SeparationOutput:
        cfg = _decode_params(params, OnlineILRMAParams)
        stft = build_stft(stft_cfg, sample_rate)
        spec = _freq_first_spectrogram(stft, mix)
        n_freq, n_mic, _ = spec.shape
        strategy_registry = _strategy_registry()

        model = OnlineILRMA(
            n_mic=n_mic,
            n_freq=n_freq,
            n_bases=cfg.n_bases,
            ref_mic=cfg.ref_mic,
            beta=cfg.beta,
            forget=cfg.forget,
            inner_iter=cfg.inner_iter,
            keep_h=cfg.keep_h,
            cov_scale=cfg.cov_scale,
            random_state=cfg.random_state,
            spatial=_create_spatial_strategy(strategy_registry, cfg.spatial),
            covariance=_create_covariance_strategy(
                strategy_registry,
                cfg.covariance,
                alpha=cfg.forget,
            ),
            nmf=_create_nmf_strategy(strategy_registry, cfg.nmf),
        )
        output = model.process_stream_tf(
            spec,
            request=StreamRequest(frame_axis=2, reference_mic=cfg.ref_mic),
        )
        separated = output.estimate_tf
        if separated is None:
            raise ValueError("OnlineILRMA runner expected TF estimate output.")

        est = np.real(stft.istft(separated, f_axis=0, t_axis=2))
        extras: dict[str, object] = {
            "n_bases": cfg.n_bases,
            "beta": cfg.beta,
            "forget": cfg.forget,
            "inner_iter": cfg.inner_iter,
        }
        if cfg.return_nmf_factors:
            nmf_payload: dict[str, object] = {
                "type": "online_ilrma",
                "basis": model.basis.tolist(),
            }
            if all(len(history) > 0 for history in model._H_store):
                activ = np.stack(
                    [np.stack(history, axis=0) for history in model._H_store], axis=0
                )
                nmf_payload["activations"] = activ.tolist()
            extras["nmf_factors"] = nmf_payload

        metadata = {"reference_mic": cfg.ref_mic}
        metadata.update(extras)
        return SeparationOutput(
            estimate_time=est[: mix.shape[0], :].T,
            metadata=metadata,
        )


class OnlineISNMFRunner(BaseMethodRunner):
    """Runner for OnlineISNMF."""

    method_type = "online_isnmf"

    def run(
        self,
        mix: np.ndarray,
        params: Mapping[str, object],
        stft_cfg: STFTPlan,
        sample_rate: int,
    ) -> SeparationOutput:
        cfg = _decode_params(params, OnlineISNMFParams)
        stft = build_stft(stft_cfg, sample_rate)
        spec = _freq_first_spectrogram(stft, mix)
        n_freq, n_mic, _ = spec.shape

        if cfg.analysis_channel < 0 or cfg.analysis_channel >= n_mic:
            raise ValueError(
                f"analysis_channel={cfg.analysis_channel} out of range for n_mic={n_mic}"
            )
        n_sources = cfg.n_sources if cfg.n_sources is not None else n_mic
        if cfg.n_components < n_sources:
            raise ValueError(
                f"n_components ({cfg.n_components}) must be >= n_sources ({n_sources})"
            )
        component_to_source = (
            None
            if cfg.component_to_source is None
            else np.asarray(cfg.component_to_source, dtype=np.int64)
        )
        strategy_registry = _strategy_registry()

        model = OnlineISNMF(
            n_components=cfg.n_components,
            n_features=n_freq,
            beta=cfg.beta,
            forget=cfg.forget,
            inner_iter=cfg.inner_iter,
            keep_h=cfg.keep_h,
            eps=cfg.eps,
            n_sources=n_sources,
            component_to_source=component_to_source,
            random_state=cfg.random_state,
            nmf=_create_nmf_strategy(strategy_registry, cfg.nmf),
            assignment=_create_assignment_strategy(strategy_registry, cfg.assignment),
        )

        x = spec[:, cfg.analysis_channel, :]
        output = model.process_stream_tf(
            x,
            request=StreamRequest(
                frame_axis=1,
                n_sources=n_sources,
                component_to_source=component_to_source,
                reference_mic=cfg.ref_mic,
            ),
        )
        separated = output.estimate_tf
        if separated is None:
            raise ValueError("OnlineISNMF runner expected TF estimate output.")

        est = np.real(stft.istft(separated.transpose(1, 0, 2), f_axis=0, t_axis=2))
        extras: dict[str, object] = {
            "n_sources": n_sources,
            "analysis_channel": cfg.analysis_channel,
            "n_components": cfg.n_components,
            "beta": cfg.beta,
            "inner_iter": cfg.inner_iter,
        }
        if cfg.return_nmf_factors:
            nmf_payload: dict[str, object] = {
                "type": "online_isnmf",
                "basis": model.W.tolist(),
                "observed_magnitude": np.abs(x).tolist(),
            }
            if len(model._H_store) > 0:
                activation = np.stack(model._H_store, axis=0).T  # (K, T)
                nmf_payload["activations"] = activation.tolist()
            extras["nmf_factors"] = nmf_payload

        metadata = {"reference_mic": cfg.ref_mic}
        metadata.update(extras)
        return SeparationOutput(
            estimate_time=est[: mix.shape[0], :].T,
            metadata=metadata,
        )


@dataclass
class MethodRunnerRegistry:
    """Registry that resolves method IDs to runner instances."""

    _runners: dict[str, MethodRunner] = field(default_factory=dict)

    def register(
        self,
        method_type: str,
        runner: MethodRunner | MethodRunnerFn,
        *,
        overwrite: bool = False,
    ) -> None:
        if not overwrite and method_type in self._runners:
            raise ValueError(f"Method runner already registered: {method_type}")
        if callable(runner) and not hasattr(runner, "run"):
            self._runners[method_type] = FunctionalMethodRunner(
                cast(MethodRunnerFn, runner)
            )
        else:
            self._runners[method_type] = cast(MethodRunner, runner)

    def resolve(self, method_type: str) -> MethodRunner:
        try:
            return self._runners[method_type]
        except KeyError as exc:
            available = ", ".join(sorted(self._runners))
            raise ValueError(
                f"Unknown method type: {method_type}. Available methods: {available}"
            ) from exc

    def available(self) -> list[str]:
        return sorted(self._runners)


def default_method_runner_registry() -> MethodRunnerRegistry:
    """Create the built-in method runner registry."""
    registry = MethodRunnerRegistry()
    registry.register("batch_auxiva", BatchAuxIVARunner())
    registry.register("batch_ilrma", BatchILRMARunner())
    registry.register("blockbatch_auxiva", BlockBatchAuxIVARunner())
    registry.register("blockbatch_ilrma", BlockBatchILRMARunner())
    registry.register("online_auxiva", OnlineAuxIVARunner())
    registry.register("online_ilrma", OnlineILRMARunner())
    registry.register("online_isnmf", OnlineISNMFRunner())
    return registry
