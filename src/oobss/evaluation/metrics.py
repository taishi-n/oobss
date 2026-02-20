"""Metric computation for blind source separation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from fast_bss_eval import bss_eval_sources

from oobss.separators.core import PermutationRequest, PermutationStrategy
from oobss.separators.eval_utils import summarize_framewise_si_sdr
from oobss.separators.strategies import BssEvalPermutationStrategy

BssevalMetrics = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class FrameEvalLike(Protocol):
    """Structural type for frame-wise evaluation options."""

    window_sec: float
    hop_sec: float | None
    compute_permutation: bool
    scaling: bool


def normalize_framewise_metrics(
    framewise: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray] | None:
    """Normalize frame-wise metric keys while keeping backward compatibility.

    The framewise SI-SDR utilities return canonical keys such as
    ``mean_si_sdr`` / ``mean_si_sdr_imp``. Some downstream code expects alias
    keys with ``*_channels`` suffix. This function guarantees both key styles
    are available and fills missing aggregate keys from raw frame matrices.

    Parameters
    ----------
    framewise:
        Framewise metric dictionary, typically produced by
        :func:`oobss.separators.eval_utils.summarize_framewise_si_sdr`.
        If ``None``, ``None`` is returned.

    Returns
    -------
    dict[str, np.ndarray] | None
        A normalized dictionary where numeric values are converted to numpy
        arrays and the following aliases are guaranteed when source keys exist:
        - ``mean_si_sdr_channels``, ``median_si_sdr_channels``
        - ``mean_si_sdr_mix_channels``, ``median_si_sdr_mix_channels``
        - ``mean_si_sdr_imp_channels``, ``median_si_sdr_imp_channels``
    """
    if framewise is None:
        return None

    normalized = {key: np.asarray(value) for key, value in framewise.items()}

    if "si_sdr" in normalized:
        si_sdr = np.asarray(normalized["si_sdr"])
        if "mean_si_sdr" not in normalized:
            normalized["mean_si_sdr"] = np.nanmean(si_sdr, axis=1)
        if "median_si_sdr" not in normalized:
            normalized["median_si_sdr"] = np.nanmedian(si_sdr, axis=1)

    if "mean_si_sdr" in normalized and "mean_si_sdr_channels" not in normalized:
        normalized["mean_si_sdr_channels"] = np.asarray(normalized["mean_si_sdr"])
    if "median_si_sdr" in normalized and "median_si_sdr_channels" not in normalized:
        normalized["median_si_sdr_channels"] = np.asarray(normalized["median_si_sdr"])

    if "si_sdr_mix" in normalized:
        si_sdr_mix = np.asarray(normalized["si_sdr_mix"])
        if "mean_si_sdr_mix" not in normalized:
            normalized["mean_si_sdr_mix"] = np.nanmean(si_sdr_mix, axis=1)
        if "median_si_sdr_mix" not in normalized:
            normalized["median_si_sdr_mix"] = np.nanmedian(si_sdr_mix, axis=1)
        if "mean_si_sdr_mix_channels" not in normalized:
            normalized["mean_si_sdr_mix_channels"] = np.asarray(
                normalized["mean_si_sdr_mix"]
            )
        if "median_si_sdr_mix_channels" not in normalized:
            normalized["median_si_sdr_mix_channels"] = np.asarray(
                normalized["median_si_sdr_mix"]
            )

    if "si_sdr_imp" in normalized:
        si_sdr_imp = np.asarray(normalized["si_sdr_imp"])
        if "mean_si_sdr_imp" not in normalized:
            normalized["mean_si_sdr_imp"] = np.nanmean(si_sdr_imp, axis=1)
        if "median_si_sdr_imp" not in normalized:
            normalized["median_si_sdr_imp"] = np.nanmedian(si_sdr_imp, axis=1)
        if "mean_si_sdr_imp_channels" not in normalized:
            normalized["mean_si_sdr_imp_channels"] = np.asarray(
                normalized["mean_si_sdr_imp"]
            )
        if "median_si_sdr_imp_channels" not in normalized:
            normalized["median_si_sdr_imp_channels"] = np.asarray(
                normalized["median_si_sdr_imp"]
            )

    return normalized


@dataclass
class MetricsBundle:
    sdr_mix: np.ndarray
    sdr_est: np.ndarray
    sir_est: np.ndarray
    sar_est: np.ndarray
    permutation: np.ndarray
    framewise: dict[str, np.ndarray] | None

    def to_summary(self) -> dict[str, object]:
        sdr_imp = np.asarray(self.sdr_est) - np.asarray(self.sdr_mix)
        stats = {
            "sdr_mix_mean": float(np.nanmean(self.sdr_mix)),
            "sdr_est_mean": float(np.nanmean(self.sdr_est)),
            "sdr_imp_mean": float(np.nanmean(sdr_imp)),
            "sdr_est_median": float(np.nanmedian(self.sdr_est)),
            "sdr_imp_median": float(np.nanmedian(sdr_imp)),
            "sir_mean": float(np.nanmean(self.sir_est)),
            "sar_mean": float(np.nanmean(self.sar_est)),
        }
        stats["sdr_mix_channels"] = [float(v) for v in np.ravel(self.sdr_mix)]
        stats["sdr_est_channels"] = [float(v) for v in np.ravel(self.sdr_est)]
        stats["sdr_imp_channels"] = [float(v) for v in np.ravel(sdr_imp)]
        stats["sir_channels"] = [float(v) for v in np.ravel(self.sir_est)]
        stats["sar_channels"] = [float(v) for v in np.ravel(self.sar_est)]
        if self.framewise:
            for key in (
                "mean_si_sdr",
                "median_si_sdr",
                "mean_si_sdr_imp",
                "median_si_sdr_imp",
                "mean_si_sdr_mix",
                "median_si_sdr_mix",
            ):
                if key in self.framewise:
                    stats[key] = [float(v) for v in np.ravel(self.framewise[key])]
        return stats


def compute_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    mixture: np.ndarray,
    sample_rate: int,
    *,
    filter_length: int,
    frame_cfg: FrameEvalLike | None,
    compute_permutation: bool = True,
    permutation_strategy: PermutationStrategy | None = None,
) -> MetricsBundle:
    """Compute batch and optional frame-wise separation metrics.

    Parameters
    ----------
    reference:
        Reference signals with shape ``(n_ref, n_samples)``.
    estimate:
        Estimated signals with shape ``(n_est, n_samples)``.
    mixture:
        Mixture baseline with shape ``(n_mix, n_samples)`` or ``(n_samples,)``.
        For reference-mic baseline evaluation, pass a single channel.
    compute_permutation:
        Whether to allow permutation solving in ``fast_bss_eval``. This should
        be enabled for under/over-determined cases where channel counts differ.
    """
    if reference.ndim != 2 or estimate.ndim != 2:
        raise ValueError("reference and estimate must be 2-D arrays (n_src, n_samples)")
    if reference.shape[-1] != estimate.shape[-1]:
        raise ValueError(
            f"reference and estimate must share sample length, got {reference.shape} and {estimate.shape}"
        )
    mixture_eval = np.asarray(mixture)
    if mixture_eval.ndim == 1:
        mixture_eval = mixture_eval[None, :]
    if mixture_eval.ndim != 2:
        raise ValueError("mixture must be 1-D or 2-D array")
    if mixture_eval.shape[-1] != reference.shape[-1]:
        raise ValueError(
            "mixture must share sample length with reference/estimate, "
            f"got {mixture_eval.shape[-1]} and {reference.shape[-1]}"
        )
    if not compute_permutation and estimate.shape[0] != reference.shape[0]:
        raise ValueError(
            "compute_permutation=False requires matching channels for reference and estimate."
        )
    if not compute_permutation and mixture_eval.shape[0] != reference.shape[0]:
        if mixture_eval.shape[0] == 1:
            mixture_eval = np.repeat(mixture_eval, reference.shape[0], axis=0)
        else:
            raise ValueError(
                "compute_permutation=False requires matching channels for reference and mixture."
            )

    sdr_mix, _, _, _ = cast(
        BssevalMetrics,
        bss_eval_sources(
            reference,
            mixture_eval,
            filter_length=filter_length,
            compute_permutation=compute_permutation,
        ),
    )
    if compute_permutation:
        sdr_est, sir_est, sar_est, perm_default = cast(
            BssevalMetrics,
            bss_eval_sources(
                reference,
                estimate,
                filter_length=filter_length,
                compute_permutation=True,
            ),
        )
        strategy = (
            permutation_strategy
            if permutation_strategy is not None
            else BssEvalPermutationStrategy(filter_length=filter_length)
        )
        perm_idx = strategy.solve(
            PermutationRequest(
                score=np.asarray(sdr_est),
                reference=reference,
                estimate=estimate,
                filter_length=filter_length,
                default_perm=np.asarray(perm_default, dtype=np.int64),
            )
        )
    else:
        sdr_est, sir_est, sar_est = cast(
            tuple[np.ndarray, np.ndarray, np.ndarray],
            bss_eval_sources(
                reference,
                estimate,
                filter_length=filter_length,
                compute_permutation=False,
            ),
        )
        perm_idx = np.arange(estimate.shape[0], dtype=np.int64)

    if (
        estimate.shape[0] == reference.shape[0]
        and perm_idx.shape[0] == estimate.shape[0]
        and np.all((0 <= perm_idx) & (perm_idx < estimate.shape[0]))
    ):
        estimate_perm = estimate[np.argsort(perm_idx), :]
    else:
        estimate_perm = estimate

    framewise = None
    if frame_cfg:
        frame_mixture = mixture_eval
        frame_compute_perm = bool(frame_cfg.compute_permutation)
        if not frame_compute_perm and frame_mixture.shape[0] != reference.shape[0]:
            if frame_mixture.shape[0] == 1:
                frame_mixture = np.repeat(frame_mixture, reference.shape[0], axis=0)
            else:
                raise ValueError(
                    "frame.compute_permutation=False requires matching channels for reference and mixture."
                )
        framewise = summarize_framewise_si_sdr(
            reference,
            estimate_perm,
            sample_rate,
            mixture=frame_mixture,
            window_sec=float(frame_cfg.window_sec),
            hop_sec=frame_cfg.hop_sec,
            scaling=bool(frame_cfg.scaling),
            compute_permutation=frame_compute_perm,
        )
        framewise = normalize_framewise_metrics(framewise)

    return MetricsBundle(
        sdr_mix=np.asarray(sdr_mix),
        sdr_est=np.asarray(sdr_est),
        sir_est=np.asarray(sir_est),
        sar_est=np.asarray(sar_est),
        permutation=perm_idx,
        framewise=framewise,
    )
