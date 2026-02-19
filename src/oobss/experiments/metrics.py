"""Metric computation for blind source separation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from fast_bss_eval import bss_eval_sources

from oobss.separators.core import PermutationRequest, PermutationStrategy
from oobss.separators.eval_utils import summarize_framewise_si_sdr
from oobss.separators.strategies import BssEvalPermutationStrategy

from .config_schema import FrameEvalConfig

BssevalMetrics = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


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
        stats = {
            "sdr_mix_mean": float(np.nanmean(self.sdr_mix)),
            "sdr_est_mean": float(np.nanmean(self.sdr_est)),
            "sdr_imp_mean": float(np.nanmean(self.sdr_est - self.sdr_mix)),
            "sdr_est_median": float(np.nanmedian(self.sdr_est)),
            "sdr_imp_median": float(np.nanmedian(self.sdr_est - self.sdr_mix)),
            "sir_mean": float(np.nanmean(self.sir_est)),
            "sar_mean": float(np.nanmean(self.sar_est)),
        }
        stats["sdr_mix_channels"] = [float(v) for v in np.ravel(self.sdr_mix)]
        stats["sdr_est_channels"] = [float(v) for v in np.ravel(self.sdr_est)]
        stats["sdr_imp_channels"] = [
            float(est - mix) for est, mix in zip(self.sdr_est, self.sdr_mix)
        ]
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
    frame_cfg: FrameEvalConfig | None,
    permutation_strategy: PermutationStrategy | None = None,
) -> MetricsBundle:
    if reference.shape != estimate.shape:
        raise ValueError(
            f"Reference {reference.shape} and estimate {estimate.shape} must match"
        )

    sdr_mix, _, _, _ = cast(
        BssevalMetrics,
        bss_eval_sources(reference, mixture, filter_length=filter_length),
    )
    sdr_est, sir_est, sar_est, perm_default = cast(
        BssevalMetrics,
        bss_eval_sources(reference, estimate, filter_length=filter_length),
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

    estimate_perm = estimate[perm_idx, :]

    framewise = None
    if frame_cfg:
        framewise = summarize_framewise_si_sdr(
            reference,
            estimate_perm,
            sample_rate,
            mixture=mixture,
            window_sec=float(frame_cfg.window_sec),
            hop_sec=frame_cfg.hop_sec,
            scaling=bool(frame_cfg.scaling),
            compute_permutation=bool(frame_cfg.compute_permutation),
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
