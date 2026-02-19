"""Reusable utilities for evaluating blind source separation outputs."""

from __future__ import annotations

import logging
from typing import Iterable, Iterator, Tuple

import numpy as np
from .metrics import si_bss_eval


LOGGER = logging.getLogger(__name__)


class Framing:
    """Iterator for overlapping window slices along the last axis."""

    def __init__(self, window: int, hop: int, length: int) -> None:
        if window <= 0 or hop <= 0:
            raise ValueError("window and hop must be positive integers.")
        self.window = window
        self.hop = hop
        self.length = length
        self._index = 0

    def __iter__(self) -> Iterator[slice]:
        return self

    def __next__(self) -> slice:
        if self._index >= self.nwin:
            raise StopIteration
        start = self._index * self.hop
        stop = min(start + self.window, self.length)
        self._index += 1
        return slice(start, stop)

    @property
    def nwin(self) -> int:
        if self.window >= self.length:
            return 1
        return int(np.floor((self.length - self.window + self.hop) / self.hop))


def align_lengths(arrays: Iterable[np.ndarray]) -> Tuple[list[np.ndarray], int]:
    """
    Trim all arrays to the shortest shared length along the last axis.

    Returns
    -------
    trimmed : list of ndarray
        Arrays cropped along the last axis to the shared minimum length.
    min_len : int
        The length applied to all outputs.
    """
    arrays = list(arrays)
    min_len = min(arr.shape[-1] for arr in arrays)
    trimmed = [arr[..., :min_len] for arr in arrays]
    return trimmed, min_len


def calc_si_sdr_framewise(
    ref: np.ndarray,
    est: np.ndarray,
    window: int,
    hop: int,
    *,
    scaling: bool = True,
    compute_permutation: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute frame-wise SI-SDR using sliding windows.

    Parameters
    ----------
    ref : ndarray (n_src, n_samples)
    est : ndarray (n_src, n_samples)
    window : int
        Window length in samples.
    hop : int
        Hop size in samples.
    scaling : bool
        Whether to compute the scale-invariant metric.
    compute_permutation : bool
        Whether to solve the permutation problem per frame.
    """
    if ref.shape != est.shape:
        raise ValueError(f"Shape mismatch between ref {ref.shape} and est {est.shape}")

    n_src, n_samples = ref.shape
    window = min(window, n_samples)
    hop = min(hop, window) if window > 0 else hop

    windows = Framing(window, hop, n_samples)
    n_win = windows.nwin
    si_sdr = np.zeros((n_src, n_win), dtype=float)
    perm = np.zeros((n_src, n_win), dtype=int)

    for t, slc in enumerate(windows):
        ref_seg = ref[:, slc].T  # (n_samples_frame, n_src)
        est_seg = est[:, slc].T
        try:
            if compute_permutation:
                si_sdr[:, t], _, _, perm[:, t] = si_bss_eval(
                    ref_seg, est_seg, scaling=scaling
                )
            else:
                for ch in range(n_src):
                    sdr_ch, _, _, _ = si_bss_eval(
                        ref_seg[:, [ch]],
                        est_seg[:, [ch]],
                        scaling=scaling,
                    )
                    si_sdr[ch, t] = sdr_ch[0]
                perm[:, t] = np.arange(n_src)
        except ValueError as exc:
            LOGGER.error("SI-SDR evaluation failed at frame %s: %s", t, exc)
            si_sdr[:, t] = np.nan
            perm[:, t] = np.arange(n_src)

    return si_sdr, perm


def framewise_si_sdr_summary(
    ref: np.ndarray,
    est: np.ndarray,
    *,
    mixture: np.ndarray | None = None,
    window: int,
    hop: int,
    scaling: bool = True,
    compute_permutation: bool = True,
) -> dict[str, np.ndarray]:
    """
    Compute frame-wise SI-SDR for estimates (and optional mixture baseline).

    Returns
    -------
    summary : dict
        Keys include:
          - "si_sdr": ndarray (n_src, n_win)
          - "perm": ndarray (n_src, n_win)
          - "si_sdr_mix": ndarray (n_src, n_win) when mixture is provided.
          - "si_sdr_imp": ndarray (n_src, n_win) when mixture is provided.
    """
    arrays: list[np.ndarray] = [ref, est]
    has_mixture = mixture is not None
    if has_mixture:
        arrays.append(mixture)  # type: ignore[arg-type]
    aligned, _ = align_lengths(arrays)
    ref_aligned, est_aligned = aligned[:2]
    mix_aligned = aligned[2] if has_mixture else None

    si_sdr_est, perm = calc_si_sdr_framewise(
        ref_aligned,
        est_aligned,
        window,
        hop,
        scaling=scaling,
        compute_permutation=compute_permutation,
    )
    summary: dict[str, np.ndarray] = {
        "si_sdr": si_sdr_est,
        "perm": perm,
    }
    if has_mixture and mix_aligned is not None:
        si_sdr_mix, _ = calc_si_sdr_framewise(
            ref_aligned,
            mix_aligned,
            window,
            hop,
            scaling=scaling,
            compute_permutation=False,
        )
        summary["si_sdr_mix"] = si_sdr_mix
        summary["si_sdr_imp"] = si_sdr_est - si_sdr_mix
    return summary


def summarize_framewise_si_sdr(
    ref: np.ndarray,
    est: np.ndarray,
    fs: int,
    *,
    window_sec: float = 5.0,
    hop_sec: float | None = None,
    mixture: np.ndarray | None = None,
    scaling: bool = True,
    compute_permutation: bool = True,
) -> dict[str, np.ndarray]:
    """
    Convenience wrapper returning aggregate statistics of frame-wise SI-SDR.
    """
    window = max(1, int(round(window_sec * fs)))
    hop = window if hop_sec is None else max(1, int(round(hop_sec * fs)))
    summary = framewise_si_sdr_summary(
        ref,
        est,
        mixture=mixture,
        window=window,
        hop=hop,
        scaling=scaling,
        compute_permutation=compute_permutation,
    )

    result: dict[str, np.ndarray] = dict(summary)
    result["mean_si_sdr"] = np.nanmean(summary["si_sdr"], axis=1)
    result["median_si_sdr"] = np.nanmedian(summary["si_sdr"], axis=1)
    if "si_sdr_imp" in summary:
        result["mean_si_sdr_imp"] = np.nanmean(summary["si_sdr_imp"], axis=1)
        result["median_si_sdr_imp"] = np.nanmedian(summary["si_sdr_imp"], axis=1)
    return result
