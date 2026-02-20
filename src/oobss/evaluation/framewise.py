"""Frame-wise SI-SDR utilities."""

from __future__ import annotations

import logging
from typing import Iterable, Iterator

import numpy as np

from .si_sdr import si_bss_eval


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


def align_lengths(arrays: Iterable[np.ndarray]) -> tuple[list[np.ndarray], int]:
    """Trim all arrays to the shortest shared length along the last axis."""
    array_list = list(arrays)
    min_len = min(arr.shape[-1] for arr in array_list)
    trimmed = [arr[..., :min_len] for arr in array_list]
    return trimmed, min_len


def calc_si_sdr_framewise(
    ref: np.ndarray,
    est: np.ndarray,
    window: int,
    hop: int,
    *,
    scaling: bool = True,
    compute_permutation: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute frame-wise SI-SDR with sliding windows."""
    if ref.ndim != 2 or est.ndim != 2:
        raise ValueError("ref and est must be 2-D arrays (n_src, n_samples)")
    if ref.shape[-1] != est.shape[-1]:
        raise ValueError(
            f"Shape mismatch between ref {ref.shape} and est {est.shape} (sample length)"
        )
    if not compute_permutation and ref.shape[0] != est.shape[0]:
        raise ValueError(
            "compute_permutation=False requires equal number of channels in ref and est"
        )

    _, n_samples = ref.shape
    n_est = est.shape[0]
    window = min(window, n_samples)
    hop = min(hop, window) if window > 0 else hop

    windows = Framing(window, hop, n_samples)
    n_win = windows.nwin
    si_sdr = np.zeros((n_est, n_win), dtype=float)
    perm = np.zeros((n_est, n_win), dtype=int)

    for t, slc in enumerate(windows):
        ref_seg = ref[:, slc].T
        est_seg = est[:, slc].T
        try:
            if compute_permutation:
                si_sdr[:, t], _, _, perm[:, t] = si_bss_eval(
                    ref_seg,
                    est_seg,
                    scaling=scaling,
                )
            else:
                for ch in range(n_est):
                    sdr_ch, _, _, _ = si_bss_eval(
                        ref_seg[:, [ch]],
                        est_seg[:, [ch]],
                        scaling=scaling,
                    )
                    si_sdr[ch, t] = sdr_ch[0]
                perm[:, t] = np.arange(n_est)
        except ValueError as exc:
            LOGGER.error("SI-SDR evaluation failed at frame %s: %s", t, exc)
            si_sdr[:, t] = np.nan
            perm[:, t] = np.arange(n_est)

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
    """Compute frame-wise SI-SDR and optional mixture baseline."""
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
    summary: dict[str, np.ndarray] = {"si_sdr": si_sdr_est, "perm": perm}
    if has_mixture and mix_aligned is not None:
        si_sdr_mix, _ = calc_si_sdr_framewise(
            ref_aligned,
            mix_aligned,
            window,
            hop,
            scaling=scaling,
            compute_permutation=compute_permutation,
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
    """Return frame-wise SI-SDR and aggregate statistics."""
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
