"""Low-level SI-SDR/SI-SIR/SI-SAR metric utilities."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def si_bss_eval(
    reference_signals: np.ndarray,
    estimated_signals: np.ndarray,
    scaling: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute SI-SDR family metrics and permutation.

    Parameters
    ----------
    reference_signals:
        Reference matrix shaped ``(n_samples, n_channels)``.
    estimated_signals:
        Estimated matrix shaped ``(n_samples, n_channels)``.
    scaling:
        If ``True``, compute scale-invariant scores.
    """
    _, n_chan = estimated_signals.shape
    rss = np.dot(reference_signals.transpose(), reference_signals)

    sdr = np.zeros((n_chan, n_chan))
    sir = np.zeros((n_chan, n_chan))
    sar = np.zeros((n_chan, n_chan))

    for ref_idx in range(n_chan):
        for est_idx in range(n_chan):
            sdr[ref_idx, est_idx], sir[ref_idx, est_idx], sar[ref_idx, est_idx] = (
                _compute_measures(
                    estimated_signals[:, est_idx],
                    reference_signals,
                    rss,
                    ref_idx,
                    scaling=scaling,
                )
            )

    row_idx, perm = _linear_sum_assignment_with_inf(-sir)
    return sdr[row_idx, perm], sir[row_idx, perm], sar[row_idx, perm], perm


def _compute_measures(
    estimated_signal: np.ndarray,
    reference_signals: np.ndarray,
    rss: np.ndarray,
    ref_idx: int,
    *,
    scaling: bool = True,
) -> tuple[float, float, float]:
    """Compute SI-SDR/SIR/SAR for one reference-estimate pair."""
    target = reference_signals[:, ref_idx]

    if scaling:
        scale = np.dot(target, estimated_signal) / rss[ref_idx, ref_idx]
    else:
        scale = 1.0

    e_true = scale * target
    e_res = estimated_signal - e_true

    sss = (e_true**2).sum()
    snn = (e_res**2).sum()
    sdr = 10.0 * math.log10(sss / snn)

    rsr = np.dot(reference_signals.transpose(), e_res)
    b = np.linalg.solve(rss, rsr)

    e_interf = np.dot(reference_signals, b)
    e_artif = e_res - e_interf

    sir = 10.0 * math.log10(sss / (e_interf**2).sum())
    sar = 10.0 * math.log10(sss / (e_artif**2).sum())
    return sdr, sir, sar


def _linear_sum_assignment_with_inf(
    cost_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve linear assignment while tolerating ``inf`` entries."""
    matrix = np.asarray(cost_matrix)
    has_min_inf = np.isneginf(matrix).any()
    has_max_inf = np.isposinf(matrix).any()
    if has_min_inf and has_max_inf:
        raise ValueError("matrix contains both inf and -inf")

    if has_min_inf or has_max_inf:
        matrix = matrix.copy()
        finite_values = matrix[~np.isinf(matrix)]
        min_value = finite_values.min()
        max_value = finite_values.max()
        n = min(matrix.shape)
        positive = n * (max_value - min_value + np.abs(max_value) + np.abs(min_value) + 1)
        if has_max_inf:
            place_holder = (max_value + (n - 1) * (max_value - min_value)) + positive
        else:
            place_holder = (min_value + (n - 1) * (min_value - max_value)) - positive
        matrix[np.isinf(matrix)] = place_holder

    return linear_sum_assignment(matrix)
