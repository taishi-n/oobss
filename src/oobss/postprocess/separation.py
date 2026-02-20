"""Reusable post-processing helpers for reference-microphone separation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from oobss.separators.utils import projection_back, projection_back_scale


class ISTFTLike(Protocol):
    """Protocol for STFT classes that support inverse transforms."""

    def istft(self, spectrogram: np.ndarray) -> np.ndarray:
        """Reconstruct a time-domain signal from spectrogram."""


@dataclass(frozen=True)
class ParameterEstimationResult:
    """Estimated parameters and raw demixed spectra for one method."""

    method_id: str
    demixed_tf_raw: np.ndarray
    demixing_matrix: np.ndarray
    source_model: np.ndarray
    stft: ISTFTLike


@dataclass(frozen=True)
class PerReferenceSeparationResult:
    """Per-reference separation outputs."""

    ref_mic: int
    estimate: np.ndarray
    projected_tf: np.ndarray
    mixing_matrix: np.ndarray


def gaussian_source_model_weight(demixed_tfm: np.ndarray) -> np.ndarray:
    """Create Gaussian source-model weights from demixed spectra.

    This utility computes frame/source power and broadcasts it across the
    frequency axis so the output can be used as a simple Gaussian source model.

    Parameters
    ----------
    demixed_tfm:
        Demixed STFT with shape ``(T, F, M)`` where ``T`` is the number of
        frames, ``F`` is the number of frequency bins, and ``M`` is the number
        of sources/channels.

    Returns
    -------
    np.ndarray
        Broadcast source model weights with shape ``(T, F, M)`` and
        ``float64``-compatible numeric values.
    """
    n_freq = max(int(demixed_tfm.shape[1]), 1)
    power_tm = (np.linalg.norm(demixed_tfm, axis=1) ** 2) / float(n_freq)
    return np.broadcast_to(power_tm[:, None, :], demixed_tfm.shape).copy()


def _mixing_matrix_single_frame(
    demixing_matrix: np.ndarray,
    *,
    ref_mic: int,
) -> np.ndarray:
    source_scale = projection_back_scale(
        demixing_matrix,
        ref_mic=ref_mic,
        n_src=demixing_matrix.shape[1],
    )
    safe_scale = np.where(
        np.abs(source_scale) > 1.0e-12,
        source_scale,
        np.ones_like(source_scale),
    )
    return np.asarray(np.linalg.inv(demixing_matrix) / safe_scale[:, None, :])


def mixing_matrix_from_demixing_for_reference(
    demixing_matrix: np.ndarray,
    *,
    ref_mic: int,
) -> np.ndarray:
    """Return reference-normalized mixing matrices from demixing matrices.

    The function first inverts each demixing matrix, then applies
    projection-back scaling such that the selected reference microphone row
    becomes source-consistent for each source.

    Parameters
    ----------
    demixing_matrix:
        Demixing matrix array. Supported shapes are:
        - ``(F, M, M)`` for batch separators.
        - ``(T, F, M, M)`` for online separators.
    ref_mic:
        Reference microphone index used for projection-back normalization.

    Returns
    -------
    np.ndarray
        Reference-normalized mixing matrix with the same leading dimensionality
        as ``demixing_matrix``:
        - ``(F, M, M)`` for batch.
        - ``(T, F, M, M)`` for online.

    Raises
    ------
    ValueError
        If ``demixing_matrix`` is not 3-D or 4-D.
    """
    if demixing_matrix.ndim == 3:
        return _mixing_matrix_single_frame(demixing_matrix, ref_mic=ref_mic)
    if demixing_matrix.ndim == 4:
        return np.stack(
            [
                _mixing_matrix_single_frame(demixing_matrix[t], ref_mic=ref_mic)
                for t in range(demixing_matrix.shape[0])
            ],
            axis=0,
        )
    raise ValueError(f"Unsupported demixing_matrix ndim: {demixing_matrix.ndim}")


def _project_for_reference(
    *,
    demixed_tf_raw: np.ndarray,
    demixing_matrix: np.ndarray,
    ref_mic: int,
) -> tuple[np.ndarray, np.ndarray]:
    if demixing_matrix.ndim == 3:
        projected_tf = projection_back(
            demixed_tf_raw,
            demixing_matrix,
            ref_mic=ref_mic,
        )
        mixing_matrix = mixing_matrix_from_demixing_for_reference(
            demixing_matrix,
            ref_mic=ref_mic,
        )
        return projected_tf, mixing_matrix

    if demixing_matrix.ndim == 4:
        if (
            demixed_tf_raw.ndim != 3
            or demixed_tf_raw.shape[0] != demixing_matrix.shape[0]
        ):
            raise ValueError(
                "Online demixing_matrix requires demixed_tf_raw shape (T, F, M) with matching T."
            )
        projected_tf = np.stack(
            [
                projection_back(
                    demixed_tf_raw[t],
                    demixing_matrix[t],
                    ref_mic=ref_mic,
                )
                for t in range(demixed_tf_raw.shape[0])
            ],
            axis=0,
        )
        mixing_matrix = mixing_matrix_from_demixing_for_reference(
            demixing_matrix,
            ref_mic=ref_mic,
        )
        return projected_tf, mixing_matrix

    raise ValueError(f"Unsupported demixing_matrix ndim: {demixing_matrix.ndim}")


def separate_with_reference(
    params: ParameterEstimationResult,
    *,
    ref_mic: int,
    n_samples: int,
) -> PerReferenceSeparationResult:
    """Apply reference-wise projection-back and reconstruct time signals.

    Parameters
    ----------
    params:
        Parameter estimation bundle that includes demixed STFT, demixing matrix,
        and an STFT object implementing ``istft``.
    ref_mic:
        Reference microphone index for projection-back normalization.
    n_samples:
        Number of time-domain samples to keep after iSTFT. The reconstructed
        output is cropped to this length.

    Returns
    -------
    PerReferenceSeparationResult
        Result containing:
        - ``estimate``: separated time-domain signals ``(M, N)``.
        - ``projected_tf``: projection-back scaled STFT.
        - ``mixing_matrix``: reference-normalized mixing matrix.

    Raises
    ------
    ValueError
        If demixing and demixed STFT shapes are unsupported or inconsistent.
    """
    projected_tf, mixing_matrix = _project_for_reference(
        demixed_tf_raw=params.demixed_tf_raw,
        demixing_matrix=params.demixing_matrix,
        ref_mic=ref_mic,
    )
    estimate = np.real(params.stft.istft(projected_tf.transpose(2, 1, 0)))
    estimate = np.asarray(estimate, dtype=np.float64)[:, : int(n_samples)]
    return PerReferenceSeparationResult(
        ref_mic=int(ref_mic),
        estimate=estimate,
        projected_tf=projected_tf,
        mixing_matrix=mixing_matrix,
    )
