"""Permutation strategies for source alignment."""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy.optimize import linear_sum_assignment

from oobss.separators.core import PermutationRequest, PermutationStrategy


class IdentityPermutationStrategy(PermutationStrategy):
    """Return the identity permutation."""

    def solve(self, request: PermutationRequest) -> np.ndarray:
        score = request.score
        if score.ndim == 0:
            raise ValueError("score must provide at least one source dimension.")
        n_src = int(score.shape[0])
        return np.arange(n_src, dtype=np.int64)


class ScoreMatrixPermutationStrategy(PermutationStrategy):
    """Solve permutation by maximizing a source similarity score matrix.

    Notes
    -----
    Input ``score`` must be a square matrix with shape ``(n_src, n_src)``,
    where ``score[i, j]`` indicates similarity between reference source ``i``
    and estimated source ``j``.
    """

    def solve(self, request: PermutationRequest) -> np.ndarray:
        score = request.score
        if score.ndim != 2 or score.shape[0] != score.shape[1]:
            raise ValueError("score must be a square 2-D array.")
        row_idx, col_idx = linear_sum_assignment(-score)
        perm = np.zeros(score.shape[0], dtype=np.int64)
        perm[row_idx] = col_idx
        return perm


class BssEvalPermutationStrategy(PermutationStrategy):
    """Resolve permutation using ``fast_bss_eval.bss_eval_sources``."""

    def __init__(self, filter_length: int = 512) -> None:
        self.filter_length = int(filter_length)

    def solve(self, request: PermutationRequest) -> np.ndarray:
        default_perm = request.default_perm
        if isinstance(default_perm, np.ndarray):
            return np.asarray(default_perm, dtype=np.int64)

        reference = request.reference
        estimate = request.estimate
        filter_length = int(request.filter_length or self.filter_length)
        if not isinstance(reference, np.ndarray) or not isinstance(
            estimate, np.ndarray
        ):
            raise ValueError(
                "BssEvalPermutationStrategy requires reference and estimate arrays."
            )

        from fast_bss_eval import bss_eval_sources

        metrics = cast(
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            bss_eval_sources(reference, estimate, filter_length=filter_length),
        )
        perm = np.asarray(metrics[3], dtype=np.int64)
        return perm
