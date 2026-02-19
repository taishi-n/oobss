"""Reconstruction strategies for separated output signals."""

from __future__ import annotations

import numpy as np

from oobss.separators.core import (
    ReconstructionRequest,
    ReconstructionResult,
    ReconstructionStrategy,
)
from oobss.separators.utils import demix, projection_back


class DemixReconstructionStrategy(ReconstructionStrategy):
    """Reconstruct separated spectra by applying demixing filters."""

    def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResult:
        mixture = request.mixture
        demix_filter = request.demix_filter
        if not isinstance(demix_filter, np.ndarray):
            raise TypeError("demix_filter must be provided as np.ndarray")
        return ReconstructionResult(estimate=demix(mixture, demix_filter), mask=None)


class RatioMaskReconstructionStrategy(ReconstructionStrategy):
    """Reconstruct source spectra using Wiener-style ratio masks."""

    def __init__(self, eps: float = 1.0e-12) -> None:
        self.eps = float(eps)

    def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResult:
        mixture = request.mixture
        source_power = request.source_power
        if not isinstance(source_power, np.ndarray):
            raise TypeError("source_power must be provided as np.ndarray")
        denom = np.maximum(source_power.sum(axis=0, keepdims=True), self.eps)
        mask = source_power / denom
        separated = mask * mixture[None, :]
        return ReconstructionResult(estimate=separated, mask=mask)


class ProjectionBackDemixReconstructionStrategy(ReconstructionStrategy):
    """Reconstruct outputs with demixing + projection back.

    For each frequency $f$, define $A_f = W_f^{-1}$. The
    source image at reference microphone $m_{\\mathrm{ref}}$ is:

    $$
       \\hat{y}_{k,f,t} = a_{k,f}[m_{\\mathrm{ref}}] y_{k,f,t}
    $$
    """

    def __init__(self, ref_mic: int = 0) -> None:
        self.ref_mic = int(ref_mic)

    def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResult:
        mixture = request.mixture
        demix_filter = request.demix_filter
        if not isinstance(demix_filter, np.ndarray):
            raise TypeError("demix_filter must be provided as np.ndarray")

        est = demix_filter @ mixture[:, :, None]
        out = projection_back(est[:, :, 0], demix_filter, ref_mic=self.ref_mic)
        return ReconstructionResult(estimate=out, mask=None)
