"""Normalization and scale-restoration strategies."""

from __future__ import annotations

import numpy as np

from oobss.separators.core import NormalizationRequest, NormalizationStrategy
from oobss.separators.utils import projection_back


class IdentityNormalization(NormalizationStrategy):
    """No-op normalization strategy."""

    def apply(self, request: NormalizationRequest) -> np.ndarray:
        return request.estimate


class ProjectionBackNormalization(NormalizationStrategy):
    """Apply projection-back scale restoration.

    For each frequency $f$, define $A_f = W_f^{-1}$. The
    source image at reference microphone $m_{\\mathrm{ref}}$ is:

    $$
       \\hat{y}_{k,f,t} = a_{k,f}[m_{\\mathrm{ref}}] y_{k,f,t}
    $$
    """

    def __init__(self, ref_mic: int = 0) -> None:
        self.ref_mic = int(ref_mic)

    def apply(self, request: NormalizationRequest) -> np.ndarray:
        estimate = request.estimate
        demix_filter = request.demix_filter
        if not isinstance(demix_filter, np.ndarray):
            raise TypeError("demix_filter must be provided as np.ndarray")
        return projection_back(estimate, demix_filter, ref_mic=self.ref_mic)
