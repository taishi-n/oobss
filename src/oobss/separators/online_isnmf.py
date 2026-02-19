"""Online IS-NMF separator."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .core import (
    BaseStreamingSeparator,
    ComponentAssignmentRequest,
    ComponentAssignmentStrategy,
    NMFUpdateRequest,
    OnlineFrameRequest,
    OnlineNMFUpdateStrategy,
    ReconstructionRequest,
    ReconstructionStrategy,
)
from .strategies import (
    ModuloAssignmentStrategy,
    MultiplicativeNMFStrategy,
    RatioMaskReconstructionStrategy,
)


class OnlineISNMF(BaseStreamingSeparator):
    """Online Itakura-Saito NMF with source-wise ratio-mask reconstruction.

    Procedure
    ---------
    ```text

       input: STFT frame sequence {x_t}_{t=1..T}, x_t in C^F
       initialize NMF basis W, statistics A/B
       for each frame x_t:
           v_t <- |x_t|^2
           update activation h_t and basis statistics by online MU
           if update timing reached: refresh W from A/B with forgetting
           compute source power p_{s,f} by component-to-source assignment
           compute ratio mask m_{s,f} = p_{s,f} / sum_s p_{s,f}
           emit separated sources y_{s,f} = m_{s,f} x_f
    ```

    Update Equations
    ----------------
    With $m = Wh + \\varepsilon$, the online NMF updates are:

    $$
       h \\leftarrow h \\odot
       \\frac{W^{\\top}(v \\oslash m^2)}
       {\\max\\left(W^{\\top}(1 \\oslash m), \\varepsilon\\right)}
       A \\leftarrow A + \\left((v \\oslash m^2)h^{\\top}\\right) \\odot W^2,
       \\quad
       B \\leftarrow B + (1 \\oslash m)h^{\\top}
    $$

    Basis refresh (every $\\beta$ frames):

    $$
       \\rho = \\alpha^{\\beta / t}, \\quad
       A \\leftarrow \\rho A, \\; B \\leftarrow \\rho B, \\quad
       W \\leftarrow \\sqrt{A \\oslash B}
    $$

    Source-wise power aggregation and Wiener-style ratio mask:

    $$
       p_{s,f} = \\sum_{k:\\pi(k)=s} W_{f,k} h_k, \\quad
       m_{s,f} = \\frac{p_{s,f}}{\\sum_{s'} p_{s',f} + \\varepsilon},
       \\quad
       \\hat{y}_{s,f} = m_{s,f} x_f
    $$

    Examples
    --------
    Separate one-channel STFT stream into two sources:

    ```python

       import numpy as np
       from scipy.signal import ShortTimeFFT, get_window
       from oobss import OnlineISNMF, StreamRequest

       fs = 16000
       fft_size = 1024
       hop_size = 256
       win = get_window("hann", fft_size, fftbins=True)
       stft = ShortTimeFFT(win=win, hop=hop_size, fs=fs)

       mixture_time = np.random.randn(fs * 2, 1)  # mono mixture
       X_ft = stft.stft(mixture_time[:, 0])       # (n_freq, n_frame)

       model = OnlineISNMF(
           n_components=16,
           n_features=X_ft.shape[0],
           n_sources=2,
           beta=2,
           forget=0.99,
           inner_iter=10,
           random_state=0,
       )
       out = model.process_stream_tf(
           X_ft,
           request=StreamRequest(frame_axis=1, n_sources=2),
       )
       Y_sft = out.estimate_tf  # (n_sources, n_freq, n_frame)
       if Y_sft is None:
           raise ValueError("OnlineISNMF did not return TF estimates.")
    ```

    Request masks instead of separated spectra:

    ```python

       out = model.process_stream_tf(
           X_ft,
           request=StreamRequest(frame_axis=1, n_sources=2, return_mask=True),
       )
       mask = out.estimate_tf  # (n_sources, n_freq, n_frame)
    ```
    """

    def __init__(
        self,
        n_components: int,
        n_features: int,
        *,
        beta: int = 50,
        forget: float = 0.9,
        inner_iter: int = 30,
        keep_h: bool | str = "auto",
        eps: float = 1.0e-12,
        n_sources: Optional[int] = None,
        component_to_source: np.ndarray | None = None,
        random_state: Optional[int] = None,
        nmf: OnlineNMFUpdateStrategy | None = None,
        assignment: ComponentAssignmentStrategy | None = None,
        reconstruction_strategy: ReconstructionStrategy | None = None,
    ) -> None:
        self.F, self.K = int(n_features), int(n_components)
        self.beta = int(beta)
        self.r = float(forget)
        self.inner_iter = int(inner_iter)
        self.eps = float(eps)
        self._random_state = random_state

        self._default_n_sources = n_sources
        self._default_component_to_source = component_to_source

        self.reconstruction_strategy = (
            reconstruction_strategy
            if reconstruction_strategy is not None
            else RatioMaskReconstructionStrategy(eps=self.eps)
        )
        self.nmf_strategy = nmf if nmf is not None else MultiplicativeNMFStrategy()
        self.assignment_strategy = (
            assignment if assignment is not None else ModuloAssignmentStrategy()
        )

        if keep_h == "auto":
            self.keep_h = self.beta < 1000
        else:
            self.keep_h = bool(keep_h)
        self.reset()

    def reset(self) -> None:
        """Reset online NMF parameters and sufficient statistics."""
        rng = np.random.default_rng(self._random_state)
        self.W = rng.random((self.F, self.K)) + self.eps
        self.A = np.zeros_like(self.W)
        self.B = np.zeros_like(self.W)
        self._l1_normalise_W()
        self._t = 0
        self._batch_counter = 0
        self._H_store: list[np.ndarray] = []

    def partial_fit(self, v: np.ndarray) -> np.ndarray:
        """Update online NMF model from one power spectrum frame."""
        if v.ndim != 1 or v.shape[0] != self.F:
            raise ValueError("v must be 1-D array of length F")

        h_prev = self._H_store[-1].copy() if self.keep_h and self._H_store else None
        result = self.nmf_strategy.update(
            NMFUpdateRequest(
                v=v,
                basis=self.W,
                stat_a=self.A,
                stat_b=self.B,
                inner_iter=self.inner_iter,
                beta=self.beta,
                alpha=self.r,
                batch_counter=self._batch_counter,
                t=self._t,
                eps=self.eps,
                h_prev=h_prev,
            )
        )

        self.W = result.basis
        self.A = result.stat_a
        self.B = result.stat_b
        self._batch_counter = result.batch_counter
        self._t = result.t

        if self.keep_h:
            self._H_store.append(result.h)
        return result.h

    def _l1_normalise_W(self) -> None:
        colsum = np.maximum(self.W.sum(axis=0, keepdims=True), self.eps)
        self.W /= colsum
        self.A /= colsum
        self.B *= colsum

    def source_power(
        self,
        h: np.ndarray,
        *,
        n_sources: int,
        component_to_source: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute source-wise power model from component activation."""
        if h.ndim != 1 or h.shape[0] != self.K:
            raise ValueError("h must be a 1-D array of length K")

        assignment = self.assignment_strategy.resolve(
            ComponentAssignmentRequest(
                n_components=self.K,
                n_sources=n_sources,
                component_to_source=component_to_source,
            )
        )
        component_power = self.W * h[None, :]
        source_power = np.zeros((n_sources, self.F), dtype=component_power.dtype)
        np.add.at(source_power, assignment, component_power.T)
        return source_power

    def separate_frame(
        self,
        x: np.ndarray,
        *,
        n_sources: int,
        component_to_source: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Separate one complex STFT frame via source-wise ratio masking."""
        if x.ndim != 1 or x.shape[0] != self.F:
            raise ValueError("x must be a 1-D complex STFT frame of length F")

        h = self.partial_fit(np.abs(x) ** 2)
        source_power = self.source_power(
            h,
            n_sources=n_sources,
            component_to_source=component_to_source,
        )
        recon = self.reconstruction_strategy.reconstruct(
            ReconstructionRequest(
                mixture=x,
                source_power=source_power,
            )
        )
        if recon.mask is None:
            raise ValueError("OnlineISNMF reconstruction strategy must return a mask")
        return recon.estimate, recon.mask

    @property
    def n_sources(self) -> int:
        """Return configured source count (fallback to 1 when unspecified)."""
        return int(self._default_n_sources) if self._default_n_sources else 1

    def process_frame(
        self,
        frame: np.ndarray,
        request: OnlineFrameRequest | None = None,
    ) -> np.ndarray:
        """Process one frame using default or explicitly provided settings."""
        request_obj = request if request is not None else OnlineFrameRequest()
        n_sources = (
            request_obj.n_sources
            if request_obj.n_sources is not None
            else self._default_n_sources
        )
        if n_sources is None:
            raise ValueError("n_sources must be provided for OnlineISNMF separation")

        component_to_source = (
            request_obj.component_to_source
            if request_obj.component_to_source is not None
            else self._default_component_to_source
        )
        separated, mask = self.separate_frame(
            frame,
            n_sources=int(n_sources),
            component_to_source=component_to_source,
        )
        return mask if request_obj.return_mask else separated
