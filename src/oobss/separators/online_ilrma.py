"""Online independent low-rank matrix analysis."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .core import (
    BaseStreamingSeparator,
    CovarianceRequest,
    CovarianceUpdateStrategy,
    NMFUpdateRequest,
    OnlineFrameRequest,
    OnlineNMFUpdateStrategy,
    ReconstructionRequest,
    ReconstructionStrategy,
    SpatialUpdateStrategy,
)
from .strategies import (
    EMACovarianceStrategy,
    IP1SpatialStrategy,
    MultiplicativeNMFStrategy,
    ProjectionBackDemixReconstructionStrategy,
)


class OnlineILRMA(BaseStreamingSeparator):
    """Online independent low-rank matrix analysis.

    Procedure
    ---------
    ```text

       input: frame sequence {x_{f,t}}_{f=1..F, t=1..T}
       initialize W_{f,0} = I_M, V_{k,f,0}
       initialize source-wise NMF parameters b_{k,f,\\ell}, c_{k,\\ell,t}
       for each frame t:
           repeat inner_iter times:
               y_{f,t} <- W_{f,t} x_{f,t}
               for each source k:
                   update c_{k,\\ell,t}, sufficient stats, and b_{k,f,\\ell}
                   r_{k,f,t} <- sum_\\ell b_{k,f,\\ell} c_{k,\\ell,t}
               V_{k,f,t} <- (1-alpha) * x_{f,t} x_{f,t}^H / r_{k,f,t}
                          + alpha * V_{k,f,t-1}
               update w_{k,f,t} by IP1
           projection-back by reference microphone
           emit separated frame
    ```

    Update Equations
    ----------------
    Indices are $k=1,\\dots,K$ (source), $m=1,\\dots,M$
    (channel), $f=1,\\dots,F$ (frequency), $t=1,\\dots,T$
    (time frame), and $\\ell=1,\\dots,L$ (NMF basis index). In the
    determined case, $K=M$.

    Demixing:

    $$
       \\bm{x}_{f,t} \\in \\mathbb{C}^{M}, \\quad
       \\bm{y}_{f,t} = W_{f,t}\\bm{x}_{f,t}, \\quad
       y_{k,f,t} = \\bm{w}_{k,f,t}^{\\mathsf{H}}\\bm{x}_{f,t}
    $$

    For each source $k$, define:

    $$
       v_{k,f,t} = |y_{k,f,t}|, \\quad
       m_{k,f,t} =
       \\sum_{\\ell=1}^{L}b_{k,f,\\ell}c_{k,\\ell,t} + \\varepsilon
    $$

    The online MU update for $C$ is:

    $$
       c_{k,\\ell,t}
       \\leftarrow
       c_{k,\\ell,t}
       \\frac{
       \\sum_{f=1}^{F}
       b_{k,f,\\ell}v_{k,f,t}m_{k,f,t}^{-2}
       }{
       \\max\\left(
       \\sum_{f=1}^{F}b_{k,f,\\ell}m_{k,f,t}^{-1},
       \\varepsilon
       \\right)
       }
    $$

    Sufficient statistics and basis update:

    $$
       P_{k,f,\\ell,t}
       \\leftarrow
       P_{k,f,\\ell,t-1}
       +
       v_{k,f,t}m_{k,f,t}^{-2}c_{k,\\ell,t}b_{k,f,\\ell}^2,
       \\quad
       Q_{k,f,\\ell,t}
       \\leftarrow
       Q_{k,f,\\ell,t-1} + m_{k,f,t}^{-1}c_{k,\\ell,t}
    $$

    $$
       \\rho_{k,t} = \\alpha^{\\beta/t_k}, \\quad
       P_{k,f,\\ell,t} \\leftarrow \\rho_{k,t} P_{k,f,\\ell,t}, \\;
       Q_{k,f,\\ell,t} \\leftarrow \\rho_{k,t} Q_{k,f,\\ell,t}, \\quad
       b_{k,f,\\ell} \\leftarrow
       \\sqrt{P_{k,f,\\ell,t} \\oslash Q_{k,f,\\ell,t}}
    $$

    Source variance and covariance update:

    $$
       r_{k,f,t}
       = \\max\\left(
       \\sum_{\\ell=1}^{L}b_{k,f,\\ell}c_{k,\\ell,t},
       \\varepsilon
       \\right)
    $$

    $$
       V_{k,f,t}
       \\leftarrow
       (1-\\alpha)
       \\frac{
       \\bm{x}_{f,t}\\bm{x}_{f,t}^{\\mathsf{H}}
       }{r_{k,f,t}}
       + \\alpha V_{k,f,t-1}
    $$

    Demixing and common projection back:

    $$
       \\tilde{\\bm{w}}_{k,f,t}
       = \\left(W_{f,t}V_{k,f,t}\\right)^{-1}\\bm{e}_k, \\quad
       \\bm{w}_{k,f,t}
       = \\frac{\\tilde{\\bm{w}}_{k,f,t}}
       {\\sqrt{
       \\tilde{\\bm{w}}_{k,f,t}^{\\mathsf{H}}
       V_{k,f,t}
       \\tilde{\\bm{w}}_{k,f,t}
       }}
    $$

    $$
       A_{f,t} = W_{f,t}^{-1}, \\quad
       \\hat{y}_{k,f,t} = a_{k,f,t}[m_{\\mathrm{ref}}] y_{k,f,t}
    $$

    This is shared with batch AuxIVA/ILRMA via
    :func:`oobss.separators.utils.projection_back`.

    Examples
    --------
    Process a stream with online ILRMA:

    ```python

       import numpy as np
       from scipy.signal import ShortTimeFFT, get_window
       from oobss import OnlineILRMA, StreamRequest

       fs = 16000
       fft_size = 2048
       hop_size = 512
       win = get_window("hann", fft_size, fftbins=True)
       stft = ShortTimeFFT(win=win, hop=hop_size, fs=fs)

       mixture_time = np.random.randn(fs * 2, 2)  # (n_samples, n_mic)
       X_fmt = stft.stft(mixture_time.T).transpose(1, 0, 2)  # (F, M, T)

       model = OnlineILRMA(
           n_mic=2,
           n_freq=X_fmt.shape[0],
           n_bases=8,
           ref_mic=0,
           beta=1,
           forget=0.99,
           inner_iter=5,
           random_state=0,
       )
       out = model.process_stream_tf(
           X_fmt,
           request=StreamRequest(frame_axis=2, reference_mic=0),
       )
       Y_fmt = out.estimate_tf
       if Y_fmt is None:
           raise ValueError("OnlineILRMA did not return TF estimates.")

       y_time = np.real(stft.istft(Y_fmt, f_axis=0, t_axis=2)).T
    ```

    Plug-and-play NMF updater while keeping spatial update fixed:

    ```python

       from oobss.separators.strategies import MultiplicativeNMFStrategy

       model = OnlineILRMA(
           n_mic=2,
           n_freq=X_fmt.shape[0],
           n_bases=8,
           nmf=MultiplicativeNMFStrategy(),
       )
    ```

    References
    ----------
    [1] T. Nakashima and N. Ono, "Online independent low-rank matrix analysis
    as a lightweight and trainable model for real-time multichannel music
    source separation," in *Proc. AAAI 2026 Workshop on Audio-Centric AI:
    Towards Real-World Multimodal Reasoning and Application Use Cases
    (Audio-AAAI)*, Jan. 2026.
    """

    def __init__(
        self,
        n_mic: int,
        n_freq: int,
        n_bases: int,
        *,
        ref_mic: int = 0,
        beta: int = 1,
        forget: float = 0.9,
        inner_iter: int = 30,
        keep_h: bool | str = False,
        eps: float = 1.0e-12,
        cov_scale: float = 1.0e-6,
        random_state: Optional[int] = None,
        spatial: SpatialUpdateStrategy | None = None,
        covariance: CovarianceUpdateStrategy | None = None,
        nmf: OnlineNMFUpdateStrategy | None = None,
        reconstruction_strategy: ReconstructionStrategy | None = None,
    ) -> None:
        self.n_mic = int(n_mic)
        self.n_freq = int(n_freq)
        self.n_bases = int(n_bases)
        self._random_state = random_state

        self.ref_mic = int(ref_mic)
        self.beta = int(beta)
        self.alpha = float(forget)
        self.inner_iter = int(inner_iter)
        self.eps = float(eps)
        self._cov_scale = float(cov_scale)

        self.spatial_strategy = spatial if spatial is not None else IP1SpatialStrategy()
        self.covariance_strategy = (
            covariance
            if covariance is not None
            else EMACovarianceStrategy(alpha=self.alpha)
        )
        self.nmf_strategy = nmf if nmf is not None else MultiplicativeNMFStrategy()
        self.reconstruction_strategy = (
            reconstruction_strategy
            if reconstruction_strategy is not None
            else ProjectionBackDemixReconstructionStrategy(ref_mic=self.ref_mic)
        )

        if keep_h == "auto":
            self.keep_h = self.beta < 1000
        else:
            self.keep_h = bool(keep_h)
        self.reset()

    def reset(self) -> None:
        """Reset online state and NMF statistics to initial values."""
        self.rng = np.random.default_rng(self._random_state)
        self.demix = np.tile(np.eye(self.n_mic, dtype=complex), (self.n_freq, 1, 1))
        self.cov = (
            np.tile(np.eye(self.n_mic, dtype=complex), (self.n_mic, self.n_freq, 1, 1))
            * self._cov_scale
        )

        self.basis = self.rng.random((self.n_mic, self.n_freq, self.n_bases)) + self.eps
        self.A = np.zeros_like(self.basis)
        self.B = np.zeros_like(self.basis)
        self._l1_normalise_W()

        self._t = np.zeros((self.n_mic,), dtype=np.int64)
        self._batch_counter = np.zeros((self.n_mic,), dtype=np.int64)
        self._H_store: list[list[np.ndarray]] = [[] for _ in range(self.n_mic)]

    def partial_fit(
        self,
        x: np.ndarray,
        *,
        reference_mic: int | None = None,
    ) -> np.ndarray:
        """Update model with one frame and return separated spectra."""
        if x.shape != (self.n_freq, self.n_mic):
            raise ValueError(
                "x must have shape (n_freq, n_mic) "
                f"= ({self.n_freq}, {self.n_mic}), got {x.shape}"
            )

        source_model = np.ones((self.n_freq, self.n_mic), dtype=np.float64)
        for _ in range(self.inner_iter):
            demixed = (self.demix @ x[:, :, None])[:, :, 0]
            for k in range(self.n_mic):
                prev_h = (
                    self._H_store[k][-1].copy()
                    if self.keep_h and self._H_store[k]
                    else None
                )
                result = self.nmf_strategy.update(
                    NMFUpdateRequest(
                        v=np.abs(demixed[:, k]),
                        basis=self.basis[k],
                        stat_a=self.A[k],
                        stat_b=self.B[k],
                        inner_iter=1,
                        beta=self.beta,
                        alpha=self.alpha,
                        batch_counter=int(self._batch_counter[k]),
                        t=int(self._t[k]),
                        eps=self.eps,
                        h_prev=prev_h,
                    )
                )
                self.basis[k] = result.basis
                self.A[k] = result.stat_a
                self.B[k] = result.stat_b
                self._batch_counter[k] = result.batch_counter
                self._t[k] = result.t
                if self.keep_h:
                    self._H_store[k].append(result.h)
                source_model[:, k] = np.maximum(result.basis @ result.h, self.eps)

            self.cov = self.covariance_strategy.update(
                CovarianceRequest(
                    observed=x[None, :, :],
                    source_model=source_model[None, :, :],
                    prev_cov=self.cov,
                    alpha=self.alpha,
                )
            )
            for row_idx in self.spatial_strategy.row_groups(self.n_mic):
                self.demix = self.spatial_strategy.update(
                    self.cov,
                    self.demix,
                    row_idx=row_idx,
                )

        ref = self.ref_mic if reference_mic is None else int(reference_mic)
        recon_strategy = self.reconstruction_strategy
        if isinstance(recon_strategy, ProjectionBackDemixReconstructionStrategy):
            recon_strategy = ProjectionBackDemixReconstructionStrategy(ref_mic=ref)

        output = recon_strategy.reconstruct(
            ReconstructionRequest(
                mixture=x,
                demix_filter=self.demix,
            )
        )
        return output.estimate

    def fit(self, spectrogram: np.ndarray) -> None:
        """Process a full spectrogram sequentially."""
        for n in range(spectrogram.shape[1]):
            self.partial_fit(spectrogram[:, n])

    def _l1_normalise_W(self) -> None:
        colsum = np.maximum(self.basis.sum(axis=0, keepdims=True), self.eps)
        self.basis /= colsum
        self.A /= colsum
        self.B *= colsum

    @property
    def n_sources(self) -> int:
        """Return number of separated sources."""
        return int(self.n_mic)

    def process_frame(
        self,
        frame: np.ndarray,
        request: OnlineFrameRequest | None = None,
    ) -> np.ndarray:
        """Process one TF frame and return separated frame."""
        ref_mic = None if request is None else request.reference_mic
        return self.partial_fit(frame, reference_mic=ref_mic)
