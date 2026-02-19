"""Online auxiliary-function-based independent vector analysis."""

from __future__ import annotations

import numpy as np

from .core import (
    BaseStreamingSeparator,
    CovarianceRequest,
    CovarianceUpdateStrategy,
    OnlineFrameRequest,
    ReconstructionRequest,
    ReconstructionStrategy,
    SourceModelRequest,
    SourceModelStrategy,
    SeparatorState,
    StreamingSeparatorState,
    SpatialUpdateStrategy,
)
from .strategies import (
    EMACovarianceStrategy,
    GaussSourceStrategy,
    IP1SpatialStrategy,
    ProjectionBackDemixReconstructionStrategy,
)


class OnlineAuxIVA(BaseStreamingSeparator):
    """Online auxiliary-function-based independent vector analysis.

    Procedure
    ---------
    ```text

       input: frame sequence {x_{f,t}}_{f=1..F, t=1..T}
       initialize W_{f,0} = I_M and V_{k,f,0}
       for each frame t:
           repeat inner_iter times:
               y_{f,t} <- W_{f,t} x_{f,t}
               compute r_{k,t} and phi(r_{k,t}) (Gauss)
               V_{k,f,t} <- (1-alpha) * phi(r_{k,t}) x_{f,t} x_{f,t}^H
                          + alpha * V_{k,f,t-1}
               update w_{k,f,t} by IP1
           projection-back by reference microphone
           emit separated frame
    ```

    Update Equations
    ----------------
    Indices are $k=1,\\dots,K$ (source), $m=1,\\dots,M$
    (channel), $f=1,\\dots,F$ (frequency), and
    $t=1,\\dots,T$ (time frame). In the determined case, $K=M$.

    Per-frame demixing:

    $$
       \\bm{x}_{f,t} \\in \\mathbb{C}^{M}, \\quad
       \\bm{y}_{f,t} = W_{f,t}\\bm{x}_{f,t}, \\quad
       y_{k,f,t} = \\bm{w}_{k,f,t}^{\\mathsf{H}}\\bm{x}_{f,t}
    $$

    The default source model is AuxIVA Gauss:

    $$
       r_{k,t} = \\left(\\sum_{f=1}^{F}|y_{k,f,t}|^2\\right)^{1/2}, \\quad
       \\varphi(r_{k,t}) = \\frac{1}{\\max(r_{k,t}^2/F,\\varepsilon)}
    $$

    Online covariance recursion:

    $$
       \\widehat{V}_{k,f,t}
       = \\varphi(r_{k,t})
       \\bm{x}_{f,t}\\bm{x}_{f,t}^{\\mathsf{H}},
       \\quad
       V_{k,f,t}
       \\leftarrow
       (1-\\alpha)\\widehat{V}_{k,f,t} + \\alpha V_{k,f,t-1}
    $$

    Demixing row update (IP1):

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

    Common projection-back post-processing:

    $$
       A_{f,t} = W_{f,t}^{-1}, \\quad
       \\hat{y}_{k,f,t} = a_{k,f,t}[m_{\\mathrm{ref}}] y_{k,f,t}
    $$

    This is shared with batch AuxIVA/ILRMA via
    :func:`oobss.separators.utils.projection_back`.

    Parameters
    ----------
    n_mic:
        Number of microphones / separated sources.
    n_freq:
        Number of frequency bins.
    ref_mic:
        Reference microphone for projection-back reconstruction.
    forget:
        Forgetting factor in covariance smoothing.
    inner_iter:
        Number of per-frame inner updates.
    eps:
        Numerical stability constant.
    cov_scale:
        Initial diagonal covariance scale.
    spatial:
        Strategy used to update demixing filters.
    source:
        Strategy used to compute source model per frame.
    covariance:
        Strategy used to update weighted covariance matrices.
    reconstruction_strategy:
        Strategy used to reconstruct output spectra.

    Examples
    --------
    Process a full stream at once:

    ```python

       import numpy as np
       from scipy.signal import ShortTimeFFT, get_window
       from oobss import OnlineAuxIVA, StreamRequest

       fs = 16000
       fft_size = 2048
       hop_size = 512
       win = get_window("hann", fft_size, fftbins=True)
       stft = ShortTimeFFT(win=win, hop=hop_size, fs=fs)

       mixture_time = np.random.randn(fs * 2, 2)  # (n_samples, n_mic)
       # channel-first STFT: (n_mic, n_freq, n_frame)
       X_cft = stft.stft(mixture_time.T)
       # online input: (n_freq, n_mic, n_frame)
       X_fmt = X_cft.transpose(1, 0, 2)

       model = OnlineAuxIVA(
           n_mic=2,
           n_freq=X_fmt.shape[0],
           ref_mic=0,
           forget=0.99,
           inner_iter=5,
       )
       out = model.process_stream_tf(
           X_fmt,
           request=StreamRequest(frame_axis=2, reference_mic=0),
       )
       Y_fmt = out.estimate_tf
       if Y_fmt is None:
           raise ValueError("OnlineAuxIVA did not return TF estimates.")

       # inverse STFT expects channel-first axes
       y_time = np.real(stft.istft(Y_fmt, f_axis=0, t_axis=2)).T
    ```

    Frame-by-frame update with explicit state carry:

    ```python

       from oobss import StreamingSeparatorState

       state: StreamingSeparatorState | None = None
       outputs = []
       for t in range(X_fmt.shape[2]):
           frame = X_fmt[:, :, t]  # (n_freq, n_mic)
           y_frame, state = model.forward_streaming(frame, state=state)
           outputs.append(y_frame)
       Y_fmt = np.stack(outputs, axis=2)
    ```

    References
    ----------
    [1] T. Taniguchi, N. Ono, A. Kawamura, and S. Sagayama, "An
    auxiliary-function approach to online independent vector analysis for
    real-time blind source separation," in *Proc. Joint Workshop on Hands-free
    Speech Communication and Microphone Arrays (HSCMA)*, pp. 107-111, May 2014,
    doi: 10.1109/HSCMA.2014.6843261.
    """

    def __init__(
        self,
        n_mic: int,
        n_freq: int,
        *,
        ref_mic: int = 0,
        forget: float = 0.9,
        inner_iter: int = 30,
        eps: float = 1.0e-12,
        cov_scale: float = 1.0e-6,
        spatial: SpatialUpdateStrategy | None = None,
        source: SourceModelStrategy | None = None,
        covariance: CovarianceUpdateStrategy | None = None,
        reconstruction_strategy: ReconstructionStrategy | None = None,
    ) -> None:
        self.n_mic = int(n_mic)
        self.n_freq = int(n_freq)
        self.ref_mic = int(ref_mic)
        self.alpha = float(forget)
        self.inner_iter = int(inner_iter)
        self.eps = float(eps)
        self.cov_scale = float(cov_scale)

        self.spatial_strategy = spatial if spatial is not None else IP1SpatialStrategy()
        self.source_strategy = (
            source if source is not None else GaussSourceStrategy(eps=self.eps)
        )
        self.covariance_strategy = (
            covariance
            if covariance is not None
            else EMACovarianceStrategy(alpha=self.alpha)
        )
        self.reconstruction_strategy = (
            reconstruction_strategy
            if reconstruction_strategy is not None
            else ProjectionBackDemixReconstructionStrategy(ref_mic=self.ref_mic)
        )

        self.reset()

    def reset(self) -> None:
        """Reset online state to its initial values."""
        self.demix = np.tile(np.eye(self.n_mic, dtype=complex), (self.n_freq, 1, 1))
        self.cov = (
            np.tile(np.eye(self.n_mic, dtype=complex), (self.n_mic, self.n_freq, 1, 1))
            * self.cov_scale
        )
        self.source_model = np.ones((self.n_freq, self.n_mic), dtype=np.float64)
        self._t = 0

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
            source_model_result = self.source_strategy.update(
                SourceModelRequest(
                    estimated=demixed[None, :, :],
                    n_freq=self.n_freq,
                )
            )
            if source_model_result.source_model is None:
                raise ValueError("source strategy must return source_model")
            source_model = np.maximum(source_model_result.source_model[0], self.eps)

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

        self.source_model = source_model
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
        self._t += 1
        return output.estimate

    def get_state(self) -> StreamingSeparatorState:
        """Return a typed snapshot of the current online state."""
        mix_filter: np.ndarray | None
        try:
            mix_filter = np.linalg.inv(self.demix)
        except np.linalg.LinAlgError:
            mix_filter = None
        return StreamingSeparatorState(
            source_model=np.array(self.source_model, copy=True),
            demix_filter=np.array(self.demix, copy=True),
            mix_filter=None if mix_filter is None else np.array(mix_filter, copy=True),
            frame_index=int(self._t),
            metadata={"covariance": np.array(self.cov, copy=True)},
        )

    def set_state(
        self,
        state: SeparatorState | StreamingSeparatorState,
    ) -> None:
        """Restore online state from :class:`StreamingSeparatorState`."""
        if not isinstance(state, StreamingSeparatorState):
            raise TypeError(
                "OnlineAuxIVA.set_state expects StreamingSeparatorState, "
                f"got {type(state).__name__}"
            )
        demix = state.demix_filter
        if demix is None:
            raise ValueError("state.demix_filter must be provided")
        if demix.shape != (self.n_freq, self.n_mic, self.n_mic):
            raise ValueError(
                "state.demix_filter must have shape "
                f"({self.n_freq}, {self.n_mic}, {self.n_mic}), got {demix.shape}"
            )
        self.demix = np.array(demix, copy=True)

        cov = state.metadata.get("covariance")
        if cov is None:
            cov = (
                np.tile(
                    np.eye(self.n_mic, dtype=complex), (self.n_mic, self.n_freq, 1, 1)
                )
                * self.cov_scale
            )
        cov_arr = np.asarray(cov)
        if cov_arr.shape != (self.n_mic, self.n_freq, self.n_mic, self.n_mic):
            raise ValueError(
                "state.metadata['covariance'] must have shape "
                f"({self.n_mic}, {self.n_freq}, {self.n_mic}, {self.n_mic}), "
                f"got {cov_arr.shape}"
            )
        self.cov = np.array(cov_arr, copy=True)

        if state.source_model is None:
            self.source_model = np.ones((self.n_freq, self.n_mic), dtype=np.float64)
        else:
            if state.source_model.shape != (self.n_freq, self.n_mic):
                raise ValueError(
                    "state.source_model must have shape "
                    f"({self.n_freq}, {self.n_mic}), got {state.source_model.shape}"
                )
            self.source_model = np.array(state.source_model, copy=True)
        self._t = int(state.frame_index)

    def fit(self, spectrogram: np.ndarray) -> None:
        """Process a full spectrogram with shape ``(n_freq, n_frames, n_mic)``."""
        for idx in range(spectrogram.shape[1]):
            self.partial_fit(spectrogram[:, idx])

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
