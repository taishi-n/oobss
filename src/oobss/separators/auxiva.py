"""Auxiliary-function-based independent vector analysis."""

import numpy as np

from .core import (
    BaseIterativeSeparator,
    CovarianceRequest,
    CovarianceUpdateStrategy,
    ReconstructionRequest,
    ReconstructionStrategy,
    SourceModelRequest,
    SourceModelStrategy,
    SpatialUpdateStrategy,
)
from .strategies import DemixReconstructionStrategy
from .strategies import (
    BatchCovarianceStrategy,
    GaussSourceStrategy,
    IP1SpatialStrategy,
)
from .utils import tensor_H

eps = np.finfo(np.float64).eps


class AuxIVA(BaseIterativeSeparator):
    """
    Base class for auxiliary-function-based independent vector analysis.

    Procedure
    ---------
    ```text

       input: {x_{f,t}}_{f=1..F, t=1..T}, iterations I
       initialize W_f = I_M
       for i = 1..I:
           y_{f,t} <- W_f x_{f,t}
           compute r_{k,t} and phi(r_{k,t}) (Gauss or Laplace)
           V_{k,f} <- (1/T) * sum_t phi(r_{k,t}) x_{f,t} x_{f,t}^H
           update w_{k,f} by IP1/IP2 for each k
       return y_{f,t}
    ```

    Update Equations
    ----------------
    Indices are $k=1,\\dots,K$ (source), $m=1,\\dots,M$
    (channel), $f=1,\\dots,F$ (frequency), and
    $t=1,\\dots,T$ (time frame). In the determined case, $K=M$.

    Notation:

    $$
       \\bm{x}_{f,t} \\in \\mathbb{C}^{M}, \\quad
       \\bm{y}_{f,t} = W_f \\bm{x}_{f,t} \\in \\mathbb{C}^{M},
       \\quad
       y_{k,f,t} = \\bm{w}_{k,f}^{\\mathsf{H}}\\bm{x}_{f,t}
    $$

    $$
       r_{k,t} = \\left(\\sum_{f=1}^{F}|y_{k,f,t}|^2\\right)^{1/2}, \\quad
       \\varphi_{\\mathrm{Laplace}}(r) = \\frac{1}{\\max(2r,\\varepsilon)}, \\quad
       \\varphi_{\\mathrm{Gauss}}(r) = \\frac{1}{\\max(r^2/F,\\varepsilon)}
    $$

    The weighted covariance for source $k$ is:

    $$
       V_{k,f}
       = \\frac{1}{T}\\sum_{t=1}^{T}
       \\varphi(r_{k,t})\\,\\bm{x}_{f,t}\\bm{x}_{f,t}^{\\mathsf{H}}
    $$

    IP1 update for the $k$-th demixing row:

    $$
       \\tilde{\\bm{w}}_{k,f}
       = (W_f V_{k,f})^{-1}\\bm{e}_k, \\quad
       \\bm{w}_{k,f}
       = \\frac{\\tilde{\\bm{w}}_{k,f}}
       {\\sqrt{
       \\tilde{\\bm{w}}_{k,f}^{\\mathsf{H}}
       V_{k,f}
       \\tilde{\\bm{w}}_{k,f}
       }}
    $$

    Common projection-back post-processing
    --------------------------------------
    AuxIVA/ILRMA and their online variants use the same projection-back rule.
    For a reference microphone $m_{\\mathrm{ref}}$:

    $$
       A_f = W_f^{-1}, \\quad
       \\hat{y}_{k,f,t} = a_{k,f}[m_{\\mathrm{ref}}] y_{k,f,t}
    $$

    The shared implementation lives in
    :func:`oobss.separators.utils.projection_back`.

    Attributes
    ----------
    observations : ndarray of shape (n_frame, n_freq, n_src)
    spatial : SpatialUpdateStrategy
        Demixing-matrix strategy (e.g., IP1/IP2).
    source : SourceModelStrategy
        Source-model strategy (e.g., Gauss/Laplace).
    covariance : CovarianceUpdateStrategy
        Weighted covariance strategy.

    estimated : ndarray of shape (n_frame, n_freq, n_src)
    source_model : ndarray of shape (n_frame, n_freq, n_src)
    demix_filter : ndarray of shape (n_freq, n_src, n_src)
    loss : list[float]

    Examples
    --------
    Basic TF-domain usage:

    ```python

       import numpy as np
       from scipy.signal import ShortTimeFFT, get_window
       from oobss import AuxIVA

       fs = 16000
       fft_size = 2048
       hop_size = 512
       win = get_window("hann", fft_size, fftbins=True)
       stft = ShortTimeFFT(win=win, hop=hop_size, fs=fs)

       # mixture_time: (n_samples, n_mic)
       mixture_time = np.random.randn(fs * 2, 2)
       # channel-first STFT: (n_mic, n_freq, n_frame)
       X_cft = stft.stft(mixture_time.T)
       # AuxIVA input must be frame-first: (n_frame, n_freq, n_mic)
       X_tfm = X_cft.transpose(2, 1, 0)

       model = AuxIVA(X_tfm)
       out = model.fit_transform_tf(X_tfm, n_iter=30)
       Y_tfm = out.estimate_tf
       if Y_tfm is None:
           raise ValueError("AuxIVA did not return TF estimates.")

       # Reconstruct separated waveforms: (n_src, n_samples)
       y_time = np.real(stft.istft(Y_tfm.transpose(2, 1, 0)))
    ```

    Strategy plug-and-play (fix source model, swap spatial update):

    ```python

       from oobss.separators.strategies import (
           BatchCovarianceStrategy,
           GaussSourceStrategy,
           IP2SpatialStrategy,
       )

       model = AuxIVA(
           X_tfm,
           source=GaussSourceStrategy(),          # fixed source model
           covariance=BatchCovarianceStrategy(),  # fixed covariance update
           spatial=IP2SpatialStrategy(),          # swapped demixing update
       )
       model.run(20)
       Y_tfm = model.get_estimate()
    ```

    References
    ----------
    [1] N. Ono, "Stable and fast update rules for independent vector analysis
    based on auxiliary function technique," in *Proc. IEEE Workshop on
    Applications of Signal Processing to Audio and Acoustics (WASPAA)*, pp.
    189-192, Oct. 2011, doi: 10.1109/ASPAA.2011.6082320.
    """

    def __init__(
        self,
        observations,
        *,
        spatial: SpatialUpdateStrategy | None = None,
        source: SourceModelStrategy | None = None,
        covariance: CovarianceUpdateStrategy | None = None,
        reconstruction_strategy: ReconstructionStrategy | None = None,
    ):
        """Initialize parameters in AuxIVA."""
        # Setup
        self.observations = observations
        self.spatial_strategy = spatial if spatial is not None else IP1SpatialStrategy()
        self.source_strategy = (
            source if source is not None else GaussSourceStrategy(eps=float(eps))
        )
        self.covariance_strategy = (
            covariance if covariance is not None else BatchCovarianceStrategy()
        )
        self.reconstruction_strategy = (
            reconstruction_strategy
            if reconstruction_strategy is not None
            else DemixReconstructionStrategy()
        )
        # Results
        self.bind_mixture_tf(np.asarray(observations))

    def step(self):
        """Update paramters one step."""
        n_src = self.observations.shape[-1]
        # 1. Update source model
        self.source_model = self.calc_source_model()

        # 2. Update covariance
        self.covariance = self.covariance_strategy.update(
            CovarianceRequest(
                observed=self.observations,
                source_model=self.source_model,
            )
        )

        # 3. Update demixing filter
        for row_idx in self.spatial_strategy.row_groups(n_src):
            self.demix_filter[:, :, :] = self.spatial_strategy.update(
                self.covariance,
                self.demix_filter,
                row_idx=row_idx,
            )

        # 4. Update estimated sources
        recon = self.reconstruction_strategy.reconstruct(
            ReconstructionRequest(
                mixture=self.observations,
                demix_filter=self.demix_filter,
            )
        )
        self.estimated = recon.estimate

        # 5. Update loss function value
        self.loss = self.calc_loss()

    def init_demix(self):
        """Initialize demixing matrix."""
        _, n_freq, n_src = self.observations.shape
        W0 = np.zeros((n_freq, n_src, n_src), dtype=complex)
        W0[:, :, :n_src] = np.tile(np.eye(n_src, dtype=complex), (n_freq, 1, 1))
        return W0

    def bind_mixture_tf(self, mixture_tf: np.ndarray) -> None:
        """Bind a TF-domain mixture and reset internal iterative state."""
        observations = np.asarray(mixture_tf)
        if observations.ndim != 3:
            raise ValueError(
                "AuxIVA expects mixture_tf with shape (n_frame, n_freq, n_mic)."
            )

        self.observations = observations
        self.demix_filter = self.init_demix()
        recon = self.reconstruction_strategy.reconstruct(
            ReconstructionRequest(
                mixture=self.observations,
                demix_filter=self.demix_filter,
            )
        )
        self.estimated = recon.estimate
        self.source_model = None
        self.covariance = None
        self.loss = self.calc_loss()

    def calc_source_model(self):
        """
        Calculate source model.

        Returns
        -------
        ndarray of shape (n_frame, n_freq, n_src)
        """
        updated = self.source_strategy.update(
            SourceModelRequest(
                estimated=self.estimated,
                n_freq=int(self.observations.shape[1]),
            )
        )
        if updated.source_model is None:
            raise ValueError("source strategy did not return source_model.")
        return updated.source_model

    def _source_model_name_for_loss(self) -> str:
        name = getattr(self.source_strategy, "model", "Gauss")
        if not isinstance(name, str):
            return "Gauss"
        normalized = name.capitalize()
        if normalized not in {"Gauss", "Laplace"}:
            raise ValueError(
                "calc_loss supports Gauss/Laplace source models. "
                f"Got source_strategy.model={name!r}."
            )
        return normalized

    def calc_loss(self):
        """Calculate loss function value."""
        n_frames, _, _ = self.estimated.shape

        def f_norm(y):
            return np.linalg.norm(y, axis=1)

        contrast_func = {
            "Laplace": lambda y: np.sum(f_norm(y)),
            "Gauss": lambda y: np.sum(np.log(1.0 / np.maximum(eps, f_norm(y)))),
        }[self._source_model_name_for_loss()]
        target_loss = contrast_func(self.estimated)

        tfn_fnt = [1, 2, 0]
        XX = self.observations.transpose(tfn_fnt)
        YY = self.estimated.transpose(tfn_fnt)
        W_H = np.linalg.solve(XX @ tensor_H(XX), XX @ tensor_H(YY))
        _, logdet = np.linalg.slogdet(W_H)
        demix_loss = -2 * n_frames * np.sum(logdet)

        return target_loss + demix_loss

    @property
    def n_sources(self) -> int:
        """Return number of separated sources."""
        return int(self.observations.shape[-1])

    def get_estimate(self) -> np.ndarray:
        """Return current TF-domain estimate."""
        return self.estimated
