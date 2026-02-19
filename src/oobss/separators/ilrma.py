"""Independent low-rank matrix analysis."""

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
    ILRMANMFSourceStrategy,
    IP1SpatialStrategy,
)

eps = np.finfo(np.float64).eps


class ILRMA(BaseIterativeSeparator):
    """
    Base class for independent low-rank matrix analysis.

    Procedure
    ---------
    ```text

       input: {x_{f,t}}_{f=1..F, t=1..T}, NMF rank L, iterations I
       initialize W_f = I_M, b_{k,f,\\ell}, c_{k,\\ell,t}
       for i = 1..I:
           y_{f,t} <- W_f x_{f,t}
           for each source k:
               r_{k,f,t} <- sum_\\ell b_{k,f,\\ell} c_{k,\\ell,t}
               update b_{k,f,\\ell}, c_{k,\\ell,t} by MU using |y_{k,f,t}|^2
           V_{k,f} <- (1/T) * sum_t x_{f,t} x_{f,t}^H / r_{k,f,t}
           update w_{k,f} by IP1/IP2
       return y_{f,t}
    ```

    Update Equations
    ----------------
    Indices are $k=1,\\dots,K$ (source), $m=1,\\dots,M$
    (channel), $f=1,\\dots,F$ (frequency), $t=1,\\dots,T$
    (time frame), and $\\ell=1,\\dots,L$ (NMF basis index). In the
    determined case, $K=M$.

    Notation:

    $$
       \\bm{x}_{f,t} \\in \\mathbb{C}^{M}, \\quad
       \\bm{y}_{f,t} = W_f\\bm{x}_{f,t}, \\quad
       y_{k,f,t} = \\bm{w}_{k,f}^{\\mathsf{H}}\\bm{x}_{f,t}
    $$

    Source variance model with NMF basis entries $b_{k,f,\\ell}$ and
    activation entries $c_{k,\\ell,t}$:

    $$
       r_{k,f,t} = \\sum_{\\ell=1}^{L} b_{k,f,\\ell}c_{k,\\ell,t}
    $$

    For each source $k$, the multiplicative updates are:

    $$
       b_{k,f,\\ell} \\leftarrow b_{k,f,\\ell}
       \\frac{
       \\sum_{t=1}^{T}|y_{k,f,t}|^2r_{k,f,t}^{-2}c_{k,\\ell,t}
       }{
       \\sum_{t=1}^{T}r_{k,f,t}^{-1}c_{k,\\ell,t}
       },
       \\quad
       c_{k,\\ell,t} \\leftarrow c_{k,\\ell,t}
       \\frac{
       \\sum_{f=1}^{F}|y_{k,f,t}|^2r_{k,f,t}^{-2}b_{k,f,\\ell}
       }{
       \\sum_{f=1}^{F}r_{k,f,t}^{-1}b_{k,f,\\ell}
       }
    $$

    Weighted covariance:

    $$
       V_{k,f} = \\frac{1}{T}\\sum_{t=1}^{T}
       \\frac{
       \\bm{x}_{f,t}\\bm{x}_{f,t}^{\\mathsf{H}}
       }{
       r_{k,f,t}
       }
    $$

    IP1 update for source $k$:

    $$
       \\tilde{\\bm{w}}_{k,f}
       = (W_fV_{k,f})^{-1}\\bm{e}_{k}, \\quad
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
        Source-model strategy (typically ILRMA NMF MU).
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
       from oobss import ILRMA

       fs = 16000
       fft_size = 2048
       hop_size = 512
       win = get_window("hann", fft_size, fftbins=True)
       stft = ShortTimeFFT(win=win, hop=hop_size, fs=fs)

       mixture_time = np.random.randn(fs * 2, 2)  # (n_samples, n_mic)
       X_tfm = stft.stft(mixture_time.T).transpose(2, 1, 0)  # (T, F, M)

       model = ILRMA(X_tfm, n_basis=8, random_state=0)
       out = model.fit_transform_tf(X_tfm, n_iter=50)
       Y_tfm = out.estimate_tf
       if Y_tfm is None:
           raise ValueError("ILRMA did not return TF estimates.")

       y_time = np.real(stft.istft(Y_tfm.transpose(2, 1, 0)))
    ```

    Warm-start NMF factors and demixing:

    ```python

       # Initial factors: basis0=(n_src, n_freq, n_basis),
       # activ0=(n_src, n_frame, n_basis)
       basis0 = np.abs(np.random.randn(2, X_tfm.shape[1], 8)) + 1e-6
       activ0 = np.abs(np.random.randn(2, X_tfm.shape[0], 8)) + 1e-6

       model = ILRMA(
           X_tfm,
           n_basis=8,
           basis0=basis0,
           activ0=activ0,
           random_state=0,
       )
       model.run(30)
       Y_tfm = model.get_estimate()
    ```

    References
    ----------
    [1] D. Kitamura, N. Ono, H. Sawada, H. Kameoka, and H. Saruwatari,
    "Determined blind source separation unifying independent vector analysis and
    nonnegative matrix factorization," *IEEE/ACM Trans. Audio, Speech, and
    Language Processing*, vol. 24, no. 9, pp. 1622-1637, Sep. 2016,
    doi: 10.1109/TASLP.2016.2577880.
    """

    def __init__(
        self,
        observations,
        *,
        n_basis: int = 10,
        basis0=None,
        activ0=None,
        random_state: int | None = None,
        rng: np.random.Generator | None = None,
        spatial: SpatialUpdateStrategy | None = None,
        source: SourceModelStrategy | None = None,
        covariance: CovarianceUpdateStrategy | None = None,
        reconstruction_strategy: ReconstructionStrategy | None = None,
    ):
        """Initialize parameters in ILRMA."""
        # Setup
        self.observations = observations
        self.n_basis = int(n_basis)
        self.rng = np.random.default_rng(random_state) if rng is None else rng
        self._basis0 = None if basis0 is None else np.array(basis0, copy=True)
        self._activ0 = None if activ0 is None else np.array(activ0, copy=True)
        self.spatial_strategy = spatial if spatial is not None else IP1SpatialStrategy()
        self.source_strategy = (
            source if source is not None else ILRMANMFSourceStrategy(eps=float(eps))
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
        y_power = np.square(np.abs(self.estimated))

        row_groups = self.spatial_strategy.row_groups(n_src)
        pairwise_mode = any(np.ndim(row_idx) > 0 for row_idx in row_groups)

        if pairwise_mode:
            for row_idx in row_groups:
                for src_idx in np.atleast_1d(np.asarray(row_idx, dtype=np.int64)):
                    b = self.basis[src_idx]
                    a = self.activ[src_idx]
                    self.basis[src_idx], self.activ[src_idx] = self.calc_source_model(
                        b, a, y_power[:, :, src_idx]
                    )
                    self.source_model[:, :, src_idx] = (
                        self.activ[src_idx] @ self.basis[src_idx].T
                    )
                self.covariance = self.covariance_strategy.update(
                    CovarianceRequest(
                        observed=self.observations,
                        source_model=self.source_model,
                    )
                )
                self.demix_filter[:, :, :] = self.spatial_strategy.update(
                    self.covariance,
                    self.demix_filter,
                    row_idx=row_idx,
                )
        else:
            for s in range(n_src):
                b = self.basis[s]
                a = self.activ[s]
                self.basis[s], self.activ[s] = self.calc_source_model(
                    b, a, y_power[:, :, s]
                )
                self.source_model[:, :, s] = self.activ[s] @ self.basis[s].T
            self.covariance = self.covariance_strategy.update(
                CovarianceRequest(
                    observed=self.observations,
                    source_model=self.source_model,
                )
            )
            for row_idx in row_groups:
                self.demix_filter[:, :, :] = self.spatial_strategy.update(
                    self.covariance,
                    self.demix_filter,
                    row_idx=row_idx,
                )

        # 3. Update estimated sources
        recon = self.reconstruction_strategy.reconstruct(
            ReconstructionRequest(
                mixture=self.observations,
                demix_filter=self.demix_filter,
            )
        )
        self.estimated = recon.estimate

        # 4. Update loss function value
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
                "ILRMA expects mixture_tf with shape (n_frame, n_freq, n_mic)."
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

        if self._basis0 is None:
            self.basis = self.init_basis()
        else:
            self.basis = np.array(self._basis0, copy=True)
        if self._activ0 is None:
            self.activ = self.init_activ()
        else:
            self.activ = np.array(self._activ0, copy=True)

        if self.basis.shape[:2] != (
            self.observations.shape[-1],
            self.observations.shape[1],
        ):
            raise ValueError(
                "basis0 shape mismatch: expected (n_src, n_freq, n_basis) for mixture_tf."
            )
        if self.activ.shape[:2] != (
            self.observations.shape[-1],
            self.observations.shape[0],
        ):
            raise ValueError(
                "activ0 shape mismatch: expected (n_src, n_frame, n_basis) for mixture_tf."
            )

        self.source_model = self.init_source_model()
        self.covariance = None
        self.loss = self.calc_loss()

    def init_basis(self):
        """Initialize basis matrix."""
        _, n_freq, n_src = self.observations.shape
        return np.ones((n_src, n_freq, self.n_basis))

    def init_activ(self):
        """Initialize activation matrix."""
        n_frame, _, n_src = self.observations.shape
        return self.rng.uniform(
            low=0.1,
            high=1.0,
            size=(n_src, n_frame, self.n_basis),
        )

    def init_source_model(self):
        """Initialize source variance model ``R`` with shape ``(T, F, N)``."""
        return self._compose_source_model()

    def _compose_source_model(self) -> np.ndarray:
        """Compose source variance model from basis/activation factors.

        Returns
        -------
        np.ndarray
            Source variance model ``R`` with shape ``(n_frame, n_freq, n_src)``.
        """
        return np.einsum(
            "sfk,stk->tfs",
            self.basis,
            self.activ,
            optimize=True,
        )

    def calc_source_model(self, B, A, y_power):
        """
        Calculate source model.
        By overriding this method, various source models (e.g., Student t, ILRMA-T, generalized Kullback---Leibler divergence, or IDLMA) can be applied.

        Parameters
        ----------
        B : ndarray of shape (n_freq, n_basis)
            Basis matrix
        A : ndarray of shape (n_frame, n_basis)
            Activation matrix
        y_power : ndarray of shape (n_frame, n_freq)
            Power spectrograms of estimated source

        Returns
        -------
        """
        updated = self.source_strategy.update(
            SourceModelRequest(
                basis=B,
                activ=A,
                y_power=y_power,
            )
        )
        if updated.basis is None or updated.activ is None:
            raise ValueError("source strategy did not return basis/activ.")
        return updated.basis, updated.activ

    def calc_loss(self, axis=None):
        """
        Calculate loss function value of ILRMA.

        Parameters
        ----------
        axis : int or None, default=None

        Raises
        ------
        ValueError:
            If `cost` is infinite or not a number.
        """
        # (n_frame, n_freq, n_src)
        y_power = np.square(np.abs(self.estimated))

        # basis: (n_src, n_freq, n_basis)
        # activ: (n_src, n_frame, n_basis)

        src_var = self._compose_source_model()

        # (n_freq,)
        target_loss = -2 * np.linalg.slogdet(self.demix_filter)[1]

        # (n_frame, n_freq)
        demix_loss = np.sum(y_power / src_var + np.log(src_var), axis=2)

        cost = np.sum(demix_loss + target_loss[None, :], axis=axis)
        if np.isinf(cost).any() or np.isnan(cost).any():
            raise ValueError("Cost cannot be calculated.")
        else:
            return cost

    @property
    def n_sources(self) -> int:
        """Return number of separated sources."""
        return int(self.observations.shape[-1])

    def get_estimate(self) -> np.ndarray:
        """Return current TF-domain estimate."""
        return self.estimated
