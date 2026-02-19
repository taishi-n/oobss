from __future__ import annotations

from typing import cast

import numpy as np
from fast_bss_eval import bss_eval_sources
from scipy.signal import ShortTimeFFT, get_window

from oobss import AuxIVA, OnlineISNMF
from oobss.separators.core import NormalizationRequest, StreamRequest
from oobss.separators.strategies import (
    LaplaceSourceStrategy,
    ProjectionBackNormalization,
)

BssevalMetrics = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _build_synthetic_mix(seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_samples = 4096
    sources = rng.laplace(size=(2, n_samples)).astype(np.float64)
    sources /= np.std(sources, axis=1, keepdims=True)
    mixing = np.array([[1.0, 0.5], [0.3, 1.2]], dtype=np.float64)
    mixture = (mixing @ sources).T  # (n_samples, n_mic)
    return sources, mixture


def _stft() -> ShortTimeFFT:
    return ShortTimeFFT(win=get_window("hann", 256, fftbins=True), hop=128, fs=8000)


def test_auxiva_si_sdr_regression() -> None:
    ref, mix = _build_synthetic_mix()
    stft = _stft()

    obs = stft.stft(mix.T).transpose(2, 1, 0)  # (n_frame, n_freq, n_mic)
    model = AuxIVA(obs, source=LaplaceSourceStrategy())
    model.run(10)

    projected = ProjectionBackNormalization(ref_mic=0).apply(
        NormalizationRequest(
            estimate=model.estimated,
            observations=obs,
            demix_filter=model.demix_filter,
        )
    )
    est = np.real(stft.istft(projected.transpose(2, 1, 0)))[:, : ref.shape[1]]

    sdr_mix, _, _, _ = cast(
        BssevalMetrics, bss_eval_sources(ref, mix.T, filter_length=1)
    )
    sdr_est, _, _, _ = cast(BssevalMetrics, bss_eval_sources(ref, est, filter_length=1))
    sdr_mix = np.asarray(sdr_mix, dtype=np.float64)
    sdr_est = np.asarray(sdr_est, dtype=np.float64)
    sdr_imp = sdr_est - sdr_mix

    expected_sdr_est = np.array([9.50653783, 6.47766787])
    expected_sdr_imp = np.array([3.49289370, -5.56692646])
    np.testing.assert_allclose(sdr_est, expected_sdr_est, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(sdr_imp, expected_sdr_imp, atol=1e-6, rtol=0.0)


def test_online_isnmf_si_sdr_regression() -> None:
    ref, mix = _build_synthetic_mix()
    stft = _stft()

    spec = stft.stft(mix[:, 0]).astype(np.complex128)  # (n_freq, n_frame)
    model = OnlineISNMF(
        n_components=8,
        n_features=spec.shape[0],
        n_sources=2,
        inner_iter=5,
        beta=3,
        random_state=0,
    )
    out = model.process_stream_tf(spec, request=StreamRequest(frame_axis=1))
    assert out.estimate_tf is not None
    separated = out.estimate_tf
    est = np.real(stft.istft(separated.transpose(1, 0, 2), f_axis=0, t_axis=2))
    est = est[: ref.shape[1], :].T

    mix_baseline = np.tile(mix[:, 0][None, :], (2, 1))
    sdr_mix, _, _, _ = cast(
        BssevalMetrics, bss_eval_sources(ref, mix_baseline, filter_length=1)
    )
    sdr_est, _, _, _ = cast(BssevalMetrics, bss_eval_sources(ref, est, filter_length=1))
    sdr_mix = np.asarray(sdr_mix, dtype=np.float64)
    sdr_est = np.asarray(sdr_est, dtype=np.float64)
    sdr_imp = sdr_est - sdr_mix

    expected_sdr_est = np.array([4.79318731, -6.29350249])
    expected_sdr_imp = np.array([-1.22045681, -0.26798303])
    np.testing.assert_allclose(sdr_est, expected_sdr_est, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(sdr_imp, expected_sdr_imp, atol=1e-6, rtol=0.0)
