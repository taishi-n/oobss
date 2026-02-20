import numpy as np
import pytest

from oobss.benchmark.config_schema import FrameEvalConfig
from oobss.evaluation.metrics import compute_metrics, normalize_framewise_metrics


def test_normalize_framewise_metrics_adds_channel_aliases() -> None:
    raw = {
        "si_sdr": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "perm": np.array([[0, 0], [1, 1]], dtype=np.int64),
        "si_sdr_mix": np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64),
        "si_sdr_imp": np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64),
    }
    normalized = normalize_framewise_metrics(raw)
    assert normalized is not None
    for key in (
        "mean_si_sdr_channels",
        "median_si_sdr_channels",
        "mean_si_sdr_mix_channels",
        "median_si_sdr_mix_channels",
        "mean_si_sdr_imp_channels",
        "median_si_sdr_imp_channels",
    ):
        assert key in normalized


def test_compute_metrics_returns_normalized_framewise_keys() -> None:
    n_samples = 32
    fs = 8000
    t = np.arange(n_samples, dtype=np.float64) / float(fs)
    reference = np.stack(
        [
            np.sin(2.0 * np.pi * 200.0 * t),
            np.cos(2.0 * np.pi * 350.0 * t),
        ],
        axis=0,
    )
    estimate = np.array(reference, copy=True)
    mixture = 0.6 * reference

    bundle = compute_metrics(
        reference,
        estimate,
        mixture,
        fs,
        filter_length=1,
        frame_cfg=FrameEvalConfig(window_sec=0.002, hop_sec=0.001),
    )
    assert bundle.framewise is not None
    assert "mean_si_sdr_channels" in bundle.framewise
    assert "mean_si_sdr_imp_channels" in bundle.framewise


def test_compute_metrics_supports_single_channel_mixture_with_permutation() -> None:
    rng = np.random.default_rng(42)
    n_samples = 256
    reference = rng.standard_normal((3, n_samples))
    estimate = np.array(reference, copy=True)
    mixture_ref = np.sum(reference, axis=0, keepdims=True)  # (1, N)

    bundle = compute_metrics(
        reference,
        estimate,
        mixture_ref,
        16000,
        filter_length=1,
        frame_cfg=FrameEvalConfig(window_sec=0.01, hop_sec=0.005),
        compute_permutation=True,
    )
    assert bundle.sdr_mix.shape == (1,)
    assert bundle.sdr_est.shape == (3,)
    summary = bundle.to_summary()
    channels = summary["sdr_imp_channels"]
    assert isinstance(channels, list)
    assert len(channels) == 3


def test_compute_metrics_rejects_channel_mismatch_without_permutation() -> None:
    rng = np.random.default_rng(0)
    reference = rng.standard_normal((3, 128))
    estimate = rng.standard_normal((2, 128))
    mixture = rng.standard_normal((1, 128))

    with pytest.raises(ValueError):
        compute_metrics(
            reference,
            estimate,
            mixture,
            16000,
            filter_length=1,
            frame_cfg=None,
            compute_permutation=False,
        )
