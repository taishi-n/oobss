import numpy as np

from oobss.experiments.config_schema import FrameEvalConfig
from oobss.experiments.metrics import compute_metrics, normalize_framewise_metrics


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
