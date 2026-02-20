from pathlib import Path

import matplotlib
import numpy as np

from oobss.visualization import plot_nmf_factors, save_channel_spectrograms


matplotlib.use("Agg")


def test_plot_nmf_factors_returns_figure() -> None:
    x = np.abs(np.random.randn(16, 20))
    basis = np.abs(np.random.randn(16, 3))
    activations = np.abs(np.random.randn(3, 20))
    fig = plot_nmf_factors(x, basis, activations, vmin=-40, vmax=20)
    assert fig is not None


def test_save_channel_spectrograms_writes_files(tmp_path: Path) -> None:
    spec = np.random.randn(12, 10, 2) + 1j * np.random.randn(12, 10, 2)
    saved = save_channel_spectrograms(spec, "demo", tmp_path)
    assert len(saved) == 2
    assert all(path.exists() for path in saved)
