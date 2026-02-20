"""Plotting utilities for spectrogram-like representations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_nmf_factors(
    x: np.ndarray,
    basis: np.ndarray,
    activations: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    """Plot NMF factors and reconstructed spectrogram in a compact grid."""
    if x.ndim != 2:
        raise ValueError("x must be a 2-D array shaped (n_freq, n_frame)")
    if basis.ndim != 2:
        raise ValueError("basis must be a 2-D array shaped (n_freq, n_components)")
    if activations.ndim != 2:
        raise ValueError(
            "activations must be a 2-D array shaped (n_components, n_frame)"
        )

    n_freq, n_frame = x.shape
    _, n_components = basis.shape
    if activations.shape != (n_components, n_frame):
        raise ValueError(
            "activations shape must match (n_components, n_frame): "
            f"got {activations.shape}, expected {(n_components, n_frame)}"
        )

    boxwidth = 0.1
    line_width = 0.2
    freq_axis = np.arange(n_freq)
    time_axis = np.arange(n_frame)

    fig_width, fig_height = plt.gcf().get_size_inches()
    margin_ratio = 1.0 / 4.0
    dx = fig_width * margin_ratio
    dy = fig_height * margin_ratio

    fig, axes = plt.subplots(
        nrows=n_components + 1,
        ncols=n_components + 1,
        figsize=(dx + fig_width, dy + fig_height),
        width_ratios=[dx / n_components] * n_components + [fig_width],
        height_ratios=[fig_height] + [dy / n_components] * n_components,
    )

    reconstructed = np.maximum(basis @ activations, 1.0e-12)
    axes[0, n_components].imshow(
        10.0 * np.log10(reconstructed),
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )

    for k in range(n_components):
        axes[0, k].plot(-basis[:, k], freq_axis, linewidth=line_width)
        axes[k + 1, n_components].plot(
            time_axis, activations[k, :], linewidth=line_width
        )

    for row in range(n_components):
        for col in range(n_components):
            fig.delaxes(axes[row + 1, col])

    for ax in axes.flat:
        ax.tick_params(
            axis="both",
            length=0,
            width=0,
            labelbottom=False,
            labelleft=False,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(boxwidth)

    return fig


def save_channel_spectrograms(
    spec: np.ndarray,
    name: str,
    outdir: str | Path,
    *,
    vmin: float = -40.0,
    vmax: float = 20.0,
) -> list[Path]:
    """Save one spectrogram image per channel from a frame-first spectrogram."""
    if spec.ndim != 3:
        raise ValueError("spec must be 3-D shaped (n_frame, n_freq, n_channel)")

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    n_channel = spec.shape[-1]
    for ch in range(n_channel):
        path = output_dir / f"{name}-{ch}.pdf"
        power = np.maximum(np.abs(spec[:, :, ch].T), 1.0e-12)
        plt.imshow(10.0 * np.log10(power), vmin=vmin, vmax=vmax, rasterized=True)
        plt.axis("off")
        plt.savefig(path)
        plt.close()
        saved.append(path)

    return saved
