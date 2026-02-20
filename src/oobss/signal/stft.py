"""Shared STFT planning and construction utilities."""

from __future__ import annotations

from dataclasses import dataclass

from scipy.signal import ShortTimeFFT, get_window


@dataclass(frozen=True)
class STFTPlan:
    """STFT configuration shared across benchmark runners."""

    fft_size: int
    hop_size: int
    window: str


def build_stft(plan: STFTPlan, sample_rate: int) -> ShortTimeFFT:
    """Build a :class:`scipy.signal.ShortTimeFFT` instance from ``plan``."""
    win = get_window(plan.window, plan.fft_size, fftbins=True)
    return ShortTimeFFT(win=win, hop=plan.hop_size, fs=sample_rate)
