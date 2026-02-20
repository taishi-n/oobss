"""Shared helpers for runnable example scripts.

The helpers intentionally keep only example-specific glue while delegating
method execution, parameter defaults, and metric computation to ``oobss``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import yaml

from oobss.benchmark.config_loader import load_method_configs
from oobss.benchmark.methods import default_method_runner_registry
from oobss.evaluation.metrics import compute_metrics
from oobss.separators.core import SeparationOutput
from oobss.signal import STFTPlan, build_stft
from oobss.visualization import plot_nmf_factors, save_channel_spectrograms


_METHOD_CONFIG_DIR = (
    Path(__file__).resolve().parent / "benchmark" / "config" / "methods"
)

DEFAULT_OUTPUT_FILENAMES: dict[str, str] = {
    "batch_auxiva": "auxiva.wav",
    "batch_ilrma": "ilrma.wav",
    "blockbatch_auxiva": "mini_batch_auxiva.wav",
    "blockbatch_ilrma": "mini_batch_ilrma.wav",
    "online_auxiva": "online_auxiva.wav",
    "online_ilrma": "online_ilrma.wav",
    "online_isnmf": "online_isnmf.wav",
}

NMF_FACTOR_METHODS: set[str] = {
    "batch_ilrma",
    "blockbatch_ilrma",
    "online_ilrma",
    "online_isnmf",
}


@dataclass(frozen=True)
class SiSdrSummary:
    """SI-SDR statistics computed on aligned channels/samples."""

    sdr_mix: np.ndarray
    sdr_est: np.ndarray
    sdr_imp: np.ndarray
    permutation: np.ndarray
    n_sources: int
    n_samples: int


@lru_cache(maxsize=1)
def _default_method_params() -> dict[str, dict[str, object]]:
    methods = load_method_configs(_METHOD_CONFIG_DIR)
    return {method.id: dict(method.params) for method in methods}


def ensure_output_dir(path: str | Path) -> Path:
    """Create and return an output directory."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def available_methods() -> list[str]:
    """Return built-in method IDs."""
    return default_method_runner_registry().available()


def resolve_methods(requested: Sequence[str]) -> list[str]:
    """Resolve requested method list, expanding ``all``."""
    values = [item.strip() for item in requested if item.strip()]
    if not values:
        values = ["all"]
    methods = available_methods() if "all" in values else values
    known = set(available_methods())
    unknown = sorted(set(methods) - known)
    if unknown:
        raise ValueError(
            f"Unknown methods: {', '.join(unknown)}. Available: {', '.join(sorted(known))}"
        )
    dedup: list[str] = []
    seen: set[str] = set()
    for method in methods:
        if method not in seen:
            seen.add(method)
            dedup.append(method)
    return dedup


def parse_key_values(items: Sequence[str]) -> dict[str, object]:
    """Parse ``key=value`` CLI segments with YAML scalar decoding."""
    parsed: dict[str, object] = {}
    for raw in items:
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}': expected key=value")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{item}': empty key")
        parsed[key] = yaml.safe_load(value)
    return parsed


def parse_method_key_values(items: Sequence[str]) -> dict[str, dict[str, object]]:
    """Parse ``method.key=value`` CLI segments."""
    parsed: dict[str, dict[str, object]] = {}
    for raw in items:
        item = raw.strip()
        if not item:
            continue
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(
                f"Invalid method override '{item}': expected method.key=value"
            )
        lhs, rhs = item.split("=", 1)
        method, key = lhs.split(".", 1)
        method = method.strip()
        key = key.strip()
        if not method or not key:
            raise ValueError(f"Invalid method override '{item}'")
        method_entry = parsed.setdefault(method, {})
        method_entry[key] = yaml.safe_load(rhs)
    return parsed


def merge_method_params(
    method: str,
    *,
    common_overrides: Mapping[str, object] | None = None,
    method_overrides: Mapping[str, object] | None = None,
    n_sources_hint: int | None = None,
) -> dict[str, object]:
    """Merge method defaults from benchmark config with CLI overrides."""
    params = dict(_default_method_params().get(method, {}))
    if common_overrides:
        params.update(dict(common_overrides))
    if method_overrides:
        params.update(dict(method_overrides))
    if (
        method == "online_isnmf"
        and "n_sources" not in params
        and n_sources_hint is not None
    ):
        params["n_sources"] = int(n_sources_hint)
    return params


def build_plan(*, fft_size: int, hop_size: int, window: str) -> STFTPlan:
    """Build a typed STFT plan from CLI options."""
    return STFTPlan(fft_size=int(fft_size), hop_size=int(hop_size), window=str(window))


def load_mixture(path: Path) -> tuple[np.ndarray, int]:
    """Load a mixture waveform as ``(n_samples, n_mic)``."""
    audio, sample_rate = sf.read(path, always_2d=True)
    return np.asarray(audio), int(sample_rate)


def load_reference_stems(
    directory: Path,
    *,
    stems: Sequence[str] | None = None,
) -> tuple[np.ndarray, int, list[str]]:
    """Load reference stems as ``(n_src, n_samples, n_mic)``."""
    files = list(stems) if stems else sorted(p.name for p in directory.glob("*.wav"))
    if not files:
        raise ValueError(f"No reference wav files found in: {directory}")

    waves: list[np.ndarray] = []
    sample_rate: int | None = None
    stem_names: list[str] = []
    for name in files:
        wav_path = directory / name
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing reference stem: {wav_path}")
        data, stem_sr = sf.read(wav_path, always_2d=True)
        if sample_rate is None:
            sample_rate = int(stem_sr)
        elif int(stem_sr) != sample_rate:
            raise ValueError(
                f"Sampling rate mismatch in {wav_path}: {stem_sr} (expected {sample_rate})"
            )
        waves.append(np.asarray(data))
        stem_names.append(Path(name).stem)

    assert sample_rate is not None
    min_len = min(wave.shape[0] for wave in waves)
    stacked = np.stack([wave[:min_len, :] for wave in waves], axis=0)
    return stacked, sample_rate, stem_names


def run_method(
    *,
    method: str,
    mix: np.ndarray,
    sample_rate: int,
    plan: STFTPlan,
    params: Mapping[str, object],
) -> SeparationOutput:
    """Run one built-in method through the shared method registry."""
    registry = default_method_runner_registry()
    runner = registry.resolve(method)
    return runner.run(mix, dict(params), plan, sample_rate)


def save_estimate(path: Path, estimate: np.ndarray, sample_rate: int) -> None:
    """Save estimate shaped ``(n_src, n_samples)`` to WAV."""
    sf.write(path, estimate.T, sample_rate)


def save_method_plots(
    *,
    estimate: np.ndarray,
    sample_rate: int,
    plan: STFTPlan,
    output_dir: Path,
    prefix: str,
) -> dict[str, list[str]]:
    """Save spectrogram and source-model plots for one method output."""
    stft = build_stft(plan, sample_rate)
    estimate_tf = stft.stft(estimate).transpose(2, 1, 0)  # (T, F, K)
    power_model = np.square(np.abs(estimate_tf))

    spec_paths = save_channel_spectrograms(
        estimate_tf,
        f"{prefix}_spectrogram",
        output_dir,
    )
    model_paths = save_channel_spectrograms(
        power_model,
        f"{prefix}_source_model",
        output_dir,
    )
    return {
        "spectrograms": [str(path) for path in spec_paths],
        "source_models": [str(path) for path in model_paths],
    }


def save_nmf_factor_plots_from_metadata(
    *,
    method: str,
    metadata: Mapping[str, object],
    output_dir: Path,
    prefix: str,
    vmin: float = -40.0,
    vmax: float = 20.0,
) -> list[str]:
    """Save NMF factor plots from method metadata and return written paths."""
    payload_obj = metadata.get("nmf_factors")
    if not isinstance(payload_obj, dict):
        return []
    payload = {str(k): v for k, v in payload_obj.items()}

    kind = str(payload.get("type", ""))
    basis_obj = payload.get("basis")
    activ_obj = payload.get("activations")
    if not isinstance(basis_obj, list) or not isinstance(activ_obj, list):
        return []

    saved: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    if kind in {"ilrma", "online_ilrma"}:
        basis = np.asarray(basis_obj, dtype=np.float64)  # (K, F, L)
        activ = np.asarray(activ_obj, dtype=np.float64)  # (K, T, L)
        if basis.ndim != 3 or activ.ndim != 3:
            return []
        n_src = min(basis.shape[0], activ.shape[0])
        for src in range(n_src):
            b = np.maximum(basis[src], 1.0e-12)  # (F, L)
            c = np.maximum(activ[src].T, 1.0e-12)  # (L, T)
            x = np.maximum(b @ c, 1.0e-12)
            fig = plot_nmf_factors(x, b, c, vmin=vmin, vmax=vmax)
            path = output_dir / f"{prefix}_nmf_factors-{src}.pdf"
            fig.savefig(path)
            plt.close(fig)
            saved.append(str(path))
        return saved

    if kind == "online_isnmf":
        basis = np.asarray(basis_obj, dtype=np.float64)  # (F, K)
        activ = np.asarray(activ_obj, dtype=np.float64)  # (K, T)
        if basis.ndim != 2 or activ.ndim != 2:
            return []
        observed_obj = payload.get("observed_magnitude")
        if isinstance(observed_obj, list):
            observed = np.asarray(observed_obj, dtype=np.float64)
        else:
            observed = np.maximum(basis @ activ, 1.0e-12)
        x = np.maximum(observed, 1.0e-12)
        b = np.maximum(basis, 1.0e-12)
        c = np.maximum(activ, 1.0e-12)
        fig = plot_nmf_factors(x, b, c, vmin=vmin, vmax=vmax)
        path = output_dir / f"{prefix}_nmf_factors.pdf"
        fig.savefig(path)
        plt.close(fig)
        saved.append(str(path))
        return saved

    return []


def _align_for_eval(
    reference: np.ndarray,
    estimate: np.ndarray,
    mixture: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    if reference.ndim != 2 or estimate.ndim != 2:
        raise ValueError("reference and estimate must be 2-D arrays (n_src, n_samples).")

    mixture_eval = np.asarray(mixture)
    if mixture_eval.ndim == 1:
        mixture_samples = mixture_eval.shape[0]
    elif mixture_eval.ndim == 2:
        mixture_samples = mixture_eval.shape[1]
    else:
        raise ValueError("mixture must be 1-D or 2-D array.")

    n_sources = min(reference.shape[0], estimate.shape[0])
    if n_sources <= 0:
        raise ValueError("No common source channels available for SI-SDR evaluation.")

    n_samples = min(reference.shape[1], estimate.shape[1], mixture_samples)
    if n_samples <= 0:
        raise ValueError("No common samples available for SI-SDR evaluation.")

    ref = np.asarray(reference[:n_sources, :n_samples], dtype=np.float64)
    est = np.asarray(estimate[:n_sources, :n_samples], dtype=np.float64)
    if mixture_eval.ndim == 1:
        mix = np.asarray(mixture_eval[:n_samples], dtype=np.float64)
    else:
        mix = np.asarray(mixture_eval[:, :n_samples], dtype=np.float64)
    return ref, est, mix, n_sources, n_samples


def compute_si_sdr(
    *,
    reference: np.ndarray,
    estimate: np.ndarray,
    mixture: np.ndarray,
    filter_length: int = 1,
    compute_permutation: bool = True,
) -> SiSdrSummary:
    """Compute SI-SDR metrics through :func:`oobss.evaluation.metrics.compute_metrics`."""
    ref, est, mix, n_sources, n_samples = _align_for_eval(reference, estimate, mixture)

    metrics = compute_metrics(
        ref,
        est,
        mix,
        sample_rate=1,
        filter_length=int(filter_length),
        frame_cfg=None,
        compute_permutation=bool(compute_permutation),
    )
    sdr_mix = np.asarray(metrics.sdr_mix, dtype=np.float64)
    sdr_est = np.asarray(metrics.sdr_est, dtype=np.float64)
    permutation = np.asarray(metrics.permutation, dtype=np.int64)
    return SiSdrSummary(
        sdr_mix=sdr_mix,
        sdr_est=sdr_est,
        sdr_imp=sdr_est - sdr_mix,
        permutation=permutation,
        n_sources=n_sources,
        n_samples=n_samples,
    )


def to_jsonable(value: Any) -> Any:
    """Convert numpy-backed values into JSON-serializable objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
