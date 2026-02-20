"""Compare multiple separation methods on one mixture WAV.

This script runs multiple methods, saves each estimate to ``outputs/``, and
optionally evaluates SI-SDR when reference stems are provided.

Usage
-----
Run all methods:

``uv run python examples/compare_wav_methods.py examples/data/mixture.wav``

Run selected methods with SI-SDR evaluation:

``uv run python examples/compare_wav_methods.py examples/data/mixture.wav --methods batch_auxiva online_ilrma --reference-dir examples/data/ref --compute-permutation --filter-length 1``

Enable plot outputs:

``uv run python examples/compare_wav_methods.py examples/data/mixture.wav --plot``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from _sample_utils import (
    DEFAULT_OUTPUT_FILENAMES,
    NMF_FACTOR_METHODS,
    build_plan,
    compute_si_sdr,
    ensure_output_dir,
    load_mixture,
    load_reference_stems,
    merge_method_params,
    parse_key_values,
    parse_method_key_values,
    resolve_methods,
    run_method,
    save_estimate,
    save_method_plots,
    save_nmf_factor_plots_from_metadata,
    to_jsonable,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and compare multiple BSS methods on one mixture WAV.",
    )
    parser.add_argument("input_wav", type=Path, help="Path to input mixture WAV.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Methods to run. Use 'all' to run all built-ins.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for WAV and optional plot files.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs/comparison_summary.json"),
        help="Path to save JSON summary.",
    )
    parser.add_argument("--fft-size", type=int, default=2048, help="STFT FFT size.")
    parser.add_argument("--hop-size", type=int, default=1024, help="STFT hop size.")
    parser.add_argument("--window", type=str, default="hann", help="STFT window name.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Common parameter override in key=value form (applied to all methods).",
    )
    parser.add_argument(
        "--method-set",
        action="append",
        default=[],
        help="Method-specific override in method.key=value form.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Directory containing reference stems for SI-SDR evaluation.",
    )
    parser.add_argument(
        "--stems",
        nargs="*",
        default=None,
        help="Optional ordered stem filenames in reference directory.",
    )
    parser.add_argument(
        "--filter-length",
        type=int,
        default=1,
        help="BSSeval filter length. Use 1 for SI-SDR-style evaluation.",
    )
    parser.add_argument(
        "--compute-permutation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Solve source permutation during metric computation.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save spectrogram and source-model plots for each method.",
    )
    return parser.parse_args(argv)


def _print_metric_row(method: str, sdr_mix: np.ndarray, sdr_est: np.ndarray) -> None:
    improvement = sdr_est - sdr_mix
    print(f"[{method}] SI-SDR mix (dB): {sdr_mix}")
    print(f"[{method}] SI-SDR est (dB): {sdr_est}")
    print(f"[{method}] SI-SDR imp (dB): {improvement}")


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    methods = resolve_methods(args.methods)
    output_dir = ensure_output_dir(args.output_dir)

    mix, sample_rate = load_mixture(args.input_wav)
    plan = build_plan(
        fft_size=args.fft_size,
        hop_size=args.hop_size,
        window=args.window,
    )

    common_overrides = parse_key_values(args.set)
    method_overrides = parse_method_key_values(args.method_set)

    references: np.ndarray | None = None
    reference_stems: list[str] = []
    if args.reference_dir is not None:
        references, ref_sample_rate, reference_stems = load_reference_stems(
            args.reference_dir,
            stems=args.stems,
        )
        if ref_sample_rate != sample_rate:
            raise ValueError(
                f"Sampling rate mismatch: mixture={sample_rate}, reference={ref_sample_rate}"
            )

    summary: dict[str, dict[str, Any]] = {}
    for method in methods:
        params = merge_method_params(
            method,
            common_overrides=common_overrides,
            method_overrides=method_overrides.get(method),
            n_sources_hint=(
                references.shape[0] if references is not None else mix.shape[1]
            ),
        )
        if args.plot and method in NMF_FACTOR_METHODS:
            params["return_nmf_factors"] = True
            if method in {"online_ilrma", "online_isnmf"}:
                params["keep_h"] = True

        result = run_method(
            method=method,
            mix=mix,
            sample_rate=sample_rate,
            plan=plan,
            params=params,
        )
        if result.estimate_time is None:
            raise ValueError(f"Method '{method}' did not return estimate_time.")
        estimate = result.estimate_time
        metadata = dict(result.metadata)

        output_name = DEFAULT_OUTPUT_FILENAMES.get(method, f"{method}.wav")
        wav_path = output_dir / output_name
        save_estimate(wav_path, estimate, sample_rate)

        method_summary: dict[str, Any] = {
            "method": method,
            "output_wav": str(wav_path),
            "params": dict(params),
        }

        if metadata:
            compact_metadata = dict(metadata)
            compact_metadata.pop("nmf_factors", None)
            if compact_metadata:
                method_summary["method_metadata"] = to_jsonable(compact_metadata)

        if args.plot:
            plot_dir = output_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            method_summary["plots"] = save_method_plots(
                estimate=estimate,
                sample_rate=sample_rate,
                plan=plan,
                output_dir=plot_dir,
                prefix=method,
            )
            if method in NMF_FACTOR_METHODS:
                nmf_paths = save_nmf_factor_plots_from_metadata(
                    method=method,
                    metadata=metadata,
                    output_dir=plot_dir,
                    prefix=method,
                )
                if nmf_paths:
                    method_summary.setdefault("plots", {})
                    plots = method_summary["plots"]
                    if isinstance(plots, dict):
                        plots["nmf_factors"] = nmf_paths

        if references is not None:
            ref_mic = int(metadata.get("reference_mic", params.get("ref_mic", 0)))
            reference = references[:, :, ref_mic]
            metrics = compute_si_sdr(
                reference=reference,
                estimate=estimate,
                mixture=mix[:, ref_mic],
                filter_length=args.filter_length,
                compute_permutation=args.compute_permutation,
            )
            _print_metric_row(method, metrics.sdr_mix, metrics.sdr_est)
            method_summary["evaluation"] = {
                "aligned_n_sources": metrics.n_sources,
                "aligned_n_samples": metrics.n_samples,
                "reference_stems": reference_stems[: metrics.n_sources],
                "si_sdr_mix": metrics.sdr_mix.tolist(),
                "si_sdr_est": metrics.sdr_est.tolist(),
                "si_sdr_imp": metrics.sdr_imp.tolist(),
                "permutation": metrics.permutation.tolist(),
                "mean_si_sdr": float(np.mean(metrics.sdr_est)),
                "mean_si_sdr_imp": float(np.mean(metrics.sdr_imp)),
            }

        summary[method] = method_summary

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(summary), fh, indent=2)
    print(f"Saved summary: {args.summary_json}")


if __name__ == "__main__":
    main()
