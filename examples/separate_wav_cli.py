"""CLI example: separate one WAV file with one method and save outputs.

Usage
-----
Run with default method:

``uv run python examples/separate_wav_cli.py examples/data/mixture.wav``

Run with custom parameters:

``uv run python examples/separate_wav_cli.py examples/data/mixture.wav --method online_ilrma --set n_bases=6 --set inner_iter=3``
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from _sample_utils import (
    DEFAULT_OUTPUT_FILENAMES,
    NMF_FACTOR_METHODS,
    available_methods,
    build_plan,
    ensure_output_dir,
    load_mixture,
    merge_method_params,
    parse_key_values,
    run_method,
    save_estimate,
    save_method_plots,
    save_nmf_factor_plots_from_metadata,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Separate one mixture WAV file with one BSS method.",
    )
    parser.add_argument("input_wav", type=Path, help="Path to input mixture WAV.")
    parser.add_argument(
        "--method",
        type=str,
        default="batch_auxiva",
        choices=available_methods(),
        help="Method ID to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for separated WAV and optional figures.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output WAV filename (default is method-specific).",
    )
    parser.add_argument("--fft-size", type=int, default=2048, help="STFT FFT size.")
    parser.add_argument("--hop-size", type=int, default=1024, help="STFT hop size.")
    parser.add_argument("--window", type=str, default="hann", help="STFT window name.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Method parameter override in key=value form.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save spectrogram and source-model plots under output dir.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = ensure_output_dir(args.output_dir)

    mix, sample_rate = load_mixture(args.input_wav)
    plan = build_plan(
        fft_size=args.fft_size,
        hop_size=args.hop_size,
        window=args.window,
    )

    overrides = parse_key_values(args.set)
    params = merge_method_params(
        args.method,
        common_overrides=overrides,
        n_sources_hint=mix.shape[1],
    )
    if args.plot and args.method in NMF_FACTOR_METHODS:
        params["return_nmf_factors"] = True
        if args.method in {"online_ilrma", "online_isnmf"}:
            params["keep_h"] = True

    result = run_method(
        method=args.method,
        mix=mix,
        sample_rate=sample_rate,
        plan=plan,
        params=params,
    )
    if result.estimate_time is None:
        raise ValueError(f"Method '{args.method}' did not return estimate_time.")
    estimate = result.estimate_time
    metadata = dict(result.metadata)

    output_name = (
        args.output_name
        if args.output_name
        else DEFAULT_OUTPUT_FILENAMES.get(args.method, f"{args.method}.wav")
    )
    wav_path = output_dir / output_name
    save_estimate(wav_path, estimate, sample_rate)
    print(f"Saved separated waveform: {wav_path}")

    if args.plot:
        plots = save_method_plots(
            estimate=estimate,
            sample_rate=sample_rate,
            plan=plan,
            output_dir=output_dir,
            prefix=args.method,
        )
        print(f"Saved spectrogram plots: {plots['spectrograms']}")
        print(f"Saved source-model plots: {plots['source_models']}")
        if args.method in NMF_FACTOR_METHODS:
            nmf_paths = save_nmf_factor_plots_from_metadata(
                method=args.method,
                metadata=metadata,
                output_dir=output_dir,
                prefix=args.method,
            )
            if nmf_paths:
                print(f"Saved NMF factor plots: {nmf_paths}")

    if metadata:
        compact_metadata = dict(metadata)
        compact_metadata.pop("nmf_factors", None)
        if compact_metadata:
            print(f"Method metadata: {compact_metadata}")


if __name__ == "__main__":
    main()
