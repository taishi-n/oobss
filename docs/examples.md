# Examples

This page summarizes the current example scripts and expected inputs/outputs.

## Prerequisites

```bash
uv sync
```

All commands assume repository root as the current directory.

## 1. Single-WAV CLI Separation

Script: `examples/separate_wav_cli.py`

### Command

```bash
uv run python examples/separate_wav_cli.py examples/data/mixture.wav --method batch_auxiva
```

### Expected Input

- One mixture WAV file path.

### Expected Output

- One separated waveform saved under `outputs/` (method-specific filename).
- Optional spectrogram/source-model plots when `--plot` is enabled.

## 2. Multi-Method Comparison on One WAV

Script: `examples/compare_wav_methods.py`

### Command

```bash
uv run python examples/compare_wav_methods.py \
    examples/data/mixture.wav \
    --methods all \
    --reference-dir examples/data/ref \
    --compute-permutation \
    --filter-length 1 \
    --plot
```

### Expected Input

- One mixture WAV file path.
- Optional reference stem directory (`--reference-dir`) for SI-SDR evaluation.
- SI-SDR baseline is computed from the selected reference microphone channel of
  the mixture (`mix[:, ref_mic]`).
- `--compute-permutation/--no-compute-permutation` switches permutation solving
  for determined/overdetermined evaluations.
- `--filter-length 1` computes SI-SDR-style batch metrics.

### Expected Output

- Separated waveforms in `outputs/` for each selected method.
- `outputs/comparison_summary.json` (or `--summary-json` path).
- Optional plots under `outputs/plots/`:
    - `<method>_spectrogram-*.pdf`
    - `<method>_source_model-*.pdf`
    - `<method>_nmf_factors*.pdf` (for NMF-based methods)
- Console SI-SDR logs when references are provided.

## 3. Dataset-Wide Benchmark

Script: `examples/benchmark_dataset.py`

This script is a thin compatibility wrapper around
`oobss.benchmark.cli run`.

Note: this command discovers tracks during planning, so even `--dry-run`
requires a valid dataset root.

### Dry-run Command

```bash
uv run python examples/benchmark_dataset.py \
    --dry-run \
    --sample-limit 2 \
    --set dataset.root=/path/to/cmu_arctic_torchrir_dynamic_dataset
```

### Actual Run Command

```bash
uv run python examples/benchmark_dataset.py \
    --sample-limit 2 \
    --workers 1 \
    --set dataset.root=/path/to/cmu_arctic_torchrir_dynamic_dataset
```

Equivalent direct command:

```bash
uv run python -m oobss.benchmark.cli run \
    --sample-limit 2 \
    --workers 1 \
    --set dataset.root=/path/to/cmu_arctic_torchrir_dynamic_dataset
```

### Expected Input

- Common config: `examples/benchmark/config/common.yaml`
- Method configs: `examples/benchmark/config/methods/*.yaml`
- Dataset root with torchrir dynamic layout (`dataset.type: torchrir_dynamic`):
    - `<root>/scene_xxxx/mixture.wav`
    - `<root>/scene_xxxx/source_00.wav`, `source_01.wav`, ...
    - Optional metadata files:
      `metadata.json`, `source_info.json`, `room_layout_2d.mp4`, `room_layout_3d.mp4`

### Expected Output

Under `runtime.output_dir` (default `outputs/dataset_benchmark`) a run
directory is created:

- `run_YYYYMMDD-HHMMSS/results.jsonl`
- `run_YYYYMMDD-HHMMSS/configs/common.yaml`
- `run_YYYYMMDD-HHMMSS/configs/methods/*.yaml`
- `run_YYYYMMDD-HHMMSS/<method_id>/<track_id>/detail.json`
- `run_YYYYMMDD-HHMMSS/<method_id>/<track_id>/estimate.wav` (if enabled)
- `run_YYYYMMDD-HHMMSS/reports/*` (CSV/JSON/PDF aggregate reports)
