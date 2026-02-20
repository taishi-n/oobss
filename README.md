# oobss

Open online blind source separation toolkit.

This repository contains classical and online blind source separation algorithms,
plus utilities for configuration, logging, and documentation.

## Installation

Install from PyPI:

```bash
pip install oobss
```

Or with `uv`:

```bash
uv add oobss
```

Install with realtime extras (`torch`, `torchrir`):

```bash
pip install "oobss[realtime]"
```

```bash
uv add "oobss[realtime]"
```

## Development Setup

Create the development environment and run tests:

```bash
uv sync
uv run pytest
```

Run the multi-method comparison example:

```bash
uv run python examples/compare_wav_methods.py \
  examples/data/mixture.wav \
  --methods all \
  --compute-permutation \
  --filter-length 1
```

## Documentation

Build docs locally with Material for MkDocs:

```bash
uv run mkdocs build
```

Preview docs locally:

```bash
uv run mkdocs serve
```

## Example Scripts

- Single WAV + single method CLI:

  ```bash
  uv run python examples/separate_wav_cli.py examples/data/mixture.wav --method batch_auxiva
  ```

- Single WAV + multi-method comparison (optional SI-SDR with references):

  ```bash
  uv run python examples/compare_wav_methods.py \
    examples/data/mixture.wav \
    --methods all \
    --reference-dir examples/data/ref \
    --compute-permutation \
    --filter-length 1 \
    --plot
  ```

  Notes: SI-SDR baseline uses the selected reference-microphone channel
  (`mix[:, ref_mic]`), `--compute-permutation/--no-compute-permutation` can be
  switched for determined/overdetermined evaluation setups, and
  `--filter-length 1` gives SI-SDR-style batch metrics.

- Dataset-wide benchmark with dataloader + aggregation/visualization:

  ```bash
  uv run python examples/benchmark_dataset.py \
    --sample-limit 2 \
    --workers 1 \
    --set dataset.root=/path/to/cmu_arctic_torchrir_dynamic_dataset
  ```

- CMU ARCTIC + torchrir dataset build (torchrir side):

  ```bash
  torchrir-build-dynamic-cmu-arctic \
    --cmu-root /path/to/cmu_arctic \
    --dataset-root outputs/cmu_arctic_torchrir_dynamic_dataset \
    --n-scenes 10 \
    --overwrite-dataset
  ```

  Alternative module form:

  ```bash
  python -m torchrir.datasets.dynamic_cmu_arctic \
    --cmu-root /path/to/cmu_arctic \
    --dataset-root outputs/cmu_arctic_torchrir_dynamic_dataset \
    --n-scenes 10 \
    --overwrite-dataset
  ```

  Optional layout video flags (torchrir):
  `--save-layout-mp4/--no-save-layout-mp4`,
  `--save-layout-mp4-3d/--no-save-layout-mp4-3d`,
  `--layout-video-fps`, `--layout-video-no-audio`.

## Major APIs

For benchmark and multi-method execution, prefer the `oobss.benchmark`
entrypoints (`ExperimentEngine`, `default_method_runner_registry`).
Direct separator classes are lower-level building blocks.

### 1. Batch Separators (`AuxIVA`, `ILRMA`)

Use TF-domain mixtures with shape `(n_frame, n_freq, n_mic)` and run iterative updates.

```python
import numpy as np
from scipy.signal import ShortTimeFFT, get_window

from oobss import AuxIVA

fs = 16000
fft_size = 2048
hop_size = 512
win = get_window("hann", fft_size, fftbins=True)
stft = ShortTimeFFT(win=win, hop=hop_size, fs=fs)

# mixture_time: (n_samples, n_mic)
mixture_time = np.random.randn(fs, 2)
obs = stft.stft(mixture_time.T).transpose(2, 1, 0)  # (n_frame, n_freq, n_mic)

separator = AuxIVA(obs)
separator.run(30)
est_tf = separator.get_estimate()  # (n_frame, n_freq, n_src)
```

Strategy plug-and-play example:

```python
from oobss import AuxIVA
from oobss.separators.strategies import (
    BatchCovarianceStrategy,
    GaussSourceStrategy,
    IP1SpatialStrategy,
)

separator = AuxIVA(
    obs,
    spatial=IP1SpatialStrategy(),
    source=GaussSourceStrategy(),
    covariance=BatchCovarianceStrategy(),
)
separator.run(30)
```

### 2. Online Separators (`OnlineAuxIVA`, `OnlineILRMA`, `OnlineISNMF`)

Use frame-wise streaming with a shared API:

- `process_frame(frame, request=None)`
- `process_stream(stream, frame_axis=-1, request=None)`

```python
import numpy as np

from oobss import OnlineILRMA, StreamRequest

# stream: (n_freq, n_mic, n_frame)
stream = np.random.randn(513, 2, 100) + 1j * np.random.randn(513, 2, 100)

model = OnlineILRMA(
    n_mic=2,
    n_freq=513,
    n_bases=4,
    ref_mic=0,
    beta=1,
    forget=0.99,
    inner_iter=5,
)
out = model.process_stream_tf(
    stream,
    request=StreamRequest(frame_axis=2, reference_mic=0),
)
separated = out.estimate_tf  # (n_freq, n_src, n_frame)
```

### 3. Unified Separator Contract

Use typed requests for batch/stream execution:
- `fit_transform_tf(..., request=BatchRequest(...))`
- `process_stream_tf(..., request=StreamRequest(...))`

```python
from oobss import AuxIVA, BatchRequest

separator = AuxIVA(obs)  # obs: (n_frame, n_freq, n_mic)
output = separator.fit_transform_tf(
    obs,
    n_iter=50,
    request=BatchRequest(reference_mic=0),
)
estimate_tf = output.estimate_tf
```

### 4. Experiment Engine (`oobss.benchmark`)

Run method sweeps, aggregate results, and generate reports.

- `ExperimentEngine`: task planning and execution
- `expand_method_grids`: method-parameter grid expansion
- `generate_experiment_report`: aggregate CSV/JSON/PDF outputs
- `oobss.dataloaders.create_loader`: dataset loader factory (`torchrir_dynamic` built-in)

```bash
uv run python examples/benchmark_dataset.py \
  --sample-limit 2 \
  --workers 1 \
  --grid batch_ilrma.n_basis=2,4 \
  --grid batch_ilrma.n_iter=50,100
```

Programmatic example:

```python
from pathlib import Path

from oobss.benchmark.config_loader import (
    load_common_config_schema,
    load_method_configs,
)
from oobss.benchmark.config_schema import common_config_to_dict
from oobss.benchmark.engine import ExperimentEngine, parse_grid_overrides
from oobss.benchmark.recipe import recipe_from_common_config
from oobss.benchmark.reporting import generate_experiment_report

cfg = load_common_config_schema(
    Path("examples/benchmark/config/common.yaml")
)
methods = load_method_configs(Path("examples/benchmark/config/methods"))
recipe = recipe_from_common_config(common_config_to_dict(cfg))
grid = parse_grid_overrides(["batch_ilrma.n_basis=2,4"])

engine = ExperimentEngine()
artifacts = engine.run(
    recipe=recipe,
    methods=methods,
    output_root=Path("outputs/dataset_benchmark"),
    workers=1,
    overwrite=True,
    save_framewise=True,
    summary_precision=6,
    save_audio=True,
    method_grid=grid,
)
generate_experiment_report(artifacts.results_path, artifacts.run_root / "reports")
```

## License

This project is distributed under the terms in `LICENSE`.
It is based on Apache License 2.0 with additional restrictions, including:

- non-commercial use only
- required attribution for redistribution/deployment/derived use

Refer to `LICENSE` for the complete and binding terms.

## Future Work

The following roadmap is planned for future iterations:

1. Define a stable dataset contract for `oobss` (track-level manifest, required fields, and directory layout).
2. Introduce a recipe system (`recipe.yaml`) to convert arbitrary raw datasets into the `oobss` data contract.
3. Implement recipe execution modules in `oobss` (validation, conversion, manifest generation, and failure reporting).
4. Extend dataset adapters beyond `torchrir_dynamic` and provide ready-to-use recipes for common public datasets.
5. Strengthen validation tooling for converted datasets (schema checks, duration/channel checks, missing-file diagnostics).
6. Standardize benchmark outputs (`results.jsonl`, per-track details, optional frame-wise metrics) and keep plotting optional.
7. Extend CLI commands for end-to-end workflows:
   - recipe validation and conversion
   - benchmark run, summarize, and plotting
8. Add example recipes for multiple dataset structures and keep examples as thin wrappers around library APIs.
9. Add integration tests for recipe conversion and small-scale benchmark runs to prevent regressions.
