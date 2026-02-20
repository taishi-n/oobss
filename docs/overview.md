# Overview

oobss provides online and batch blind source separation algorithms with a small,
consistent Python API and command-line entrypoint.

## Module Roles

- `oobss.separators`: Core separation algorithms and shared separator framework.
  This includes algorithm implementations (`AuxIVA`, `ILRMA`, online variants),
  reusable strategy interfaces, and concrete strategy injections.
- `oobss.evaluation`: Objective evaluation utilities.
  Batch metric computation (`compute_metrics`) and low-level/frame-wise SI-SDR
  helpers live here.
- `oobss.benchmark`: Experiment orchestration for dataset-scale evaluations.
  It provides config loading/validation, method runners, task pipeline,
  reporting, and CLI integration.
- `oobss.dataloaders`: Dataset adapters and track loading interfaces.
- `oobss.signal`: Signal processing utilities such as STFT plan/build helpers.
- `oobss.postprocess`: Post-separation utilities for reference-microphone
  projection and waveform reconstruction.
- `oobss.visualization`: Plotting helpers used by examples and reports.
- `oobss.configs` and `oobss.logging_utils`: Cross-cutting configuration and
  lightweight logging helpers.
