"""Compatibility wrapper for experiment-suite benchmark execution.

This script delegates to ``oobss.benchmark.cli`` so examples do not
duplicate benchmark orchestration logic.

Usage
-----
Dry-run:

``uv run python examples/benchmark_dataset.py --dry-run --sample-limit 2 --set dataset.root=/path/to/dataset``

Execute:

``uv run python examples/benchmark_dataset.py --sample-limit 2 --workers 1 --set dataset.root=/path/to/dataset``
"""

from __future__ import annotations

import sys
from typing import Sequence

from oobss.benchmark.cli import main as run_suite_main


def main(argv: Sequence[str] | None = None) -> None:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    try:
        run_suite_main(["run", *forwarded])
    except FileNotFoundError as exc:
        message = str(exc)
        if "Dataset root not found" not in message:
            raise
        raise SystemExit(
            "Dataset root not found. Build the torchrir dynamic dataset first or "
            "pass '--set dataset.root=/path/to/dataset'."
        ) from exc


if __name__ == "__main__":
    main()
