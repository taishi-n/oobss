"""Compatibility wrapper for experiment-suite benchmark execution.

This script delegates to ``oobss.experiments.run_suite`` so examples do not
duplicate benchmark orchestration logic.

Usage
-----
Dry-run:

``uv run python examples/benchmark_dataset.py --dry-run --sample-limit 2``

Execute:

``uv run python examples/benchmark_dataset.py --sample-limit 2 --workers 1``
"""

from __future__ import annotations

import sys
from typing import Sequence

from oobss.experiments.run_suite import main as run_suite_main


def main(argv: Sequence[str] | None = None) -> None:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    run_suite_main(["run", *forwarded])


if __name__ == "__main__":
    main()
