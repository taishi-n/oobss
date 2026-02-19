"""Small JSONL logging utilities for streaming experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


class JsonlLogger:
    """Append JSON-serializable records to a JSON Lines file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def log_steps_jsonl(path: str | Path, steps: Iterable[Mapping[str, Any]]) -> None:
    """Write many step dictionaries to JSONL."""
    logger = JsonlLogger(path)
    for step in steps:
        logger.write(step)
