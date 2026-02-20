"""Experiment execution engine for large-scale separation benchmarks.

This module provides a reusable orchestration layer that supports:

1. Method configuration expansion via Cartesian parameter grids.
2. Task planning from recipes.
3. Sequential or multi-process execution.
4. Artifact persistence through the pipeline writer.

The engine is intentionally DI-friendly: task building, task execution, and
result writing functions can be replaced in tests or downstream integrations.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, cast

import numpy as np
import yaml

from .config_loader import MethodConfig
from .methods import MethodRunnerRegistry
from .pipeline import (
    ExperimentOutput,
    ExperimentTask,
    build_tasks_from_recipe,
    generate_run_directory,
    run_task,
    write_outputs,
)
from .recipe import ExperimentRecipe

LOGGER = logging.getLogger(__name__)

BuildTasksFn = Callable[..., list[ExperimentTask]]
RunTaskFn = Callable[[ExperimentTask, MethodRunnerRegistry | None], ExperimentOutput]
WriteOutputsFn = Callable[[Iterable[ExperimentOutput], Path, bool, int, bool], Path]


@dataclass(frozen=True)
class EngineRunArtifacts:
    """Artifacts produced by one engine run.

    Attributes
    ----------
    run_root:
        Root output directory for this run.
    results_path:
        JSONL file containing one summary record per task.
    task_count:
        Number of scheduled tasks.
    method_count:
        Number of method variants (after grid expansion).
    """

    run_root: Path
    results_path: Path
    task_count: int
    method_count: int


def _slug_value(value: object) -> str:
    text = str(value)
    return (
        text.replace(" ", "")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(",", "_")
        .replace(":", "_")
    )


def _set_nested(mapping: dict[str, object], dotted_key: str, value: object) -> None:
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        raise ValueError(f"Invalid grid parameter key: {dotted_key!r}")
    cursor: dict[str, object] = mapping
    for key in parts[:-1]:
        current = cursor.get(key)
        if current is None:
            current = {}
            cursor[key] = current
        if not isinstance(current, dict):
            raise TypeError(
                f"Cannot assign grid key '{dotted_key}' because '{key}' is not a mapping."
            )
        cursor = cast(dict[str, object], current)
    cursor[parts[-1]] = value


def expand_method_grids(
    methods: Sequence[MethodConfig],
    method_grid: Mapping[str, Mapping[str, Sequence[object]]] | None,
) -> list[MethodConfig]:
    """Expand method configurations by Cartesian product grids.

    Parameters
    ----------
    methods:
        Base method definitions loaded from YAML.
    method_grid:
        Mapping ``method_id -> {param_key: [values...]}`` where ``param_key`` is
        interpreted as a path inside ``MethodConfig.params``. Example:
        ``{"batch_ilrma": {"n_basis": [2, 4], "n_iter": [50, 100]}}``.

    Returns
    -------
    list[MethodConfig]
        Expanded method variants. Methods without grid definitions are returned
        unchanged.
    """
    if not method_grid:
        return list(methods)

    known_ids = {method.id for method in methods}
    unknown = sorted(set(method_grid) - known_ids)
    if unknown:
        unknown_list = ", ".join(unknown)
        raise ValueError(f"Grid references unknown method IDs: {unknown_list}")

    expanded: list[MethodConfig] = []
    for method in methods:
        grid = method_grid.get(method.id)
        if not grid:
            expanded.append(method)
            continue

        keys = list(grid.keys())
        values_per_key: list[list[object]] = []
        for key in keys:
            raw_values = list(grid[key])
            if not raw_values:
                raise ValueError(
                    f"Grid for method '{method.id}' and key '{key}' has no values."
                )
            values_per_key.append(raw_values)

        combo_count = int(np.prod([len(values) for values in values_per_key]))
        if combo_count <= 0:
            expanded.append(method)
            continue

        for combo_idx in range(combo_count):
            params = dict(method.params)
            remainder = combo_idx
            assignments: list[tuple[str, object]] = []
            for key, values in zip(keys, values_per_key):
                base = len(values)
                value = values[remainder % base]
                remainder //= base
                assignments.append((key, value))
                _set_nested(params, key, value)

            suffix_items = [f"{key}={_slug_value(value)}" for key, value in assignments]
            suffix = "__".join(suffix_items)
            label_details = ", ".join(f"{key}={value}" for key, value in assignments)
            expanded.append(
                MethodConfig(
                    id=f"{method.id}__{suffix}",
                    label=f"{method.label} [{label_details}]",
                    type=method.type,
                    enabled=True,
                    params=params,
                )
            )

    return expanded


def parse_grid_overrides(items: Iterable[str]) -> dict[str, dict[str, list[object]]]:
    """Parse CLI-style grid overrides into ``method_grid`` format.

    Each item must use:
    ``<method_id>.<param_path>=<value1>,<value2>,...``

    Values are parsed with ``yaml.safe_load`` so numbers and booleans keep their
    native Python types.
    """
    grid: dict[str, dict[str, list[object]]] = {}
    for raw_item in items:
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid grid override '{item}'. Expected '<method>.<param>=v1,v2'."
            )
        lhs, rhs = item.split("=", 1)
        if "." not in lhs:
            raise ValueError(
                f"Invalid grid override '{item}'. Missing method or parameter path."
            )
        method_id, param_path = lhs.split(".", 1)
        values = [segment.strip() for segment in rhs.split(",") if segment.strip()]
        if not values:
            raise ValueError(f"Grid override '{item}' has no values.")
        parsed = [yaml.safe_load(value) for value in values]
        method_entry = grid.setdefault(method_id, {})
        method_entry[param_path] = parsed
    return grid


def merge_method_grids(
    base: Mapping[str, Mapping[str, Sequence[object]]] | None,
    overrides: Mapping[str, Mapping[str, Sequence[object]]] | None,
) -> dict[str, dict[str, list[object]]]:
    """Merge two method grid mappings with override precedence."""
    merged: dict[str, dict[str, list[object]]] = {}
    for method_id, params in (base or {}).items():
        merged[method_id] = {key: list(values) for key, values in params.items()}
    for method_id, params in (overrides or {}).items():
        method_entry = merged.setdefault(method_id, {})
        for key, values in params.items():
            method_entry[key] = list(values)
    return merged


def _run_task_with_registry(
    task: ExperimentTask,
    runner_registry: MethodRunnerRegistry | None,
) -> ExperimentOutput:
    return run_task(task, runner_registry=runner_registry)


@dataclass
class ExperimentEngine:
    """Orchestrate planning and execution of benchmark experiments."""

    build_tasks_fn: BuildTasksFn = build_tasks_from_recipe
    run_task_fn: RunTaskFn = _run_task_with_registry
    write_outputs_fn: WriteOutputsFn | None = None

    def __post_init__(self) -> None:
        if self.write_outputs_fn is None:
            self.write_outputs_fn = _write_outputs_adapter

    def build_tasks(
        self,
        recipe: ExperimentRecipe,
        methods: Sequence[MethodConfig],
        *,
        method_grid: Mapping[str, Mapping[str, Sequence[object]]] | None = None,
    ) -> list[ExperimentTask]:
        """Build tasks from recipe and method definitions."""
        expanded_methods = expand_method_grids(methods, method_grid)
        return self.build_tasks_fn(recipe, methods=expanded_methods)

    def run(
        self,
        *,
        recipe: ExperimentRecipe,
        methods: Sequence[MethodConfig],
        output_root: Path,
        workers: int | None,
        overwrite: bool,
        save_framewise: bool,
        summary_precision: int,
        save_audio: bool,
        method_grid: Mapping[str, Mapping[str, Sequence[object]]] | None = None,
        runner_registry: MethodRunnerRegistry | None = None,
    ) -> EngineRunArtifacts:
        """Execute planned experiments and persist outputs."""
        tasks = self.build_tasks(recipe, methods, method_grid=method_grid)
        run_root = generate_run_directory(output_root, overwrite=overwrite)
        outputs = self._execute_tasks(
            tasks,
            workers=workers,
            runner_registry=runner_registry,
        )

        assert self.write_outputs_fn is not None  # assigned in __post_init__
        results_path = self.write_outputs_fn(
            outputs,
            run_root,
            save_framewise,
            summary_precision,
            save_audio,
        )
        method_count = len({task.method.id for task in tasks})
        return EngineRunArtifacts(
            run_root=run_root,
            results_path=results_path,
            task_count=len(tasks),
            method_count=method_count,
        )

    def _execute_tasks(
        self,
        tasks: Sequence[ExperimentTask],
        *,
        workers: int | None,
        runner_registry: MethodRunnerRegistry | None,
    ) -> list[ExperimentOutput]:
        worker_count = workers or (os.cpu_count() or 1)
        worker_count = max(1, int(worker_count))
        outputs: list[ExperimentOutput] = []

        if worker_count > 1 and runner_registry is not None:
            LOGGER.warning(
                "Custom runner registry was provided; falling back to single-process execution."
            )
            worker_count = 1

        if worker_count == 1:
            for idx, task in enumerate(tasks, start=1):
                outputs.append(self.run_task_fn(task, runner_registry))
                LOGGER.info("Completed %d/%d", idx, len(tasks))
            return outputs

        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(self.run_task_fn, task, runner_registry): task
                for task in tasks
            }
            for idx, future in enumerate(as_completed(futures), start=1):
                outputs.append(future.result())
                LOGGER.info("Completed %d/%d", idx, len(tasks))
        return outputs


def _write_outputs_adapter(
    outputs: Iterable[ExperimentOutput],
    run_root: Path,
    save_framewise: bool,
    summary_precision: int,
    save_audio: bool,
) -> Path:
    return write_outputs(
        outputs,
        run_root,
        save_framewise=save_framewise,
        summary_precision=summary_precision,
        save_audio=save_audio,
    )
