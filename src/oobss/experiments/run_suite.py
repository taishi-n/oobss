"""CLI for running, summarizing, and inspecting experiment suites."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

import yaml

from .config_loader import (
    MethodConfig,
    load_common_config_schema,
    load_method_configs,
)
from .config_schema import CommonExperimentConfig, common_config_to_dict
from .engine import (
    ExperimentEngine,
    expand_method_grids,
    merge_method_grids,
    parse_grid_overrides,
)
from .recipe import recipe_from_common_config
from .reporting import generate_experiment_report


def _resolve_reports_dir(results_path: Path, requested_dir: Path | str | None) -> Path:
    """Return the directory where reports should be written."""
    if requested_dir:
        candidate = Path(requested_dir).expanduser()
        if candidate.is_absolute():
            return candidate
        return (results_path.parent / candidate).resolve()
    return results_path.parent / "reports"


def _snapshot_configs(
    run_root: Path,
    common_cfg: dict[str, Any],
    methods: Sequence[MethodConfig],
) -> None:
    """Persist resolved run configuration into the output directory."""
    config_dir = run_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    with (config_dir / "common.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(common_cfg, fh, sort_keys=False)

    methods_dir = config_dir / "methods"
    methods_dir.mkdir(parents=True, exist_ok=True)
    for method in methods:
        data = {
            "id": method.id,
            "label": method.label,
            "type": method.type,
            "enabled": method.enabled,
            "params": method.params,
        }
        with (methods_dir / f"{method.id}.yaml").open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False)


def configure_logging(level: str) -> None:
    """Configure logging format and level for CLI commands."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )


def _resolve_method_grid(
    cfg: CommonExperimentConfig, grid_args: Sequence[str]
) -> dict[str, dict[str, list[Any]]]:
    cfg_grid_obj = cfg.experiments.grid
    cfg_grid = cfg_grid_obj if isinstance(cfg_grid_obj, dict) else {}
    cli_grid = parse_grid_overrides(grid_args)
    return merge_method_grids(cfg_grid, cli_grid)


def run_command(args: argparse.Namespace) -> None:
    """Run configured experiment tasks and generate reports."""
    cfg = load_common_config_schema(args.config, overrides=args.set or None)
    cfg_map = common_config_to_dict(cfg)

    configure_logging(cfg.runtime.log_level)

    methods = load_method_configs(
        args.method_config_dir,
        selected=args.method,
        overrides=args.method_set or None,
    )
    if not methods:
        raise ValueError("No methods selected. Check YAML files or --method filters.")

    dataset_cfg = cfg_map["dataset"]
    stft_cfg = cfg_map["stft"]
    evaluation_cfg = dict(cfg_map["evaluation"])
    evaluation_cfg.setdefault("ref_mic", cfg.dataset.ref_mic)
    metrics_cfg = cfg_map["metrics"]
    method_grid = _resolve_method_grid(cfg, args.grid or [])

    recipe = recipe_from_common_config(
        {
            "dataset": dataset_cfg,
            "stft": stft_cfg,
            "evaluation": evaluation_cfg,
            "metrics": metrics_cfg,
            "runtime": cfg_map["runtime"],
        },
        track_subset=args.track if args.track else None,
        sample_limit=args.sample_limit,
    )

    engine = ExperimentEngine()
    tasks = engine.build_tasks(recipe, methods, method_grid=method_grid)
    logging.info(
        "Prepared %d tasks for %d method variants",
        len(tasks),
        len({task.method.id for task in tasks}),
    )
    if args.dry_run:
        for task in tasks:
            logging.info("PLAN | method=%s track=%s", task.method.id, task.track_id)
        return

    artifacts = engine.run(
        recipe=recipe,
        methods=methods,
        output_root=Path(cfg.runtime.output_dir),
        workers=args.workers or cfg.runtime.workers,
        overwrite=args.overwrite,
        save_framewise=cfg.metrics.save_framewise,
        summary_precision=cfg.metrics.summary_precision,
        save_audio=not args.no_save_audio,
        method_grid=method_grid,
    )
    logging.info("Run directory: %s", artifacts.run_root)

    expanded_methods = expand_method_grids(methods, method_grid)
    _snapshot_configs(artifacts.run_root, cfg_map, expanded_methods)

    reports_dir_override = cfg.runtime.reports_dir
    reports_dir = _resolve_reports_dir(artifacts.results_path, reports_dir_override)
    generate_experiment_report(
        artifacts.results_path,
        reports_dir,
    )
    logging.info("Finished run. Results: %s", artifacts.results_path)


def summarize_command(args: argparse.Namespace) -> None:
    """Generate reports from an existing ``results.jsonl`` file."""
    configure_logging("INFO")
    resolved_results = args.results.expanduser()
    reports_dir = _resolve_reports_dir(resolved_results, args.output)
    generate_experiment_report(
        resolved_results,
        reports_dir,
    )


def show_config_command(args: argparse.Namespace) -> None:
    """Print fully resolved configuration including method variants."""
    cfg = load_common_config_schema(args.config, overrides=args.set or None)
    cfg_map = common_config_to_dict(cfg)
    methods = load_method_configs(
        args.method_config_dir,
        overrides=args.method_set or None,
    )
    method_grid = _resolve_method_grid(cfg, args.grid or [])
    expanded = expand_method_grids(methods, method_grid)

    print("Common configuration:\n")
    print(cfg_map)
    print("\nEnabled methods:\n")
    for method in expanded:
        print(f"- {method.id}: {method.label} ({method.type})")


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for experiment suite commands."""
    parser = argparse.ArgumentParser(
        description="Run and analyze blind source separation benchmark samples."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run benchmark tasks.")
    run_parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/benchmark/config/common.yaml"),
        help="Path to common config.",
    )
    run_parser.add_argument(
        "--method-config-dir",
        type=Path,
        default=Path("examples/benchmark/config/methods"),
        help="Directory containing method YAMLs.",
    )
    run_parser.add_argument(
        "--method",
        nargs="*",
        default=None,
        help="Filter methods by ID.",
    )
    run_parser.add_argument(
        "--track",
        nargs="*",
        default=None,
        help="Subset of track IDs to process.",
    )
    run_parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit number of tracks (post-filtering).",
    )
    run_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list planned tasks without executing.",
    )
    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting run directory if timestamp collides.",
    )
    run_parser.add_argument(
        "--no-save-audio",
        action="store_true",
        help="Disable writing separated waveforms.",
    )
    run_parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="OmegaConf-style common config override, e.g. stft.fft_size=1024",
    )
    run_parser.add_argument(
        "--method-set",
        action="append",
        default=[],
        help=("OmegaConf-style method override, e.g. online_ilrma.params.beta=4"),
    )
    run_parser.add_argument(
        "--grid",
        action="append",
        default=[],
        help=(
            "Grid override: <method_id>.<param>=v1,v2,... "
            "(example: batch_ilrma.n_basis=2,4)"
        ),
    )
    run_parser.set_defaults(handler=run_command)

    summarize_parser = sub.add_parser(
        "summarize", help="Generate reports from results."
    )
    summarize_parser.add_argument(
        "results",
        type=Path,
        help="Path to results.jsonl produced by run command.",
    )
    summarize_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for generated plots.",
    )
    summarize_parser.set_defaults(handler=summarize_command)

    show_parser = sub.add_parser("show-config", help="Print resolved configuration.")
    show_parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/benchmark/config/common.yaml"),
        help="Common config path.",
    )
    show_parser.add_argument(
        "--method-config-dir",
        type=Path,
        default=Path("examples/benchmark/config/methods"),
        help="Method YAML directory.",
    )
    show_parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="OmegaConf-style common config override, e.g. stft.fft_size=1024",
    )
    show_parser.add_argument(
        "--method-set",
        action="append",
        default=[],
        help=("OmegaConf-style method override, e.g. online_ilrma.params.beta=4"),
    )
    show_parser.add_argument(
        "--grid",
        action="append",
        default=[],
        help=(
            "Grid override: <method_id>.<param>=v1,v2,... "
            "(example: batch_ilrma.n_basis=2,4)"
        ),
    )
    show_parser.set_defaults(handler=show_config_command)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
