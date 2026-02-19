"""Experiment report generation utilities.

This module generates aggregated benchmark reports from ``results.jsonl`` files.
It produces:

- variant-level and family-level CSV summaries
- JSON summary with top-performing variants
- compact plots for SI-SDR improvement and real-time factor tradeoffs
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _split_method_variant(method_id: str) -> tuple[str, str]:
    if "__" in method_id:
        base, variant = method_id.split("__", 1)
        return base, variant
    return method_id, ""


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.median(values))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_variant_sdri(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    sorted_rows = sorted(
        rows,
        key=lambda item: _safe_float(item.get("sdr_imp_mean_mean")) or -np.inf,
        reverse=True,
    )
    labels = [str(item["method_id"]) for item in sorted_rows]
    values = [
        _safe_float(item.get("sdr_imp_mean_mean")) or float("nan")
        for item in sorted_rows
    ]

    fig, ax = plt.subplots(figsize=(max(8.0, 0.35 * len(labels)), 4.0))
    ax.bar(np.arange(len(labels)), values, color="#4c78a8")
    ax.set_ylabel("Mean SI-SDR Improvement [dB]")
    ax.set_title("Variant Ranking by Mean SI-SDR Improvement")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _plot_tradeoff(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    x = np.array([_safe_float(item.get("rtf_mean")) for item in rows], dtype=float)
    y = np.array(
        [_safe_float(item.get("sdr_imp_mean_mean")) for item in rows], dtype=float
    )
    labels = [str(item["method_id"]) for item in rows]
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(x[valid], y[valid], c="#f58518", edgecolors="black", linewidths=0.4)
    for xi, yi, label in zip(x[valid], y[valid], np.array(labels)[valid]):
        ax.annotate(
            label, (xi, yi), fontsize=7, xytext=(4, 2), textcoords="offset points"
        )
    ax.set_xlabel("Mean Real-Time Factor")
    ax.set_ylabel("Mean SI-SDR Improvement [dB]")
    ax.set_title("Quality-Speed Tradeoff")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def generate_experiment_report(
    results_path: Path,
    reports_dir: Path,
) -> None:
    """Generate aggregate reports for a benchmark run.

    Parameters
    ----------
    results_path:
        Path to ``results.jsonl`` written by the experiment pipeline.
    reports_dir:
        Destination directory for report artifacts.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            records.append(json.loads(text))

    if not records:
        LOGGER.warning("No records found in %s", results_path)
        return

    variant_acc: dict[str, dict[str, Any]] = {}
    family_acc: dict[str, dict[str, Any]] = {}
    for record in records:
        method_id = str(record.get("method_id", "unknown"))
        base_id, variant_name = _split_method_variant(method_id)
        label = str(record.get("method_label", method_id))
        success = bool(record.get("success", False))
        sdr_imp_mean = _safe_float(record.get("sdr_imp_mean"))
        rtf = _safe_float(record.get("real_time_factor"))

        variant = variant_acc.setdefault(
            method_id,
            {
                "method_id": method_id,
                "base_method_id": base_id,
                "variant": variant_name,
                "method_label": label,
                "num_runs": 0,
                "num_success": 0,
                "sdr_imp_mean_values": [],
                "rtf_values": [],
            },
        )
        variant["num_runs"] += 1
        if success:
            variant["num_success"] += 1
            if sdr_imp_mean is not None:
                variant["sdr_imp_mean_values"].append(sdr_imp_mean)
            if rtf is not None:
                variant["rtf_values"].append(rtf)

        family = family_acc.setdefault(
            base_id,
            {
                "base_method_id": base_id,
                "num_runs": 0,
                "num_success": 0,
                "sdr_imp_mean_values": [],
                "rtf_values": [],
            },
        )
        family["num_runs"] += 1
        if success:
            family["num_success"] += 1
            if sdr_imp_mean is not None:
                family["sdr_imp_mean_values"].append(sdr_imp_mean)
            if rtf is not None:
                family["rtf_values"].append(rtf)

    variant_rows: list[dict[str, Any]] = []
    for item in variant_acc.values():
        num_runs = int(item["num_runs"])
        num_success = int(item["num_success"])
        sdri_values = list(item["sdr_imp_mean_values"])
        rtf_values = list(item["rtf_values"])
        variant_rows.append(
            {
                "method_id": item["method_id"],
                "base_method_id": item["base_method_id"],
                "variant": item["variant"],
                "method_label": item["method_label"],
                "num_runs": num_runs,
                "num_success": num_success,
                "success_rate": (num_success / num_runs)
                if num_runs > 0
                else float("nan"),
                "sdr_imp_mean_mean": _mean(sdri_values),
                "sdr_imp_mean_median": _median(sdri_values),
                "rtf_mean": _mean(rtf_values),
            }
        )
    variant_rows.sort(
        key=lambda row: _safe_float(row.get("sdr_imp_mean_mean")) or -np.inf,
        reverse=True,
    )

    family_rows: list[dict[str, Any]] = []
    for item in family_acc.values():
        num_runs = int(item["num_runs"])
        num_success = int(item["num_success"])
        sdri_values = list(item["sdr_imp_mean_values"])
        rtf_values = list(item["rtf_values"])
        family_rows.append(
            {
                "base_method_id": item["base_method_id"],
                "num_runs": num_runs,
                "num_success": num_success,
                "success_rate": (num_success / num_runs)
                if num_runs > 0
                else float("nan"),
                "sdr_imp_mean_mean": _mean(sdri_values),
                "sdr_imp_mean_median": _median(sdri_values),
                "rtf_mean": _mean(rtf_values),
            }
        )
    family_rows.sort(
        key=lambda row: _safe_float(row.get("sdr_imp_mean_mean")) or -np.inf,
        reverse=True,
    )

    best_by_family: list[dict[str, Any]] = []
    for family in family_rows:
        base_id = str(family["base_method_id"])
        candidates = [row for row in variant_rows if row["base_method_id"] == base_id]
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda row: _safe_float(row.get("sdr_imp_mean_mean")) or -np.inf,
        )
        best_by_family.append(
            {
                "base_method_id": base_id,
                "best_method_id": best["method_id"],
                "best_sdr_imp_mean_mean": best["sdr_imp_mean_mean"],
                "best_rtf_mean": best["rtf_mean"],
            }
        )

    _write_csv(
        reports_dir / "aggregate_by_variant.csv",
        variant_rows,
        [
            "method_id",
            "base_method_id",
            "variant",
            "method_label",
            "num_runs",
            "num_success",
            "success_rate",
            "sdr_imp_mean_mean",
            "sdr_imp_mean_median",
            "rtf_mean",
        ],
    )
    _write_csv(
        reports_dir / "aggregate_by_family.csv",
        family_rows,
        [
            "base_method_id",
            "num_runs",
            "num_success",
            "success_rate",
            "sdr_imp_mean_mean",
            "sdr_imp_mean_median",
            "rtf_mean",
        ],
    )
    _write_csv(
        reports_dir / "best_variant_per_family.csv",
        best_by_family,
        [
            "base_method_id",
            "best_method_id",
            "best_sdr_imp_mean_mean",
            "best_rtf_mean",
        ],
    )

    summary = {
        "num_records": len(records),
        "num_variants": len(variant_rows),
        "num_families": len(family_rows),
        "best_variants": best_by_family,
    }
    (reports_dir / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    _plot_variant_sdri(variant_rows, reports_dir / "variant_sdri_bar.pdf")
    _plot_tradeoff(variant_rows, reports_dir / "variant_tradeoff_scatter.pdf")
