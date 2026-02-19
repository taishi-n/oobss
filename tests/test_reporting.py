from __future__ import annotations

import json
from pathlib import Path

from oobss.experiments.reporting import generate_experiment_report


def test_generate_experiment_report_writes_aggregate_files(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    rows = [
        {
            "track_id": "001",
            "method_id": "batch_ilrma__n_basis=2",
            "method_label": "Batch ILRMA [n_basis=2]",
            "success": True,
            "sdr_imp_mean": 1.0,
            "real_time_factor": 0.5,
        },
        {
            "track_id": "002",
            "method_id": "batch_ilrma__n_basis=4",
            "method_label": "Batch ILRMA [n_basis=4]",
            "success": True,
            "sdr_imp_mean": 2.0,
            "real_time_factor": 0.8,
        },
        {
            "track_id": "001",
            "method_id": "online_auxiva",
            "method_label": "Online AuxIVA",
            "success": True,
            "sdr_imp_mean": 0.2,
            "real_time_factor": 0.1,
        },
    ]
    results.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    reports_dir = tmp_path / "reports"
    generate_experiment_report(results, reports_dir)

    assert (reports_dir / "aggregate_by_variant.csv").exists()
    assert (reports_dir / "aggregate_by_family.csv").exists()
    assert (reports_dir / "best_variant_per_family.csv").exists()
    assert (reports_dir / "aggregate_summary.json").exists()
    assert (reports_dir / "variant_sdri_bar.pdf").exists()
    assert (reports_dir / "variant_tradeoff_scatter.pdf").exists()
