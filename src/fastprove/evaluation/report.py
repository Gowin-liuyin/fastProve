"""Derive CSV, figures, and a limitations-aware Markdown report."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .artifacts import validate_run_record
from .report_figures import (
    _series_groups,
    _write_accuracy_figure,
    _write_softmax_figure,
)
from .report_format import (
    _as_float,
    _first,
    _format_value,
    _get,
    _markdown_table,
)
from .report_markdown import _write_report_markdown
from .report_rows import _cohort_key, _rows
from .report_tables import _build_tables
from .report_tradeoff import (
    _best_tradeoff,
    _pareto_frontier,
    _tradeoff_candidates,
)


@dataclass(frozen=True)
class ReportOutputs:
    """Paths written by :func:`build_report`."""

    summary_csv: Path
    accuracy_figure: Path
    softmax_figure: Path
    report_markdown: Path


def build_report(
    *,
    records: Sequence[dict],
    output_root: Path,
    experiment_status: str,
    pretrained_status: str,
    raw_sources: Optional[Sequence[Path]] = None,
) -> ReportOutputs:
    """Build all derived artifacts strictly from supplied raw records."""

    if not records:
        raise ValueError("no raw run records were supplied")
    for record in records:
        validate_run_record(record)
    root = Path(output_root)
    table_dir = root / "tables"
    figure_dir = root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows(records)
    summary_csv = table_dir / "summary.csv"
    fieldnames = list(rows[0].keys()) if rows else [
        "run_id",
        "status",
        "mode",
        "tau_max",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successful = [row for row in rows if row["status"] == "success"]
    non_success_count = len(rows) - len(successful)
    accuracy_figure = figure_dir / "accuracy_vs_tau.png"
    _write_accuracy_figure(successful, non_success_count, accuracy_figure)

    softmax_figure = figure_dir / "softmax_vs_tau.png"
    _write_softmax_figure(successful, non_success_count, softmax_figure)

    overall_best = _best_tradeoff(rows)
    topk_best = _best_tradeoff(rows, mode="topk_preserving")
    free_best = _best_tradeoff(rows, mode="free_bounded")
    topk_frontier = _pareto_frontier(rows, mode="topk_preserving")
    free_frontier = _pareto_frontier(rows, mode="free_bounded")

    status_counts = {
        status: sum(1 for row in rows if row.get("status") == status)
        for status in ("success", "failure", "skipped")
    }
    raw_source_lines = (
        "\n".join("- `%s`" % path for path in (raw_sources or []))
        or "- 未提供 raw source 路径；数值仍来自传入的已验证记录。"
    )
    tables = _build_tables(rows)

    report_markdown = root / "REPORT.md"
    _write_report_markdown(
        report_markdown=report_markdown,
        experiment_status=experiment_status,
        pretrained_status=pretrained_status,
        status_counts=status_counts,
        raw_source_lines=raw_source_lines,
        provenance_table=tables.provenance_table,
        exact_table=tables.exact_table,
        accuracy_table=tables.accuracy_table,
        softmax_table=tables.softmax_table,
        performance_table=tables.performance_table,
        non_success_table=tables.non_success_table,
        overall_best=overall_best,
        topk_best=topk_best,
        free_best=free_best,
        topk_frontier=topk_frontier,
        free_frontier=free_frontier,
    )
    return ReportOutputs(
        summary_csv=summary_csv,
        accuracy_figure=accuracy_figure,
        softmax_figure=softmax_figure,
        report_markdown=report_markdown,
    )
