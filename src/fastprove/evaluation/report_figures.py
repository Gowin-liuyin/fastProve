"""Figure-writing helpers for report derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .report_format import _as_float, _format_value
from .report_rows import _cohort_key


def _series_groups(
    rows: Sequence[Dict[str, object]],
) -> Dict[str, List[Dict[str, object]]]:
    """Group curves without connecting incompatible alpha/Top-k schedules."""

    groups: Dict[str, List[Dict[str, object]]] = {}
    cohort_keys = {_cohort_key(row) for row in rows}
    include_cohort = len(cohort_keys) > 1
    for row in rows:
        mode = row.get("mode")
        if mode == "topk_preserving":
            label = "topk_preserving/alpha=%s/k=%s" % (
                row.get("alpha"),
                row.get("preserve_top_k"),
            )
        elif mode == "free_bounded":
            label = "free_bounded"
        else:
            continue
        if include_cohort:
            cohort = _cohort_key(row)
            label = "%s/cohort=%s" % (
                label,
                "|".join(_format_value(value) for value in cohort),
            )
        groups.setdefault(label, []).append(row)
    for selected in groups.values():
        selected.sort(key=lambda row: float(row.get("tau_max") or 0))
    return groups


def _write_accuracy_figure(
    successful: Sequence[Dict[str, object]],
    non_success_count: int,
    accuracy_figure: Path,
) -> None:
    """Render the accuracy-versus-tau scatter figure."""

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for label, selected in _series_groups(successful).items():
        top1_points = [
            (
                _as_float(row.get("tau_max")),
                _as_float(row.get("top1_absolute_drop")),
            )
            for row in selected
        ]
        top1_points = [
            (x, y)
            for x, y in top1_points
            if x is not None and y is not None
        ]
        ppl_points = [
            (
                _as_float(row.get("tau_max")),
                _as_float(row.get("perplexity_relative_increase")),
            )
            for row in selected
        ]
        ppl_points = [
            (x, y)
            for x, y in ppl_points
            if x is not None and y is not None
        ]
        if top1_points:
            axes[0].scatter(
                [point[0] for point in top1_points],
                [point[1] for point in top1_points],
                label=label,
            )
        if ppl_points:
            axes[1].scatter(
                [point[0] for point in ppl_points],
                [point[1] for point in ppl_points],
                label=label,
            )
    axes[0].set(xlabel="tau_max", ylabel="top-1 absolute drop")
    axes[1].set(xlabel="tau_max", ylabel="perplexity relative increase")
    for axis in axes:
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    if not any(axis.collections for axis in axes):
        fig.text(
            0.5,
            0.5,
            "No successful numeric accuracy records; no points are drawn.",
            ha="center",
            fontsize=10,
        )
    if non_success_count:
        fig.text(
            0.5,
            0.01,
            "Successful records are unconnected scatter points; "
            "%d failure/skipped runs appear only in REPORT.md."
            % non_success_count,
            ha="center",
            fontsize=8,
        )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(accuracy_figure, dpi=150)
    plt.close(fig)


def _write_softmax_figure(
    successful: Sequence[Dict[str, object]],
    non_success_count: int,
    softmax_figure: Path,
) -> None:
    """Render the Softmax-versus-tau scatter figure."""

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for label, selected in _series_groups(successful).items():
        js_points = [
            (
                _as_float(row.get("tau_max")),
                _as_float(row.get("softmax_js")),
            )
            for row in selected
        ]
        js_points = [
            (x, y) for x, y in js_points if x is not None and y is not None
        ]
        overlap_points = [
            (
                _as_float(row.get("tau_max")),
                _as_float(row.get("topk_overlap")),
            )
            for row in selected
        ]
        overlap_points = [
            (x, y)
            for x, y in overlap_points
            if x is not None and y is not None
        ]
        if js_points:
            axes[0].scatter(
                [point[0] for point in js_points],
                [point[1] for point in js_points],
                label=label,
            )
        if overlap_points:
            axes[1].scatter(
                [point[0] for point in overlap_points],
                [point[1] for point in overlap_points],
                label=label,
            )
    axes[0].set(xlabel="tau_max", ylabel="JS divergence")
    axes[1].set(xlabel="tau_max", ylabel="Top-k overlap")
    for axis in axes:
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    if not any(axis.collections for axis in axes):
        fig.text(
            0.5,
            0.5,
            "No successful numeric Softmax records; no points are drawn.",
            ha="center",
            fontsize=10,
        )
    if non_success_count:
        fig.text(
            0.5,
            0.01,
            "Successful records are unconnected scatter points; "
            "%d failure/skipped runs appear only in REPORT.md."
            % non_success_count,
            ha="center",
            fontsize=8,
        )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(softmax_figure, dpi=150)
    plt.close(fig)
