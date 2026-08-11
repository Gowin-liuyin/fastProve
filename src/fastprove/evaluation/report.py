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


@dataclass(frozen=True)
class ReportOutputs:
    """Paths written by :func:`build_report`."""

    summary_csv: Path
    accuracy_figure: Path
    softmax_figure: Path
    report_markdown: Path


def _get(
    record: dict, *path: str, default: Any = None
) -> Optional[float]:
    value = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first(mapping: object, *keys: str, default: object = None) -> object:
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def _as_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_value(value: object, digits: int = 6) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "not_available" if value is None else str(value)
    return ("%%.%dg" % digits) % numeric


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    rendered = [list(str(value) for value in row) for row in rows]
    if not rendered:
        return "_没有可用记录。_"
    head = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"
        for row in rendered
    ]
    return "\n".join([head, separator, *body])


def _rows(records: Sequence[dict]) -> List[Dict[str, object]]:
    rows = []
    for record in records:
        validate_run_record(record)
        config = record.get("config", {})
        model = record.get("model", {})
        dataset = record.get("dataset", {})
        environment = record.get("environment", {})
        checkpoint_manifest = (
            model.get("checkpoint_manifest", {})
            if isinstance(model, dict)
            and isinstance(model.get("checkpoint_manifest"), dict)
            else {}
        )
        evaluation = (
            config.get("evaluation", {})
            if isinstance(config.get("evaluation"), dict)
            else {}
        )
        error = record.get("error", {})
        reason = record.get("reason", {})
        model_pretrained = _first(model, "pretrained", default=False) is True
        data_meaningful = (
            _first(dataset, "meaningful_lm_evidence", default=False) is True
        )
        rows.append(
            {
                "run_id": record.get("run_id"),
                "status": record.get("status"),
                "timestamp_utc": record.get("timestamp_utc"),
                "seed": record.get("seed"),
                "sample_count": record.get("sample_count"),
                "elapsed_seconds": record.get("elapsed_seconds"),
                "stage": _first(
                    record,
                    "stage",
                    default=_first(config, "stage", default="unspecified"),
                ),
                "selected_spec_ids_file": _first(
                    config, "selected_spec_ids_file"
                ),
                "base_config_sha256": _first(config, "base_config_sha256"),
                "sweep_config_sha256": _first(config, "sweep_config_sha256"),
                "model_id": _first(model, "identifier", "id"),
                "model_revision": _first(
                    model,
                    "revision",
                    default=_first(checkpoint_manifest, "upstream_revision"),
                ),
                "model_weight_sha256": _first(
                    model,
                    "weight_sha256",
                    "weights_sha256",
                    default=_first(
                        checkpoint_manifest,
                        "weights_sha256",
                        "weight_sha256",
                    ),
                ),
                "model_config_sha256": _first(
                    model,
                    "config_sha256",
                    default=_first(checkpoint_manifest, "config_sha256"),
                ),
                "model_tokenizer_sha256": _first(
                    model,
                    "tokenizer_sha256",
                    default=_first(checkpoint_manifest, "tokenizer_sha256"),
                ),
                "model_pretrained": model_pretrained,
                "dataset_id": _first(dataset, "identifier", "id"),
                "dataset_split": _first(dataset, "split"),
                "sample_ids_sha256": _first(dataset, "sample_ids_sha256"),
                "tokenized_inputs_sha256": _first(
                    dataset,
                    "tokenized_inputs_sha256",
                    "tokenized_input_sha256",
                    "tokenized_sha256",
                    "cache_content_sha256",
                ),
                "meaningful_lm_evidence": (
                    model_pretrained and data_meaningful
                ),
                "evaluation_scope": _first(
                    dataset,
                    "evaluation_scope",
                    default=_first(
                        config,
                        "evaluation_scope",
                        default=_first(model, "evidence_scope"),
                    ),
                ),
                "evidence_kind": _first(config, "evidence"),
                "device": _first(
                    environment, "actual_device", "device", default="unknown"
                ),
                "dtype": _first(
                    environment,
                    "activation_dtype",
                    "dtype",
                    default="unknown",
                ),
                "checkpoint_compute_dtype": _first(
                    environment, "checkpoint_compute_dtype"
                ),
                "sequence_length": _first(
                    evaluation,
                    "sequence_length",
                    default=_first(config, "sequence_length"),
                ),
                "batch_size": _first(
                    evaluation,
                    "batch_size",
                    default=_first(
                        environment,
                        "batch_size",
                        default=_first(config, "batch_size"),
                    ),
                ),
                "generation_tokens": _first(
                    evaluation, "generation_tokens", default=_first(config, "generation_tokens")
                ),
                "raw_source": record.get("_raw_source"),
                "raw_line": record.get("_raw_line"),
                "error_type": _first(error, "type"),
                "error_message": _first(error, "message"),
                "reason_code": _first(reason, "code"),
                "reason_message": _first(reason, "message"),
                "last_completed_sample_id": record.get(
                    "last_completed_sample_id"
                ),
                "partial_metrics_available": record.get(
                    "partial_metrics_available"
                ),
                "mode": config.get("mode"),
                "tau_max": config.get("tau_max"),
                "alpha": config.get("alpha"),
                "preserve_top_k": config.get("preserve_top_k"),
                "exact_gate_passed": record.get("metrics", {})
                .get("exact_gate", {})
                .get("passed"),
                "plaintext_nll": _get(
                    record,
                    "metrics",
                    "plaintext",
                    "negative_log_likelihood",
                ),
                "obfuscated_nll": _get(
                    record,
                    "metrics",
                    "obfuscated",
                    "negative_log_likelihood",
                ),
                "nll_absolute_increase": _get(
                    record,
                    "metrics",
                    "degradation",
                    "nll_absolute_increase",
                ),
                "nll_relative_increase": _get(
                    record,
                    "metrics",
                    "degradation",
                    "nll_relative_increase",
                ),
                "plaintext_perplexity": _get(
                    record, "metrics", "plaintext", "perplexity"
                ),
                "obfuscated_perplexity": _get(
                    record, "metrics", "obfuscated", "perplexity"
                ),
                "perplexity_absolute_increase": _get(
                    record,
                    "metrics",
                    "degradation",
                    "perplexity_absolute_increase",
                ),
                "perplexity_relative_increase": _get(
                    record,
                    "metrics",
                    "degradation",
                    "perplexity_relative_increase",
                ),
                "plaintext_top1": _get(
                    record,
                    "metrics",
                    "plaintext",
                    "next_token_top1_accuracy",
                ),
                "obfuscated_top1": _get(
                    record,
                    "metrics",
                    "obfuscated",
                    "next_token_top1_accuracy",
                ),
                "top1_absolute_drop": _get(
                    record,
                    "metrics",
                    "degradation",
                    "top1_absolute_drop",
                ),
                "top1_relative_drop": _get(
                    record,
                    "metrics",
                    "degradation",
                    "top1_relative_drop",
                ),
                "bootstrap_replicates": _get(
                    record, "metrics", "bootstrap", "replicates"
                ),
                "bootstrap_top1_drop_low": _get(
                    record,
                    "metrics",
                    "bootstrap",
                    "metrics",
                    "top1_absolute_drop",
                    "low",
                ),
                "bootstrap_top1_drop_high": _get(
                    record,
                    "metrics",
                    "bootstrap",
                    "metrics",
                    "top1_absolute_drop",
                    "high",
                ),
                "bootstrap_ppl_relative_low": _get(
                    record,
                    "metrics",
                    "bootstrap",
                    "metrics",
                    "perplexity_relative_increase",
                    "low",
                ),
                "bootstrap_ppl_relative_high": _get(
                    record,
                    "metrics",
                    "bootstrap",
                    "metrics",
                    "perplexity_relative_increase",
                    "high",
                ),
                "plaintext_top5": _get(
                    record,
                    "metrics",
                    "plaintext",
                    "next_token_top5_accuracy",
                ),
                "obfuscated_top5": _get(
                    record,
                    "metrics",
                    "obfuscated",
                    "next_token_top5_accuracy",
                ),
                "top5_absolute_drop": _get(
                    record,
                    "metrics",
                    "degradation",
                    "top5_absolute_drop",
                ),
                "top5_relative_drop": _get(
                    record,
                    "metrics",
                    "degradation",
                    "top5_relative_drop",
                ),
                "top1_agreement": _get(
                    record,
                    "metrics",
                    "agreement",
                    "next_token_top1_agreement",
                ),
                "greedy_token_match": _get(
                    record,
                    "metrics",
                    "greedy",
                    "greedy_token_exact_match",
                ),
                "greedy_sequence_match": _get(
                    record,
                    "metrics",
                    "greedy",
                    "greedy_sequence_exact_match",
                ),
                "softmax_kl": _get(
                    record, "metrics", "softmax", "kl_divergence"
                ),
                "softmax_js": _get(
                    record, "metrics", "softmax", "js_divergence"
                ),
                "topk_overlap": _get(
                    record, "metrics", "softmax", "topk_overlap"
                ),
                "rank_correlation": _get(
                    record, "metrics", "softmax", "rank_correlation"
                ),
                "topk_changed_fraction": _get(
                    record,
                    "metrics",
                    "softmax",
                    "topk_changed_fraction",
                ),
                "actual_noise_inf": _get(
                    record,
                    "metrics",
                    "softmax",
                    "actual_noise_infinity_norm",
                ),
                "zero_noise_query_fraction": _get(
                    record,
                    "metrics",
                    "softmax",
                    "zero_noise_query_fraction",
                ),
                "clean_boundary_margin_min": _get(
                    record,
                    "metrics",
                    "softmax",
                    "clean_boundary_margin_min",
                ),
                "clean_boundary_margin_mean": _get(
                    record,
                    "metrics",
                    "softmax",
                    "clean_boundary_margin_mean",
                ),
                "clean_boundary_margin_max": _get(
                    record,
                    "metrics",
                    "softmax",
                    "clean_boundary_margin_max",
                ),
                "attention_output_relative_l2_error": _get(
                    record,
                    "metrics",
                    "softmax",
                    "attention_output_relative_l2_error",
                ),
                "nan_inf_count": _get(
                    record, "metrics", "nan_inf_count"
                ),
                "logits_max_absolute_error": _get(
                    record,
                    "metrics",
                    "layer",
                    "logits",
                    "max_absolute_error",
                ),
                "logits_mean_absolute_error": _get(
                    record,
                    "metrics",
                    "layer",
                    "logits",
                    "mean_absolute_error",
                ),
                "logits_relative_l2_error": _get(
                    record,
                    "metrics",
                    "layer",
                    "logits",
                    "relative_l2_error",
                ),
                "logits_cosine_similarity": _get(
                    record,
                    "metrics",
                    "layer",
                    "logits",
                    "cosine_similarity",
                ),
                "qk_max_absolute_error": _get(
                    record,
                    "metrics",
                    "layer",
                    "qk_scores",
                    "max_absolute_error",
                ),
                "qk_mean_absolute_error": _get(
                    record,
                    "metrics",
                    "layer",
                    "qk_scores",
                    "mean_absolute_error",
                ),
                "qk_relative_l2_error": _get(
                    record,
                    "metrics",
                    "layer",
                    "qk_scores",
                    "relative_l2_error",
                ),
                "exact_softmax_max_absolute_error": _get(
                    record,
                    "metrics",
                    "layer",
                    "exact_softmax_probabilities",
                    "max_absolute_error",
                ),
                "conversion_time_seconds": _get(
                    record,
                    "metrics",
                    "performance",
                    "conversion_time_seconds",
                ),
                "prefill_plaintext_mean_seconds": _get(
                    record,
                    "metrics",
                    "performance",
                    "prefill",
                    "plaintext",
                    "mean_seconds",
                ),
                "prefill_obfuscated_mean_seconds": _get(
                    record,
                    "metrics",
                    "performance",
                    "prefill",
                    "obfuscated",
                    "mean_seconds",
                ),
                "prefill_obfuscated_tokens_per_second": _get(
                    record,
                    "metrics",
                    "performance",
                    "prefill",
                    "obfuscated_tokens_per_second",
                ),
                "decode_status": record.get("metrics", {})
                .get("performance", {})
                .get("decode", {})
                .get("status"),
                "decode_plaintext_tpot_seconds": _get(
                    record,
                    "metrics",
                    "performance",
                    "decode",
                    "plaintext",
                    "tpot_seconds",
                ),
                "decode_plaintext_tokens_per_second": _get(
                    record,
                    "metrics",
                    "performance",
                    "decode",
                    "plaintext",
                    "tokens_per_second",
                ),
                "decode_obfuscated_tpot_seconds": _get(
                    record,
                    "metrics",
                    "performance",
                    "decode",
                    "obfuscated",
                    "tpot_seconds",
                ),
                "decode_obfuscated_tokens_per_second": _get(
                    record,
                    "metrics",
                    "performance",
                    "decode",
                    "obfuscated",
                    "tokens_per_second",
                ),
                "peak_memory_status": record.get("metrics", {})
                .get("performance", {})
                .get("peak_memory", {})
                .get("status"),
                "peak_memory_bytes": _get(
                    record,
                    "metrics",
                    "performance",
                    "peak_memory",
                    "bytes",
                ),
                "kv_cache_status": record.get("metrics", {})
                .get("performance", {})
                .get("kv_cache", {})
                .get("status"),
                "kv_cache_plaintext_bytes": _get(
                    record,
                    "metrics",
                    "performance",
                    "kv_cache",
                    "plaintext_bytes",
                ),
                "kv_cache_obfuscated_bytes": _get(
                    record,
                    "metrics",
                    "performance",
                    "kv_cache",
                    "obfuscated_bytes",
                ),
            }
        )
    return rows


def _cohort_key(row: Dict[str, object]) -> Tuple[object, ...]:
    return (
        row.get("model_id"),
        row.get("model_revision"),
        row.get("model_weight_sha256"),
        row.get("model_config_sha256"),
        row.get("model_tokenizer_sha256"),
        row.get("dataset_id"),
        row.get("dataset_split"),
        row.get("seed"),
        row.get("stage"),
        row.get("device"),
        row.get("dtype"),
        row.get("checkpoint_compute_dtype"),
        row.get("sequence_length"),
        row.get("batch_size"),
        row.get("generation_tokens"),
        row.get("sample_count"),
        row.get("tokenized_inputs_sha256"),
        row.get("sample_ids_sha256"),
        row.get("base_config_sha256"),
        row.get("sweep_config_sha256"),
    )


def _tradeoff_candidates(
    rows: Sequence[Dict[str, object]],
    *,
    mode: Optional[str] = None,
) -> List[Dict[str, object]]:
    meaningful = [
        row
        for row in rows
        if row.get("status") == "success"
        and row.get("meaningful_lm_evidence") is True
    ]
    cohorts = {_cohort_key(row) for row in meaningful}
    if len(cohorts) != 1:
        return []
    cohort = next(iter(cohorts))
    exact_gate_passed = any(
        row.get("mode") == "exact"
        and row.get("exact_gate_passed") is True
        and _cohort_key(row) == cohort
        for row in meaningful
    )
    if not exact_gate_passed:
        return []
    candidates: List[Dict[str, object]] = []
    for row in meaningful:
        row_mode = row.get("mode")
        if row_mode not in ("topk_preserving", "free_bounded"):
            continue
        if mode is not None and row_mode != mode:
            continue
        noise = _as_float(row.get("actual_noise_inf"))
        top1_drop = _as_float(row.get("top1_absolute_drop"))
        agreement = _as_float(row.get("top1_agreement"))
        ppl_increase = _as_float(row.get("perplexity_relative_increase"))
        if (
            noise is None
            or top1_drop is None
            or agreement is None
            or ppl_increase is None
            or noise <= 0
        ):
            continue
        candidates.append(row)
    return candidates


def _best_tradeoff(
    rows: Sequence[Dict[str, object]],
    *,
    mode: Optional[str] = None,
) -> Optional[str]:
    candidates = _tradeoff_candidates(rows, mode=mode)
    if not candidates:
        return None
    constrained = [
        row
        for row in candidates
        if float(row["top1_absolute_drop"]) <= 0.01
        and float(row["top1_agreement"]) >= 0.99
    ]
    if constrained:
        best = max(
            constrained,
            key=lambda row: (
                float(row["actual_noise_inf"]),
                -float(row["perplexity_relative_increase"]),
            ),
        )
    else:
        best = min(
            candidates,
            key=lambda row: (
                float(row["top1_absolute_drop"])
                + max(0.0, float(row["perplexity_relative_increase"])),
                -float(row["actual_noise_inf"]),
            ),
        )
    return str(best["run_id"])


def _pareto_frontier(
    rows: Sequence[Dict[str, object]], *, mode: str
) -> List[str]:
    candidates = _tradeoff_candidates(rows, mode=mode)
    frontier: List[Dict[str, object]] = []
    for candidate in candidates:
        c_top1 = float(candidate["top1_absolute_drop"])
        c_ppl = float(candidate["perplexity_relative_increase"])
        c_noise = float(candidate["actual_noise_inf"])
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            o_top1 = float(other["top1_absolute_drop"])
            o_ppl = float(other["perplexity_relative_increase"])
            o_noise = float(other["actual_noise_inf"])
            no_worse = (
                o_top1 <= c_top1
                and o_ppl <= c_ppl
                and o_noise >= c_noise
            )
            strictly_better = (
                o_top1 < c_top1
                or o_ppl < c_ppl
                or o_noise > c_noise
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    frontier.sort(
        key=lambda row: (
            float(row["actual_noise_inf"]),
            str(row["run_id"]),
        )
    )
    return [str(row["run_id"]) for row in frontier]


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

    softmax_figure = figure_dir / "softmax_vs_tau.png"
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

    overall_best = _best_tradeoff(rows)
    topk_best = _best_tradeoff(rows, mode="topk_preserving")
    free_best = _best_tradeoff(rows, mode="free_bounded")
    topk_frontier = _pareto_frontier(rows, mode="topk_preserving")
    free_frontier = _pareto_frontier(rows, mode="free_bounded")

    provenance_rows = []
    seen_provenance = set()
    for row in rows:
        provenance = (
            row.get("model_id"),
            row.get("model_revision"),
            row.get("model_weight_sha256"),
            row.get("model_config_sha256"),
            row.get("dataset_id"),
            row.get("dataset_split"),
            row.get("tokenized_inputs_sha256"),
            row.get("sample_ids_sha256"),
            row.get("seed"),
            row.get("sample_count"),
            row.get("device"),
            row.get("dtype"),
            row.get("sequence_length"),
            row.get("batch_size"),
            row.get("stage"),
            row.get("selected_spec_ids_file"),
            row.get("base_config_sha256"),
            row.get("sweep_config_sha256"),
            row.get("raw_source"),
            row.get("raw_line"),
        )
        if provenance in seen_provenance:
            continue
        seen_provenance.add(provenance)
        provenance_rows.append(
            [
                row.get("model_id"),
                row.get("model_revision"),
                _format_value(row.get("model_weight_sha256")),
                _format_value(row.get("model_config_sha256")),
                row.get("dataset_id"),
                row.get("dataset_split"),
                _format_value(row.get("tokenized_inputs_sha256")),
                _format_value(row.get("sample_ids_sha256")),
                row.get("seed"),
                row.get("sample_count"),
                row.get("device"),
                row.get("dtype"),
                _format_value(row.get("sequence_length")),
                _format_value(row.get("batch_size")),
                row.get("stage"),
                _format_value(row.get("selected_spec_ids_file")),
                _format_value(row.get("base_config_sha256")),
                _format_value(row.get("sweep_config_sha256")),
                _format_value(row.get("raw_source")),
                _format_value(row.get("raw_line")),
            ]
        )

    exact_rows = [
        row
        for row in rows
        if row.get("status") == "success" and row.get("mode") == "exact"
    ]
    accuracy_rows = [
        row
        for row in rows
        if row.get("status") == "success"
        and row.get("mode") in ("plaintext", "exact", "topk_preserving", "free_bounded")
    ]
    approximate_rows = [
        row
        for row in rows
        if row.get("status") == "success"
        and row.get("mode") in ("topk_preserving", "free_bounded")
    ]
    performance_rows = [
        row for row in rows if row.get("status") == "success"
    ]
    non_success_rows = [
        row for row in rows if row.get("status") != "success"
    ]

    status_counts = {
        status: sum(1 for row in rows if row.get("status") == status)
        for status in ("success", "failure", "skipped")
    }
    raw_source_lines = (
        "\n".join("- `%s`" % path for path in (raw_sources or []))
        or "- 未提供 raw source 路径；数值仍来自传入的已验证记录。"
    )
    provenance_table = _markdown_table(
        [
            "模型",
            "revision",
            "weight SHA-256",
            "config SHA-256",
            "数据",
            "split",
            "tokenized SHA-256",
            "sample IDs SHA-256",
            "seed",
            "样本数",
            "device",
            "dtype",
            "sequence length",
            "batch size",
            "stage",
            "candidate manifest",
            "base config SHA-256",
            "sweep config SHA-256",
            "raw source",
            "raw line",
        ],
        provenance_rows,
    )
    exact_table = _markdown_table(
        [
            "run_id",
            "raw source",
            "raw line",
            "logits max abs",
            "QK max abs",
            "Softmax max abs",
            "token agreement",
            "greedy token",
            "sequence",
            "gate",
        ],
        (
            [
                row.get("run_id"),
                _format_value(row.get("raw_source")),
                _format_value(row.get("raw_line")),
                _format_value(row.get("logits_max_absolute_error")),
                _format_value(row.get("qk_max_absolute_error")),
                _format_value(row.get("exact_softmax_max_absolute_error")),
                _format_value(row.get("top1_agreement")),
                _format_value(row.get("greedy_token_match")),
                _format_value(row.get("greedy_sequence_match")),
                row.get("exact_gate_passed"),
            ]
            for row in exact_rows
        ),
    )
    accuracy_table = _markdown_table(
        [
            "run_id",
            "raw source",
            "raw line",
            "mode",
            "evaluation scope",
            "tau",
            "明文 NLL",
            "混淆 NLL",
            "NLL 绝对增量",
            "NLL 相对增量",
            "明文 PPL",
            "混淆 PPL",
            "PPL 绝对增量",
            "PPL 相对增量",
            "明文 top-1",
            "混淆 top-1",
            "top-1 绝对下降",
            "top-1 相对下降",
            "bootstrap reps",
            "top-1 drop CI low",
            "top-1 drop CI high",
            "PPL relative CI low",
            "PPL relative CI high",
            "明文 top-5",
            "混淆 top-5",
            "top-5 绝对下降",
            "top-5 相对下降",
            "token agreement",
            "greedy token",
            "greedy sequence",
        ],
        (
            [
                row.get("run_id"),
                _format_value(row.get("raw_source")),
                _format_value(row.get("raw_line")),
                row.get("mode"),
                row.get("evaluation_scope"),
                _format_value(row.get("tau_max")),
                _format_value(row.get("plaintext_nll")),
                _format_value(row.get("obfuscated_nll")),
                _format_value(row.get("nll_absolute_increase")),
                _format_value(row.get("nll_relative_increase")),
                _format_value(row.get("plaintext_perplexity")),
                _format_value(row.get("obfuscated_perplexity")),
                _format_value(row.get("perplexity_absolute_increase")),
                _format_value(row.get("perplexity_relative_increase")),
                _format_value(row.get("plaintext_top1")),
                _format_value(row.get("obfuscated_top1")),
                _format_value(row.get("top1_absolute_drop")),
                _format_value(row.get("top1_relative_drop")),
                _format_value(row.get("bootstrap_replicates")),
                _format_value(row.get("bootstrap_top1_drop_low")),
                _format_value(row.get("bootstrap_top1_drop_high")),
                _format_value(row.get("bootstrap_ppl_relative_low")),
                _format_value(row.get("bootstrap_ppl_relative_high")),
                _format_value(row.get("plaintext_top5")),
                _format_value(row.get("obfuscated_top5")),
                _format_value(row.get("top5_absolute_drop")),
                _format_value(row.get("top5_relative_drop")),
                _format_value(row.get("top1_agreement")),
                _format_value(row.get("greedy_token_match")),
                _format_value(row.get("greedy_sequence_match")),
            ]
            for row in accuracy_rows
        ),
    )
    softmax_table = _markdown_table(
        [
            "run_id",
            "raw source",
            "raw line",
            "mode",
            "tau",
            "actual ||noise||∞",
            "KL",
            "JS",
            "Top-k overlap",
            "changed fraction",
            "rank correlation",
            "zero-noise fraction",
            "attention output rel-L2",
        ],
        (
            [
                row.get("run_id"),
                _format_value(row.get("raw_source")),
                _format_value(row.get("raw_line")),
                row.get("mode"),
                _format_value(row.get("tau_max")),
                _format_value(row.get("actual_noise_inf")),
                _format_value(row.get("softmax_kl")),
                _format_value(row.get("softmax_js")),
                _format_value(row.get("topk_overlap")),
                _format_value(row.get("topk_changed_fraction")),
                _format_value(row.get("rank_correlation")),
                _format_value(row.get("zero_noise_query_fraction")),
                _format_value(
                    row.get("attention_output_relative_l2_error")
                ),
            ]
            for row in approximate_rows
        ),
    )
    performance_table = _markdown_table(
        [
            "run_id",
            "raw source",
            "raw line",
            "conversion s",
            "prefill plain s",
            "prefill obf s",
            "decode plain TPOT",
            "decode obf TPOT",
            "prefill obf tok/s",
            "decode plain tok/s",
            "decode obf tok/s",
            "peak bytes",
            "KV plain bytes",
            "KV obf bytes",
        ],
        (
            [
                row.get("run_id"),
                _format_value(row.get("raw_source")),
                _format_value(row.get("raw_line")),
                _format_value(row.get("conversion_time_seconds")),
                _format_value(row.get("prefill_plaintext_mean_seconds")),
                _format_value(row.get("prefill_obfuscated_mean_seconds")),
                _format_value(row.get("decode_plaintext_tpot_seconds")),
                _format_value(row.get("decode_obfuscated_tpot_seconds")),
                _format_value(
                    row.get("prefill_obfuscated_tokens_per_second")
                ),
                _format_value(
                    row.get("decode_plaintext_tokens_per_second")
                ),
                _format_value(
                    row.get("decode_obfuscated_tokens_per_second")
                ),
                _format_value(row.get("peak_memory_bytes")),
                _format_value(row.get("kv_cache_plaintext_bytes")),
                _format_value(row.get("kv_cache_obfuscated_bytes")),
            ]
            for row in performance_rows
        ),
    )
    non_success_table = _markdown_table(
        [
            "run_id",
            "status",
            "mode",
            "tau",
            "alpha",
            "k",
            "raw source",
            "raw line",
            "stage",
            "code/type",
            "message",
            "last sample",
            "partial metrics",
        ],
        (
            [
                row.get("run_id"),
                row.get("status"),
                row.get("mode"),
                _format_value(row.get("tau_max")),
                _format_value(row.get("alpha")),
                _format_value(row.get("preserve_top_k")),
                _format_value(row.get("raw_source")),
                _format_value(row.get("raw_line")),
                row.get("stage"),
                row.get("reason_code") or row.get("error_type"),
                row.get("reason_message") or row.get("error_message"),
                row.get("last_completed_sample_id"),
                row.get("partial_metrics_available"),
            ]
            for row in non_success_rows
        ),
    )

    report_markdown = root / "REPORT.md"
    report = f"""# fastProve 实验报告

> 状态：`{experiment_status}`。预训练评测：`{pretrained_status}`。
> 记录统计：success={status_counts["success"]}，failure={status_counts["failure"]}，
> skipped={status_counts["skipped"]}。

## 1. 实验目标

公平比较 plaintext、exact、topk_preserving 与 free_bounded 四种模式，并
分别回答 exact 数值误差、精度下降、排序变化、最佳折中和安全非结论。

## 2. 数学实现摘要

原型使用良态增广基、显式非线性检查点、RoPE 后共同正交 Q/K 变换、
混合 Value 路径、SwiGLU 置换/缩放补偿，以及仅作用于有效 logits 的
FP32 有界噪声。

## 3. 环境、模型和数据

所有数值只能来自 raw JSONL。随机 Tiny LM（random tiny model）仅用于
correctness，不能作为有意义的语言模型准确率证据。

Raw 证据：

{raw_source_lines}

{provenance_table}

## 4. 精确模式正确性

{exact_table}

只有 `exact_gate_passed=True` 且 greedy/token agreement 均为 1 的记录
才能支持“仅浮点级误差”的本次实验结论；否则必须先调查。

## 5. 噪声强度—精度曲线

见 `figures/accuracy_vs_tau.png` 与 `tables/summary.csv`。成功记录以
不连线的散点显示；failure/skipped 点不插值，并在下表中直接追溯 raw 行。
若状态为 `completed_selected_subset`，曲线仅覆盖候选 manifest 选中的点，
不能解释为完整 72 点扫描。若记录启用 sample-unit bootstrap，accuracy 表同时
列出 top-1 drop 和 PPL relative increase 的 95% 区间。

{accuracy_table}

## 6. Softmax 排序与 Top-k 变化

见 `figures/softmax_vs_tau.png`；该图同样只画不连线的散点。Top-k 模式
保证的是集合而非集合内顺序；free-bounded 允许但不保证发生排名改变。

{softmax_table}

## 7. 最佳折中配置

透明规则：先筛选 top-1 绝对下降不超过 0.01 且 token agreement 至少
0.99 的非零噪声点，再选实际噪声最大者；若无满足点，则最小化
top-1 drop 与 perplexity 相对增幅之和。只有明确标记为预训练且数据可作为
LM 证据、同一 cohort 且 exact gate 通过的记录才参与选择。

- 整体：`{overall_best or "not_available"}`
- topk_preserving：`{topk_best or "not_available"}`
- free_bounded：`{free_best or "not_available"}`
- topk_preserving Pareto：`{", ".join(topk_frontier) or "not_available"}`
- free_bounded Pareto：`{", ".join(free_frontier) or "not_available"}`

## 8. 性能开销

转换、prefill、decode/TPOT、tokens/s、峰值内存和 KV-cache 内存必须由
正式运行记录；当前状态不得从代码结构推测数值。

{performance_table}

失败与跳过点不插值、不删除，列示如下：

{non_success_table}

## 9. 威胁模型和安全限制

本原型只考察 honest-but-curious 观察者看到的持久张量和普通框架输出。
“API 不返回概率”不是密码学安全。能够修改内核、挂任意 hook、读取融合
检查点内部、dump 寄存器/临时缓冲区或读取客户端秘密的攻击者不受保护。
本项目不是 FHE、端到端加密，也不因任何 KDF 而获得 LWE 安全性。

## 10. 完整复现命令

```bash
python3 -m pytest -q
python3 scripts/run_correctness.py --config configs/tiny_exact.yaml --execute
python3 scripts/run_accuracy_sweep.py --config configs/eval_sweep.yaml --execute-deferred
python3 scripts/build_report.py --raw results/raw --output results --execute
```
"""
    report_markdown.write_text(report, encoding="utf-8")
    return ReportOutputs(
        summary_csv=summary_csv,
        accuracy_figure=accuracy_figure,
        softmax_figure=softmax_figure,
        report_markdown=report_markdown,
    )
