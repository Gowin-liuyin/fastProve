"""Markdown table-building helpers for report derivation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

from .report_format import _format_value, _markdown_table


@dataclass(frozen=True)
class ReportTables:
    """Rendered Markdown tables produced by :func:`_build_tables`."""

    provenance_table: str
    exact_table: str
    accuracy_table: str
    softmax_table: str
    performance_table: str
    non_success_table: str


def _build_tables(
    rows: Sequence[Dict[str, object]],
) -> ReportTables:
    """Build every Markdown table from derived report rows."""

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


    return ReportTables(
        provenance_table=provenance_table,
        exact_table=exact_table,
        accuracy_table=accuracy_table,
        softmax_table=softmax_table,
        performance_table=performance_table,
        non_success_table=non_success_table,
    )
