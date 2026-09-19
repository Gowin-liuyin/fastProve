"""Row extraction helpers for report derivation."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .artifacts import validate_run_record
from .report_format import _first, _get


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
