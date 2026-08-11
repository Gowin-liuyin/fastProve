from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import pytest
from matplotlib.figure import Figure

from fastprove.evaluation.artifacts import build_run_record
from fastprove.evaluation.report import (
    _cohort_key,
    _rows,
    build_report,
)


def _metrics(tau: float) -> dict[str, Any]:
    return {
        "plaintext": {
            "negative_log_likelihood": 1.0,
            "perplexity": math.e,
            "next_token_top1_accuracy": 0.5,
            "next_token_top5_accuracy": 0.9,
        },
        "obfuscated": {
            "negative_log_likelihood": 1.0 + tau,
            "perplexity": math.e + tau,
            "next_token_top1_accuracy": 0.5 - tau,
            "next_token_top5_accuracy": 0.9 - tau,
        },
        "agreement": {"next_token_top1_agreement": 1.0 - tau},
        "degradation": {
            "nll_absolute_increase": tau,
            "nll_relative_increase": tau,
            "perplexity_absolute_increase": tau,
            "perplexity_relative_increase": tau / math.e,
            "top1_absolute_drop": tau,
            "top1_relative_drop": tau / 0.5,
            "top5_absolute_drop": tau,
            "top5_relative_drop": tau / 0.9,
        },
        "greedy": {
            "greedy_token_exact_match": 1.0 - tau,
            "greedy_sequence_exact_match": 1.0 - tau,
        },
        "softmax": {
            "kl_divergence": tau,
            "js_divergence": tau / 2,
            "topk_overlap": 1.0 - tau,
            "rank_correlation": 1.0 - tau,
            "topk_changed_fraction": tau,
            "actual_noise_infinity_norm": tau,
            "zero_noise_query_fraction": 1.0 if tau == 0 else 0.0,
            "attention_output_relative_l2_error": tau,
        },
        "performance": {
            "conversion_time_seconds": 0.01,
            "prefill_latency_seconds": 0.02,
            "decode_tpot_seconds": 0.003,
            "tokens_per_second": 333.0,
            "peak_memory_bytes": 1024,
            "kv_cache_memory_bytes": 512,
        },
    }


def _success_record(
    run_id: str,
    *,
    mode: str = "free_bounded",
    tau: float = 0.01,
    raw_source: str | None = "results/raw/attempt/eval.jsonl",
    raw_line: int | None = 1,
) -> dict[str, Any]:
    record = build_run_record(
        run_id=run_id,
        status="success",
        config={
            "mode": mode,
            "tau_max": tau,
            "alpha": 0.8,
            "preserve_top_k": 8,
            "stage": "full_evaluation",
            "evaluation": {
                "sequence_length": 128,
                "batch_size": 2,
            },
        },
        seed=52,
        model={
            "id": "unit-model",
            "revision": "revision-a",
            "weight_sha256": "weight-sha-a",
            "pretrained": False,
        },
        dataset={
            "id": "unit-data",
            "split": "test",
            "sample_ids_sha256": "sample-sha-a",
            "tokenized_inputs_sha256": "tokenized-sha-a",
            "meaningful_lm_evidence": False,
        },
        environment={
            "actual_device": "cpu",
            "activation_dtype": "float32",
            "batch_size": 2,
        },
        sample_count=4,
        metrics=_metrics(tau),
        elapsed_seconds=0.1,
    )
    if raw_source is not None:
        record["_raw_source"] = raw_source
    if raw_line is not None:
        record["_raw_line"] = raw_line
    return record


def _failure_record(
    run_id: str,
    *,
    status: str = "failure",
    tau: float = 0.05,
    raw_line: int = 9,
) -> dict[str, Any]:
    common = {
        "run_id": run_id,
        "status": status,
        "config": {
            "mode": "topk_preserving",
            "tau_max": tau,
            "alpha": 0.95,
            "preserve_top_k": 16,
            "stage": "full_evaluation",
            "evaluation": {
                "sequence_length": 128,
                "batch_size": 2,
            },
        },
        "seed": 52,
        "model": {
            "id": "unit-model",
            "revision": "revision-a",
            "weight_sha256": "weight-sha-a",
            "pretrained": False,
        },
        "dataset": {
            "id": "unit-data",
            "split": "test",
            "sample_ids_sha256": "sample-sha-a",
            "tokenized_inputs_sha256": "tokenized-sha-a",
            "meaningful_lm_evidence": False,
        },
        "environment": {
            "actual_device": "cpu",
            "activation_dtype": "float32",
            "batch_size": 2,
        },
        "sample_count": 0,
        "metrics": {},
        "elapsed_seconds": 0.1,
    }
    if status == "failure":
        record = build_run_record(
            **common,
            error={"type": "RuntimeError", "message": "intentional failure"},
            stage="full_evaluation",
            last_completed_sample_id=None,
            partial_metrics_available=False,
        )
    else:
        record = build_run_record(
            **common,
            reason={
                "code": "not_selected",
                "message": "not selected after calibration",
            },
        )
    record["_raw_source"] = "results/raw/attempt/eval.jsonl"
    record["_raw_line"] = raw_line
    return record


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("model_revision", "revision-b"),
        ("model_weight_sha256", "weight-sha-b"),
        ("model_config_sha256", "config-sha-b"),
        ("model_tokenizer_sha256", "tokenizer-sha-b"),
        ("dataset_split", "validation"),
        ("device", "mps"),
        ("dtype", "bfloat16"),
        ("sequence_length", 256),
        ("batch_size", 4),
        ("generation_tokens", 8),
        ("sample_count", 8),
        ("tokenized_inputs_sha256", "tokenized-sha-b"),
        ("sample_ids_sha256", "sample-sha-b"),
    ],
)
def test_cohort_key_separates_every_fairness_dimension(
    field: str, changed: object
) -> None:
    baseline = {
        "model_id": "model",
        "model_revision": "revision-a",
        "model_weight_sha256": "weight-sha-a",
        "model_config_sha256": "config-sha-a",
        "model_tokenizer_sha256": "tokenizer-sha-a",
        "dataset_id": "data",
        "dataset_split": "test",
        "seed": 52,
        "stage": "full",
        "device": "cpu",
        "dtype": "float32",
        "sequence_length": 128,
        "batch_size": 2,
        "generation_tokens": 4,
        "sample_count": 4,
        "tokenized_inputs_sha256": "tokenized-sha-a",
        "sample_ids_sha256": "sample-sha-a",
    }
    candidate = copy.deepcopy(baseline)
    candidate[field] = changed

    assert _cohort_key(candidate) != _cohort_key(baseline)


def test_rows_preserve_fairness_and_direct_raw_provenance() -> None:
    row = _rows([_success_record("traceable")])[0]

    assert row["sequence_length"] == 128
    assert row["batch_size"] == 2
    assert row["tokenized_inputs_sha256"] == "tokenized-sha-a"
    assert row["raw_source"] == "results/raw/attempt/eval.jsonl"
    assert row["raw_line"] == 1


def test_rows_extract_qwen_hashes_from_checkpoint_manifest_and_cache() -> None:
    record = _success_record("nested-provenance")
    record["model"].pop("revision")
    record["model"].pop("weight_sha256")
    record["model"]["checkpoint_manifest"] = {
        "upstream_revision": "qwen-revision",
        "weights_sha256": "qwen-weight-sha",
        "config_sha256": "qwen-config-sha",
    }
    record["dataset"].pop("tokenized_inputs_sha256")
    record["dataset"]["cache_content_sha256"] = "cache-content-sha"

    row = _rows([record])[0]

    assert row["model_revision"] == "qwen-revision"
    assert row["model_weight_sha256"] == "qwen-weight-sha"
    assert row["model_config_sha256"] == "qwen-config-sha"
    assert row["tokenized_inputs_sha256"] == "cache-content-sha"


def test_report_figures_render_successes_as_unconnected_scatter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots: list[tuple[int, int]] = []

    def inspect_figure(self: Figure, *_args: object, **_kwargs: object) -> None:
        line_count = sum(len(axis.lines) for axis in self.axes)
        collection_count = sum(len(axis.collections) for axis in self.axes)
        snapshots.append((line_count, collection_count))

    monkeypatch.setattr(Figure, "savefig", inspect_figure)
    records = [
        _success_record("low", tau=0.01, raw_line=1),
        _failure_record("failed-middle", tau=0.05, raw_line=2),
        _success_record("high", tau=0.1, raw_line=3),
    ]

    output = build_report(
        records=records,
        output_root=tmp_path,
        experiment_status="unit_test",
        pretrained_status="not_run",
    )
    report = output.report_markdown.read_text(encoding="utf-8")

    assert snapshots == [(0, 2), (0, 2)]
    assert "不连线的散点" in report


def test_report_tables_include_raw_locations_and_failure_parameters(
    tmp_path: Path,
) -> None:
    exact = _success_record("exact", mode="exact", tau=0.0, raw_line=4)
    approximate = _success_record("approx", raw_line=5)
    failure = _failure_record("failed", raw_line=6)
    output = build_report(
        records=[exact, approximate, failure],
        output_root=tmp_path,
        experiment_status="unit_test",
        pretrained_status="not_run",
    )
    report = output.report_markdown.read_text(encoding="utf-8")

    assert "weight SHA-256" in report
    assert "tokenized SHA-256" in report
    assert "sample IDs SHA-256" in report
    assert "weight-sha-a" in report
    assert "tokenized-sha-a" in report
    assert report.count("raw source") >= 6
    assert report.count("raw line") >= 6
    assert "results/raw/attempt/eval.jsonl" in report
    assert "| failed | failure | topk_preserving | 0.05 | 0.95 | 16 |" in report


def test_missing_direct_raw_location_is_explicitly_not_available(
    tmp_path: Path,
) -> None:
    output = build_report(
        records=[
            _success_record(
                "unlocated",
                mode="exact",
                tau=0.0,
                raw_source=None,
                raw_line=None,
            )
        ],
        output_root=tmp_path,
        experiment_status="unit_test",
        pretrained_status="not_run",
    )
    report = output.report_markdown.read_text(encoding="utf-8")

    assert "| unlocated | not_available | not_available |" in report
