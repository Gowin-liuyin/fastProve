from __future__ import annotations

import json
import csv
import copy
from pathlib import Path

import pytest

from fastprove.evaluation.artifacts import (
    append_jsonl_record,
    build_run_record,
    validate_run_record,
)
from fastprove.evaluation.report import _series_groups, build_report
from fastprove.evaluation.sweep import enumerate_sweep_specs


def test_required_sweep_enumerates_every_point_without_duplicates() -> None:
    specs = enumerate_sweep_specs(Path("configs/eval_sweep.yaml"))
    assert len(specs) == 72
    counts = {}
    for spec in specs:
        counts[spec.mode] = counts.get(spec.mode, 0) + 1
    assert counts == {
        "plaintext": 1,
        "exact": 1,
        "topk_preserving": 63,
        "free_bounded": 7,
    }
    assert len({spec.run_id for spec in specs}) == len(specs)
    topk = [spec for spec in specs if spec.mode == "topk_preserving"]
    assert {spec.tau_max for spec in topk} == {
        0.0,
        0.001,
        0.003,
        0.01,
        0.03,
        0.05,
        0.1,
    }
    assert {spec.alpha for spec in topk} == {0.5, 0.8, 0.95}
    assert {spec.preserve_top_k for spec in topk} == {4, 8, 16}


def _record(run_id: str, mode: str, tau: float) -> dict:
    return build_run_record(
        run_id=run_id,
        status="success",
        config={
            "mode": mode,
            "tau_max": tau,
            "tau_error": 0.1,
            "alpha": 0.8,
            "preserve_top_k": 4,
        },
        seed=52,
        model={
            "id": "unit-test-model",
            "path": None,
            "revision": "unit-test",
            "weight_sha256": None,
            "random_model_correctness_only": True,
        },
        dataset={
            "id": "unit-test-data",
            "split": "test",
            "sample_ids_sha256": "abc",
            "public": False,
        },
        environment={
            "python": "test",
            "torch": "test",
            "device": "cpu",
            "dtype": "float32",
            "source_sha256": "def",
        },
        sample_count=2,
        metrics={
            "plaintext": {
                "negative_log_likelihood": 1.0,
                "perplexity": 2.718,
                "next_token_top1_accuracy": 0.5,
                "next_token_top5_accuracy": 0.9,
            },
            "obfuscated": {
                "negative_log_likelihood": 1.0 + tau,
                "perplexity": 2.718 + tau,
                "next_token_top1_accuracy": 0.5 - tau,
                "next_token_top5_accuracy": 0.9 - tau,
            },
            "agreement": {"next_token_top1_agreement": 1.0 - tau},
            "degradation": {
                "nll_absolute_increase": tau,
                "perplexity_absolute_increase": tau,
                "perplexity_relative_increase": tau / 2.718,
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
        },
        elapsed_seconds=0.1,
    )


def test_run_record_schema_and_jsonl_preserve_success_and_failure(
    tmp_path: Path,
) -> None:
    success = _record("success-0", "exact", 0.0)
    validate_run_record(success)
    failure = build_run_record(
        run_id="failure-0",
        status="failure",
        config={"mode": "free_bounded", "tau_max": 0.1},
        seed=52,
        model={"id": "unit"},
        dataset={"id": "unit"},
        environment={"device": "cpu"},
        sample_count=0,
        metrics={},
        elapsed_seconds=0.01,
        error={"type": "RuntimeError", "message": "intentional unit failure"},
        stage="unit_test",
        last_completed_sample_id=None,
        partial_metrics_available=False,
    )
    validate_run_record(failure)
    output = tmp_path / "raw.jsonl"
    append_jsonl_record(output, success)
    append_jsonl_record(output, failure)
    parsed = [json.loads(line) for line in output.read_text().splitlines()]
    assert [item["status"] for item in parsed] == ["success", "failure"]
    assert parsed[1]["error"]["message"] == "intentional unit failure"


def test_jsonl_append_rejects_duplicate_run_ids(tmp_path: Path) -> None:
    output = tmp_path / "raw.jsonl"
    record = _record("duplicate", "exact", 0.0)
    append_jsonl_record(output, record)
    with pytest.raises(ValueError, match="duplicate run_id"):
        append_jsonl_record(output, record)
    assert len(output.read_text().splitlines()) == 1


def test_success_record_rejects_empty_metric_payload() -> None:
    with pytest.raises(ValueError, match="success record must include metrics"):
        build_run_record(
            run_id="empty-success",
            status="success",
            config={"mode": "exact"},
            seed=52,
            model={"id": "unit"},
            dataset={"id": "unit"},
            environment={"device": "cpu"},
            sample_count=1,
            metrics={},
            elapsed_seconds=0.01,
        )


def test_failure_record_requires_progress_context() -> None:
    with pytest.raises(ValueError, match="failure record must include stage"):
        build_run_record(
            run_id="incomplete-failure",
            status="failure",
            config={"mode": "exact"},
            seed=52,
            model={"id": "unit"},
            dataset={"id": "unit"},
            environment={"device": "cpu"},
            sample_count=0,
            metrics={},
            elapsed_seconds=0.01,
            error={"type": "RuntimeError", "message": "intentional"},
        )


def test_failure_record_rejects_coerced_progress_context_types() -> None:
    with pytest.raises(
        ValueError, match="failure record must include stage"
    ):
        build_run_record(
            run_id="typed-failure-stage",
            status="failure",
            config={"mode": "exact"},
            seed=52,
            model={"id": "unit"},
            dataset={"id": "unit"},
            environment={"device": "cpu"},
            sample_count=0,
            metrics={},
            elapsed_seconds=0.01,
            error={"type": "RuntimeError", "message": "intentional"},
            stage=123,  # type: ignore[arg-type]
            last_completed_sample_id=None,
            partial_metrics_available=False,
        )
    with pytest.raises(
        ValueError, match="partial_metrics_available"
    ):
        build_run_record(
            run_id="typed-failure-partial",
            status="failure",
            config={"mode": "exact"},
            seed=52,
            model={"id": "unit"},
            dataset={"id": "unit"},
            environment={"device": "cpu"},
            sample_count=0,
            metrics={},
            elapsed_seconds=0.01,
            error={"type": "RuntimeError", "message": "intentional"},
            stage="unit",
            last_completed_sample_id=None,
            partial_metrics_available="false",  # type: ignore[arg-type]
        )


def test_skipped_record_requires_structured_reason() -> None:
    with pytest.raises(ValueError, match="skipped record must include reason"):
        build_run_record(
            run_id="incomplete-skip",
            status="skipped",
            config={"mode": "free_bounded"},
            seed=52,
            model={"id": "unit"},
            dataset={"id": "unit"},
            environment={"device": "cpu"},
            sample_count=0,
            metrics={},
            elapsed_seconds=0.0,
        )


def test_report_builder_derives_csv_figures_and_pending_disclosure(
    tmp_path: Path,
) -> None:
    records = [
        _record("plain", "plaintext", 0.0),
        _record("exact", "exact", 0.0),
        _record("topk", "topk_preserving", 0.03),
        _record("free", "free_bounded", 0.05),
    ]
    outputs = build_report(
        records=records,
        output_root=tmp_path,
        experiment_status="unit_test_synthetic_records",
        pretrained_status="not_run",
    )
    assert outputs.summary_csv.exists()
    assert outputs.accuracy_figure.exists()
    assert outputs.softmax_figure.exists()
    assert outputs.report_markdown.exists()
    with outputs.summary_csv.open(newline="", encoding="utf-8") as handle:
        summary_row = next(csv.DictReader(handle))
    required_columns = {
        "plaintext_top5",
        "obfuscated_top5",
        "top5_absolute_drop",
        "top5_relative_drop",
        "attention_output_relative_l2_error",
        "zero_noise_query_fraction",
        "clean_boundary_margin_mean",
        "nan_inf_count",
        "logits_max_absolute_error",
        "qk_max_absolute_error",
        "conversion_time_seconds",
        "prefill_plaintext_mean_seconds",
        "prefill_obfuscated_mean_seconds",
        "decode_plaintext_tpot_seconds",
        "decode_obfuscated_tpot_seconds",
        "peak_memory_bytes",
        "kv_cache_plaintext_bytes",
        "kv_cache_obfuscated_bytes",
    }
    assert required_columns.issubset(summary_row)
    report = outputs.report_markdown.read_text()
    assert "威胁模型和安全限制" in report
    assert "unit_test_synthetic_records" in report
    assert "not_run" in report
    assert "random" in report.lower()


def test_report_is_traceable_and_reports_each_strategy_best_and_failures(
    tmp_path: Path,
) -> None:
    records = [
        _record("plain", "plaintext", 0.0),
        _record("exact", "exact", 0.0),
        _record("topk", "topk_preserving", 0.005),
        _record("free", "free_bounded", 0.006),
    ]
    for record in records:
        record["model"] = {
            "id": "pretrained-unit-model",
            "revision": "unit-revision",
            "weight_sha256": "model-sha",
            "pretrained": True,
        }
        record["dataset"] = {
            "id": "public-unit-data",
            "split": "test",
            "sample_ids_sha256": "sample-sha",
            "meaningful_lm_evidence": True,
        }
    records[1]["metrics"]["exact_gate"] = {"passed": True, "reasons": []}
    failure = build_run_record(
        run_id="failed-point",
        status="failure",
        config={"mode": "topk_preserving", "tau_max": 0.1},
        seed=52,
        model={"id": "pretrained-unit-model", "pretrained": True},
        dataset={"id": "public-unit-data", "meaningful_lm_evidence": True},
        environment={"device": "cpu"},
        sample_count=3,
        metrics={"partial": {"completed_samples": 3}},
        elapsed_seconds=0.2,
        error={"type": "RuntimeError", "message": "intentional failure"},
        stage="calibration",
        last_completed_sample_id="sample-2",
        partial_metrics_available=True,
    )
    skipped = build_run_record(
        run_id="skipped-point",
        status="skipped",
        config={"mode": "free_bounded", "tau_max": 0.1},
        seed=52,
        model={"id": "pretrained-unit-model", "pretrained": True},
        dataset={"id": "public-unit-data", "meaningful_lm_evidence": True},
        environment={"device": "cpu"},
        sample_count=0,
        metrics={},
        elapsed_seconds=0.0,
        reason={
            "code": "calibration_only",
            "message": "not selected for full evaluation",
        },
    )
    records.extend([failure, skipped])

    outputs = build_report(
        records=records,
        output_root=tmp_path,
        experiment_status="completed_with_recorded_failures",
        pretrained_status="completed",
        raw_sources=[Path("results/raw/unit/eval.jsonl")],
    )
    report = outputs.report_markdown.read_text(encoding="utf-8")

    assert "results/raw/unit/eval.jsonl" in report
    assert "明文 NLL" in report
    assert "明文 PPL" in report
    assert "绝对增量" in report
    assert "top-5 绝对下降" in report
    assert "greedy sequence" in report
    assert "decode obf tok/s" in report
    assert "topk_preserving：`topk`" in report
    assert "free_bounded：`free`" in report
    assert "failed-point" in report
    assert "intentional failure" in report
    assert "skipped-point" in report
    assert "calibration_only" in report
    assert "Pareto" in report


def test_report_rejects_invalid_success_record_instead_of_crashing(
    tmp_path: Path,
) -> None:
    invalid = copy.deepcopy(_record("invalid", "free_bounded", 0.01))
    invalid["metrics"] = {}
    with pytest.raises(ValueError, match="success record must include metrics"):
        build_report(
            records=[invalid],
            output_root=tmp_path,
            experiment_status="invalid",
            pretrained_status="not_run",
        )


def test_report_series_do_not_connect_different_topk_schedules() -> None:
    rows = [
        {
            "mode": "topk_preserving",
            "alpha": 0.5,
            "preserve_top_k": 4,
            "tau_max": 0.01,
        },
        {
            "mode": "topk_preserving",
            "alpha": 0.8,
            "preserve_top_k": 8,
            "tau_max": 0.01,
        },
        {
            "mode": "free_bounded",
            "alpha": None,
            "preserve_top_k": None,
            "tau_max": 0.01,
        },
    ]
    groups = _series_groups(rows)
    assert set(groups) == {
        "topk_preserving/alpha=0.5/k=4",
        "topk_preserving/alpha=0.8/k=8",
        "free_bounded",
    }


def test_report_series_separate_distinct_model_data_seed_cohorts() -> None:
    rows = [
        {
            "mode": "free_bounded",
            "tau_max": 0.01,
            "model_id": "model-a",
            "dataset_id": "data",
            "seed": 1,
            "stage": "full",
            "sample_ids_sha256": "hash-a",
        },
        {
            "mode": "free_bounded",
            "tau_max": 0.03,
            "model_id": "model-b",
            "dataset_id": "data",
            "seed": 1,
            "stage": "full",
            "sample_ids_sha256": "hash-a",
        },
    ]
    groups = _series_groups(rows)
    assert len(groups) == 2
