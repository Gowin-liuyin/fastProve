from __future__ import annotations

import copy
import math
from typing import Any

import pytest

from fastprove.evaluation.artifacts import (
    build_run_record,
    validate_run_record,
)


def _core_metrics() -> dict[str, Any]:
    return {
        "plaintext": {
            "negative_log_likelihood": 1.0,
            "perplexity": math.e,
            "next_token_top1_accuracy": 0.5,
            "next_token_top5_accuracy": 0.9,
        },
        "obfuscated": {
            "negative_log_likelihood": 1.01,
            "perplexity": math.e + 0.01,
            "next_token_top1_accuracy": 0.49,
            "next_token_top5_accuracy": 0.89,
        },
        "agreement": {"next_token_top1_agreement": 0.99},
        "degradation": {
            "nll_absolute_increase": 0.01,
            "perplexity_absolute_increase": 0.01,
            "perplexity_relative_increase": 0.01 / math.e,
            "top1_absolute_drop": 0.01,
            "top1_relative_drop": 0.02,
            "top5_absolute_drop": 0.01,
            "top5_relative_drop": 0.01 / 0.9,
        },
        "greedy": {
            "greedy_token_exact_match": 1.0,
            "greedy_sequence_exact_match": 1.0,
        },
        "softmax": {
            "kl_divergence": 0.0,
            "js_divergence": 0.0,
            "topk_overlap": 1.0,
            "rank_correlation": 1.0,
            "topk_changed_fraction": 0.0,
            "actual_noise_infinity_norm": 0.0,
            "zero_noise_query_fraction": 1.0,
            "attention_output_relative_l2_error": 0.0,
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


def _valid_record(**overrides: Any) -> dict[str, Any]:
    record = build_run_record(
        run_id="schema-valid",
        status="success",
        config={"mode": "exact"},
        seed=52,
        model={"id": "unit-model", "pretrained": False},
        dataset={
            "id": "unit-data",
            "meaningful_lm_evidence": False,
        },
        environment={"device": "cpu", "dtype": "float32"},
        sample_count=2,
        metrics=_core_metrics(),
        elapsed_seconds=0.1,
    )
    record.update(overrides)
    return record


@pytest.mark.parametrize(
    "field", ["config", "model", "dataset", "environment"]
)
def test_context_fields_must_be_mappings(field: str) -> None:
    record = _valid_record()
    record[field] = []

    with pytest.raises(ValueError, match=rf"{field} must be a mapping"):
        validate_run_record(record)


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("schema_version", 1, "schema_version"),
        ("schema_version", "fastprove.run.v999", "schema_version"),
        ("timestamp_utc", 1, "timestamp_utc"),
        ("timestamp_utc", "not-a-timestamp", "timestamp_utc"),
        ("timestamp_utc", "2026-07-31T12:00:00", "timestamp_utc"),
        (
            "timestamp_utc",
            "2026-07-31T12:00:00+08:00",
            "timestamp_utc",
        ),
        ("run_id", 1, "run_id"),
        ("run_id", "   ", "run_id"),
        ("status", 1, "status"),
        ("seed", True, "seed"),
        ("seed", 1.0, "seed"),
        ("sample_count", True, "sample_count"),
        ("sample_count", 1.0, "sample_count"),
        ("sample_count", -1, "sample_count"),
        ("elapsed_seconds", True, "elapsed_seconds"),
        ("elapsed_seconds", "0.1", "elapsed_seconds"),
        ("elapsed_seconds", float("nan"), "elapsed_seconds"),
        ("elapsed_seconds", float("inf"), "elapsed_seconds"),
        ("elapsed_seconds", -0.1, "elapsed_seconds"),
    ],
)
def test_required_scalar_fields_have_strict_types_and_ranges(
    field: str, invalid_value: Any, message: str
) -> None:
    record = _valid_record()
    record[field] = invalid_value

    with pytest.raises(ValueError, match=message):
        validate_run_record(record)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("run_id", 123),
        ("seed", True),
        ("sample_count", True),
        ("elapsed_seconds", True),
    ],
)
def test_builder_does_not_coerce_invalid_scalar_types(
    field: str, invalid_value: Any
) -> None:
    kwargs: dict[str, Any] = {
        "run_id": "strict-builder",
        "status": "success",
        "config": {"mode": "plaintext"},
        "seed": 7,
        "model": {"id": "unit"},
        "dataset": {"id": "unit"},
        "environment": {"device": "cpu"},
        "sample_count": 1,
        "metrics": _core_metrics(),
        "elapsed_seconds": 0.1,
    }
    kwargs[field] = invalid_value

    with pytest.raises(ValueError, match=field):
        build_run_record(**kwargs)


def test_success_rejects_arbitrary_nonempty_metric_payload() -> None:
    with pytest.raises(ValueError, match="core metrics"):
        build_run_record(
            run_id="garbage-evidence",
            status="success",
            config={"mode": "plaintext"},
            seed=7,
            model={"id": "pretrained", "pretrained": True},
            dataset={
                "id": "public-data",
                "meaningful_lm_evidence": True,
            },
            environment={"device": "cpu"},
            sample_count=1,
            metrics={"evidence_marker": 1.0},
            elapsed_seconds=0.1,
        )


def test_nonstandard_caption_scope_is_comparative_not_lm_evidence() -> None:
    record = build_run_record(
        run_id="caption-limited",
        status="success",
        config={
            "mode": "plaintext",
            "stage": "pretrained_accuracy",
            "evaluation_scope": "caption_only_nonstandard_lm_candidate",
            "evidence": "comparative_evaluation",
        },
        seed=7,
        model={
            "id": "local-qwen2",
            "pretrained": True,
            "evidence_scope": "caption_only_nonstandard_lm_candidate",
        },
        dataset={
            "id": "flickr30k-captions",
            "evaluation_scope": "caption_only_nonstandard_lm_candidate",
            "meaningful_lm_evidence": False,
        },
        environment={"device": "mps", "dtype": "bfloat16"},
        sample_count=1,
        metrics=_core_metrics(),
        elapsed_seconds=0.1,
    )

    validate_run_record(record)


def test_success_rejects_missing_core_metric_for_comparison() -> None:
    metrics = _core_metrics()
    del metrics["greedy"]["greedy_sequence_exact_match"]

    with pytest.raises(ValueError, match="greedy.greedy_sequence_exact_match"):
        build_run_record(
            run_id="missing-core",
            status="success",
            config={"mode": "free_bounded"},
            seed=7,
            model={"id": "unit"},
            dataset={"id": "unit"},
            environment={"device": "cpu"},
            sample_count=1,
            metrics=metrics,
            elapsed_seconds=0.1,
        )


def test_success_rejects_unknown_mode() -> None:
    record = _valid_record()
    record["config"]["mode"] = "not-a-mode"

    with pytest.raises(ValueError, match="config.mode"):
        validate_run_record(record)


def test_tiny_prototype_success_requires_correctness_evidence() -> None:
    metrics = _core_metrics()

    with pytest.raises(ValueError, match="layer"):
        build_run_record(
            run_id="prototype-without-layer-evidence",
            status="success",
            config={
                "mode": "exact",
                "stage": "tiny_reference",
                "prototype": {"kind": "random_tiny"},
                "evidence": {"kind": "prototype_correctness"},
            },
            seed=7,
            model={"id": "unit", "pretrained": False},
            dataset={"id": "unit", "meaningful_lm_evidence": False},
            environment={"device": "cpu"},
            sample_count=1,
            metrics=metrics,
            elapsed_seconds=0.1,
        )

    metrics["layer"] = {
        "logits": {"max_absolute_error": 0.0},
        "qk_scores": {"max_absolute_error": 0.0},
        "exact_softmax_probabilities": {"max_absolute_error": 0.0},
    }
    metrics["nan_inf_count"] = 0
    record = build_run_record(
        run_id="prototype-with-layer-evidence",
        status="success",
        config={
            "mode": "exact",
            "stage": "tiny_reference",
            "prototype": {"kind": "random_tiny"},
            "evidence": {"kind": "prototype_correctness"},
        },
        seed=7,
        model={"id": "unit", "pretrained": False},
        dataset={"id": "unit", "meaningful_lm_evidence": False},
        environment={"device": "cpu"},
        sample_count=1,
        metrics=metrics,
        elapsed_seconds=0.1,
    )

    validate_run_record(record)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stage", 1),
        ("prototype", []),
        ("evidence", {"kind": ""}),
        ("evidence", {"kind": "unsupported"}),
    ],
)
def test_success_rejects_invalid_config_discriminators(
    field: str, value: Any
) -> None:
    record = _valid_record()
    record["config"][field] = value

    with pytest.raises(ValueError, match=rf"config.{field}"):
        validate_run_record(record)


def test_failure_and_skipped_records_remain_compatible() -> None:
    failure = build_run_record(
        run_id="compatible-failure",
        status="failure",
        config={
            "mode": "exact",
            "stage": "tiny_correctness",
            "prototype": {"kind": "random_tiny"},
        },
        seed=52,
        model={"id": "unit"},
        dataset={"id": "unit"},
        environment={"device": "cpu"},
        sample_count=0,
        metrics={},
        elapsed_seconds=0.01,
        error={"type": "RuntimeError", "message": "intentional"},
        stage="unit_test",
        last_completed_sample_id=None,
        partial_metrics_available=False,
    )
    skipped = build_run_record(
        run_id="compatible-skipped",
        status="skipped",
        config={"mode": "free_bounded"},
        seed=52,
        model={"id": "unit"},
        dataset={"id": "unit"},
        environment={"device": "cpu"},
        sample_count=0,
        metrics={},
        elapsed_seconds=0.0,
        reason={"code": "not_selected", "message": "calibration filter"},
    )

    validate_run_record(failure)
    validate_run_record(skipped)


def test_nested_nonfinite_metric_is_rejected() -> None:
    record = _valid_record()
    record["metrics"] = copy.deepcopy(record["metrics"])
    record["metrics"]["softmax"]["kl_divergence"] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        validate_run_record(record)
