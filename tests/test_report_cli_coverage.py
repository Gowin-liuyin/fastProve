from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastprove.evaluation.artifacts import build_run_record
from scripts import build_report as report_cli


FOUR_MODES = (
    "plaintext",
    "exact",
    "topk_preserving",
    "free_bounded",
)


def _comparison_metrics() -> dict:
    return {
        "plaintext": {
            "negative_log_likelihood": 1.0,
            "perplexity": 2.0,
            "next_token_top1_accuracy": 0.5,
            "next_token_top5_accuracy": 0.9,
        },
        "obfuscated": {
            "negative_log_likelihood": 1.0,
            "perplexity": 2.0,
            "next_token_top1_accuracy": 0.5,
            "next_token_top5_accuracy": 0.9,
        },
        "agreement": {"next_token_top1_agreement": 1.0},
        "degradation": {
            "nll_absolute_increase": 0.0,
            "perplexity_absolute_increase": 0.0,
            "perplexity_relative_increase": 0.0,
            "top1_absolute_drop": 0.0,
            "top1_relative_drop": 0.0,
            "top5_absolute_drop": 0.0,
            "top5_relative_drop": 0.0,
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
            "conversion_time_seconds": 0.0,
            "prefill_latency_seconds": 0.0,
            "decode_tpot_seconds": 0.0,
            "tokens_per_second": 0.0,
            "peak_memory_bytes": 0,
            "kv_cache_memory_bytes": 0,
        },
    }


def _success(
    run_id: str,
    *,
    mode: str,
    experiment_name: str | None = None,
    stage: str | None = None,
    expected_run_ids: list[str] | None = None,
    pretrained: bool = False,
) -> dict:
    config: dict[str, object] = {"mode": mode}
    if experiment_name is not None:
        config["experiment_name"] = experiment_name
    if stage is not None:
        config["stage"] = stage
    if expected_run_ids is not None:
        config["expected_run_ids"] = expected_run_ids
    metrics = _comparison_metrics()
    if mode == "exact":
        metrics["exact_gate"] = {"passed": True, "reasons": []}
    return build_run_record(
        run_id=run_id,
        status="success",
        config=config,
        seed=7,
        model={"id": "model", "pretrained": pretrained},
        dataset={
            "id": "dataset",
            "meaningful_lm_evidence": pretrained,
        },
        environment={"device": "cpu"},
        sample_count=1,
        metrics=metrics,
        elapsed_seconds=0.01,
    )


def test_load_records_adds_in_memory_source_after_schema_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw = tmp_path / "attempt.jsonl"
    record = _success("one", mode="exact")
    original = "\n" + json.dumps(record, allow_nan=False) + "\n"
    raw.write_text(original, encoding="utf-8")
    validated_records: list[dict] = []

    def strict_validate(candidate: dict) -> None:
        assert "_raw_source" not in candidate
        assert "_raw_line" not in candidate
        validated_records.append(dict(candidate))

    monkeypatch.setattr(report_cli, "validate_run_record", strict_validate)

    loaded = report_cli._load_records([raw])

    assert validated_records == [record]
    assert loaded[0]["_raw_source"] == str(raw.resolve())
    assert loaded[0]["_raw_line"] == 2
    assert raw.read_text(encoding="utf-8") == original


def test_full_sweep_requires_manifest_success_and_all_four_modes() -> None:
    expected = ["plain", "exact", "topk", "free"]
    complete = [
        _success(
            run_id,
            mode=mode,
            experiment_name="pretrained-softmax-sweep",
            stage="full_sweep",
            expected_run_ids=expected,
            pretrained=True,
        )
        for run_id, mode in zip(expected, FOUR_MODES)
    ]

    assert report_cli._derived_status(complete) == "completed"
    assert report_cli._derived_pretrained_status(complete) == "completed"

    missing = complete[:-1]
    assert report_cli._derived_status(missing) == "partial_or_failed"
    assert (
        report_cli._derived_pretrained_status(missing)
        == "partial_or_failed"
    )

    exact_gate_missing = [
        {
            **record,
            "metrics": {
                key: value
                for key, value in record["metrics"].items()
                if key != "exact_gate"
            },
        }
        if record["config"]["mode"] == "exact"
        else record
        for record in complete
    ]
    assert (
        report_cli._derived_status(exact_gate_missing)
        == "partial_or_failed"
    )

    incomplete_manifest_declaration = [
        {
            **record,
            "config": {
                key: value
                for key, value in record["config"].items()
                if key != "expected_run_ids"
            },
        }
        if record["run_id"] == "free"
        else record
        for record in complete
    ]
    assert (
        report_cli._derived_status(incomplete_manifest_declaration)
        == "partial_or_failed"
    )

    no_free_mode = [
        {**record, "config": {**record["config"], "mode": "topk_preserving"}}
        if record["run_id"] == "free"
        else record
        for record in complete
    ]
    assert report_cli._derived_status(no_free_mode) == "partial_or_failed"
    assert (
        report_cli._derived_pretrained_status(no_free_mode)
        == "partial_or_failed"
    )

    failed = [
        {
            **record,
            "status": "failure",
            "metrics": {},
            "error": {"type": "RuntimeError", "message": "boom"},
        }
        if record["run_id"] == "free"
        else record
        for record in complete
    ]
    assert report_cli._derived_status(failed) == "partial_or_failed"
    assert (
        report_cli._derived_pretrained_status(failed)
        == "partial_or_failed"
    )


def test_selected_candidate_run_is_not_labeled_as_full_72_point_completion() -> None:
    expected = ["plain", "exact", "topk", "free"]
    selected = [
        _success(
            run_id,
            mode=mode,
            experiment_name="pretrained-softmax-sweep",
            stage="full_sweep",
            expected_run_ids=expected,
            pretrained=True,
        )
        for run_id, mode in zip(expected, FOUR_MODES)
    ]
    selected = [
        {
            **record,
            "config": {
                **record["config"],
                "selected_spec_ids_file": "/tmp/candidate_manifest.json",
            },
        }
        for record in selected
    ]

    assert report_cli._derived_status(selected) == "completed_selected_subset"
    assert (
        report_cli._derived_pretrained_status(selected)
        == "completed_selected_subset"
    )

def test_single_plaintext_point_is_not_mislabeled_as_completed_sweep() -> None:
    ambiguous = [_success("plain", mode="plaintext", pretrained=True)]
    named_sweep = [
        _success(
            "plain",
            mode="plaintext",
            experiment_name="pretrained-softmax-sweep",
            stage="full_sweep",
            pretrained=True,
        )
    ]

    assert report_cli._derived_status(ambiguous) == "partial_or_failed"
    assert (
        report_cli._derived_pretrained_status(ambiguous)
        == "partial_or_failed"
    )
    assert report_cli._derived_status(named_sweep) == "partial_or_failed"
    assert (
        report_cli._derived_pretrained_status(named_sweep)
        == "partial_or_failed"
    )


def test_one_explicit_non_sweep_stage_still_requires_manifest() -> None:
    stage = [
        _success(
            "exact-correctness",
            mode="exact",
            experiment_name="tiny-correctness",
            stage="exact_gate",
        )
    ]

    assert report_cli._derived_status(stage) == "partial_or_failed"


def test_invalid_manifest_cannot_complete_a_named_stage() -> None:
    malformed = _success(
        "exact-correctness",
        mode="exact",
        experiment_name="tiny-correctness",
        stage="exact_gate",
    )
    malformed["config"]["expected_run_ids"] = "exact-correctness"

    assert report_cli._derived_status([malformed]) == "partial_or_failed"


def test_cli_status_override_cannot_elevate_missing_evidence(
    tmp_path: Path,
    capsys,
) -> None:
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        json.dumps(
            _success("plain", mode="plaintext", pretrained=True),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as raised:
        report_cli.main(
            [
                "--raw",
                str(raw),
                "--output",
                str(tmp_path / "derived"),
                "--experiment-status",
                "completed",
                "--pretrained-status",
                "completed",
            ]
        )

    assert raised.value.code == 2
    assert "cannot elevate" in capsys.readouterr().err
