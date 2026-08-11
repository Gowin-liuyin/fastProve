from __future__ import annotations

from typing import Any

from fastprove.evaluation.selection import select_sweep_candidates
from tests.test_report_integrity import _success_record


def _calibration_records() -> list[dict[str, Any]]:
    records = [
        _success_record("plaintext", mode="plaintext", tau=0.0),
        _success_record("exact", mode="exact", tau=0.0),
        _success_record("topk-zero", mode="topk_preserving", tau=0.0),
        _success_record("topk-noisy", mode="topk_preserving", tau=0.1),
        _success_record("free-zero", mode="free_bounded", tau=0.0),
        _success_record("free-noisy", mode="free_bounded", tau=0.1),
    ]
    for record in records:
        record["config"]["expected_run_ids"] = [item["run_id"] for item in records]
        record["metrics"]["exact_gate"] = {
            "passed": record["run_id"] == "exact",
            "reasons": [],
        }
    return records


def test_selector_keeps_baselines_endpoints_and_frontier() -> None:
    manifest = select_sweep_candidates(_calibration_records(), source="calibration.jsonl")

    assert manifest["status"] == "ready_for_full_evaluation"
    assert manifest["baseline_run_ids"] == ["plaintext", "exact"]
    assert set(("topk-zero", "topk-noisy", "free-zero", "free-noisy")).issubset(
        set(manifest["selected_run_ids"])
    )
    assert manifest["failure_run_ids"] == []
    assert manifest["source"] == "calibration.jsonl"
    assert set(manifest["unselected_run_ids"]) == set()


def test_selector_blocks_when_exact_gate_fails() -> None:
    records = _calibration_records()
    records[1]["metrics"]["exact_gate"] = {"passed": False, "reasons": ["drift"]}

    manifest = select_sweep_candidates(records)

    assert manifest["status"] == "blocked_exact_gate"


def test_selector_blocks_an_incomplete_declared_calibration() -> None:
    records = _calibration_records()
    expected = [item["run_id"] for item in records] + ["missing-point"]
    for record in records:
        record["config"]["expected_run_ids"] = expected

    manifest = select_sweep_candidates(records)

    assert manifest["status"] == "incomplete_calibration"
    assert manifest["missing_expected_run_ids"] == ["missing-point"]
