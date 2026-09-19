"""``fastprove.attack.v1`` cost records: schema, censoring and amortisation.

These tests pin the rules that make the work-factor ledger auditable: a
right-censored run must not present its budget ceiling as a measured cost, an
unexecuted attack must say why, and the amortisation formula must follow
docs/work_factor_security.md section 3 at all three required scales.
"""

from __future__ import annotations

import json

import pytest

from fastprove.evaluation.attack_records import (
    ATTACK_RECORD_SCHEMA,
    AttackCost,
    append_attack_jsonl,
    build_attack_record,
    cost_dict,
    validate_attack_record,
)


def _cost(**overrides) -> AttackCost:
    values = dict(
        estimated_flops=1.0e6,
        wall_clock_seconds=0.5,
        peak_memory_bytes=4096,
        active_queries=40,
        passive_observations=0,
        training_data_records=0,
        setup_seconds=0.5,
        per_target_seconds=1e-4,
    )
    values.update(overrides)
    return AttackCost(**values)


def _record(**overrides) -> dict:
    values = dict(
        attack_id="test-attack",
        tier=0,
        claim="a test claim",
        coa_view="cloud_coa",
        requires={"requires_knowledge_of_the_mixing_basis": False},
        recovery={"metric": "heldout_rel_error", "value": 1e-15},
        status="executed",
        config={"d": 32, "r": 8},
        environment={"torch_version": "test", "platform": "test"},
        cost=cost_dict(
            estimated_flops=1.0e6,
            wall_clock_seconds=0.5,
            peak_memory_bytes=4096,
            active_queries=40,
            setup_seconds=0.5,
            per_target_seconds=1e-4,
            model="unit test",
        ),
    )
    values.update(overrides)
    return build_attack_record(**values)


def test_executed_record_round_trips_and_validates() -> None:
    record = _record()
    assert record["schema"] == ATTACK_RECORD_SCHEMA
    assert record["status"] == "executed"
    assert record["right_censored"] is False
    assert json.loads(json.dumps(record)) == record
    validate_attack_record(record)


def test_missing_required_key_is_rejected() -> None:
    record = _record()
    del record["tier"]
    with pytest.raises(ValueError, match="missing keys"):
        validate_attack_record(record)


def test_unknown_coa_view_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown COA view"):
        _record(coa_view="side_channel")


def test_requires_entries_must_be_boolean_premises() -> None:
    record = _record()
    record["requires"] = {"knowledge_of_the_basis": True}
    with pytest.raises(ValueError, match="requires_"):
        validate_attack_record(record)


def test_executed_record_requires_a_cost_vector() -> None:
    record = _record()
    del record["cost"]
    with pytest.raises(ValueError, match="cost vector"):
        validate_attack_record(record)


def test_right_censored_record_must_store_budget_separately() -> None:
    with pytest.raises(ValueError, match="budget limit separately"):
        _record(
            status="right_censored",
            recovery={"metric": "token_top1", "value": None},
        )
    record = _record(
        status="right_censored",
        recovery={"metric": "token_top1", "value": None},
        censoring={"budget_seconds": 3600.0, "target": "token_top1=0.5"},
    )
    validate_attack_record(record)
    assert record["right_censored"] is True


def test_right_censored_record_rejects_measured_value() -> None:
    with pytest.raises(ValueError, match="must not report a measured"):
        _record(
            status="right_censored",
            recovery={"metric": "token_top1", "value": 0.2},
            censoring={"budget_seconds": 3600.0},
        )


def test_not_executed_record_requires_reason_and_no_cost() -> None:
    with pytest.raises(ValueError, match="explicit reason"):
        build_attack_record(
            attack_id="blocked",
            tier=0,
            claim="blocked attack",
            coa_view="cloud_coa",
            requires={"requires_public_base_checkpoint": True},
            recovery={"metric": "token_top1", "value": None},
            status="not_executed",
            config={},
            environment={"torch_version": "test", "platform": "test"},
        )
    record = build_attack_record(
        attack_id="blocked",
        tier=0,
        claim="blocked attack",
        coa_view="cloud_coa",
        requires={"requires_public_base_checkpoint": True},
        recovery={"metric": "token_top1", "value": None},
        status="not_executed",
        blocked_reason="checkpoint absent",
        config={},
        environment={"torch_version": "test", "platform": "test"},
    )
    validate_attack_record(record)


def test_not_executed_record_rejects_cost_numbers() -> None:
    with pytest.raises(ValueError, match="must not carry measured cost"):
        _record(
            status="not_executed",
            recovery={"metric": "m", "value": None},
            blocked_reason="checkpoint absent",
        )


def test_amortised_seconds_matches_three_required_scales() -> None:
    cost = _cost(setup_seconds=100.0, per_target_seconds=1.0)
    assert cost.amortised_seconds(2, 1) == pytest.approx(201.0)
    assert cost.amortised_seconds(2, 100) == pytest.approx(3.0)
    assert cost.amortised_seconds(2, 10_000) == pytest.approx(1.02)
    with pytest.raises(ValueError):
        cost.amortised_seconds(0, 10)


def test_cost_rejects_negative_and_non_finite_values() -> None:
    with pytest.raises(ValueError, match="estimated_flops"):
        _cost(estimated_flops=-1.0)
    with pytest.raises(ValueError, match="wall_clock_seconds"):
        _cost(wall_clock_seconds=float("nan"))
    with pytest.raises(ValueError, match="active_queries"):
        _cost(active_queries=-3)
    with pytest.raises(ValueError, match="active_queries"):
        _cost(active_queries=True)


def test_append_attack_jsonl_appends_validated_records(tmp_path) -> None:
    path = tmp_path / "attack_costs.jsonl"
    append_attack_jsonl(path, _record())
    append_attack_jsonl(path, _record(attack_id="second"))
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["attack_id"] == "second"
    with pytest.raises(ValueError):
        append_attack_jsonl(path, {"schema": ATTACK_RECORD_SCHEMA})
