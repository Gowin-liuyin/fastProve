"""Attack cost records (``fastprove.attack.v1``).

The work-factor methodology of ``docs/work_factor_security.md`` section 2 is a
cost vector ``(F, T, M_peak, Q, O, D_train)`` plus one-time setup and
per-target costs, with a primary reference scalar. This module provides the
parallel record schema for attacks; it deliberately does not extend
``artifacts.py`` (``fastprove.run.v1``), whose mode whitelist and required
metrics describe inference runs, not attacks.

Rules encoded here:

* the cost vector is always recorded in full, never collapsed to one number;
* FLOPs may be analytic estimates (section 4.1 wording: an upper estimate, not
  an optimal-complexity lower bound);
* a record that did not reach its target inside a budget is
  ``status="right_censored"`` with the budget stored separately: the budget
  ceiling must never be presented as a measured cost (section 6);
* a configured attack that could not run must be recorded as
  ``status="not_executed"`` with an explicit reason; silent omission is
  forbidden (AGENTS.md);
* the observation view is declared through ``coa_views.COAViewContract`` and
  its input manifest is embedded in the record.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from ..coa_views import VALID_VIEWS, build_coa_view_contract

ATTACK_RECORD_SCHEMA = "fastprove.attack.v1"

VALID_TIERS = (0, 1, 2)
VALID_STATUS = ("executed", "right_censored", "not_executed")

_REQUIRED_KEYS = (
    "schema",
    "attack_id",
    "tier",
    "claim",
    "coa_view",
    "view_manifest",
    "requires",
    "recovery",
    "status",
    "config",
    "environment",
)


@dataclass(frozen=True)
class AttackCost:
    """Cost vector of ``docs/work_factor_security.md`` section 2.

    ``setup_seconds`` is the once-per-key-epoch preparation cost and
    ``per_target_seconds`` the per-target decode cost.
    """

    estimated_flops: float
    wall_clock_seconds: float
    peak_memory_bytes: int
    active_queries: int  # Q
    passive_observations: int  # O
    training_data_records: int  # D_train; zero for algebraic attacks
    setup_seconds: float  # C_setup
    per_target_seconds: float  # C_target

    def __post_init__(self) -> None:
        for name in (
            "estimated_flops",
            "wall_clock_seconds",
            "setup_seconds",
            "per_target_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("%s must be finite and non-negative" % name)
        for name in (
            "peak_memory_bytes",
            "active_queries",
            "passive_observations",
            "training_data_records",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("%s must be a non-negative integer" % name)

    def amortised_seconds(self, key_epochs: int, targets: int) -> float:
        """Return ``E * C_setup / m + C_target`` (section 3)."""

        if key_epochs < 1 or targets < 1:
            raise ValueError("key_epochs and targets must be at least one")
        return key_epochs * self.setup_seconds / targets + self.per_target_seconds


def cost_dict(
    *,
    estimated_flops: float,
    wall_clock_seconds: float,
    peak_memory_bytes: int,
    active_queries: int,
    passive_observations: int = 0,
    training_data_records: int = 0,
    setup_seconds: float,
    per_target_seconds: float,
    model: str,
) -> dict[str, Any]:
    """Return a JSON-ready cost dict with the analytic model documented."""

    cost = AttackCost(
        estimated_flops=estimated_flops,
        wall_clock_seconds=wall_clock_seconds,
        peak_memory_bytes=peak_memory_bytes,
        active_queries=active_queries,
        passive_observations=passive_observations,
        training_data_records=training_data_records,
        setup_seconds=setup_seconds,
        per_target_seconds=per_target_seconds,
    )
    return {**asdict(cost), "model": model}


def build_attack_record(
    *,
    attack_id: str,
    tier: int,
    claim: str,
    coa_view: str,
    requires: Mapping[str, bool],
    recovery: Mapping[str, Any],
    status: str,
    config: Mapping[str, Any],
    environment: Mapping[str, Any],
    cost: Mapping[str, Any] | None = None,
    censoring: Mapping[str, Any] | None = None,
    blocked_reason: str | None = None,
    key_epoch_count: int = 1,
    observation_budget_per_epoch: int | None = None,
) -> dict[str, Any]:
    """Build one validated ``fastprove.attack.v1`` record.

    ``requires`` values are ``requires_*`` boolean premise bits.
    ``observation_budget_per_epoch`` defaults to the recorded query plus
    observation counts when a cost vector is present.
    """

    if status not in VALID_STATUS:
        raise ValueError("status must be one of %s" % (VALID_STATUS,))
    budget = observation_budget_per_epoch
    if budget is None:
        if cost is None:
            budget = 1
        else:
            budget = max(
                1,
                int(cost.get("active_queries", 0))
                + int(cost.get("passive_observations", 0)),
            )
    contract = build_coa_view_contract(
        coa_view,
        key_epoch_count=key_epoch_count,
        observation_budget_per_epoch=int(budget),
    )
    record: dict[str, Any] = {
        "schema": ATTACK_RECORD_SCHEMA,
        "attack_id": str(attack_id),
        "tier": int(tier),
        "claim": str(claim),
        "coa_view": str(coa_view),
        "view_manifest": contract.input_manifest(),
        "requires": {str(key): bool(value) for key, value in requires.items()},
        "recovery": dict(recovery),
        "status": status,
        "right_censored": status == "right_censored",
        "config": dict(config),
        "environment": dict(environment),
    }
    if cost is not None:
        record["cost"] = dict(cost)
    if censoring is not None:
        record["censoring"] = dict(censoring)
    if blocked_reason is not None:
        record["blocked_reason"] = str(blocked_reason)
    validate_attack_record(record)
    return record


def validate_attack_record(record: Mapping[str, Any]) -> None:
    """Enforce required keys, finiteness and the status/budget rules."""

    missing = [key for key in _REQUIRED_KEYS if key not in record]
    if missing:
        raise ValueError("attack record missing keys: %s" % ", ".join(missing))
    if record["schema"] != ATTACK_RECORD_SCHEMA:
        raise ValueError("unexpected schema %r" % record["schema"])
    if record["tier"] not in VALID_TIERS:
        raise ValueError("tier must be one of %s" % (VALID_TIERS,))
    if record["status"] not in VALID_STATUS:
        raise ValueError("status must be one of %s" % (VALID_STATUS,))
    if record["coa_view"] not in VALID_VIEWS:
        raise ValueError("coa_view must be one of %s" % (VALID_VIEWS,))
    if record["view_manifest"]["view"] != record["coa_view"]:
        raise ValueError("view_manifest does not match coa_view")
    requires = record["requires"]
    if not isinstance(requires, Mapping) or not requires:
        raise ValueError("requires must be a non-empty mapping")
    for key, value in requires.items():
        if not key.startswith("requires_") or not isinstance(value, bool):
            raise ValueError(
                "requires entries must be requires_* booleans; got %r=%r"
                % (key, value)
            )
    recovery = record["recovery"]
    if not isinstance(recovery, Mapping) or "metric" not in recovery:
        raise ValueError("recovery must be a mapping with a metric key")
    cost = record.get("cost")
    if record["status"] == "executed":
        if not isinstance(record.get("recovery", {}).get("value"), (int, float)):
            raise ValueError("executed records need a numeric recovery value")
        if not isinstance(recovery["value"], (int, float)) or not math.isfinite(
            float(recovery["value"])
        ):
            raise ValueError("recovery value must be finite")
        if cost is None:
            raise ValueError("executed records require a cost vector")
    if record["status"] == "right_censored":
        if cost is None:
            raise ValueError("right-censored records require the budget vector")
        censoring = record.get("censoring")
        if not isinstance(censoring, Mapping) or "budget_seconds" not in censoring:
            raise ValueError(
                "right-censored records must store the budget limit separately "
                "instead of presenting it as a measured cost"
            )
        budget = censoring["budget_seconds"]
        if not isinstance(budget, (int, float)) or not math.isfinite(float(budget)):
            raise ValueError("censoring budget_seconds must be finite")
        if recovery.get("value") is not None:
            raise ValueError(
                "right-censored records must not report a measured recovery "
                "value (the budget ceiling is not a measurement)"
            )
    if record["status"] == "not_executed":
        if not record.get("blocked_reason"):
            raise ValueError("not_executed records require an explicit reason")
        if "cost" in record:
            raise ValueError(
                "not_executed records must not carry measured cost numbers"
            )
    if cost is not None:
        for key in (
            "estimated_flops",
            "wall_clock_seconds",
            "peak_memory_bytes",
            "active_queries",
            "passive_observations",
            "training_data_records",
            "setup_seconds",
            "per_target_seconds",
        ):
            if key not in cost:
                raise ValueError("cost vector missing %r" % key)
        for key in (
            "estimated_flops",
            "wall_clock_seconds",
            "setup_seconds",
            "per_target_seconds",
        ):
            if not math.isfinite(float(cost[key])) or float(cost[key]) < 0.0:
                raise ValueError("cost.%s must be finite and non-negative" % key)


def append_attack_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Validate then append one record to a JSONL file."""

    validate_attack_record(record)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
