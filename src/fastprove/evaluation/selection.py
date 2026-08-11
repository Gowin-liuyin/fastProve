"""Deterministic candidate selection from a completed calibration JSONL.

The selector is deliberately conservative: it never deletes or rewrites raw
records, requires the plaintext/exact baselines, retains schedule endpoints,
and adds the non-dominated accuracy/perturbation frontier for every
approximate schedule.  The resulting manifest is an input to a later full
evaluation, not a result artifact itself.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .artifacts import validate_run_record


SELECTION_SCHEMA_VERSION = "fastprove.sweep_selection.v1"
REQUIRED_BASELINE_MODES = frozenset({"plaintext", "exact"})
APPROXIMATE_MODES = frozenset({"topk_preserving", "free_bounded"})


def _path(record: Mapping[str, Any], *parts: str) -> Any:
    value: Any = record
    for part in parts:
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _float(record: Mapping[str, Any], *parts: str) -> float | None:
    value = _path(record, *parts)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _cohort_key(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    model = record.get("model", {})
    dataset = record.get("dataset", {})
    environment = record.get("environment", {})
    config = record.get("config", {})
    evaluation = config.get("evaluation", {}) if isinstance(config, Mapping) else {}
    checkpoint = (
        model.get("checkpoint_manifest", {})
        if isinstance(model, Mapping)
        and isinstance(model.get("checkpoint_manifest"), Mapping)
        else {}
    )
    return (
        model.get("identifier", model.get("id")) if isinstance(model, Mapping) else None,
        model.get("revision", checkpoint.get("upstream_revision"))
        if isinstance(model, Mapping)
        else None,
        model.get("weight_sha256", checkpoint.get("weights_sha256"))
        if isinstance(model, Mapping)
        else None,
        model.get("config_sha256", checkpoint.get("config_sha256"))
        if isinstance(model, Mapping)
        else None,
        model.get("tokenizer_sha256", checkpoint.get("tokenizer_sha256"))
        if isinstance(model, Mapping)
        else None,
        dataset.get("identifier", dataset.get("id")) if isinstance(dataset, Mapping) else None,
        dataset.get("split") if isinstance(dataset, Mapping) else None,
        dataset.get("tokenized_inputs_sha256", dataset.get("cache_content_sha256"))
        if isinstance(dataset, Mapping)
        else None,
        dataset.get("sample_ids_sha256") if isinstance(dataset, Mapping) else None,
        record.get("seed"),
        config.get("stage") if isinstance(config, Mapping) else None,
        environment.get("actual_device", environment.get("device"))
        if isinstance(environment, Mapping)
        else None,
        environment.get("activation_dtype", environment.get("dtype"))
        if isinstance(environment, Mapping)
        else None,
        environment.get("checkpoint_compute_dtype")
        if isinstance(environment, Mapping)
        else None,
        evaluation.get("sequence_length") if isinstance(evaluation, Mapping) else None,
        evaluation.get("batch_size") if isinstance(evaluation, Mapping) else None,
        evaluation.get("generation_tokens") if isinstance(evaluation, Mapping) else None,
        record.get("sample_count"),
        config.get("base_config_sha256") if isinstance(config, Mapping) else None,
        config.get("sweep_config_sha256") if isinstance(config, Mapping) else None,
    )


def _schedule_key(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    config = record.get("config", {})
    mode = config.get("mode") if isinstance(config, Mapping) else None
    if mode == "topk_preserving":
        return (mode, config.get("alpha"), config.get("preserve_top_k"))
    return (mode,)


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Whether left is no worse in perturbation/accuracy and strictly better."""

    left_noise = _float(left, "metrics", "softmax", "actual_noise_infinity_norm")
    right_noise = _float(right, "metrics", "softmax", "actual_noise_infinity_norm")
    left_drop = _float(left, "metrics", "degradation", "top1_absolute_drop")
    right_drop = _float(right, "metrics", "degradation", "top1_absolute_drop")
    left_ppl = _float(left, "metrics", "degradation", "perplexity_relative_increase")
    right_ppl = _float(right, "metrics", "degradation", "perplexity_relative_increase")
    if None in (left_noise, right_noise, left_drop, right_drop, left_ppl, right_ppl):
        return False
    no_worse = (
        left_noise >= right_noise
        and left_drop <= right_drop
        and left_ppl <= right_ppl
    )
    strictly_better = (
        left_noise > right_noise
        or left_drop < right_drop
        or left_ppl < right_ppl
    )
    return bool(no_worse and strictly_better)


def _frontier(records: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    eligible = [
        record
        for record in records
        if record.get("status") == "success"
        and _float(record, "metrics", "softmax", "actual_noise_infinity_norm") is not None
        and _float(record, "metrics", "degradation", "top1_absolute_drop") is not None
        and _float(record, "metrics", "degradation", "perplexity_relative_increase") is not None
    ]
    return [
        candidate
        for candidate in eligible
        if not any(
            other is not candidate and _dominates(other, candidate)
            for other in eligible
        )
    ]


def _endpoint_records(records: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    successful = [record for record in records if record.get("status") == "success"]
    if not successful:
        return []
    tau_values = [
        _float(record, "config", "tau_max")
        for record in successful
    ]
    finite_tau = [value for value in tau_values if value is not None]
    if not finite_tau:
        return []
    minimum = min(finite_tau)
    maximum = max(finite_tau)
    return [
        record
        for record, tau in zip(successful, tau_values)
        if tau in (minimum, maximum)
    ]


def select_sweep_candidates(
    records: Sequence[Mapping[str, Any]],
    *,
    source: str | None = None,
) -> Dict[str, Any]:
    """Build a reproducible full-evaluation candidate manifest."""

    if not records:
        raise ValueError("calibration records are empty")
    normalized: List[Mapping[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("every calibration record must be a mapping")
        validate_run_record(dict(record))
        run_id = record.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("calibration run_id must be a non-empty string")
        if run_id in seen:
            raise ValueError("duplicate calibration run_id: %s" % run_id)
        seen.add(run_id)
        normalized.append(record)

    cohorts = {_cohort_key(record) for record in normalized}
    if len(cohorts) != 1:
        raise ValueError("calibration records do not share one fair cohort")

    manifest_declarations: set[Tuple[str, ...]] = set()
    for record in normalized:
        config = record.get("config")
        if not isinstance(config, Mapping) or "expected_run_ids" not in config:
            continue
        expected = config.get("expected_run_ids")
        if (
            not isinstance(expected, list)
            or not expected
            or any(not isinstance(item, str) or not item for item in expected)
            or len(set(expected)) != len(expected)
        ):
            raise ValueError("expected_run_ids must be a unique non-empty string list")
        manifest_declarations.add(tuple(expected))
    if len(manifest_declarations) > 1:
        raise ValueError("calibration records disagree on expected_run_ids")
    expected_run_ids = (
        list(next(iter(manifest_declarations)))
        if manifest_declarations
        else None
    )
    observed_run_ids = [str(record["run_id"]) for record in normalized]
    missing_expected_ids = (
        sorted(set(expected_run_ids).difference(observed_run_ids))
        if expected_run_ids is not None
        else []
    )

    successful = [record for record in normalized if record.get("status") == "success"]
    exact = [
        record
        for record in successful
        if _path(record, "config", "mode") == "exact"
    ]
    plaintext = [
        record
        for record in successful
        if _path(record, "config", "mode") == "plaintext"
    ]
    exact_passed = any(
        _path(record, "metrics", "exact_gate", "passed") is True
        for record in exact
    )
    baseline_ids = [record["run_id"] for record in plaintext[:1] + exact[:1]]

    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for record in successful:
        mode = _path(record, "config", "mode")
        if mode in APPROXIMATE_MODES:
            grouped.setdefault(_schedule_key(record), []).append(record)

    selected: Dict[str, Mapping[str, Any]] = {
        str(record["run_id"]): record
        for record in normalized
        if record.get("run_id") in baseline_ids
    }
    schedule_details: List[Dict[str, Any]] = []
    for key in sorted(grouped, key=str):
        group = grouped[key]
        endpoints = _endpoint_records(group)
        frontier = _frontier(group)
        chosen = {str(record["run_id"]): record for record in endpoints + frontier}
        selected.update(chosen)
        schedule_details.append(
            {
                "schedule": list(key),
                "observed_successes": len(group),
                "endpoint_run_ids": [record["run_id"] for record in endpoints],
                "frontier_run_ids": [record["run_id"] for record in frontier],
                "selected_run_ids": sorted(chosen),
            }
        )

    order = {str(record["run_id"]): index for index, record in enumerate(normalized)}
    selected_ids = sorted(selected, key=lambda run_id: order[run_id])
    failure_ids = [
        str(record["run_id"])
        for record in normalized
        if record.get("status") != "success"
    ]
    observed_modes = sorted(
        {
            str(_path(record, "config", "mode"))
            for record in normalized
            if _path(record, "config", "mode") is not None
        }
    )
    unselected_ids = [
        run_id for run_id in observed_run_ids if run_id not in selected
    ]
    record_status = {
        str(record["run_id"]): str(record["status"])
        for record in normalized
    }
    if expected_run_ids is None:
        status = "missing_calibration_manifest"
    elif missing_expected_ids:
        status = "incomplete_calibration"
    elif not exact_passed:
        status = "blocked_exact_gate"
    elif not plaintext or not exact:
        status = "insufficient_baselines"
    elif not any(
        _path(record, "config", "mode") in APPROXIMATE_MODES
        and record.get("status") == "success"
        for record in normalized
    ):
        status = "insufficient_approximate_successes"
    else:
        status = "ready_for_full_evaluation"

    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": status,
        "source": source,
        "record_count": len(normalized),
        "status_counts": dict(Counter(str(record["status"]) for record in normalized)),
        "expected_run_ids": expected_run_ids,
        "observed_run_ids": observed_run_ids,
        "missing_expected_run_ids": missing_expected_ids,
        "failure_run_ids": failure_ids,
        "observed_modes": observed_modes,
        "baseline_run_ids": baseline_ids,
        "selected_run_ids": selected_ids,
        "unselected_run_ids": unselected_ids,
        "selection_status_by_run_id": {
            run_id: (
                "selected_for_full"
                if run_id in selected
                else (
                    "calibration_only"
                    if record_status[run_id] == "success"
                    else "failure_retained"
                )
            )
            for run_id in observed_run_ids
        },
        "selection_rule": {
            "description": (
                "retain plaintext/exact, every successful schedule endpoint, "
                "and the non-dominated noise/accuracy/perplexity frontier"
            ),
            "objectives": {
                "maximize": ["actual_noise_infinity_norm"],
                "minimize": [
                    "top1_absolute_drop",
                    "perplexity_relative_increase",
                ],
            },
        },
        "schedule_details": schedule_details,
        "exact_gate": {
            "passed": exact_passed,
            "run_ids": [record["run_id"] for record in exact],
        },
        "cohort_key": list(next(iter(cohorts))),
    }


def load_jsonl_records(path: str) -> List[Dict[str, Any]]:
    """Load and validate one raw JSONL file for the selector."""

    from pathlib import Path
    import json

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError("raw calibration file does not exist: %s" % source)
    records: List[Dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("%s:%d: invalid JSON" % (source, line_number)) from error
        if not isinstance(value, dict):
            raise ValueError("%s:%d: record must be an object" % (source, line_number))
        records.append(value)
    return records
