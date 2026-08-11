"""Machine-readable experiment record contracts."""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

SCHEMA_VERSION = "fastprove.run.v1"
REQUIRED_FIELDS = {
    "schema_version",
    "run_id",
    "status",
    "timestamp_utc",
    "config",
    "seed",
    "model",
    "dataset",
    "environment",
    "sample_count",
    "metrics",
    "elapsed_seconds",
}
SUPPORTED_MODES = {
    "plaintext",
    "exact",
    "topk_preserving",
    "free_bounded",
}
SUPPORTED_EVIDENCE_KINDS = {
    "comparative_evaluation",
    "language_model_accuracy",
    "prototype_correctness",
}
COMPARISON_METRIC_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("plaintext", "negative_log_likelihood"),
    ("plaintext", "perplexity"),
    ("plaintext", "next_token_top1_accuracy"),
    ("plaintext", "next_token_top5_accuracy"),
    ("obfuscated", "negative_log_likelihood"),
    ("obfuscated", "perplexity"),
    ("obfuscated", "next_token_top1_accuracy"),
    ("obfuscated", "next_token_top5_accuracy"),
    ("agreement", "next_token_top1_agreement"),
    ("degradation", "nll_absolute_increase"),
    ("degradation", "perplexity_absolute_increase"),
    ("degradation", "perplexity_relative_increase"),
    ("degradation", "top1_absolute_drop"),
    ("degradation", "top1_relative_drop"),
    ("degradation", "top5_absolute_drop"),
    ("degradation", "top5_relative_drop"),
    ("greedy", "greedy_token_exact_match"),
    ("greedy", "greedy_sequence_exact_match"),
    ("softmax", "kl_divergence"),
    ("softmax", "js_divergence"),
    ("softmax", "topk_overlap"),
    ("softmax", "rank_correlation"),
    ("softmax", "topk_changed_fraction"),
    ("softmax", "actual_noise_infinity_norm"),
    ("softmax", "zero_noise_query_fraction"),
    ("softmax", "attention_output_relative_l2_error"),
)
OPTIONAL_UNDEFINED_METRIC_PATHS = {
    ("degradation", "top1_relative_drop"),
    ("degradation", "top5_relative_drop"),
}


def build_run_record(
    *,
    run_id: str,
    status: str,
    config: Dict[str, Any],
    seed: int,
    model: Dict[str, Any],
    dataset: Dict[str, Any],
    environment: Dict[str, Any],
    sample_count: int,
    metrics: Dict[str, Any],
    elapsed_seconds: float,
    error: Optional[Dict[str, Any]] = None,
    stage: Optional[str] = None,
    last_completed_sample_id: Optional[str] = None,
    partial_metrics_available: Optional[bool] = None,
    reason: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build and validate one raw run record."""

    record: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": config,
        "seed": seed,
        "model": model,
        "dataset": dataset,
        "environment": environment,
        "sample_count": sample_count,
        "metrics": metrics,
        "elapsed_seconds": elapsed_seconds,
    }
    if error is not None:
        record["error"] = error
    if status == "failure":
        record["last_completed_sample_id"] = last_completed_sample_id
    if stage is not None:
        record["stage"] = stage
    if partial_metrics_available is not None:
        record["partial_metrics_available"] = partial_metrics_available
    if reason is not None:
        record["reason"] = reason
    validate_run_record(record)
    return record


def _validate_utc_timestamp(value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp_utc must be a non-empty UTC ISO timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        timestamp = dt.datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(
            "timestamp_utc must be a valid UTC ISO timestamp"
        ) from error
    if timestamp.tzinfo is None or timestamp.utcoffset() != dt.timedelta(0):
        raise ValueError("timestamp_utc must include an explicit UTC offset")


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _metric_value(
    metrics: Mapping[str, Any], path: Sequence[str]
) -> Tuple[bool, Any]:
    current: Any = metrics
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _require_numeric_metric_paths(
    metrics: Mapping[str, Any],
    paths: Sequence[Tuple[str, ...]],
) -> None:
    missing = []
    invalid = []
    for path in paths:
        present, value = _metric_value(metrics, path)
        label = ".".join(path)
        if not present:
            missing.append(label)
            continue
        if value is None and path in OPTIONAL_UNDEFINED_METRIC_PATHS:
            continue
        if not _is_finite_number(value):
            invalid.append(label)
    if missing:
        raise ValueError(
            "success record missing core metrics: %s"
            % ", ".join(sorted(missing))
        )
    if invalid:
        raise ValueError(
            "success core metrics must be finite numbers: %s"
            % ", ".join(sorted(invalid))
        )


def _validate_performance_metrics(metrics: Mapping[str, Any]) -> None:
    performance = metrics.get("performance")
    if not isinstance(performance, Mapping):
        raise ValueError(
            "success record missing core metrics: performance"
        )
    conversion = performance.get("conversion_time_seconds")
    if not _is_finite_number(conversion):
        raise ValueError(
            "success core metrics must include finite "
            "performance.conversion_time_seconds"
        )
    nested_keys = ("prefill", "decode", "peak_memory", "kv_cache")
    legacy_keys = (
        "prefill_latency_seconds",
        "decode_tpot_seconds",
        "tokens_per_second",
        "peak_memory_bytes",
        "kv_cache_memory_bytes",
    )
    nested = all(
        isinstance(performance.get(key), Mapping) for key in nested_keys
    )
    legacy = all(
        _is_finite_number(performance.get(key)) for key in legacy_keys
    )
    if not nested and not legacy:
        raise ValueError(
            "success record missing core metrics: performance must use "
            "the nested runner layout or complete legacy report layout"
        )


def _evidence_kind(config: Mapping[str, Any]) -> str:
    stage = config.get("stage")
    if stage is not None and (
        not isinstance(stage, str) or not stage.strip()
    ):
        raise ValueError("config.stage must be a non-empty string")
    prototype = config.get("prototype")
    if prototype is not None and not isinstance(prototype, Mapping):
        raise ValueError("config.prototype must be a mapping")
    evidence = config.get("evidence")
    explicit_kind: Optional[str] = None
    if evidence is not None:
        if isinstance(evidence, str):
            explicit_kind = evidence
        elif isinstance(evidence, Mapping):
            explicit_kind = evidence.get("kind")
        else:
            raise ValueError(
                "config.evidence must be a supported string or mapping"
            )
        if (
            not isinstance(explicit_kind, str)
            or explicit_kind not in SUPPORTED_EVIDENCE_KINDS
        ):
            raise ValueError(
                "config.evidence kind must be one of: %s"
                % ", ".join(sorted(SUPPORTED_EVIDENCE_KINDS))
            )
        return explicit_kind
    if prototype is not None:
        return "prototype_correctness"
    normalized_stage = stage.lower() if isinstance(stage, str) else ""
    if (
        "correctness" in normalized_stage
        or "tiny_reference" in normalized_stage
        or "prototype" in normalized_stage
    ):
        return "prototype_correctness"
    return "comparative_evaluation"


def _validate_success_record(
    *,
    config: Mapping[str, Any],
    model: Mapping[str, Any],
    dataset: Mapping[str, Any],
    metrics: Mapping[str, Any],
    sample_count: int,
) -> None:
    mode = config.get("mode")
    if not isinstance(mode, str) or mode not in SUPPORTED_MODES:
        raise ValueError(
            "config.mode must be one of: %s"
            % ", ".join(sorted(SUPPORTED_MODES))
        )
    if sample_count == 0:
        raise ValueError("success sample_count must be positive")
    for context, field in (
        (model, "pretrained"),
        (dataset, "meaningful_lm_evidence"),
    ):
        if field in context and not isinstance(context[field], bool):
            raise ValueError("%s evidence flag must be boolean" % field)
    evidence_kind = _evidence_kind(config)
    if evidence_kind == "language_model_accuracy" and not (
        model.get("pretrained") is True
        and dataset.get("meaningful_lm_evidence") is True
    ):
        raise ValueError(
            "language_model_accuracy evidence requires a pretrained model "
            "and meaningful_lm_evidence dataset"
        )
    _require_numeric_metric_paths(metrics, COMPARISON_METRIC_PATHS)
    _validate_performance_metrics(metrics)
    if evidence_kind == "prototype_correctness":
        prototype_paths = [
            ("layer", "logits", "max_absolute_error"),
            ("nan_inf_count",),
        ]
        if mode != "plaintext":
            prototype_paths.append(
                ("layer", "qk_scores", "max_absolute_error")
            )
        if mode == "exact":
            prototype_paths.append(
                (
                    "layer",
                    "exact_softmax_probabilities",
                    "max_absolute_error",
                )
            )
        _require_numeric_metric_paths(metrics, prototype_paths)


def validate_run_record(record: Dict[str, Any]) -> None:
    """Enforce required fields and explicit success/failure semantics."""

    if not isinstance(record, Mapping):
        raise ValueError("run record must be a mapping")
    missing = REQUIRED_FIELDS.difference(record)
    if missing:
        raise ValueError("run record missing fields: %s" % sorted(missing))
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema_version must be %s" % SCHEMA_VERSION)
    _validate_utc_timestamp(record["timestamp_utc"])
    if (
        not isinstance(record["run_id"], str)
        or not record["run_id"].strip()
    ):
        raise ValueError("run_id must be a non-empty string")
    if not isinstance(record["status"], str) or record["status"] not in (
        "success",
        "failure",
        "skipped",
    ):
        raise ValueError("run status must be success, failure, or skipped")
    if (
        not isinstance(record["seed"], int)
        or isinstance(record["seed"], bool)
    ):
        raise ValueError("seed must be an integer, not a boolean")
    if (
        not isinstance(record["sample_count"], int)
        or isinstance(record["sample_count"], bool)
        or record["sample_count"] < 0
    ):
        raise ValueError("sample_count must be a non-negative integer")
    if (
        not _is_finite_number(record["elapsed_seconds"])
        or record["elapsed_seconds"] < 0
    ):
        raise ValueError(
            "elapsed_seconds must be a finite non-negative number"
        )
    for field in ("config", "model", "dataset", "environment"):
        if not isinstance(record[field], Mapping):
            raise ValueError("%s must be a mapping" % field)
    status = record["status"]
    metrics = record["metrics"]
    if not isinstance(metrics, Mapping):
        raise ValueError("metrics must be a mapping")
    if status == "success":
        if "error" in record:
            raise ValueError("success record must not include an error")
        if not metrics:
            raise ValueError("success record must include metrics")
        _validate_success_record(
            config=record["config"],
            model=record["model"],
            dataset=record["dataset"],
            metrics=metrics,
            sample_count=record["sample_count"],
        )
    if status == "failure":
        error = record.get("error")
        if not isinstance(error, Mapping):
            raise ValueError("failure record must include an error")
        if not isinstance(error.get("type"), str) or not error["type"]:
            raise ValueError("failure error must include a non-empty type")
        if not isinstance(error.get("message"), str):
            raise ValueError("failure error must include a message")
        if not isinstance(record.get("stage"), str) or not record["stage"]:
            raise ValueError("failure record must include stage")
        if "last_completed_sample_id" not in record:
            raise ValueError(
                "failure record must include last_completed_sample_id"
            )
        if (
            record["last_completed_sample_id"] is not None
            and not isinstance(record["last_completed_sample_id"], str)
        ):
            raise ValueError(
                "last_completed_sample_id must be a string or null"
            )
        if not isinstance(record.get("partial_metrics_available"), bool):
            raise ValueError(
                "failure record must include partial_metrics_available"
            )
    if status == "skipped":
        reason = record.get("reason")
        if not isinstance(reason, Mapping):
            raise ValueError("skipped record must include reason")
        if not isinstance(reason.get("code"), str) or not reason["code"]:
            raise ValueError("skipped reason must include a non-empty code")
        if not isinstance(reason.get("message"), str):
            raise ValueError("skipped reason must include a message")
    try:
        json.dumps(record, allow_nan=False, sort_keys=True)
    except ValueError as error:
        raise ValueError(
            "run record must contain only finite JSON numbers"
        ) from error
    except TypeError as error:
        raise ValueError("run record must be JSON-serializable") from error


def require_fresh_jsonl_destination(path: Path) -> None:
    """Require a missing or empty raw destination before a multi-run command.

    Raw evidence is append-only, while the prototype's run IDs are stable for
    reproducibility. Refusing an occupied destination before inference avoids
    wasted compute and ambiguous partial-resume semantics.
    """

    destination = Path(path)
    if destination.exists() and (
        not destination.is_file() or destination.stat().st_size > 0
    ):
        raise FileExistsError(
            "raw JSONL destination is not empty: %s; choose a new --output "
            "path to preserve append-only evidence" % destination
        )


def append_jsonl_record(path: Path, record: Dict[str, Any]) -> None:
    """Append one validated record without rewriting prior raw evidence."""

    validate_run_record(record)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        for line_number, line in enumerate(
            destination.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                existing = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "existing JSONL is invalid at line %d" % line_number
                ) from error
            if existing.get("run_id") == record["run_id"]:
                raise ValueError(
                    "duplicate run_id in raw artifact: %s"
                    % record["run_id"]
                )
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                record, ensure_ascii=False, allow_nan=False, sort_keys=True
            )
            + "\n"
        )
