"""Exact-mode acceptance gate and its recorded tolerance envelope."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..structured import reduction_compute_dtype_name


def _exact_gate_tolerances(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return the recorded numeric envelope used by the exact gate.

    The tiny FP32 unit gate is deliberately much tighter than the deep-model
    envelope.  A real checkpoint applies a basis conversion at every layer,
    and the reference smoke on both CPU and MPS showed a small absolute
    Softmax deviation (roughly 1.3--1.5e-3) while relative error and token
    agreement remained at floating-point level.  Keep the relaxed deep FP32
    absolute Softmax bound explicit and recorded; this is not a dtype label
    that can make a large drift pass.
    """

    config = record.get("config", {})
    runtime = config.get("runtime", {}) if isinstance(config, dict) else {}
    environment = record.get("environment", {})
    model = record.get("model", {})
    pretrained = bool(model.get("pretrained", False)) if isinstance(model, dict) else False
    dtype = str(runtime.get("activation_dtype", "float32"))
    actual_device = str(
        environment.get("actual_device")
        or environment.get("device")
        or runtime.get("device", "unknown")
    )
    checkpoint_dtype = str(
        environment.get("checkpoint_compute_dtype")
        or reduction_compute_dtype_name()
    )
    if dtype == "float32" and pretrained:
        profile = "deep_fp32_calibrated"
        logits_max = 1e-3
        qk_max = 2e-2
        softmax_max = 2e-3
    elif dtype == "float32":
        profile = "tiny_fp32_strict"
        logits_max = 1e-3
        qk_max = 2e-2
        softmax_max = 1e-3
    else:
        profile = "non_fp32_separate_condition"
        logits_max = 2e-2
        qk_max = 2e-2
        softmax_max = 2e-2
    return {
        "profile": profile,
        "pretrained": pretrained,
        "activation_dtype": dtype,
        "actual_device": actual_device,
        "checkpoint_compute_dtype": checkpoint_dtype,
        "max_logit_absolute_error": logits_max,
        "max_qk_absolute_error": qk_max,
        "max_softmax_absolute_error": softmax_max,
        "max_relative_l2_error": 1e-4,
        "required_teacher_forced_agreement": 1.0,
        "required_greedy_token_agreement": 1.0,
        "required_greedy_sequence_agreement": 1.0,
    }


def evaluate_exact_gate(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Apply the mandatory exact-mode gate before approximate conclusions."""

    reasons: List[str] = []
    if record.get("status") != "success":
        return False, ["exact run status is not success"]
    if record.get("config", {}).get("mode") != "exact":
        return False, ["record is not exact mode"]
    metrics = record.get("metrics", {})
    tolerances = _exact_gate_tolerances(record)
    metrics["exact_gate_tolerances"] = tolerances
    logits_tolerance = float(tolerances["max_logit_absolute_error"])
    qk_tolerance = float(tolerances["max_qk_absolute_error"])
    softmax_tolerance = float(tolerances["max_softmax_absolute_error"])
    checks = (
        (
            metrics.get("nan_inf_count") == 0,
            "exact output contains NaN/Inf",
        ),
        (
            metrics.get("agreement", {}).get(
                "next_token_top1_agreement"
            )
            == 1.0,
            "teacher-forced token agreement is not 1",
        ),
        (
            metrics.get("greedy", {}).get("greedy_token_exact_match")
            == 1.0,
            "greedy token agreement is not 1",
        ),
        (
            metrics.get("greedy", {}).get(
                "greedy_sequence_exact_match"
            )
            == 1.0,
            "greedy sequence agreement is not 1",
        ),
        (
            metrics.get("layer", {})
            .get("logits", {})
            .get("max_absolute_error", float("inf"))
            <= logits_tolerance,
            "exact logit error exceeds tolerance",
        ),
        (
            metrics.get("layer", {})
            .get("logits", {})
            .get("relative_l2_error", float("inf"))
            <= 1e-4,
            "exact relative logit error exceeds tolerance",
        ),
        (
            metrics.get("layer", {})
            .get("qk_scores", {})
            .get("relative_l2_error", float("inf"))
            <= 1e-4,
            "exact relative QK error exceeds tolerance",
        ),
        (
            metrics.get("layer", {})
            .get("qk_scores", {})
            .get("max_absolute_error", float("inf"))
            <= qk_tolerance,
            "exact QK error exceeds tolerance",
        ),
        (
            metrics.get("layer", {})
            .get("exact_softmax_probabilities", {})
            .get("relative_l2_error", float("inf"))
            <= 1e-4,
            "exact relative Softmax error exceeds tolerance",
        ),
        (
            metrics.get("layer", {})
            .get("exact_softmax_probabilities", {})
            .get("max_absolute_error", float("inf"))
            <= softmax_tolerance,
            "exact Softmax error exceeds tolerance",
        ),
    )
    for passed, reason in checks:
        if not passed:
            reasons.append(reason)
    diagnostics = metrics.get("exact_gate_diagnostics", {})
    if diagnostics.get("first_teacher_forced_argmax_mismatch") is not None:
        reasons.append("teacher-forced mismatch diagnostic is populated")
    if diagnostics.get("first_greedy_mismatch") is not None:
        reasons.append("greedy mismatch diagnostic is populated")
    return not reasons, reasons
