"""Hard-correctness and soft utility gates (protocol §7).

Hard gates are binary mathematical claims of the covariant scheme.
Soft gates are pre-specified engineering non-inferiority margins; they must
**not** be adjusted after seeing results (protocol §6.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Pre-specified non-inferiority margins (protocol §7.2) — DO NOT retune post-hoc.
# ---------------------------------------------------------------------------

# Accuracy drop: Acc_plain − Acc_obf ≤ 0.5 percentage points.
ACCURACY_DROP_PP_MARGIN: float = 0.5

# PPL relative increase: PPL_obf / PPL_plain − 1 ≤ 1%.
PPL_RELATIVE_INCREASE_MARGIN: float = 0.01

# FP32 ChainLinear / decoded-signal max absolute error.
FP32_CHAINLINEAR_MAX_ABS_ERROR: float = 1e-4

# FP32 per-operator decoded signal (protocol §4 Layer 1 baseline).
FP32_OPERATOR_MAX_ABS_ERROR: float = 1e-4


@dataclass(frozen=True)
class GateResult:
    """Outcome of one named gate check."""

    name: str
    category: str  # "hard" | "soft"
    passed: bool
    observed: Any
    threshold: Any
    unit: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "passed": self.passed,
            "observed": self.observed,
            "threshold": self.threshold,
            "unit": self.unit,
            "detail": self.detail,
        }


@dataclass
class GateSummary:
    """Aggregate pass/fail for --check-gates."""

    results: List[GateResult] = field(default_factory=list)

    @property
    def hard_passed(self) -> bool:
        return all(r.passed for r in self.results if r.category == "hard")

    @property
    def soft_passed(self) -> bool:
        return all(r.passed for r in self.results if r.category == "soft")

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hard_passed": self.hard_passed,
            "soft_passed": self.soft_passed,
            "all_passed": self.all_passed,
            "results": [r.to_dict() for r in self.results],
        }

    def format_table(self) -> str:
        """Human-readable PASS/FAIL table."""

        lines = [
            "Gate                                      | Cat  | Status | Observed | Threshold",
            "-" * 90,
        ]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(
                "%-42s | %-4s | %-6s | %s | %s"
                % (
                    r.name[:42],
                    r.category[:4],
                    status,
                    _fmt(r.observed),
                    _fmt(r.threshold),
                )
            )
        lines.append("-" * 90)
        lines.append(
            "HARD: %s   SOFT: %s   ALL: %s"
            % (
                "PASS" if self.hard_passed else "FAIL",
                "PASS" if self.soft_passed else "FAIL",
                "PASS" if self.all_passed else "FAIL",
            )
        )
        return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "%.6g" % value
    return str(value)


def _as_float(metrics: Mapping[str, Any], *keys: str) -> Optional[float]:
    node: Any = metrics
    for key in keys:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    if node is None:
        return None
    try:
        return float(node)
    except (TypeError, ValueError):
        return None


def check_hard_gates(metrics: Mapping[str, Any]) -> List[GateResult]:
    """Apply protocol §7.1 hard-correctness gates.

    Expected metric paths (filled by layer runners / report aggregator)::

        layer1.chain_linear.max_absolute_error
        layer2.rank_flip_rate
        layer2.top1_match
        layer2.topk_overlap
        layer2.causal_mask_match
        layer3.expert_set_match          (optional if no MoE)
        layer4.lm_head_argmax_match
        layer4.greedy_sequence_exact_match
        cache.cache_vs_nocache_identical
    """

    results: List[GateResult] = []

    def hard(
        name: str,
        observed: Optional[float],
        *,
        threshold: float,
        cmp: str,
        unit: str,
        skip_if_missing: bool = False,
        detail: str = "",
    ) -> None:
        if observed is None:
            if skip_if_missing:
                results.append(
                    GateResult(
                        name=name,
                        category="hard",
                        passed=True,
                        observed=None,
                        threshold=threshold,
                        unit=unit,
                        detail=detail or "skipped (metric unavailable)",
                    )
                )
                return
            results.append(
                GateResult(
                    name=name,
                    category="hard",
                    passed=False,
                    observed=None,
                    threshold=threshold,
                    unit=unit,
                    detail=detail or "metric missing",
                )
            )
            return
        if cmp == "<=":
            passed = observed <= threshold + 1e-15
        elif cmp == ">=":
            passed = observed >= threshold - 1e-15
        elif cmp == "==":
            passed = abs(observed - threshold) <= 1e-12
        else:
            raise ValueError("unknown cmp %s" % cmp)
        results.append(
            GateResult(
                name=name,
                category="hard",
                passed=passed,
                observed=observed,
                threshold=threshold,
                unit=unit,
                detail=detail,
            )
        )

    hard(
        "FP32 ChainLinear max abs error",
        _as_float(metrics, "layer1", "chain_linear", "max_absolute_error"),
        threshold=FP32_CHAINLINEAR_MAX_ABS_ERROR,
        cmp="<=",
        unit="abs",
    )
    hard(
        "Attention rank-flip rate",
        _as_float(metrics, "layer2", "rank_flip_rate"),
        threshold=0.0,
        cmp="==",
        unit="fraction",
        skip_if_missing=True,
        detail="skipped when Layer 2 not run",
    )
    hard(
        "Attention top-1 match",
        _as_float(metrics, "layer2", "top1_match"),
        threshold=1.0,
        cmp=">=",
        unit="fraction",
        skip_if_missing=True,
        detail="skipped when Layer 2 not run",
    )
    hard(
        "Attention top-k overlap",
        _as_float(metrics, "layer2", "topk_overlap"),
        threshold=1.0,
        cmp=">=",
        unit="fraction",
        skip_if_missing=True,
        detail="skipped when Layer 2 not run",
    )
    hard(
        "MoE expert set match",
        _as_float(metrics, "layer3", "expert_set_match"),
        threshold=1.0,
        cmp=">=",
        unit="fraction",
        skip_if_missing=True,
        detail="N/A when model has no MoE or Layer 3 not run",
    )
    hard(
        "Causal mask match",
        _as_float(metrics, "layer2", "causal_mask_match"),
        threshold=1.0,
        cmp=">=",
        unit="fraction",
        skip_if_missing=True,
        detail="skipped when Layer 2 not run",
    )
    hard(
        "Cache vs no-cache identical",
        _as_float(metrics, "cache", "cache_vs_nocache_identical"),
        threshold=1.0,
        cmp=">=",
        unit="fraction",
        skip_if_missing=True,
        detail="skipped when Layer 4 not run",
    )
    hard(
        "LM Head inverse-permuted argmax",
        _as_float(metrics, "layer4", "lm_head_argmax_match"),
        threshold=1.0,
        cmp=">=",
        unit="fraction",
        skip_if_missing=True,
        detail="skipped when Layer 4 not run",
    )
    hard(
        "Greedy release-validation sequences",
        _as_float(metrics, "layer4", "greedy_sequence_exact_match"),
        threshold=1.0,
        cmp=">=",
        unit="fraction",
        skip_if_missing=True,
        detail="skipped when Layer 4 not run",
    )
    return results


def check_soft_gates(metrics: Mapping[str, Any]) -> List[GateResult]:
    """Apply protocol §7.2 statistical utility (soft) gates."""

    results: List[GateResult] = []

    acc_drop = _as_float(metrics, "utility", "accuracy_drop_pp")
    if acc_drop is None:
        # Fall back to top-1 absolute drop from teacher-forced metrics.
        acc_drop = _as_float(
            metrics, "layer4", "top1_absolute_drop_pp"
        )
    if acc_drop is not None:
        results.append(
            GateResult(
                name="Accuracy drop ≤ 0.5 pp",
                category="soft",
                passed=acc_drop <= ACCURACY_DROP_PP_MARGIN + 1e-12,
                observed=acc_drop,
                threshold=ACCURACY_DROP_PP_MARGIN,
                unit="pp",
            )
        )
    else:
        results.append(
            GateResult(
                name="Accuracy drop ≤ 0.5 pp",
                category="soft",
                passed=True,
                observed=None,
                threshold=ACCURACY_DROP_PP_MARGIN,
                unit="pp",
                detail="metric unavailable; treated as skip",
            )
        )

    ppl_rel = _as_float(metrics, "utility", "ppl_relative_increase")
    if ppl_rel is None:
        ppl_rel = _as_float(metrics, "layer4", "ppl_relative_increase")
    if ppl_rel is not None:
        results.append(
            GateResult(
                name="PPL relative increase ≤ 1%",
                category="soft",
                passed=ppl_rel <= PPL_RELATIVE_INCREASE_MARGIN + 1e-12,
                observed=ppl_rel,
                threshold=PPL_RELATIVE_INCREASE_MARGIN,
                unit="relative",
            )
        )
    else:
        results.append(
            GateResult(
                name="PPL relative increase ≤ 1%",
                category="soft",
                passed=True,
                observed=None,
                threshold=PPL_RELATIVE_INCREASE_MARGIN,
                unit="relative",
                detail="metric unavailable; treated as skip",
            )
        )

    # 95% CI within non-inferiority bound (if CI upper bound provided).
    ci_upper_acc = _as_float(metrics, "utility", "accuracy_drop_pp_ci95_upper")
    if ci_upper_acc is not None:
        results.append(
            GateResult(
                name="Accuracy drop 95% CI within non-inf bound",
                category="soft",
                passed=ci_upper_acc <= ACCURACY_DROP_PP_MARGIN + 1e-12,
                observed=ci_upper_acc,
                threshold=ACCURACY_DROP_PP_MARGIN,
                unit="pp",
            )
        )
    return results


def evaluate_gates(metrics: Mapping[str, Any]) -> GateSummary:
    """Run all hard and soft gates and return a summary."""

    summary = GateSummary()
    summary.results.extend(check_hard_gates(metrics))
    summary.results.extend(check_soft_gates(metrics))
    return summary
