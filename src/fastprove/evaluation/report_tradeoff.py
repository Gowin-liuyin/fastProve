"""Trade-off selection helpers for report derivation."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .report_format import _as_float
from .report_rows import _cohort_key


def _tradeoff_candidates(
    rows: Sequence[Dict[str, object]],
    *,
    mode: Optional[str] = None,
) -> List[Dict[str, object]]:
    meaningful = [
        row
        for row in rows
        if row.get("status") == "success"
        and row.get("meaningful_lm_evidence") is True
    ]
    cohorts = {_cohort_key(row) for row in meaningful}
    if len(cohorts) != 1:
        return []
    cohort = next(iter(cohorts))
    exact_gate_passed = any(
        row.get("mode") == "exact"
        and row.get("exact_gate_passed") is True
        and _cohort_key(row) == cohort
        for row in meaningful
    )
    if not exact_gate_passed:
        return []
    candidates: List[Dict[str, object]] = []
    for row in meaningful:
        row_mode = row.get("mode")
        if row_mode not in ("topk_preserving", "free_bounded"):
            continue
        if mode is not None and row_mode != mode:
            continue
        noise = _as_float(row.get("actual_noise_inf"))
        top1_drop = _as_float(row.get("top1_absolute_drop"))
        agreement = _as_float(row.get("top1_agreement"))
        ppl_increase = _as_float(row.get("perplexity_relative_increase"))
        if (
            noise is None
            or top1_drop is None
            or agreement is None
            or ppl_increase is None
            or noise <= 0
        ):
            continue
        candidates.append(row)
    return candidates


def _best_tradeoff(
    rows: Sequence[Dict[str, object]],
    *,
    mode: Optional[str] = None,
) -> Optional[str]:
    candidates = _tradeoff_candidates(rows, mode=mode)
    if not candidates:
        return None
    constrained = [
        row
        for row in candidates
        if float(row["top1_absolute_drop"]) <= 0.01
        and float(row["top1_agreement"]) >= 0.99
    ]
    if constrained:
        best = max(
            constrained,
            key=lambda row: (
                float(row["actual_noise_inf"]),
                -float(row["perplexity_relative_increase"]),
            ),
        )
    else:
        best = min(
            candidates,
            key=lambda row: (
                float(row["top1_absolute_drop"])
                + max(0.0, float(row["perplexity_relative_increase"])),
                -float(row["actual_noise_inf"]),
            ),
        )
    return str(best["run_id"])


def _pareto_frontier(
    rows: Sequence[Dict[str, object]], *, mode: str
) -> List[str]:
    candidates = _tradeoff_candidates(rows, mode=mode)
    frontier: List[Dict[str, object]] = []
    for candidate in candidates:
        c_top1 = float(candidate["top1_absolute_drop"])
        c_ppl = float(candidate["perplexity_relative_increase"])
        c_noise = float(candidate["actual_noise_inf"])
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            o_top1 = float(other["top1_absolute_drop"])
            o_ppl = float(other["perplexity_relative_increase"])
            o_noise = float(other["actual_noise_inf"])
            no_worse = (
                o_top1 <= c_top1
                and o_ppl <= c_ppl
                and o_noise >= c_noise
            )
            strictly_better = (
                o_top1 < c_top1
                or o_ppl < c_ppl
                or o_noise > c_noise
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    frontier.sort(
        key=lambda row: (
            float(row["actual_noise_inf"]),
            str(row["run_id"]),
        )
    )
    return [str(row["run_id"]) for row in frontier]
