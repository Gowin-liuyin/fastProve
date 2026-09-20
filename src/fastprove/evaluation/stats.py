"""Canonical interval and degradation statistics (single source of truth).

Extracted from ``evals/stats.py`` (P8/P9 cleanup, docs/PROBLEMS.md §3.4) so
that the evaluation harness and the OSNIP-style benchmark scoring share one
bit-exact definition. ``evals.stats`` re-exports these names for backward
compatibility. The 95% z constant is the sourced table value
``1.959963984540054``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class IntervalEstimate:
    """Point estimate with a confidence interval."""

    estimate: float
    ci_low: float
    ci_high: float
    level: float
    method: str
    n: int

    def to_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "level": self.level,
            "method": self.method,
            "n": self.n,
        }


def validate_level(level: float) -> None:
    if not 0.0 < level < 1.0:
        raise ValueError("confidence level must be in (0, 1)")


def wilson_interval(
    successes: int, n: int, *, level: float = 0.95
) -> IntervalEstimate:
    """Wilson score interval for a binomial proportion."""

    validate_level(level)
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    z = _z_from_level(level)
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    margin = (
        z
        * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n)
        / denom
    )
    return IntervalEstimate(
        estimate=phat,
        ci_low=max(0.0, centre - margin),
        ci_high=min(1.0, centre + margin),
        level=level,
        method="wilson",
        n=n,
    )


def _z_from_level(level: float) -> float:
    # Approximate inverse normal CDF for common levels.
    alpha = 1.0 - level
    # Beasley-Springer-Moro or simple table for 90/95/99.
    table = {
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }
    if abs(level - 0.95) < 1e-9:
        return table[0.95]
    if abs(level - 0.90) < 1e-9:
        return table[0.90]
    if abs(level - 0.99) < 1e-9:
        return table[0.99]
    # Rational approximation (Abramowitz & Stegun 26.2.23) for general level.
    p = 1.0 - alpha / 2.0
    return _normsinv(p)


def _normsinv(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    # Acklam's approximation.
    a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614736e1,
        2.506628277459239e0,
    ]
    b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ]
    c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ]
    d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def relative_increase(plain: float, obfuscated: float) -> float:
    """Δ_rel = obf / plain − 1 for lower-is-better metrics (PPL, NLL)."""

    if plain == 0.0:
        return math.inf if obfuscated != 0.0 else 0.0
    return obfuscated / plain - 1.0


def percentage_point_drop(plain: float, obfuscated: float) -> float:
    """Δ_pp = plain − obfuscated for higher-is-better metrics (accuracy).

    Units are percentage points when inputs are percentages, or fractions
    when inputs are fractions — callers must label the unit explicitly.
    """

    return plain - obfuscated

