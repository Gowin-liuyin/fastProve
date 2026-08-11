"""Paired and stratified statistics (protocol §6.2).

* Accuracy: paired bootstrap or McNemar.
* PPL/NLL: paired bootstrap over documents/samples.
* Token agreement: Wilson or Clopper–Pearson intervals.
* Multi-key: stratified bootstrap — sample keys first, then samples.
* Always report 95% CI, never point estimates alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


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


def _validate_level(level: float) -> None:
    if not 0.0 < level < 1.0:
        raise ValueError("confidence level must be in (0, 1)")


def wilson_interval(
    successes: int, n: int, *, level: float = 0.95
) -> IntervalEstimate:
    """Wilson score interval for a binomial proportion."""

    _validate_level(level)
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


def clopper_pearson_interval(
    successes: int, n: int, *, level: float = 0.95
) -> IntervalEstimate:
    """Clopper–Pearson exact binomial interval via beta quantiles."""

    _validate_level(level)
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    alpha = 1.0 - level
    # Use scipy-free implementation via incomplete beta / F relation.
    if successes == 0:
        low = 0.0
    else:
        low = _beta_ppf(alpha / 2.0, successes, n - successes + 1)
    if successes == n:
        high = 1.0
    else:
        high = _beta_ppf(1.0 - alpha / 2.0, successes + 1, n - successes)
    return IntervalEstimate(
        estimate=successes / n,
        ci_low=low,
        ci_high=high,
        level=level,
        method="clopper_pearson",
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


def _beta_ppf(p: float, a: float, b: float) -> float:
    """Incomplete-beta quantile via scipy-free Newton on regularised beta."""

    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    # Initial guess: mean.
    x = a / (a + b)
    for _ in range(50):
        cdf = _betainc(a, b, x)
        pdf = _beta_pdf(a, b, x)
        if pdf <= 0.0:
            break
        delta = (cdf - p) / pdf
        x = min(1.0 - 1e-12, max(1e-12, x - delta))
        if abs(delta) < 1e-10:
            break
    return float(x)


def _beta_pdf(a: float, b: float, x: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return 0.0
    log_b = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return math.exp((a - 1.0) * math.log(x) + (b - 1.0) * math.log(1.0 - x) - log_b)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a,b) via continued fraction."""

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Use symmetry when helpful.
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)
    ln_front = (
        a * math.log(x)
        + b * math.log(1.0 - x)
        - math.lgamma(a)
        - math.lgamma(b)
        + math.lgamma(a + b)
    )
    # Lentz continued fraction for incomplete beta.
    front = math.exp(ln_front) / a
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        # Even step.
        aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        # Odd step.
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return front * h


def paired_bootstrap_mean_diff(
    plain: Sequence[float],
    obfuscated: Sequence[float],
    *,
    level: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> IntervalEstimate:
    """Paired bootstrap CI for mean(plain − obfuscated)."""

    _validate_level(level)
    if len(plain) != len(obfuscated):
        raise ValueError("paired sequences must have equal length")
    n = len(plain)
    if n == 0:
        raise ValueError("empty sample")
    plain_arr = np.asarray(plain, dtype=np.float64)
    obf_arr = np.asarray(obfuscated, dtype=np.float64)
    diffs = plain_arr - obf_arr
    estimate = float(diffs.mean())
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = diffs[idx].mean()
    alpha = 1.0 - level
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return IntervalEstimate(
        estimate=estimate,
        ci_low=low,
        ci_high=high,
        level=level,
        method="paired_bootstrap",
        n=n,
    )


def mcnemar_exact(
    plain_correct: Sequence[bool],
    obfuscated_correct: Sequence[bool],
) -> dict:
    """McNemar exact test for paired binary outcomes.

    Returns b, c counts and two-sided exact p-value (binomial mid-p style
    via central binomial).
    """

    if len(plain_correct) != len(obfuscated_correct):
        raise ValueError("paired sequences must have equal length")
    b = 0  # plain correct, obf wrong
    c = 0  # plain wrong, obf correct
    for p, o in zip(plain_correct, obfuscated_correct):
        if p and not o:
            b += 1
        elif o and not p:
            c += 1
    n_disc = b + c
    if n_disc == 0:
        p_value = 1.0
    else:
        # Two-sided exact binomial test under p=0.5.
        k = min(b, c)
        # P(X <= k) + P(X >= n-k) with X~Bin(n, 0.5).
        # For p=0.5, sum of two tails = 2 * cdf(k) when k < n/2, else 1.
        cdf = sum(
            math.comb(n_disc, i) for i in range(k + 1)
        ) / (2.0**n_disc)
        p_value = min(1.0, 2.0 * cdf)
    return {
        "b_plain_only": b,
        "c_obf_only": c,
        "discordant": n_disc,
        "p_value_two_sided": p_value,
        "method": "mcnemar_exact",
    }


def stratified_key_bootstrap(
    per_key_values: Sequence[Sequence[float]],
    *,
    level: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 0,
    statistic: str = "mean",
) -> IntervalEstimate:
    """Stratified bootstrap: resample keys, then samples within keys.

    Covers both key variance and sample variance (protocol §6.1–6.2).
    """

    _validate_level(level)
    keys = [np.asarray(v, dtype=np.float64) for v in per_key_values if len(v) > 0]
    if not keys:
        raise ValueError("no key strata provided")
    n_keys = len(keys)

    def _stat(strata: Sequence[np.ndarray]) -> float:
        pooled = np.concatenate(list(strata))
        if statistic == "mean":
            return float(pooled.mean())
        if statistic == "median":
            return float(np.median(pooled))
        raise ValueError("unsupported statistic")

    estimate = _stat(keys)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        key_idx = rng.integers(0, n_keys, size=n_keys)
        resampled = []
        for ki in key_idx:
            stratum = keys[int(ki)]
            m = len(stratum)
            sample_idx = rng.integers(0, m, size=m)
            resampled.append(stratum[sample_idx])
        boots[i] = _stat(resampled)
    alpha = 1.0 - level
    return IntervalEstimate(
        estimate=estimate,
        ci_low=float(np.quantile(boots, alpha / 2.0)),
        ci_high=float(np.quantile(boots, 1.0 - alpha / 2.0)),
        level=level,
        method="stratified_key_bootstrap",
        n=int(sum(len(k) for k in keys)),
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
