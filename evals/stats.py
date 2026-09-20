"""Paired and stratified statistics (protocol §6.2).

* Accuracy: paired bootstrap or McNemar.
* PPL/NLL: paired bootstrap over documents/samples.
* Token agreement: Wilson or Clopper–Pearson intervals.
* Multi-key: stratified bootstrap — sample keys first, then samples.
* Always report 95% CI, never point estimates alone.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# Re-exported for existing importers; canonical definitions live in
# fastprove.evaluation.stats (docs/PROBLEMS.md §3.4).
from fastprove.evaluation.stats import (  # noqa: F401
    IntervalEstimate,
    percentage_point_drop,
    relative_increase,
    validate_level as _validate_level,
    wilson_interval,
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


