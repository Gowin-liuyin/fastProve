"""Single-source statistics (docs/PROBLEMS.md §3.4).

The Wilson interval previously existed twice with different z constants
(``evals.stats`` 1.959963984540054 vs benchmark scoring 1.959963985). The
canonical definition now lives in ``fastprove.evaluation.stats`` and
``evals.stats`` re-exports it; these tests pin identity and bit equality.
"""

from __future__ import annotations

import math

import pytest

from evals import stats as eval_stats
from fastprove.evaluation import stats as fast_stats
from fastprove.evaluation.benchmarks import scoring


def test_evals_stats_reexports_the_canonical_definitions() -> None:
    assert eval_stats.wilson_interval is fast_stats.wilson_interval
    assert eval_stats.IntervalEstimate is fast_stats.IntervalEstimate
    assert eval_stats.relative_increase is fast_stats.relative_increase
    assert eval_stats.percentage_point_drop is fast_stats.percentage_point_drop


@pytest.mark.parametrize(
    ("correct", "n"),
    [
        (0, 1),
        (0, 4),
        (1, 8),
        (7, 10),
        (10, 10),
        (50, 100),
        (95, 100),
        (1838, 1838),
    ],
)
def test_scoring_wrapper_is_bit_identical_to_canonical(
    correct: int, n: int
) -> None:
    low, high = scoring._wilson_interval(correct, n)
    canonical = fast_stats.wilson_interval(correct, n, level=0.95)
    assert low == canonical.ci_low
    assert high == canonical.ci_high


def test_empty_record_sentinel_is_preserved() -> None:
    low, high = scoring._wilson_interval(0, 0)
    assert math.isnan(low) and math.isnan(high)


def test_95_percent_z_uses_the_sourced_table_value() -> None:
    assert fast_stats._z_from_level(0.95) == 1.959963984540054


def test_relative_increase_zero_baseline_semantics() -> None:
    assert math.isinf(fast_stats.relative_increase(0.0, 1.0))
    assert fast_stats.relative_increase(0.0, 0.0) == 0.0
    assert fast_stats.relative_increase(2.0, 3.0) == 0.5
