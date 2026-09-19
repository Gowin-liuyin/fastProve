"""Bit-parity tests between ``evals/metrics_common.py`` and ``fastprove``.

The refactor of the ``evals`` metrics (docs/CLEANUP_PLAN.md P6, option A+C) is
only allowed after proving that the two implementations agree on fixed,
deterministic inputs.  This module is that proof: every value produced by the
``evals`` public helpers is compared against the corresponding key of
``fastprove.evaluation.metrics.attention_distribution_metrics`` (KL, JS, top-k
overlap) or ``fastprove.metrics.tensor_error_metrics`` (max-abs, relative L2).

The comparison is exact: floats are compared by their big-endian IEEE-754
bytes, so ``-0.0 != 0.0`` and any ULP difference fails loudly.

Covered edge cases: identical distributions, one-hot distributions, all-zero
rows, invalid/masked entries, fully masked rows, and multi-dimensional leading
shapes.  ``topk_set_overlap`` additionally returns top-1 match and rank-flip
rates; those have no fastprove counterpart and are not compared here.
"""

from __future__ import annotations

import struct

import pytest
import torch

from evals.metrics_common import (
    js_divergence,
    kl_divergence,
    max_abs_error,
    relative_l2_error,
    topk_set_overlap,
)
from fastprove.evaluation.metrics import (
    attention_distribution_metrics,
    tensor_error_metrics,
)


def _bits(value: float) -> str:
    return struct.pack(">d", float(value)).hex()


def assert_bit_identical(name: str, evals_value: float, fastprove_value: float) -> None:
    assert _bits(evals_value) == _bits(fastprove_value), (
        "%s differs: evals=%r (%s) fastprove=%r (%s)"
        % (
            name,
            evals_value,
            _bits(evals_value),
            fastprove_value,
            _bits(fastprove_value),
        )
    )


def _attention_metrics(
    clean_probabilities: torch.Tensor,
    noisy_probabilities: torch.Tensor,
    *,
    valid: torch.Tensor,
    clean_logits: torch.Tensor,
    noisy_logits: torch.Tensor,
    top_k: int,
) -> dict:
    return attention_distribution_metrics(
        clean_probabilities=clean_probabilities,
        noisy_probabilities=noisy_probabilities,
        clean_logits=clean_logits,
        noisy_logits=noisy_logits,
        valid_mask=valid,
        noise=torch.zeros_like(clean_probabilities),
        margin=torch.zeros(
            clean_probabilities.shape[:-1] + (1,), dtype=torch.float64
        ),
        top_k=top_k,
    )


def _compare_divergences_and_overlap(case) -> None:
    (
        clean_probabilities,
        noisy_probabilities,
        valid,
        clean_logits,
        noisy_logits,
        top_k,
    ) = case
    metrics = _attention_metrics(
        clean_probabilities,
        noisy_probabilities,
        valid=valid,
        clean_logits=clean_logits,
        noisy_logits=noisy_logits,
        top_k=top_k,
    )
    assert_bit_identical(
        "kl_divergence",
        kl_divergence(clean_probabilities, noisy_probabilities, valid=valid),
        metrics["kl_divergence"],
    )
    assert_bit_identical(
        "js_divergence",
        js_divergence(clean_probabilities, noisy_probabilities, valid=valid),
        metrics["js_divergence"],
    )
    _top1, overlap, _flip = topk_set_overlap(
        clean_logits, noisy_logits, k=top_k, valid=valid
    )
    assert_bit_identical("topk_overlap", overlap, metrics["topk_overlap"])


def _identical_case():
    probabilities = torch.tensor(
        [
            [0.6, 0.3, 0.1, 0.0],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=torch.float64,
    )
    valid = torch.ones_like(probabilities, dtype=torch.bool)
    logits = torch.tensor(
        [[3.0, 2.0, 1.0, -1.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.float64
    )
    return probabilities, probabilities.clone(), valid, logits, logits.clone(), 2


def _one_hot_case():
    clean = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    noisy = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    valid = torch.ones_like(clean, dtype=torch.bool)
    clean_logits = torch.tensor(
        [[5.0, -5.0, -5.0], [-5.0, 5.0, -5.0]], dtype=torch.float64
    )
    noisy_logits = torch.tensor(
        [[-5.0, 5.0, -5.0], [5.0, -5.0, -5.0]], dtype=torch.float64
    )
    return clean, noisy, valid, clean_logits, noisy_logits, 1


def _zero_case():
    clean = torch.zeros(2, 3, dtype=torch.float64)
    noisy = torch.zeros(2, 3, dtype=torch.float64)
    valid = torch.ones(2, 3, dtype=torch.bool)
    logits = torch.zeros(2, 3, dtype=torch.float64)
    return clean, noisy, valid, logits, logits.clone(), 3


def _masked_case():
    generator = torch.Generator().manual_seed(20240920)
    clean_raw = torch.rand(3, 2, 5, generator=generator, dtype=torch.float64) + 0.01
    clean = clean_raw / clean_raw.sum(dim=-1, keepdim=True)
    noisy_raw = torch.rand(3, 2, 5, generator=generator, dtype=torch.float64) + 0.01
    noisy = noisy_raw / noisy_raw.sum(dim=-1, keepdim=True)
    valid = torch.tensor(
        [
            [[True, True, True, True, True], [True, True, True, False, False]],
            [[True, True, False, False, False], [False, False, False, False, False]],
            [[True, True, True, True, False], [True, False, False, False, False]],
        ]
    )
    clean_logits = torch.randn(3, 2, 5, generator=generator, dtype=torch.float64)
    noisy_logits = clean_logits.clone()
    noisy_logits[0, 0, 3], noisy_logits[0, 0, 4] = (
        noisy_logits[0, 0, 4].clone(),
        noisy_logits[0, 0, 3].clone(),
    )
    return clean, noisy, valid, clean_logits, noisy_logits, 2


def _fully_masked_case():
    clean = torch.tensor(
        [[[0.5, 0.5]], [[0.2, 0.8]]], dtype=torch.float64
    )
    noisy = torch.tensor(
        [[[0.4, 0.6]], [[0.7, 0.3]]], dtype=torch.float64
    )
    valid = torch.zeros_like(clean, dtype=torch.bool)
    logits = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float64
    )
    return clean, noisy, valid, logits, logits.clone(), 2


def _four_dimensional_case():
    clean = torch.tensor(
        [
            [
                [
                    [0.7, 0.2, 0.05, 0.05],
                    [0.1, 0.2, 0.3, 0.4],
                ]
            ]
        ],
        dtype=torch.float64,
    )
    noisy = torch.tensor(
        [
            [
                [
                    [0.2, 0.7, 0.05, 0.05],
                    [0.4, 0.3, 0.2, 0.1],
                ]
            ]
        ],
        dtype=torch.float64,
    )
    valid = torch.tensor(
        [
            [
                [
                    [True, True, True, True],
                    [True, True, True, False],
                ]
            ]
        ]
    )
    clean_logits = torch.tensor(
        [[[[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]]]],
        dtype=torch.float64,
    )
    noisy_logits = torch.tensor(
        [[[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]]],
        dtype=torch.float64,
    )
    return clean, noisy, valid, clean_logits, noisy_logits, 2


CASES = {
    "identical": _identical_case,
    "one_hot": _one_hot_case,
    "zeros": _zero_case,
    "masked": _masked_case,
    "fully_masked": _fully_masked_case,
    "four_dimensional": _four_dimensional_case,
}


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_projection_divergences_match(case_name: str) -> None:
    _compare_divergences_and_overlap(CASES[case_name]())


ERROR_CASES = {
    "identical": (
        torch.tensor([[1.0, -2.0, 3.0], [0.0, 0.5, -0.5]], dtype=torch.float64),
        torch.tensor([[1.0, -2.0, 3.0], [0.0, 0.5, -0.5]], dtype=torch.float64),
    ),
    "scaled": (
        torch.tensor([[1.0, 2.0, -4.0]], dtype=torch.float64),
        torch.tensor([[1.5, 1.8, -4.4]], dtype=torch.float64),
    ),
    "zero_reference": (
        torch.zeros(2, 3, dtype=torch.float64),
        torch.zeros(2, 3, dtype=torch.float64),
    ),
    "zero_reference_nonzero_actual": (
        torch.zeros(2, 3, dtype=torch.float64),
        torch.ones(2, 3, dtype=torch.float64),
    ),
    "four_dimensional": (
        torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) / 7.0,
        torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) / 7.0 + 0.125,
    ),
}


@pytest.mark.parametrize("case_name", sorted(ERROR_CASES))
def test_tensor_error_metrics_match(case_name: str) -> None:
    reference, actual = ERROR_CASES[case_name]
    metrics = tensor_error_metrics(reference, actual)
    assert_bit_identical(
        "max_abs_error",
        max_abs_error(reference, actual),
        metrics["max_absolute_error"],
    )
    assert_bit_identical(
        "relative_l2_error",
        relative_l2_error(reference, actual),
        metrics["relative_l2_error"],
    )


def test_masked_rows_excluded_from_mean() -> None:
    """Rows with no valid entry must be excluded from the row mean."""

    clean = torch.tensor(
        [[[0.5, 0.5]], [[0.9, 0.1]]], dtype=torch.float64
    )
    noisy = torch.tensor(
        [[[0.5, 0.5]], [[0.1, 0.9]]], dtype=torch.float64
    )
    valid = torch.tensor([[[False, False]], [[True, True]]])
    metrics = _attention_metrics(
        clean,
        noisy,
        valid=valid,
        clean_logits=clean,
        noisy_logits=noisy,
        top_k=1,
    )
    assert_bit_identical(
        "kl_divergence",
        kl_divergence(clean, noisy, valid=valid),
        metrics["kl_divergence"],
    )
    assert_bit_identical(
        "js_divergence",
        js_divergence(clean, noisy, valid=valid),
        metrics["js_divergence"],
    )


def test_no_valid_mask_defaults_to_all_valid() -> None:
    clean = torch.tensor([[0.7, 0.3]], dtype=torch.float64)
    noisy = torch.tensor([[0.4, 0.6]], dtype=torch.float64)
    all_valid = torch.ones_like(clean, dtype=torch.bool)
    metrics = _attention_metrics(
        clean,
        noisy,
        valid=all_valid,
        clean_logits=clean,
        noisy_logits=noisy,
        top_k=2,
    )
    assert_bit_identical(
        "kl_divergence",
        kl_divergence(clean, noisy),
        metrics["kl_divergence"],
    )
    assert_bit_identical(
        "js_divergence",
        js_divergence(clean, noisy),
        metrics["js_divergence"],
    )
