"""Dependency-free tensor error metrics.

Kept out of ``fastprove.evaluation`` so that ``models/obfuscated.py`` can use
it without importing the evaluation package (layering fix, see
docs/CLEANUP_PLAN.md P3). ``fastprove.evaluation.metrics`` re-exports it.
"""

from __future__ import annotations

import math
from typing import Dict, Union

import torch


MetricValue = Union[int, float]

def tensor_error_metrics(
    reference: torch.Tensor, actual: torch.Tensor
) -> Dict[str, MetricValue]:
    """Compute the required tensor-level error and finite-value metrics."""

    if reference.shape != actual.shape:
        raise ValueError("reference and actual shapes must match")
    # Metric reductions are scalar/debug work, not model execution.  Move
    # tensors to CPU before widening to FP64 so the same evaluator works on
    # MPS, whose backend does not implement FP64 tensors.
    reference_fp64 = reference.detach().to(device="cpu").to(dtype=torch.float64)
    actual_fp64 = actual.detach().to(device="cpu").to(dtype=torch.float64)
    nan_count = int(torch.isnan(actual_fp64).sum().item())
    inf_count = int(torch.isinf(actual_fp64).sum().item())
    difference = (actual_fp64 - reference_fp64).abs()
    difference_for_reduction = torch.where(
        torch.isnan(difference),
        torch.full_like(difference, torch.inf),
        difference,
    )
    max_absolute = float(difference_for_reduction.max().item())
    mean_absolute = float(difference_for_reduction.mean().item())
    denominator = torch.linalg.vector_norm(reference_fp64.reshape(-1))
    numerator = torch.linalg.vector_norm(
        (actual_fp64 - reference_fp64).reshape(-1)
    )
    if denominator.item() == 0:
        relative_l2 = 0.0 if numerator.item() == 0 else math.inf
    else:
        relative_l2 = float((numerator / denominator).item())
    if (
        nan_count == 0
        and inf_count == 0
        and torch.equal(reference_fp64, actual_fp64)
    ):
        cosine = 1.0
    else:
        reference_flat = reference_fp64.reshape(-1)
        actual_flat = actual_fp64.reshape(-1)
        reference_norm = torch.linalg.vector_norm(reference_flat)
        actual_norm = torch.linalg.vector_norm(actual_flat)
        if reference_norm.item() == 0 or actual_norm.item() == 0:
            cosine = 1.0 if torch.equal(reference_flat, actual_flat) else 0.0
        else:
            cosine = float(
                torch.dot(reference_flat, actual_flat)
                .div(reference_norm * actual_norm)
                .item()
            )
    return {
        "max_absolute_error": max_absolute,
        "mean_absolute_error": mean_absolute,
        "relative_l2_error": relative_l2,
        "cosine_similarity": cosine,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }
