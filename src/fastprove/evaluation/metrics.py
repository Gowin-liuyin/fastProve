"""Numerically explicit correctness and Softmax diagnostics."""

from __future__ import annotations

import math
from typing import Dict, Optional, Union

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


def _stable_descending_ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, descending=True, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(
        values.numel(), dtype=torch.float64, device=values.device
    )
    return ranks


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = torch.linalg.vector_norm(
        left_centered
    ) * torch.linalg.vector_norm(right_centered)
    if denominator.item() == 0:
        return 1.0 if torch.equal(left, right) else 0.0
    return float(torch.dot(left_centered, right_centered).div(denominator).item())


def attention_distribution_metrics(
    *,
    clean_probabilities: torch.Tensor,
    noisy_probabilities: torch.Tensor,
    clean_logits: torch.Tensor,
    noisy_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    noise: torch.Tensor,
    margin: torch.Tensor,
    top_k: int,
    clean_output: Optional[torch.Tensor] = None,
    noisy_output: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute divergences, Top-k/rank, noise, margin, and output metrics."""

    shape = clean_probabilities.shape
    tensors = (
        noisy_probabilities,
        clean_logits,
        noisy_logits,
        valid_mask,
        noise,
    )
    if any(tensor.shape != shape for tensor in tensors):
        raise ValueError("attention tensors must have identical shapes")
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean")
    if margin.shape != shape[:-1] + (1,):
        raise ValueError("margin must have one value per query")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    # This function only produces scalar diagnostics.  Keep all rank/divergence
    # reductions on CPU so callers may pass debug tensors from MPS/CUDA without
    # invoking unsupported accelerator FP64 kernels.
    clean_probabilities = clean_probabilities.detach().to(device="cpu")
    noisy_probabilities = noisy_probabilities.detach().to(device="cpu")
    clean_logits = clean_logits.detach().to(device="cpu")
    noisy_logits = noisy_logits.detach().to(device="cpu")
    valid_mask = valid_mask.detach().to(device="cpu")
    noise = noise.detach().to(device="cpu")
    margin = margin.detach().to(device="cpu")
    if clean_output is not None:
        clean_output = clean_output.detach().to(device="cpu")
    if noisy_output is not None:
        noisy_output = noisy_output.detach().to(device="cpu")

    valid = valid_mask
    p = clean_probabilities.to(dtype=torch.float64)
    q = noisy_probabilities.to(dtype=torch.float64)
    tiny = torch.finfo(torch.float64).tiny
    p_safe = p.clamp_min(tiny)
    q_safe = q.clamp_min(tiny)
    kl_terms = torch.where(
        valid & (p > 0), p * (p_safe.log() - q_safe.log()), torch.zeros_like(p)
    )
    row_kl = kl_terms.sum(dim=-1)
    mixture = 0.5 * (p + q)
    mixture_safe = mixture.clamp_min(tiny)
    p_to_m = torch.where(
        valid & (p > 0),
        p * (p_safe.log() - mixture_safe.log()),
        torch.zeros_like(p),
    ).sum(dim=-1)
    q_to_m = torch.where(
        valid & (q > 0),
        q * (q_safe.log() - mixture_safe.log()),
        torch.zeros_like(q),
    ).sum(dim=-1)
    row_js = 0.5 * (p_to_m + q_to_m)

    flat_clean = clean_logits.reshape(-1, shape[-1])
    flat_noisy = noisy_logits.reshape(-1, shape[-1])
    flat_valid = valid.reshape(-1, shape[-1])
    overlaps = []
    changes = []
    correlations = []
    for row_index in range(flat_clean.shape[0]):
        mask = flat_valid[row_index]
        clean_values = flat_clean[row_index][mask].to(dtype=torch.float64)
        noisy_values = flat_noisy[row_index][mask].to(dtype=torch.float64)
        count = clean_values.numel()
        if count == 0:
            continue
        effective_k = min(top_k, count)
        clean_top = torch.topk(
            clean_values, effective_k, largest=True, sorted=False
        ).indices
        noisy_top = torch.topk(
            noisy_values, effective_k, largest=True, sorted=False
        ).indices
        clean_set = set(int(item) for item in clean_top.cpu().tolist())
        noisy_set = set(int(item) for item in noisy_top.cpu().tolist())
        overlap = len(clean_set.intersection(noisy_set)) / float(effective_k)
        overlaps.append(overlap)
        changes.append(0.0 if clean_set == noisy_set else 1.0)
        if count >= 2:
            correlations.append(
                _pearson(
                    _stable_descending_ranks(clean_values),
                    _stable_descending_ranks(noisy_values),
                )
            )

    masked_noise = torch.where(valid, noise, torch.zeros_like(noise))
    row_noise_norm = masked_noise.abs().amax(dim=-1)
    valid_rows = valid.any(dim=-1)
    if valid_rows.any():
        actual_noise = float(row_noise_norm[valid_rows].max().item())
        zero_fraction = float(
            (row_noise_norm[valid_rows] <= 1e-12)
            .to(dtype=torch.float64)
            .mean()
            .item()
        )
        kl_value = float(row_kl[valid_rows].mean().item())
        js_value = float(row_js[valid_rows].mean().item())
    else:
        actual_noise = 0.0
        zero_fraction = 1.0
        kl_value = 0.0
        js_value = 0.0

    margin_values = margin.to(dtype=torch.float64).reshape(-1)
    output_relative = 0.0
    if (clean_output is None) != (noisy_output is None):
        raise ValueError("clean_output and noisy_output must be provided together")
    if clean_output is not None and noisy_output is not None:
        if clean_output.shape != noisy_output.shape:
            raise ValueError("attention output shapes must match")
        difference_norm = torch.linalg.vector_norm(
            (noisy_output.double() - clean_output.double()).reshape(-1)
        )
        clean_norm = torch.linalg.vector_norm(clean_output.double().reshape(-1))
        if clean_norm.item() == 0:
            output_relative = (
                0.0 if difference_norm.item() == 0 else math.inf
            )
        else:
            output_relative = float((difference_norm / clean_norm).item())

    return {
        "kl_divergence": kl_value,
        "js_divergence": js_value,
        "topk_overlap": float(sum(overlaps) / len(overlaps))
        if overlaps
        else 1.0,
        "topk_changed_fraction": float(sum(changes) / len(changes))
        if changes
        else 0.0,
        "rank_correlation": float(sum(correlations) / len(correlations))
        if correlations
        else 1.0,
        "actual_noise_infinity_norm": actual_noise,
        "zero_noise_query_fraction": zero_fraction,
        "clean_boundary_margin_mean": float(margin_values.mean().item())
        if margin_values.numel()
        else 0.0,
        "clean_boundary_margin_min": float(margin_values.min().item())
        if margin_values.numel()
        else 0.0,
        "clean_boundary_margin_max": float(margin_values.max().item())
        if margin_values.numel()
        else 0.0,
        "attention_output_relative_l2_error": output_relative,
    }
