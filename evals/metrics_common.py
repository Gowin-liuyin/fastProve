"""Shared numerical helpers for the five-layer evaluation suite."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from fastprove.evaluation.metrics import (
    js_divergence as _canonical_js_divergence,
    kl_divergence as _canonical_kl_divergence,
    max_abs_error as _canonical_max_abs_error,
    relative_l2_error as _canonical_relative_l2_error,
    tensor_error_metrics,
    topk_row_metrics,
)


def max_abs_error(reference: torch.Tensor, actual: torch.Tensor) -> float:
    """Maximum absolute FP64 error; delegates to fastprove."""

    return _canonical_max_abs_error(reference, actual)


def relative_l2_error(
    reference: torch.Tensor, actual: torch.Tensor, *, eps: float = 1e-12
) -> float:
    """Relative L2 error with denominator floor ``eps``; delegates to fastprove."""

    return _canonical_relative_l2_error(reference, actual, eps=eps)


def infinity_and_relative(
    reference: torch.Tensor, actual: torch.Tensor, *, eps: float = 1e-12
) -> Dict[str, float]:
    """E_{ℓ,∞} and E_{ℓ,rel} from protocol §4 Layer 1."""

    return {
        "max_absolute_error": max_abs_error(reference, actual),
        "relative_l2_error": relative_l2_error(reference, actual, eps=eps),
        **{
            k: v
            for k, v in tensor_error_metrics(reference, actual).items()
            if k
            not in (
                "max_absolute_error",
                "relative_l2_error",
            )
        },
    }


def kl_divergence(
    p: torch.Tensor, q: torch.Tensor, *, valid: Optional[torch.Tensor] = None
) -> float:
    """Mean row-wise KL(p ‖ q) over valid positions; delegates to fastprove."""

    return _canonical_kl_divergence(p, q, valid_mask=valid)


def js_divergence(
    p: torch.Tensor, q: torch.Tensor, *, valid: Optional[torch.Tensor] = None
) -> float:
    """Mean row-wise Jensen–Shannon divergence; delegates to fastprove."""

    return _canonical_js_divergence(p, q, valid_mask=valid)


def total_variation(
    p: torch.Tensor, q: torch.Tensor, *, valid: Optional[torch.Tensor] = None
) -> float:
    """Mean row-wise total variation distance ½‖p−q‖₁."""

    p64 = p.detach().double()
    q64 = q.detach().double()
    if valid is None:
        valid = torch.ones_like(p64, dtype=torch.bool)
    diff = torch.where(valid, (p64 - q64).abs(), torch.zeros_like(p64))
    row = 0.5 * diff.sum(dim=-1)
    row_valid = valid.any(dim=-1)
    if not row_valid.any():
        return 0.0
    return float(row[row_valid].mean().item())


def topk_set_overlap(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    *,
    k: int,
    valid: Optional[torch.Tensor] = None,
) -> Tuple[float, float, float]:
    """Return (top1_match, topk_overlap, rank_flip_rate) over valid rows.

    Rank-flip rate: fraction of valid pairs (i,j) whose relative order
    disagrees between clean and noisy scores. For the hard gate we also
    expose a simpler top-1 flip rate separately.
    """

    if clean.shape != noisy.shape:
        raise ValueError("score shapes must match")
    flat_c = clean.reshape(-1, clean.shape[-1]).double()
    flat_n = noisy.reshape(-1, noisy.shape[-1]).double()
    if valid is None:
        flat_v = torch.ones_like(flat_c, dtype=torch.bool)
    else:
        flat_v = valid.reshape(-1, valid.shape[-1])

    top1_matches = []
    overlaps = []
    flip_rates = []
    for row in range(flat_c.shape[0]):
        mask = flat_v[row]
        if not mask.any():
            continue
        c = flat_c[row][mask]
        n = flat_n[row][mask]
        # Map local indices back for set overlap only within valid cols.
        c_top1 = int(torch.argmax(c).item())
        n_top1 = int(torch.argmax(n).item())
        top1_matches.append(1.0 if c_top1 == n_top1 else 0.0)
        overlap, _changed = topk_row_metrics(c, n, top_k=k)
        overlaps.append(overlap)
        # Pairwise rank flips among valid positions.
        if c.numel() >= 2:
            # Order disagreement: sign of pairwise differences.
            c_diff = c.unsqueeze(0) - c.unsqueeze(1)
            n_diff = n.unsqueeze(0) - n.unsqueeze(1)
            # Upper triangle only.
            iu = torch.triu(
                torch.ones(c.numel(), c.numel(), dtype=torch.bool), diagonal=1
            )
            # Flip when product of diffs is negative (strict order change).
            flips = ((c_diff * n_diff) < 0) & iu
            comparable = (c_diff != 0) & iu
            if comparable.any():
                flip_rates.append(
                    float(flips.sum().item() / comparable.sum().item())
                )
            else:
                flip_rates.append(0.0)
        else:
            flip_rates.append(0.0)

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else 1.0

    return _mean(top1_matches), _mean(overlaps), _mean(flip_rates)


def top1_margin(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Per-row margin m = S_(1) − S_(2) on valid positions; shape [..., 1]."""

    shape = scores.shape
    flat = scores.reshape(-1, shape[-1]).float()
    flat_v = valid.reshape(-1, shape[-1])
    margins = torch.zeros(flat.shape[0], 1, dtype=torch.float32, device=scores.device)
    for i in range(flat.shape[0]):
        vals = flat[i][flat_v[i]]
        if vals.numel() >= 2:
            top2 = torch.topk(vals, 2, largest=True).values
            margins[i, 0] = top2[0] - top2[1]
        elif vals.numel() == 1:
            margins[i, 0] = float("inf")
    return margins.reshape(*shape[:-1], 1)


def score_max_abs_error(
    clean: torch.Tensor, noisy: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Per-row ε^S = max_{j∈V} |S̃ − S|; shape [..., 1]."""

    diff = (clean.float() - noisy.float()).abs()
    diff = torch.where(valid, diff, torch.zeros_like(diff))
    # For fully masked rows, max is 0.
    row_max = diff.amax(dim=-1, keepdim=True)
    return row_max


def margin_risk_rate(
    margin: torch.Tensor, eps: torch.Tensor
) -> float:
    """Fraction of rows where m ≤ 2ε (theorem risk region)."""

    m = margin.reshape(-1).float()
    e = eps.reshape(-1).float()
    finite = torch.isfinite(m) & torch.isfinite(e)
    if not finite.any():
        return 0.0
    risk = (m[finite] <= 2.0 * e[finite]).float().mean()
    return float(risk.item())


def softmax_probs(logits: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Stable row-wise softmax; masked positions get zero mass."""

    x = logits.float()
    if valid is not None:
        x = x.masked_fill(~valid, -torch.inf)
    any_valid = (
        valid.any(dim=-1, keepdim=True)
        if valid is not None
        else torch.ones(*x.shape[:-1], 1, dtype=torch.bool, device=x.device)
    )
    row_max = x.amax(dim=-1, keepdim=True)
    row_max = torch.where(any_valid, row_max, torch.zeros_like(row_max))
    exp = torch.exp(x - row_max)
    if valid is not None:
        exp = torch.where(valid, exp, torch.zeros_like(exp))
    denom = exp.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    probs = exp / denom
    return torch.where(any_valid, probs, torch.zeros_like(probs))
