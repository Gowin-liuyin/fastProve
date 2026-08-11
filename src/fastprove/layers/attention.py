"""Exact and bounded-approximate FP32 attention reference kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..seed import RequestContext


class AttentionMode(str, Enum):
    """Supported attention execution modes."""

    PLAINTEXT = "plaintext"
    EXACT = "exact"
    TOPK_PRESERVING = "topk_preserving"
    FREE_BOUNDED = "free_bounded"


@dataclass(frozen=True)
class ApproximationConfig:
    """Bounded logit-noise configuration."""

    tau_max: float
    tau_error: float
    alpha: float
    preserve_top_k: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.tau_max) or not math.isfinite(self.tau_error):
            raise ValueError("tau bounds must be finite")
        if self.tau_max < 0 or self.tau_error < 0:
            raise ValueError("tau bounds must be non-negative")
        if not math.isfinite(self.alpha):
            raise ValueError("alpha must be finite")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be strictly between zero and one")
        if self.preserve_top_k < 1:
            raise ValueError("preserve_top_k must be positive")


@dataclass(frozen=True)
class AttentionDebug:
    """Diagnostics returned only by the explicitly gated debug API."""

    clean_logits: torch.Tensor
    noisy_logits: torch.Tensor
    clean_probabilities: torch.Tensor
    noisy_probabilities: torch.Tensor
    valid_mask: torch.Tensor
    noise: torch.Tensor
    tau: torch.Tensor
    margin: torch.Tensor
    clean_output: torch.Tensor


def _expand_valid_mask(
    valid_mask: torch.Tensor, logits: torch.Tensor
) -> torch.Tensor:
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean")
    if valid_mask.ndim == 3:
        valid_mask = valid_mask[:, None, :, :]
    if valid_mask.ndim != 4:
        raise ValueError("valid_mask must have rank three or four")
    try:
        return torch.broadcast_to(valid_mask, logits.shape)
    except RuntimeError as exc:
        raise ValueError("valid_mask is not broadcastable to logits") from exc


def safe_masked_softmax_fp32(
    logits: torch.Tensor, valid_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask with exact ``-inf`` and return zero for fully masked rows."""

    logits_fp32 = logits.to(dtype=torch.float32)
    valid = _expand_valid_mask(valid_mask, logits_fp32)
    masked_logits = logits_fp32.masked_fill(~valid, -torch.inf)
    any_valid = valid.any(dim=-1, keepdim=True)
    row_max = masked_logits.amax(dim=-1, keepdim=True)
    row_max = torch.where(any_valid, row_max, torch.zeros_like(row_max))
    exponentials = torch.exp(masked_logits - row_max)
    exponentials = torch.where(valid, exponentials, torch.zeros_like(exponentials))
    denominator = exponentials.sum(dim=-1, keepdim=True)
    probabilities = torch.where(
        denominator > 0,
        exponentials / denominator.clamp_min(torch.finfo(torch.float32).tiny),
        torch.zeros_like(exponentials),
    )
    return probabilities, masked_logits


def compute_topk_budget(
    clean_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    config: ApproximationConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-query Top-k margin and strict bounded-noise budget."""

    logits = clean_logits.to(dtype=torch.float32)
    valid = _expand_valid_mask(valid_mask, logits)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_valid = valid.reshape(-1, valid.shape[-1])
    flat_budget = torch.zeros(
        flat_logits.shape[0], 1, dtype=torch.float32, device=logits.device
    )
    flat_margin = torch.zeros_like(flat_budget)
    k = config.preserve_top_k
    if flat_logits.shape[-1] > k:
        masked = flat_logits.masked_fill(~flat_valid, -torch.inf)
        sorted_values = torch.sort(masked, dim=-1, descending=True, stable=True).values
        raw_margin = sorted_values[:, k - 1] - sorted_values[:, k]
        valid_count = flat_valid.sum(dim=-1)
        usable = (valid_count > k) & torch.isfinite(raw_margin) & (raw_margin > 0)
        flat_margin[:, 0] = torch.where(usable, raw_margin, torch.zeros_like(raw_margin))
        margin_budget = float(config.alpha) * flat_margin[:, 0] / 2.0
        cap_max = torch.full_like(margin_budget, float(config.tau_max))
        cap_error = torch.full_like(margin_budget, float(config.tau_error))
        flat_budget[:, 0] = torch.where(
            usable,
            torch.minimum(torch.minimum(cap_max, cap_error), margin_budget),
            torch.zeros_like(margin_budget),
        )
    output_shape = logits.shape[:-1] + (1,)
    return (
        flat_budget.reshape(output_shape),
        flat_margin.reshape(output_shape),
    )


def _coordinate_value(
    context: RequestContext,
    layer_id: str,
    batch_index: int,
    head_index: int,
    query_position: int,
    key_position: int,
) -> float:
    seed = context.seed_for(
        "attention-logit-noise",
        layer_id,
        batch_index,
        head_index,
        query_position,
        key_position,
    )
    mantissa = seed & ((1 << 53) - 1)
    return 2.0 * (mantissa / float(1 << 53)) - 1.0


def _sample_bounded_noise(
    *,
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    tau: torch.Tensor,
    context: RequestContext,
    layer_id: str,
    query_positions: Optional[torch.Tensor],
    key_positions: Optional[torch.Tensor],
) -> torch.Tensor:
    """Generate coordinate-stable bounded noise without Python O(BHQK) loops.

    The hash is deliberately a reproducibility primitive, not a cryptographic
    PRF.  Coordinates use absolute query/key positions, so a cached decode and
    a full-prefix evaluation see the same noise for the same logical pair.
    The integer arithmetic is performed on CPU to keep the sequence identical
    across CPU/CUDA/MPS; only one dense tensor transfer is performed.
    """

    valid = _expand_valid_mask(valid_mask, logits)
    batch, heads, query_count, key_count = logits.shape
    if query_positions is None:
        query_positions = torch.arange(query_count, dtype=torch.long)
    if key_positions is None:
        key_positions = torch.arange(key_count, dtype=torch.long)
    if query_positions.shape != (query_count,):
        raise ValueError("query_positions must have shape [query_count]")
    if key_positions.shape != (key_count,):
        raise ValueError("key_positions must have shape [key_count]")
    seed = context.seed_for("attention-logit-noise", layer_id)
    cpu = torch.device("cpu")
    batch_index = torch.arange(batch, dtype=torch.int64, device=cpu).view(
        batch, 1, 1, 1
    )
    head_index = torch.arange(heads, dtype=torch.int64, device=cpu).view(
        1, heads, 1, 1
    )
    q_index = query_positions.detach().to(device=cpu, dtype=torch.int64).view(
        1, 1, query_count, 1
    )
    k_index = key_positions.detach().to(device=cpu, dtype=torch.int64).view(
        1, 1, 1, key_count
    )
    # LCG + xor-mix: fast, deterministic, and sufficient for a bounded-noise
    # experiment.  It must not be described as cryptographic randomness.
    coordinate = (
        torch.full((1, 1, 1, 1), seed, dtype=torch.int64, device=cpu)
        + batch_index * 6364136223846793005
        + head_index * 1442695040888963407
        + q_index * 3202034522624059733
        + k_index * 3935559000370003845
    )
    coordinate = coordinate * 6364136223846793005 + 1442695040888963407
    coordinate = coordinate ^ (coordinate >> 33)
    coordinate = coordinate * 3202034522624059733 + 1442695040888963407
    coordinate = coordinate ^ (coordinate >> 29)
    mantissa = torch.bitwise_and(coordinate, (1 << 53) - 1)
    raw = (2.0 * mantissa.to(dtype=torch.float64) / float(1 << 53) - 1.0).to(
        dtype=torch.float32
    )
    raw = raw.to(device=logits.device)
    valid_float = valid.to(dtype=torch.float32)
    counts = valid_float.sum(dim=-1, keepdim=True)
    row_mean = (raw * valid_float).sum(dim=-1, keepdim=True) / counts.clamp_min(1)
    centered = torch.where(valid, raw - row_mean, torch.zeros_like(raw))
    maximum = centered.abs().amax(dim=-1, keepdim=True)
    unit = torch.where(
        maximum > 0,
        centered / maximum.clamp_min(torch.finfo(torch.float32).tiny),
        torch.zeros_like(centered),
    )
    # ``torch.tensor(0.001, dtype=float32)`` is the first representable
    # value *above* the Python float 0.001.  Multiplying by that value can
    # therefore make the observed infinity norm exceed the configured bound
    # by one ulp (which is still a real contract violation for the recorded
    # experiment).  Use the representable predecessor toward zero before the
    # multiply.  ``nextafter`` is supported by the reference CPU/MPS paths
    # and keeps zero exactly zero.
    tau_fp32 = tau.to(device=logits.device, dtype=torch.float32)
    safe_tau = torch.nextafter(tau_fp32, torch.zeros_like(tau_fp32))
    noise = unit * safe_tau
    return torch.where(valid, noise, torch.zeros_like(noise))


class ObfuscatedAttention(nn.Module):
    """Unified plaintext, exact, and approximate attention interface."""

    def __init__(
        self,
        *,
        mode: AttentionMode,
        approximation: Optional[ApproximationConfig],
        layer_id: str,
        debug_enabled: bool,
    ) -> None:
        super().__init__()
        self.mode = AttentionMode(mode)
        if self.mode in (
            AttentionMode.TOPK_PRESERVING,
            AttentionMode.FREE_BOUNDED,
        ):
            if approximation is None:
                raise ValueError("approximate mode requires a noise configuration")
        elif approximation is not None:
            raise ValueError("plaintext/exact mode must not receive noise config")
        self.approximation = approximation
        self.layer_id = str(layer_id)
        self.debug_enabled = bool(debug_enabled)

    def _run(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        value_mixed: torch.Tensor,
        valid_mask: torch.Tensor,
        kv_index: torch.Tensor,
        request_context: RequestContext,
        query_positions: Optional[torch.Tensor],
        key_positions: Optional[torch.Tensor],
        return_debug: bool,
    ) -> Tuple[torch.Tensor, Optional[AttentionDebug]]:
        if q.ndim != 4 or k.ndim != 4 or value_mixed.ndim != 4:
            raise ValueError("q, k, and value_mixed must be rank four")
        batch, q_heads, query_count, head_dim = q.shape
        if k.shape[:2] != (batch, value_mixed.shape[1]):
            raise ValueError("K and Value batch/KV-head shapes must match")
        if k.shape[2] != value_mixed.shape[2]:
            raise ValueError("K and Value sequence lengths must match")
        if k.shape[-1] != head_dim:
            raise ValueError("Q/K head dimensions must match")
        if kv_index.shape != (q_heads,):
            raise ValueError("kv_index must map every query head")
        if torch.any(kv_index < 0) or torch.any(kv_index >= k.shape[1]):
            raise ValueError("kv_index contains an invalid KV head")
        index = kv_index.to(device=k.device)
        repeated_k = k[:, index]
        repeated_value = value_mixed[:, index]
        clean_scores = torch.einsum(
            "bhqd,bhkd->bhqk", q.float(), repeated_k.float()
        ) / math.sqrt(head_dim)
        valid = _expand_valid_mask(valid_mask, clean_scores)
        clean_probabilities: Optional[torch.Tensor] = None
        clean_logits: torch.Tensor
        if self.mode in (AttentionMode.PLAINTEXT, AttentionMode.EXACT) or return_debug:
            clean_probabilities, clean_logits = safe_masked_softmax_fp32(
                clean_scores, valid
            )
        else:
            # Approximate production only needs masked FP32 logits for the
            # margin/budget calculation.  Avoid an unnecessary clean Softmax
            # reduction and probability tensor when debug is disabled.
            clean_logits = clean_scores.to(dtype=torch.float32).masked_fill(
                ~valid, -torch.inf
            )
        margin = torch.zeros_like(clean_logits[..., :1])
        tau = torch.zeros_like(margin)
        noise = torch.zeros_like(clean_logits)
        if self.mode == AttentionMode.TOPK_PRESERVING:
            assert self.approximation is not None
            tau, margin = compute_topk_budget(
                clean_logits, valid, self.approximation
            )
            noise = _sample_bounded_noise(
                logits=clean_logits,
                valid_mask=valid,
                tau=tau,
                context=request_context,
                layer_id=self.layer_id,
                query_positions=query_positions,
                key_positions=key_positions,
            )
        elif self.mode == AttentionMode.FREE_BOUNDED:
            assert self.approximation is not None
            fixed_tau = min(
                self.approximation.tau_max, self.approximation.tau_error
            )
            any_valid = valid.any(dim=-1, keepdim=True)
            tau = torch.where(
                any_valid,
                torch.full_like(tau, float(fixed_tau)),
                torch.zeros_like(tau),
            )
            noise = _sample_bounded_noise(
                logits=clean_logits,
                valid_mask=valid,
                tau=tau,
                context=request_context,
                layer_id=self.layer_id,
                query_positions=query_positions,
                key_positions=key_positions,
            )
        noisy_scores = clean_logits + noise
        if self.mode in (AttentionMode.PLAINTEXT, AttentionMode.EXACT):
            assert clean_probabilities is not None
            noisy_probabilities = clean_probabilities
            noisy_logits = clean_logits
        else:
            noisy_probabilities, noisy_logits = safe_masked_softmax_fp32(
                noisy_scores, valid
            )
        output_fp32 = torch.einsum(
            "bhqk,bhkd->bhqd", noisy_probabilities, repeated_value.float()
        )
        output = output_fp32.to(dtype=value_mixed.dtype)
        debug: Optional[AttentionDebug] = None
        if return_debug:
            assert clean_probabilities is not None
            clean_output_fp32 = torch.einsum(
                "bhqk,bhkd->bhqd",
                clean_probabilities,
                repeated_value.float(),
            )
            debug = AttentionDebug(
                clean_logits=clean_logits,
                noisy_logits=noisy_logits,
                clean_probabilities=clean_probabilities,
                noisy_probabilities=noisy_probabilities,
                valid_mask=valid,
                noise=noise,
                tau=tau,
                margin=margin,
                clean_output=clean_output_fp32.to(dtype=value_mixed.dtype),
            )
        return output, debug

    def forward(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        value_mixed: torch.Tensor,
        valid_mask: torch.Tensor,
        kv_index: torch.Tensor,
        request_context: RequestContext,
        query_positions: Optional[torch.Tensor] = None,
        key_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return only mixed attention output in the production API."""

        output, _ = self._run(
            q=q,
            k=k,
            value_mixed=value_mixed,
            valid_mask=valid_mask,
            kv_index=kv_index,
            request_context=request_context,
            query_positions=query_positions,
            key_positions=key_positions,
            return_debug=False,
        )
        return output

    def forward_debug(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        value_mixed: torch.Tensor,
        valid_mask: torch.Tensor,
        kv_index: torch.Tensor,
        request_context: RequestContext,
        query_positions: Optional[torch.Tensor] = None,
        key_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, AttentionDebug]:
        """Return diagnostics only when explicitly enabled for tests."""

        if not self.debug_enabled:
            raise PermissionError("attention debug API is disabled")
        return self._run(
            q=q,
            k=k,
            value_mixed=value_mixed,
            valid_mask=valid_mask,
            kv_index=kv_index,
            request_context=request_context,
            query_positions=query_positions,
            key_positions=key_positions,
            return_debug=True,
        )
