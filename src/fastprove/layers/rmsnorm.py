"""FP32 RMSNorm statistics and RoPE/QK covariance helpers."""

from __future__ import annotations

from typing import Tuple

import torch


def rms_no_gamma_fp32(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply gamma-free RMS normalization with FP32 reductions."""

    if x.ndim < 1:
        raise ValueError("RMSNorm input must have at least one dimension")
    if eps <= 0:
        raise ValueError("eps must be positive")
    x_fp32 = x.to(dtype=torch.float32)
    inverse_rms = torch.rsqrt(
        x_fp32.square().mean(dim=-1, keepdim=True) + float(eps)
    )
    return (x_fp32 * inverse_rms).to(dtype=x.dtype)


def rms_norm_fp32(
    x: torch.Tensor, gamma: torch.Tensor, eps: float
) -> torch.Tensor:
    """Apply learned RMSNorm with FP32 statistics and multiplication."""

    if gamma.ndim != 1 or gamma.shape[0] != x.shape[-1]:
        raise ValueError("gamma shape must match the hidden dimension")
    normalized = rms_no_gamma_fp32(x, eps).to(dtype=torch.float32)
    result = normalized * gamma.to(device=x.device, dtype=torch.float32)
    return result.to(dtype=x.dtype)


def absorbed_rms_projection(
    rotation: torch.Tensor,
    gamma: torch.Tensor,
    weight_math: torch.Tensor,
) -> torch.Tensor:
    """Return ``R.T @ diag(gamma) @ W`` in row-vector layout."""

    if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError("rotation must be square")
    hidden_dim = rotation.shape[0]
    if gamma.shape != (hidden_dim,):
        raise ValueError("gamma shape mismatch")
    if weight_math.ndim != 2 or weight_math.shape[0] != hidden_dim:
        raise ValueError("weight_math input dimension mismatch")
    dtype = weight_math.dtype
    device = weight_math.device
    rotation_fp32 = rotation.to(device=device, dtype=torch.float32)
    gamma_fp32 = gamma.to(device=device, dtype=torch.float32)
    weight_fp32 = weight_math.to(dtype=torch.float32)
    absorbed = rotation_fp32.T @ (gamma_fp32[:, None] * weight_fp32)
    return absorbed.to(dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    theta: float,
) -> torch.Tensor:
    """Apply Llama-style rotary position embedding in FP32."""

    if x.ndim != 4:
        raise ValueError("RoPE input must have shape [batch, heads, seq, dim]")
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("RoPE head dimension must be even")
    if theta <= 0:
        raise ValueError("theta must be positive")
    if positions.ndim not in (1, 2):
        raise ValueError("positions must have shape [seq] or [batch, seq]")
    if positions.shape[-1] != x.shape[-2]:
        raise ValueError("position count must match sequence length")
    if positions.ndim == 2 and positions.shape[0] != x.shape[0]:
        raise ValueError("batched positions must match batch size")

    device = x.device
    exponent = torch.arange(
        0, head_dim, 2, device=device, dtype=torch.float32
    ) / float(head_dim)
    inverse_frequency = 1.0 / (float(theta) ** exponent)
    frequencies = positions.to(device=device, dtype=torch.float32)[..., None]
    frequencies = frequencies * inverse_frequency
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    if positions.ndim == 1:
        cosine = embedding.cos()[None, None, :, :]
        sine = embedding.sin()[None, None, :, :]
    else:
        cosine = embedding.cos()[:, None, :, :]
        sine = embedding.sin()[:, None, :, :]
    x_fp32 = x.to(dtype=torch.float32)
    result = x_fp32 * cosine + _rotate_half(x_fp32) * sine
    return result.to(dtype=x.dtype)


def apply_qk_orthogonal_after_rope(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    common_by_kv_head: torch.Tensor,
    kv_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply one common post-RoPE orthogonal map per GQA KV group."""

    if q_rope.ndim != 4 or k_rope.ndim != 4:
        raise ValueError("q and k must have shape [batch, heads, seq, dim]")
    if q_rope.shape[0] != k_rope.shape[0]:
        raise ValueError("q and k batch sizes must match")
    if q_rope.shape[-1] != k_rope.shape[-1]:
        raise ValueError("q and k head dimensions must match")
    q_heads = q_rope.shape[1]
    kv_heads = k_rope.shape[1]
    head_dim = q_rope.shape[-1]
    if kv_index.shape != (q_heads,):
        raise ValueError("kv_index must map every query head")
    if kv_index.dtype not in (torch.int32, torch.int64):
        raise ValueError("kv_index must be integral")
    if torch.any(kv_index < 0) or torch.any(kv_index >= kv_heads):
        raise ValueError("kv_index contains an invalid KV head")
    if common_by_kv_head.shape != (kv_heads, head_dim, head_dim):
        raise ValueError("common Q/K transform shape mismatch")

    common = common_by_kv_head.to(
        device=q_rope.device, dtype=q_rope.dtype
    )
    q_common = common[kv_index.to(device=common.device)]
    q_prime = torch.einsum("bhqd,hde->bhqe", q_rope, q_common)
    k_prime = torch.einsum("bhkd,hde->bhke", k_rope, common)
    return q_prime, k_prime

