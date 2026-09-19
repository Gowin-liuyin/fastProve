"""Shared dataclasses and mixed-state helpers for obfuscated models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Tuple

import torch

from ..seed import make_generator
from ..structured import gram_rms_scale

if TYPE_CHECKING:
    from ..codec import TokenCodec
    from .client import ObfuscatedBlockClient

    # ``ConvertedObfuscatedBlock`` and ``ConvertedObfuscatedLM`` reference the
    # runtime classes defined in ``fastprove.models.obfuscated``. That module
    # imports this one, so the forward references stay string-annotated and
    # are aliased here rather than importing the module back.
    ObfuscatedDecoderBlock = Any
    ObfuscatedTinyCausalLM = Any


@dataclass(frozen=True)
class ConvertedObfuscatedBlock:
    """Converted module, separate client helper, and conversion timing."""

    module: "ObfuscatedDecoderBlock"
    client: ObfuscatedBlockClient
    conversion_time_seconds: float


@dataclass(frozen=True)
class ObfuscatedBlockDebug:
    """Diagnostics returned only by :meth:`forward_debug`."""

    qk_score_error: dict
    softmax_error: dict
    clean_logits: torch.Tensor
    noisy_logits: torch.Tensor
    clean_probabilities: torch.Tensor
    noisy_probabilities: torch.Tensor
    valid_mask: torch.Tensor
    logit_noise: torch.Tensor
    tau: torch.Tensor
    margin: torch.Tensor
    clean_attention_output: torch.Tensor
    attention_output: torch.Tensor
    post_attention: torch.Tensor
    attention_noise_state: torch.Tensor
    z_prime: torch.Tensor
    final_noise_state: torch.Tensor


@dataclass(frozen=True)
class ObfuscatedKVCache:
    """Transformed Key and mixed Value state for incremental attention."""

    key: torch.Tensor
    value_mixed: torch.Tensor
    key_valid: torch.Tensor
    positions: torch.Tensor
    cache_identity: str = ""
    request_seed: int = 0
    request_id: str = ""


@dataclass(frozen=True)
class ObfuscatedLMCache:
    """One transformed KV cache per decoder layer."""

    layers: Tuple[ObfuscatedKVCache, ...]


@dataclass(frozen=True)
class ConvertedObfuscatedLM:
    """Converted tiny LM, separate client helper, codec, and timing."""

    module: "ObfuscatedTinyCausalLM"
    client: ObfuscatedBlockClient
    token_codec: TokenCodec
    conversion_time_seconds: float


def _random_matrix(
    rows: int,
    columns: int,
    *,
    seed: int,
    domain: str,
    scale: float,
) -> torch.Tensor:
    generator = make_generator(seed, domain, rows, columns)
    return (
        torch.randn(rows, columns, generator=generator, dtype=torch.float32)
        * scale
    )


def _cache_tensor_digest(*tensors: torch.Tensor) -> str:
    """Return a stable identity for tensors that define a KV layout."""

    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _rms_scale_from_factors(
    signal_dim: int,
    gram_blocks: torch.Tensor,
    gram_perm: torch.Tensor,
) -> Any:
    """Build an ``O(n*b)`` RMS-scale callable from the server-side Gram.

    Only a scalar per token is produced; the signal is never reconstructed.
    ``gram_blocks`` / ``gram_perm`` are the server-side deployed artifacts
    (see the converter): the block-diagonal ``A_gram = P P.T`` blocks in
    ``perm_out`` order.
    """

    def rms_scale(mixed: torch.Tensor, eps: float) -> torch.Tensor:
        return gram_rms_scale(
            mixed,
            signal_dim,
            gram_blocks,
            gram_perm,
            eps,
            width_error="mixed width does not match the Gram artifacts",
        )

    return rms_scale
