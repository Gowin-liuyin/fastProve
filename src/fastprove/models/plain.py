"""Canonical plaintext Llama-like decoder block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..layers.attention import safe_masked_softmax_fp32
from ..layers.rmsnorm import apply_rope, rms_norm_fp32
from ..seed import derive_seed


@dataclass(frozen=True)
class PlainKVCache:
    """RoPE-applied K and plaintext V cache."""

    key: torch.Tensor
    value: torch.Tensor
    key_valid: torch.Tensor
    positions: torch.Tensor


@dataclass(frozen=True)
class PlainLMCache:
    """Per-layer plaintext KV caches for production greedy decoding."""

    layers: Tuple[PlainKVCache, ...]


@dataclass(frozen=True)
class PlainBlockDebug:
    """Explicit plaintext diagnostics for correctness tests."""

    qk_scores: torch.Tensor
    masked_logits: torch.Tensor
    probabilities: torch.Tensor
    valid_mask: torch.Tensor
    attention_output: torch.Tensor
    post_attention: torch.Tensor
    final_output: torch.Tensor


class PlainDecoderBlock(nn.Module):
    """Small deterministic Llama-style decoder block."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        seed: int,
        layer_id: int,
        debug_enabled: bool,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = int(layer_id)
        self.debug_enabled = bool(debug_enabled)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(
                derive_seed(seed, "plain-decoder-block", self.layer_id)
            )
            self.attention_norm_weight = nn.Parameter(
                torch.ones(config.hidden_size)
            )
            self.q_proj = nn.Linear(
                config.hidden_size,
                config.num_attention_heads * config.head_dim,
                bias=config.qkv_bias,
            )
            self.k_proj = nn.Linear(
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
                bias=config.qkv_bias,
            )
            self.v_proj = nn.Linear(
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
                bias=config.qkv_bias,
            )
            self.o_proj = nn.Linear(
                config.hidden_size, config.hidden_size, bias=False
            )
            self.ffn_norm_weight = nn.Parameter(
                torch.ones(config.hidden_size)
            )
            self.gate_proj = nn.Linear(
                config.hidden_size, config.intermediate_size, bias=False
            )
            self.up_proj = nn.Linear(
                config.hidden_size, config.intermediate_size, bias=False
            )
            self.down_proj = nn.Linear(
                config.intermediate_size, config.hidden_size, bias=False
            )
        kv_index = torch.arange(config.num_attention_heads) // (
            config.query_heads_per_kv_head
        )
        self.register_buffer("kv_index", kv_index.long(), persistent=True)

    def _positions(
        self, x: torch.Tensor, positions: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if positions is None:
            result = torch.arange(x.shape[1], device=x.device)
        else:
            if positions.shape != (x.shape[1],):
                raise ValueError("positions must have shape [sequence]")
            result = positions.to(device=x.device)
        if torch.any(result < 0) or torch.any(
            result >= self.config.max_sequence_length
        ):
            raise ValueError("position is outside configured context")
        return result.long()

    def _token_mask(
        self, x: torch.Tensor, token_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if token_mask is None:
            return torch.ones(
                x.shape[0], x.shape[1], dtype=torch.bool, device=x.device
            )
        if token_mask.shape != x.shape[:2] or token_mask.dtype != torch.bool:
            raise ValueError("token_mask must be boolean [batch, sequence]")
        return token_mask.to(device=x.device)

    def _project_qkv(
        self, normalized: torch.Tensor, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, _ = normalized.shape
        q = self.q_proj(normalized).view(
            batch,
            sequence,
            self.config.num_attention_heads,
            self.config.head_dim,
        )
        k = self.k_proj(normalized).view(
            batch,
            sequence,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        v = self.v_proj(normalized).view(
            batch,
            sequence,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return (
            apply_rope(q, positions, theta=self.config.rope_theta),
            apply_rope(k, positions, theta=self.config.rope_theta),
            v,
        )

    def _run(
        self,
        x: torch.Tensor,
        *,
        token_mask: Optional[torch.Tensor],
        positions: Optional[torch.Tensor],
        cache: Optional[PlainKVCache],
        use_cache: bool,
    ) -> Tuple[torch.Tensor, PlainBlockDebug, Optional[PlainKVCache]]:
        if x.ndim != 3 or x.shape[-1] != self.config.hidden_size:
            raise ValueError("block input must be [batch, sequence, hidden]")
        current_positions = self._positions(x, positions)
        current_valid = self._token_mask(x, token_mask)
        normalized = rms_norm_fp32(
            x, self.attention_norm_weight, self.config.rms_epsilon
        )
        q, current_k, current_v = self._project_qkv(
            normalized, current_positions
        )
        if cache is None:
            key = current_k
            value = current_v
            key_valid = current_valid
            key_positions = current_positions
        else:
            if cache.key.shape[:2] != (
                x.shape[0],
                self.config.num_key_value_heads,
            ):
                raise ValueError("cache batch/KV-head shape mismatch")
            if cache.key.dtype != current_k.dtype or cache.key.device != x.device:
                raise ValueError("cache dtype/device mismatch")
            key = torch.cat((cache.key, current_k), dim=2)
            value = torch.cat((cache.value, current_v), dim=2)
            key_valid = torch.cat((cache.key_valid, current_valid), dim=1)
            key_positions = torch.cat(
                (cache.positions.to(x.device), current_positions), dim=0
            )
        new_cache = (
            PlainKVCache(
                key=key,
                value=value,
                key_valid=key_valid,
                positions=key_positions,
            )
            if use_cache
            else None
        )
        repeated_k = key[:, self.kv_index]
        repeated_v = value[:, self.kv_index]
        scores = torch.einsum(
            "bhqd,bhkd->bhqk", q.float(), repeated_k.float()
        ) / (self.config.head_dim**0.5)
        causal = key_positions[None, :] <= current_positions[:, None]
        valid = (
            current_valid[:, None, :, None]
            & key_valid[:, None, None, :]
            & causal[None, None, :, :]
        )
        probabilities, masked_logits = safe_masked_softmax_fp32(
            scores, valid
        )
        context = torch.einsum(
            "bhqk,bhkd->bhqd", probabilities, repeated_v.float()
        ).to(dtype=x.dtype)
        context = context.transpose(1, 2).reshape(
            x.shape[0], x.shape[1], self.config.hidden_size
        )
        attention_output = self.o_proj(context)
        post_attention = x + attention_output
        ffn_input = rms_norm_fp32(
            post_attention,
            self.ffn_norm_weight,
            self.config.rms_epsilon,
        )
        ffn_output = self.down_proj(
            F.silu(self.gate_proj(ffn_input)) * self.up_proj(ffn_input)
        )
        final_output = post_attention + ffn_output
        debug = PlainBlockDebug(
            qk_scores=scores,
            masked_logits=masked_logits,
            probabilities=probabilities,
            valid_mask=torch.broadcast_to(valid, scores.shape),
            attention_output=attention_output,
            post_attention=post_attention,
            final_output=final_output,
        )
        return final_output, debug, new_cache

    def forward(
        self,
        x: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        *,
        positions: Optional[torch.Tensor] = None,
        cache: Optional[PlainKVCache] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, PlainKVCache]]:
        """Run the production block without returning attention diagnostics."""

        output, _, new_cache = self._run(
            x,
            token_mask=token_mask,
            positions=positions,
            cache=cache,
            use_cache=use_cache,
        )
        if use_cache:
            assert new_cache is not None
            return output, new_cache
        return output

    def forward_debug(
        self,
        x: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        *,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, PlainBlockDebug]:
        """Run the explicit debug path."""

        if not self.debug_enabled:
            raise PermissionError("plain block debug API is disabled")
        output, debug, _ = self._run(
            x,
            token_mask=token_mask,
            positions=positions,
            cache=None,
            use_cache=False,
        )
        return output, debug


class PlainTinyCausalLM(nn.Module):
    """Locally initialized tiny multi-layer causal LM for correctness only."""

    def __init__(
        self, config: ModelConfig, *, seed: int, debug_enabled: bool
    ) -> None:
        super().__init__()
        self.config = config
        self.seed = int(seed)
        self.debug_enabled = bool(debug_enabled)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(derive_seed(seed, "plain-tiny-lm", "io"))
            self.embedding = nn.Embedding(
                config.vocab_size, config.hidden_size
            )
            self.final_norm_weight = nn.Parameter(
                torch.ones(config.hidden_size)
            )
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            )
        self.blocks = nn.ModuleList(
            [
                PlainDecoderBlock(
                    config,
                    seed=seed,
                    layer_id=layer_id,
                    debug_enabled=debug_enabled,
                )
                for layer_id in range(config.num_layers)
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        *,
        positions: Optional[torch.Tensor] = None,
        cache: Optional[PlainLMCache] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, PlainLMCache]]:
        """Return next-token logits without intermediate states."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("input sequence exceeds configured context")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be integral")
        if torch.any(input_ids < 0) or torch.any(
            input_ids >= self.config.vocab_size
        ):
            raise ValueError("input token is outside vocabulary")
        if token_mask is not None and (
            token_mask.shape != input_ids.shape
            or token_mask.dtype != torch.bool
        ):
            raise ValueError("token_mask must be boolean and match input_ids")
        if positions is None:
            if cache is None:
                positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            else:
                if not cache.layers:
                    raise ValueError("LM cache must contain decoder layers")
                start = int(cache.layers[0].positions[-1].item()) + 1
                positions = torch.arange(
                    start, start + input_ids.shape[1], device=input_ids.device
                )
        elif positions.shape != (input_ids.shape[1],):
            raise ValueError("positions must have shape [sequence]")
        positions = positions.to(device=input_ids.device, dtype=torch.long)
        if cache is not None and len(cache.layers) != len(self.blocks):
            raise ValueError("LM cache layer count mismatch")
        hidden = self.embedding(input_ids)
        new_caches = []
        for index, block in enumerate(self.blocks):
            block_result = block(
                hidden,
                token_mask=token_mask,
                positions=positions,
                cache=None if cache is None else cache.layers[index],
                use_cache=use_cache,
            )
            if use_cache:
                hidden, block_cache = block_result
                new_caches.append(block_cache)
            else:
                assert isinstance(block_result, torch.Tensor)
                hidden = block_result
        hidden = rms_norm_fp32(
            hidden, self.final_norm_weight, self.config.rms_epsilon
        )
        logits = self.lm_head(hidden)
        if use_cache:
            return logits, PlainLMCache(tuple(new_caches))
        return logits

    @torch.no_grad()
    def generate_greedy(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Greedy generation using the same incremental KV-cache contract."""

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if (
            input_ids.shape[1] + max_new_tokens
            > self.config.max_sequence_length
        ):
            raise ValueError("generation exceeds configured context")
        if token_mask is None:
            token_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if token_mask.shape != input_ids.shape or token_mask.dtype != torch.bool:
            raise ValueError("token_mask must be boolean and match input_ids")
        tokens = input_ids.clone()
        prompt_positions = torch.arange(tokens.shape[1], device=tokens.device)
        logits, cache = self(
            tokens,
            token_mask=token_mask,
            positions=prompt_positions,
            use_cache=True,
        )
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        for step in range(max_new_tokens):
            tokens = torch.cat((tokens, next_token), dim=1)
            if step + 1 == max_new_tokens:
                break
            position = torch.tensor(
                [input_ids.shape[1] + step], device=tokens.device, dtype=torch.long
            )
            logits, cache = self(
                next_token,
                token_mask=torch.ones_like(next_token, dtype=torch.bool),
                positions=position,
                cache=cache,
                use_cache=True,
            )
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        return tokens
