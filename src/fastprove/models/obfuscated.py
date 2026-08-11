"""Exact and bounded-approximate obfuscated decoder-block reference."""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig, ObfuscationConfig
from ..evaluation.metrics import tensor_error_metrics
from ..layers.attention import (
    ApproximationConfig,
    AttentionDebug,
    AttentionMode,
    ObfuscatedAttention,
    safe_masked_softmax_fp32,
)
from ..layers.rmsnorm import (
    absorbed_rms_projection,
    apply_qk_orthogonal_after_rope,
    apply_rope,
    rms_no_gamma_fp32,
    rms_norm_fp32,
)
from ..layers.swiglu import (
    convert_swiglu_weights,
    generate_swiglu_transform,
    refresh_swiglu_noise,
)
from ..seed import RequestContext, make_generator
from ..state import MixedState, decode_debug, encode_debug
from ..structured import StructuredBasis, generate_structured_basis
from ..transforms import (
    BasisDescriptor,
    generate_orthogonal,
    generate_signed_permutation,
)
from .plain import PlainDecoderBlock, PlainTinyCausalLM


class ObfuscatedBlockClient:
    """Client/debug-side holder of the hidden mixing transform."""

    def __init__(
        self, transform: StructuredBasis, *, debug_enabled: bool
    ) -> None:
        self._transform = transform
        self._debug_enabled = bool(debug_enabled)

    @property
    def basis(self) -> BasisDescriptor:
        """Public basis identity without matrix material."""

        return self._transform.descriptor

    def encode_debug(
        self, signal: torch.Tensor, noise: torch.Tensor
    ) -> MixedState:
        """Encode an input only on the explicitly enabled reference path."""

        return encode_debug(
            signal, noise, self._transform, enabled=self._debug_enabled
        )

    def decode_debug(
        self, state: MixedState
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode an output only on the explicitly enabled reference path."""

        return decode_debug(
            state, self._transform, enabled=self._debug_enabled
        )


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
    swiglu_noise_state: torch.Tensor
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


def _checkpoint_compute_dtype(device: torch.device | str) -> torch.dtype:
    """Return the runtime checkpoint arithmetic dtype.

    Offline basis generation, inversion and conditioning checks remain FP64 (see
    ``structured.py``). Runtime arithmetic is FP32 on every device:

    * MPS does not implement FP64 tensors at all.
    * FP64 on CPU accounted for roughly two thirds of the measured reference
      overhead (prefill +150.3% with FP64 versus +50.7% with FP32 at d=1024,
      4 layers, seq=128) and is not required by any accuracy gate.
    * AGENTS.md requires FP32 or better for RMS statistics, margins, clipping
      and Softmax reductions; FP32 satisfies that.

    ``device`` is accepted for interface stability and forward compatibility.
    """

    del device
    return torch.float32


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


def _make_hidden_checkpoint(basis, *, signal_dimension: int):
    """Capture a basis inside the designated fused-checkpoint reference.

    The captured factors are deliberately absent from ``state_dict`` and
    ``named_buffers``. Python closure inspection, hooks, or modified kernels
    remain outside the prototype threat model.

    ``basis`` is a ``StructuredBasis``. Both directions cost ``O(n*b)``.

    Permutations are sorted once at capture time and the factor tensors are
    cached per ``(device, dtype)`` inside the closure, so the online path is
    pure blockwise arithmetic with no per-call permutation or dtype work.
    """

    descriptor = basis.descriptor
    sort_in = torch.argsort(basis.perm_in).detach().clone()
    sort_out = torch.argsort(basis.perm_out).detach().clone()
    cache: dict = {}

    def _factors(device: torch.device, dtype: torch.dtype):
        key = (device.type, device.index, dtype)
        prepared = cache.get(key)
        if prepared is None:
            prepared = (
                sort_in.to(device=device),
                basis.scales.to(device=device, dtype=dtype),
                basis.blocks.to(device=device, dtype=dtype),
                sort_out.to(device=device),
                basis.blocks.transpose(-1, -2).contiguous().to(
                    device=device, dtype=dtype
                ),
                basis.perm_out.to(device=device),
                basis.perm_in.to(device=device),
            )
            cache[key] = prepared
        return prepared

    def _apply(
        source: torch.Tensor, forward: bool
    ) -> torch.Tensor:
        sort_in, scales, blocks, sort_out, blocks_t, perm_out, perm_in = (
            _factors(source.device, source.dtype)
        )
        count = int(blocks.shape[0])
        block = int(blocks.shape[1])
        if forward:
            gathered = source[..., sort_in]
            scaled = gathered * scales
            segments = scaled.reshape(*source.shape[:-1], count, block)
            mixed = torch.einsum("...mi,mij->...mj", segments, blocks)
            return mixed.reshape(*source.shape)[..., sort_out]
        gathered = source[..., perm_out]
        segments = gathered.reshape(*source.shape[:-1], count, block)
        unblocked = torch.einsum("...mi,mij->...mj", segments, blocks_t)
        scaled = unblocked.reshape(*source.shape) / scales
        return scaled[..., perm_in]

    def mix(signal: torch.Tensor, noise: torch.Tensor) -> MixedState:
        compute_dtype = _checkpoint_compute_dtype(signal.device)
        augmented = torch.cat((signal, noise), dim=-1).to(dtype=compute_dtype)
        return MixedState(
            _apply(augmented, True).to(dtype=signal.dtype), descriptor
        )

    def unmix(state: MixedState) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.basis != descriptor:
            raise ValueError("input mixed state basis does not match checkpoint")
        compute_dtype = _checkpoint_compute_dtype(state.mixed.device)
        augmented = _apply(state.mixed.to(dtype=compute_dtype), False).to(
            dtype=state.mixed.dtype
        )
        return (
            augmented[..., :signal_dimension],
            augmented[..., signal_dimension:],
        )

    return mix, unmix


def _make_value_checkpoint(bases: Tuple["StructuredBasis", ...]):
    """Capture per-KV-head Value codecs outside persistent module state.

    Value bases are single dense orthogonal blocks, so the blockwise
    application coincides with a dense matrix multiply. Stacking the dense
    factors lets every head run in one einsum instead of one kernel per head
    (the same formulation stage B ships as deployed weights).
    """

    matrices = torch.stack(
        [item.dense().to(dtype=item.blocks.dtype) for item in bases]
    ).detach()
    inverses = torch.stack(
        [item.dense_inverse().to(dtype=item.blocks.dtype) for item in bases]
    ).detach()

    def mix(value_augmented: torch.Tensor) -> torch.Tensor:
        compute_dtype = _checkpoint_compute_dtype(value_augmented.device)
        converted = matrices.to(
            device=value_augmented.device, dtype=compute_dtype
        )
        return torch.einsum(
            "bhtd,hde->bhte",
            value_augmented.to(dtype=compute_dtype),
            converted,
        ).to(dtype=value_augmented.dtype)

    def unmix(
        mixed_context: torch.Tensor, kv_index: torch.Tensor
    ) -> torch.Tensor:
        index = kv_index.to(device=inverses.device)
        compute_dtype = _checkpoint_compute_dtype(mixed_context.device)
        converted = inverses[index].to(
            device=mixed_context.device, dtype=compute_dtype
        )
        return torch.einsum(
            "bhqd,hde->bhqe",
            mixed_context.to(dtype=compute_dtype),
            converted,
        ).to(dtype=mixed_context.dtype)

    return mix, unmix


class ObfuscatedDecoderBlock(nn.Module):
    """Reference fused-checkpoint implementation of one converted block.

    The private hidden inverse models the designated trusted/fused nonlinear
    checkpoint. It is not returned by the production API. This Python reference
    does not protect it from an operator who can inspect module internals.
    """

    def __init__(
        self,
        plain: PlainDecoderBlock,
        *,
        obfuscation: ObfuscationConfig,
        mode: AttentionMode,
        approximation: Optional[ApproximationConfig],
        seed: int,
        debug_enabled: bool,
        hidden_transform: StructuredBasis,
    ) -> None:
        super().__init__()
        self.config: ModelConfig = plain.config
        self.obfuscation = obfuscation
        self.mode = AttentionMode(mode)
        self.layer_id = plain.layer_id
        self.debug_enabled = bool(debug_enabled)
        self.noise_injection_enabled = True
        self.refresh_noise_scale = 1.0
        if obfuscation.refresh_mode == "fixed_debug" and not self.debug_enabled:
            raise ValueError(
                "fixed_debug refresh is debug-only; use per_request in production"
            )
        self.hidden_basis = hidden_transform.descriptor
        (
            self._fused_checkpoint_mix,
            self._fused_checkpoint_unmix,
        ) = _make_hidden_checkpoint(
            hidden_transform,
            signal_dimension=self.config.hidden_size,
        )
        self.register_buffer("kv_index", plain.kv_index.detach().clone())

        hidden = self.config.hidden_size
        heads = self.config.num_attention_heads
        kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim
        intermediate = self.config.intermediate_size
        hidden_noise = obfuscation.hidden_noise_dim
        value_noise = obfuscation.value_noise_dim_per_head

        attention_rotation = generate_orthogonal(
            hidden, seed=seed, domain="block-%d-attention-rms" % self.layer_id
        )
        ffn_rotation = generate_orthogonal(
            hidden, seed=seed, domain="block-%d-ffn-rms" % self.layer_id
        )
        common_qk = torch.stack(
            [
                generate_orthogonal(
                    head_dim,
                    seed=seed,
                    domain="block-%d-qk-%d" % (self.layer_id, head),
                )
                for head in range(kv_heads)
            ]
        )
        # Key material: must not enter the server-side state_dict.
        # ``common_qk`` is applied after RoPE and therefore cannot be absorbed
        # into a deployed weight; it stays a runtime tensor but is still key
        # material, so it is excluded from the checkpoint as well.
        self.register_buffer(
            "attention_rotation", attention_rotation, persistent=False
        )
        self.register_buffer("ffn_rotation", ffn_rotation, persistent=False)
        self.register_buffer("common_qk", common_qk, persistent=False)

        gamma_attention = plain.attention_norm_weight.detach()
        q_math = plain.q_proj.weight.detach().T.contiguous()
        k_math = plain.k_proj.weight.detach().T.contiguous()
        v_math = plain.v_proj.weight.detach().T.contiguous()
        self.register_buffer(
            "q_weight_math",
            absorbed_rms_projection(
                attention_rotation, gamma_attention, q_math
            ),
        )
        self.register_buffer(
            "k_weight_math",
            absorbed_rms_projection(
                attention_rotation, gamma_attention, k_math
            ),
        )
        self.register_buffer(
            "v_weight_math",
            absorbed_rms_projection(
                attention_rotation, gamma_attention, v_math
            ),
        )
        self.register_buffer(
            "q_bias",
            plain.q_proj.bias.detach().clone()
            if plain.q_proj.bias is not None
            else torch.zeros(q_math.shape[1]),
        )
        self.register_buffer(
            "k_bias",
            plain.k_proj.bias.detach().clone()
            if plain.k_proj.bias is not None
            else torch.zeros(k_math.shape[1]),
        )
        self.register_buffer(
            "v_bias",
            plain.v_proj.bias.detach().clone()
            if plain.v_proj.bias is not None
            else torch.zeros(v_math.shape[1]),
        )
        self.register_buffer(
            "o_weight_math", plain.o_proj.weight.detach().T.contiguous()
        )

        # Value bases always use a single dense orthogonal block: their own
        # width (head_dim + value_noise_dim_per_head) is small, so no block
        # partition is needed and the uniform ``basis_block_size`` (which may
        # not divide the Value width) is not applied to them.
        value_transforms = tuple(
            generate_structured_basis(
                head_dim,
                value_noise,
                seed=seed,
                domain="block-%d-value-%d" % (self.layer_id, head),
                block_size=head_dim + value_noise,
                max_condition_number=obfuscation.max_condition_number,
            )
            for head in range(kv_heads)
        )
        (
            self._fused_value_mix,
            self._fused_value_unmix,
        ) = _make_value_checkpoint(value_transforms)
        self.value_basis_fingerprints = tuple(
            item.fingerprint for item in value_transforms
        )
        self.value_basis_condition_numbers = tuple(
            float(item.condition_number) for item in value_transforms
        )
        self.cache_identity = ":".join(
            (
                str(self.layer_id),
                self.hidden_basis.fingerprint,
                *self.value_basis_fingerprints,
                _cache_tensor_digest(
                    common_qk,
                    plain.q_proj.weight,
                    plain.k_proj.weight,
                    plain.v_proj.weight,
                ),
            )
        )
        self.register_buffer(
            "value_signal_coupling",
            torch.stack(
                [
                    _random_matrix(
                        hidden,
                        value_noise,
                        seed=seed,
                        domain="block-%d-value-C-%d"
                        % (self.layer_id, head),
                        scale=0.03,
                    )
                    for head in range(kv_heads)
                ]
            ),
            persistent=False,
        )
        self.register_buffer(
            "value_side_propagator",
            torch.stack(
                [
                    _random_matrix(
                        hidden_noise,
                        value_noise,
                        seed=seed,
                        domain="block-%d-value-G-%d"
                        % (self.layer_id, head),
                        scale=0.08,
                    )
                    for head in range(kv_heads)
                ]
            ),
            persistent=False,
        )
        self.register_buffer(
            "value_fixed_refresh",
            _random_matrix(
                kv_heads,
                value_noise,
                seed=seed,
                domain="block-%d-value-xi" % self.layer_id,
                scale=0.02,
            )
            if obfuscation.refresh_mode == "fixed_debug"
            else torch.zeros(kv_heads, value_noise),
            persistent=False,
        )

        gamma_ffn = plain.ffn_norm_weight.detach()
        swiglu_transform = generate_swiglu_transform(
            intermediate,
            seed=seed,
            domain="block-%d-swiglu" % self.layer_id,
        )
        converted_swiglu = convert_swiglu_weights(
            hidden_rotation=ffn_rotation,
            gamma=gamma_ffn,
            gate_weight_math=plain.gate_proj.weight.detach().T.contiguous(),
            up_weight_math=plain.up_proj.weight.detach().T.contiguous(),
            down_weight_math=plain.down_proj.weight.detach().T.contiguous(),
            transform=swiglu_transform,
        )
        self.register_buffer(
            "gate_weight_math", converted_swiglu.gate_weight_math
        )
        self.register_buffer(
            "up_weight_math", converted_swiglu.up_weight_math
        )
        self.register_buffer(
            "down_weight_math", converted_swiglu.down_weight_math
        )

        gamma = obfuscation.noise_propagation_gamma
        self.register_buffer(
            "attention_noise_propagator",
            gamma
            * generate_signed_permutation(
                hidden_noise,
                seed=seed,
                domain="block-%d-attention-noise-G" % self.layer_id,
            ),
            persistent=False,
        )
        self.register_buffer(
            "swiglu_noise_propagator",
            gamma
            * generate_signed_permutation(
                hidden_noise,
                seed=seed,
                domain="block-%d-swiglu-noise-G" % self.layer_id,
            ),
            persistent=False,
        )
        self.register_buffer(
            "down_noise_propagator",
            gamma
            * generate_signed_permutation(
                hidden_noise,
                seed=seed,
                domain="block-%d-down-noise-G" % self.layer_id,
            ),
            persistent=False,
        )
        self.register_buffer(
            "attention_noise_coupling",
            _random_matrix(
                hidden,
                hidden_noise,
                seed=seed,
                domain="block-%d-attention-noise-C" % self.layer_id,
                scale=0.02,
            ),
            persistent=False,
        )
        self.register_buffer(
            "attention_aux_to_hidden",
            _random_matrix(
                value_noise,
                hidden_noise,
                seed=seed,
                domain="block-%d-attention-aux" % self.layer_id,
                scale=0.08,
            ),
            persistent=False,
        )
        self.register_buffer(
            "swiglu_noise_coupling",
            _random_matrix(
                intermediate,
                hidden_noise,
                seed=seed,
                domain="block-%d-swiglu-noise-C" % self.layer_id,
                scale=0.02,
            ),
            persistent=False,
        )
        self.register_buffer(
            "down_noise_coupling",
            _random_matrix(
                hidden,
                hidden_noise,
                seed=seed,
                domain="block-%d-down-noise-C" % self.layer_id,
                scale=0.02,
            ),
            persistent=False,
        )
        fixed = obfuscation.refresh_mode == "fixed_debug"
        self.register_buffer(
            "attention_fixed_refresh",
            _random_matrix(
                1,
                hidden_noise,
                seed=seed,
                domain="block-%d-attention-xi" % self.layer_id,
                scale=0.02,
            ).squeeze(0)
            if fixed
            else torch.zeros(hidden_noise),
            persistent=False,
        )
        self.register_buffer(
            "swiglu_fixed_refresh",
            _random_matrix(
                1,
                hidden_noise,
                seed=seed,
                domain="block-%d-swiglu-xi" % self.layer_id,
                scale=0.02,
            ).squeeze(0)
            if fixed
            else torch.zeros(hidden_noise),
            persistent=False,
        )
        self.register_buffer(
            "down_fixed_refresh",
            _random_matrix(
                1,
                hidden_noise,
                seed=seed,
                domain="block-%d-down-xi" % self.layer_id,
                scale=0.02,
            ).squeeze(0)
            if fixed
            else torch.zeros(hidden_noise),
            persistent=False,
        )

        self.attention = ObfuscatedAttention(
            mode=self.mode,
            approximation=approximation,
            layer_id="block-%d-attention" % self.layer_id,
            debug_enabled=debug_enabled,
        )

    @classmethod
    def from_plain(
        cls,
        plain: PlainDecoderBlock,
        *,
        obfuscation: ObfuscationConfig,
        mode: AttentionMode,
        approximation: Optional[ApproximationConfig],
        seed: int,
        debug_enabled: bool,
        hidden_transform: Optional[StructuredBasis] = None,
    ) -> ConvertedObfuscatedBlock:
        """Convert a plaintext block and return separate client/server objects."""

        started = time.perf_counter()
        transform = (
            hidden_transform
            if hidden_transform is not None
            else generate_structured_basis(
                plain.config.hidden_size,
                obfuscation.hidden_noise_dim,
                seed=seed,
                domain="shared-hidden-basis",
                block_size=obfuscation.basis_block_size,
                max_condition_number=obfuscation.max_condition_number,
            )
        )
        module = cls(
            plain,
            obfuscation=obfuscation,
            mode=mode,
            approximation=approximation,
            seed=seed,
            debug_enabled=debug_enabled,
            hidden_transform=transform,
        )
        client = ObfuscatedBlockClient(
            transform, debug_enabled=debug_enabled
        )
        return ConvertedObfuscatedBlock(
            module=module,
            client=client,
            conversion_time_seconds=time.perf_counter() - started,
        )

    def _refresh(
        self,
        fixed: torch.Tensor,
        context: RequestContext,
        domain: str,
    ) -> torch.Tensor:
        if not self.noise_injection_enabled:
            return torch.zeros_like(fixed)
        if self.obfuscation.refresh_mode == "fixed_debug":
            return fixed
        generator = context.generator_for(
            "block", self.layer_id, domain, "refresh"
        )
        value = torch.randn(
            fixed.shape, generator=generator, dtype=torch.float32
        ) * (0.02 * self.refresh_noise_scale)
        return value.to(device=fixed.device, dtype=fixed.dtype)

    def _unmix_checkpoint(
        self, state: MixedState
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._fused_checkpoint_unmix(state)

    def _mix_checkpoint(
        self, signal: torch.Tensor, noise: torch.Tensor
    ) -> MixedState:
        return self._fused_checkpoint_mix(signal, noise)

    def _mask(
        self, hidden: torch.Tensor, token_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if token_mask is None:
            return torch.ones(
                hidden.shape[0],
                hidden.shape[1],
                dtype=torch.bool,
                device=hidden.device,
            )
        if (
            token_mask.shape != hidden.shape[:2]
            or token_mask.dtype != torch.bool
        ):
            raise ValueError("token_mask must be boolean [batch, sequence]")
        return token_mask.to(device=hidden.device)

    def _run(
        self,
        state: MixedState,
        *,
        token_mask: Optional[torch.Tensor],
        positions: Optional[torch.Tensor],
        request_context: RequestContext,
        return_debug: bool,
        cache: Optional[ObfuscatedKVCache],
        use_cache: bool,
    ) -> Tuple[
        MixedState,
        Optional[ObfuscatedBlockDebug],
        Optional[ObfuscatedKVCache],
    ]:
        hidden, side_noise = self._unmix_checkpoint(state)
        batch, sequence, _ = hidden.shape
        if positions is None:
            positions = torch.arange(sequence, device=hidden.device)
        elif positions.shape != (sequence,):
            raise ValueError("positions must have shape [sequence]")
        positions = positions.to(device=hidden.device, dtype=torch.long)
        if torch.any(positions < 0) or torch.any(
            positions >= self.config.max_sequence_length
        ):
            raise ValueError("position is outside configured context")
        valid_tokens = self._mask(hidden, token_mask)

        rotated_hidden = hidden @ self.attention_rotation.to(
            device=hidden.device, dtype=hidden.dtype
        )
        normalized_rotated = rms_no_gamma_fp32(
            rotated_hidden, self.config.rms_epsilon
        )
        q = normalized_rotated @ self.q_weight_math.to(
            device=hidden.device, dtype=hidden.dtype
        ) + self.q_bias.to(device=hidden.device, dtype=hidden.dtype)
        k = normalized_rotated @ self.k_weight_math.to(
            device=hidden.device, dtype=hidden.dtype
        ) + self.k_bias.to(device=hidden.device, dtype=hidden.dtype)
        v = normalized_rotated @ self.v_weight_math.to(
            device=hidden.device, dtype=hidden.dtype
        ) + self.v_bias.to(device=hidden.device, dtype=hidden.dtype)
        q = q.view(
            batch,
            sequence,
            self.config.num_attention_heads,
            self.config.head_dim,
        ).transpose(1, 2)
        k = k.view(
            batch,
            sequence,
            self.config.num_key_value_heads,
            self.config.head_dim,
        ).transpose(1, 2)
        v = v.view(
            batch,
            sequence,
            self.config.num_key_value_heads,
            self.config.head_dim,
        ).transpose(1, 2)
        q_rope = apply_rope(q, positions, theta=self.config.rope_theta)
        k_rope = apply_rope(k, positions, theta=self.config.rope_theta)
        q_prime, k_prime = apply_qk_orthogonal_after_rope(
            q_rope,
            k_rope,
            self.common_qk,
            self.kv_index,
        )

        value_noise = torch.einsum(
            "btd,hdr->bhtr",
            hidden.float(),
            self.value_signal_coupling.float(),
        ) + torch.einsum(
            "btr,hrv->bhtv",
            side_noise.float(),
            self.value_side_propagator.float(),
        )
        value_noise = value_noise + self._refresh(
            self.value_fixed_refresh,
            request_context,
            "value",
        )[None, :, None, :]
        value_augmented = torch.cat((v.float(), value_noise), dim=-1)
        current_value_mixed = self._fused_value_mix(value_augmented).to(
            dtype=hidden.dtype
        )
        if cache is None:
            key = k_prime
            value_mixed = current_value_mixed
            key_valid = valid_tokens
            key_positions = positions
        else:
            if return_debug:
                raise ValueError("debug cache path is not supported")
            if cache.cache_identity != self.cache_identity:
                raise ValueError(
                    "cache identity mismatch; do not mix conversions or models"
                )
            if (
                cache.request_seed != int(request_context.global_seed)
                or cache.request_id != request_context.request_id
            ):
                raise ValueError(
                    "cache request identity mismatch; start a fresh request cache"
                )
            if cache.key.shape[:2] != (
                batch,
                self.config.num_key_value_heads,
            ):
                raise ValueError("cache batch/KV-head shape mismatch")
            if cache.key.shape[2] != cache.value_mixed.shape[2]:
                raise ValueError("cached Key/Value sequence mismatch")
            if cache.key.shape[-1] != self.config.head_dim:
                raise ValueError("cached Key head dimension mismatch")
            expected_value_dim = (
                self.config.head_dim
                + self.obfuscation.value_noise_dim_per_head
            )
            if cache.value_mixed.shape[-1] != expected_value_dim:
                raise ValueError("cached mixed Value dimension mismatch")
            if (
                cache.key.dtype != k_prime.dtype
                or cache.value_mixed.dtype != current_value_mixed.dtype
                or cache.key.device != hidden.device
                or cache.value_mixed.device != hidden.device
            ):
                raise ValueError("cache dtype/device mismatch")
            cached_length = cache.key.shape[2]
            if (
                cache.key_valid.shape != (batch, cached_length)
                or cache.key_valid.dtype != torch.bool
                or cache.positions.shape != (cached_length,)
                or cache.positions.dtype != torch.long
                or cache.key_valid.device != hidden.device
                or cache.positions.device != hidden.device
            ):
                raise ValueError("cache mask/position shape mismatch")
            key = torch.cat((cache.key, k_prime), dim=2)
            value_mixed = torch.cat(
                (cache.value_mixed, current_value_mixed), dim=2
            )
            key_valid = torch.cat(
                (cache.key_valid.to(hidden.device), valid_tokens), dim=1
            )
            key_positions = torch.cat(
                (cache.positions.to(hidden.device), positions), dim=0
            )
        new_cache = (
            ObfuscatedKVCache(
                key=key,
                value_mixed=value_mixed,
                key_valid=key_valid,
                positions=key_positions,
                cache_identity=self.cache_identity,
                request_seed=int(request_context.global_seed),
                request_id=request_context.request_id,
            )
            if use_cache
            else None
        )

        causal = key_positions[None, :] <= positions[:, None]
        valid = (
            valid_tokens[:, None, :, None]
            & key_valid[:, None, None, :]
            & causal[None, None, :, :]
        )
        if return_debug:
            mixed_context, attention_debug = self.attention.forward_debug(
                q=q_prime,
                k=key,
                value_mixed=value_mixed,
                valid_mask=valid,
                kv_index=self.kv_index,
                request_context=request_context,
                query_positions=positions,
                key_positions=key_positions,
            )
        else:
            mixed_context = self.attention(
                q=q_prime,
                k=key,
                value_mixed=value_mixed,
                valid_mask=valid,
                kv_index=self.kv_index,
                request_context=request_context,
                query_positions=positions,
                key_positions=key_positions,
            )
            attention_debug = None
        context_augmented = self._fused_value_unmix(
            mixed_context, self.kv_index
        )
        context_signal = context_augmented[
            ..., : self.config.head_dim
        ]
        context_auxiliary = context_augmented[
            ..., self.config.head_dim :
        ]
        context_flat = context_signal.transpose(1, 2).reshape(
            batch, sequence, self.config.hidden_size
        )
        attention_output = context_flat @ self.o_weight_math.to(
            device=hidden.device, dtype=hidden.dtype
        )
        clean_attention_output = None
        if attention_debug is not None:
            clean_context_augmented = self._fused_value_unmix(
                attention_debug.clean_output, self.kv_index
            )
            clean_context_signal = clean_context_augmented[
                ..., : self.config.head_dim
            ]
            clean_context_flat = clean_context_signal.transpose(1, 2).reshape(
                batch, sequence, self.config.hidden_size
            )
            clean_attention_output = (
                clean_context_flat
                @ self.o_weight_math.to(
                    device=hidden.device, dtype=hidden.dtype
                )
            )
        post_attention_signal = hidden + attention_output
        auxiliary_mean = context_auxiliary.mean(dim=1)
        attention_noise = (
            side_noise.float() @ self.attention_noise_propagator.float()
            + attention_output.float()
            @ self.attention_noise_coupling.float()
            + auxiliary_mean.float() @ self.attention_aux_to_hidden.float()
            + self._refresh(
                self.attention_fixed_refresh,
                request_context,
                "attention",
            )
        ).to(dtype=hidden.dtype)
        post_attention_state = self._mix_checkpoint(
            post_attention_signal, attention_noise
        )

        ffn_signal, ffn_side_noise = self._unmix_checkpoint(
            post_attention_state
        )
        ffn_rotated = ffn_signal @ self.ffn_rotation.to(
            device=hidden.device, dtype=hidden.dtype
        )
        ffn_normalized = rms_no_gamma_fp32(
            ffn_rotated, self.config.rms_epsilon
        )
        gate_prime = ffn_normalized @ self.gate_weight_math.to(
            device=hidden.device, dtype=hidden.dtype
        )
        up_prime = ffn_normalized @ self.up_weight_math.to(
            device=hidden.device, dtype=hidden.dtype
        )
        z_prime = F.silu(gate_prime) * up_prime
        swiglu_noise = refresh_swiglu_noise(
            z_prime=z_prime.float(),
            side_noise=ffn_side_noise.float(),
            coupling=self.swiglu_noise_coupling.float(),
            propagator=self.swiglu_noise_propagator.float(),
            refresh=self._refresh(
                self.swiglu_fixed_refresh,
                request_context,
                "swiglu",
            ).float(),
        ).to(dtype=hidden.dtype)
        down_signal = z_prime @ self.down_weight_math.to(
            device=hidden.device, dtype=hidden.dtype
        )
        down_noise = (
            down_signal.float() @ self.down_noise_coupling.float()
            + swiglu_noise.float() @ self.down_noise_propagator.float()
            + self._refresh(
                self.down_fixed_refresh, request_context, "down"
            ).float()
        ).to(dtype=hidden.dtype)
        final_signal = ffn_signal + down_signal
        final_noise = ffn_side_noise + down_noise
        final_state = self._mix_checkpoint(final_signal, final_noise)

        if not return_debug:
            return final_state, None, new_cache
        assert attention_debug is not None
        assert clean_attention_output is not None
        repeated_k_plain = k_rope[:, self.kv_index]
        plain_scores = torch.einsum(
            "bhqd,bhkd->bhqk", q_rope.float(), repeated_k_plain.float()
        ) / (self.config.head_dim**0.5)
        transformed_scores = torch.einsum(
            "bhqd,bhkd->bhqk",
            q_prime.float(),
            k_prime[:, self.kv_index].float(),
        ) / (self.config.head_dim**0.5)
        plain_probabilities, _ = safe_masked_softmax_fp32(
            plain_scores, valid
        )
        debug = ObfuscatedBlockDebug(
            qk_score_error=tensor_error_metrics(
                plain_scores, transformed_scores
            ),
            softmax_error=tensor_error_metrics(
                plain_probabilities, attention_debug.clean_probabilities
            ),
            clean_logits=attention_debug.clean_logits,
            noisy_logits=attention_debug.noisy_logits,
            clean_probabilities=attention_debug.clean_probabilities,
            noisy_probabilities=attention_debug.noisy_probabilities,
            valid_mask=attention_debug.valid_mask,
            logit_noise=attention_debug.noise,
            tau=attention_debug.tau,
            margin=attention_debug.margin,
            clean_attention_output=clean_attention_output,
            attention_output=attention_output,
            post_attention=post_attention_signal,
            attention_noise_state=attention_noise,
            swiglu_noise_state=swiglu_noise,
            final_noise_state=final_noise,
        )
        return final_state, debug, new_cache

    def forward(
        self,
        state: MixedState,
        token_mask: Optional[torch.Tensor] = None,
        *,
        positions: Optional[torch.Tensor] = None,
        request_context: RequestContext,
        cache: Optional[ObfuscatedKVCache] = None,
        use_cache: bool = False,
    ) -> Union[MixedState, Tuple[MixedState, ObfuscatedKVCache]]:
        """Run the production path and return only the next mixed state."""

        output, _, new_cache = self._run(
            state,
            token_mask=token_mask,
            positions=positions,
            request_context=request_context,
            return_debug=False,
            cache=cache,
            use_cache=use_cache,
        )
        if use_cache:
            assert new_cache is not None
            return output, new_cache
        return output

    def forward_debug(
        self,
        state: MixedState,
        token_mask: Optional[torch.Tensor] = None,
        *,
        positions: Optional[torch.Tensor] = None,
        request_context: RequestContext,
    ) -> Tuple[MixedState, ObfuscatedBlockDebug]:
        """Run the explicit debug path."""

        if not self.debug_enabled:
            raise PermissionError("obfuscated block debug API is disabled")
        output, debug, _ = self._run(
            state,
            token_mask=token_mask,
            positions=positions,
            request_context=request_context,
            return_debug=True,
            cache=None,
            use_cache=False,
        )
        assert debug is not None
        return output, debug


@dataclass(frozen=True)
class ConvertedObfuscatedLM:
    """Converted tiny LM, separate client helper, and conversion timing."""

    module: "ObfuscatedTinyCausalLM"
    client: ObfuscatedBlockClient
    conversion_time_seconds: float


class ObfuscatedTinyCausalLM(nn.Module):
    """Tiny exact/approximate LM sharing canonical plaintext base weights."""

    def __init__(
        self,
        plain: PlainTinyCausalLM,
        *,
        obfuscation: ObfuscationConfig,
        mode: AttentionMode,
        approximation: Optional[ApproximationConfig],
        seed: int,
        debug_enabled: bool,
        hidden_transform: StructuredBasis,
    ) -> None:
        super().__init__()
        self.config = plain.config
        self.obfuscation = obfuscation
        self.mode = AttentionMode(mode)
        self.debug_enabled = bool(debug_enabled)
        self.noise_injection_enabled = True
        self.initial_refresh_noise_scale = 1.0
        if obfuscation.refresh_mode == "fixed_debug" and not self.debug_enabled:
            raise ValueError(
                "fixed_debug refresh is debug-only; use per_request in production"
            )
        self.hidden_basis = hidden_transform.descriptor
        self.register_buffer(
            "embedding_weight", plain.embedding.weight.detach().clone()
        )
        self.register_buffer(
            "final_norm_weight",
            plain.final_norm_weight.detach().clone(),
        )
        self.register_buffer(
            "lm_head_weight", plain.lm_head.weight.detach().clone()
        )
        (
            self._fused_checkpoint_mix,
            self._fused_checkpoint_unmix,
        ) = _make_hidden_checkpoint(
            hidden_transform,
            signal_dimension=self.config.hidden_size,
        )
        self.register_buffer(
            "initial_noise_coupling",
            _random_matrix(
                self.config.hidden_size,
                obfuscation.hidden_noise_dim,
                seed=seed,
                domain="tiny-lm-initial-noise-C",
                scale=0.02,
            ),
            persistent=False,
        )
        self.register_buffer(
            "initial_fixed_refresh",
            _random_matrix(
                1,
                obfuscation.hidden_noise_dim,
                seed=seed,
                domain="tiny-lm-initial-xi",
                scale=0.02,
            ).squeeze(0)
            if obfuscation.refresh_mode == "fixed_debug"
            else torch.zeros(obfuscation.hidden_noise_dim),
            persistent=False,
        )
        converted_blocks = [
            ObfuscatedDecoderBlock.from_plain(
                block,
                obfuscation=obfuscation,
                mode=self.mode,
                approximation=approximation,
                seed=seed,
                debug_enabled=debug_enabled,
                hidden_transform=hidden_transform,
            ).module
            for block in plain.blocks
        ]
        self.blocks = nn.ModuleList(converted_blocks)
        self.value_basis_condition_numbers = tuple(
            tuple(block.value_basis_condition_numbers) for block in converted_blocks
        )
        self.value_basis_fingerprints = tuple(
            tuple(block.value_basis_fingerprints) for block in converted_blocks
        )

    @classmethod
    def from_plain(
        cls,
        plain: PlainTinyCausalLM,
        *,
        obfuscation: ObfuscationConfig,
        mode: AttentionMode,
        approximation: Optional[ApproximationConfig],
        seed: int,
        debug_enabled: bool,
    ) -> ConvertedObfuscatedLM:
        """Convert a plaintext tiny LM without changing its base weights."""

        started = time.perf_counter()
        hidden_transform = generate_structured_basis(
            plain.config.hidden_size,
            obfuscation.hidden_noise_dim,
            seed=seed,
            domain="tiny-lm-shared-hidden-basis",
            block_size=obfuscation.basis_block_size,
            max_condition_number=obfuscation.max_condition_number,
        )
        module = cls(
            plain,
            obfuscation=obfuscation,
            mode=mode,
            approximation=approximation,
            seed=seed,
            debug_enabled=debug_enabled,
            hidden_transform=hidden_transform,
        )
        client = ObfuscatedBlockClient(
            hidden_transform, debug_enabled=debug_enabled
        )
        return ConvertedObfuscatedLM(
            module=module,
            client=client,
            conversion_time_seconds=time.perf_counter() - started,
        )

    def _initial_refresh(self, context: RequestContext) -> torch.Tensor:
        if not self.noise_injection_enabled:
            return torch.zeros_like(self.initial_fixed_refresh)
        if self.obfuscation.refresh_mode == "fixed_debug":
            return self.initial_fixed_refresh
        generator = context.generator_for("tiny-lm", "initial", "refresh")
        refresh = torch.randn(
            self.initial_fixed_refresh.shape,
            generator=generator,
            dtype=torch.float32,
        ) * (0.02 * self.initial_refresh_noise_scale)
        return refresh.to(
            device=self.initial_fixed_refresh.device,
            dtype=self.initial_fixed_refresh.dtype,
        )

    def _run(
        self,
        input_ids: torch.Tensor,
        *,
        token_mask: Optional[torch.Tensor],
        request_context: RequestContext,
        return_debug: bool,
        positions: Optional[torch.Tensor],
        cache: Optional[ObfuscatedLMCache],
        use_cache: bool,
    ) -> Tuple[
        torch.Tensor,
        Tuple[ObfuscatedBlockDebug, ...],
        Optional[ObfuscatedLMCache],
    ]:
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
                positions = torch.arange(
                    input_ids.shape[1], device=input_ids.device
                )
            else:
                if not cache.layers:
                    raise ValueError("LM cache must contain decoder layers")
                start = int(cache.layers[0].positions[-1].item()) + 1
                positions = torch.arange(
                    start,
                    start + input_ids.shape[1],
                    device=input_ids.device,
                )
        elif positions.shape != (input_ids.shape[1],):
            raise ValueError("positions must have shape [sequence]")
        positions = positions.to(device=input_ids.device, dtype=torch.long)
        if cache is not None and len(cache.layers) != len(self.blocks):
            raise ValueError("LM cache layer count mismatch")
        if return_debug and (cache is not None or use_cache):
            raise ValueError("debug cache path is not supported")
        embedding = F.embedding(input_ids, self.embedding_weight)
        initial_noise = (
            embedding.float() @ self.initial_noise_coupling.float()
            + self._initial_refresh(request_context)
        ).to(dtype=embedding.dtype)
        state = self._fused_checkpoint_mix(embedding, initial_noise)
        debug_records = []
        new_layer_caches = []
        for layer_index, block in enumerate(self.blocks):
            if return_debug:
                state, block_debug = block.forward_debug(
                    state,
                    token_mask=token_mask,
                    positions=positions,
                    request_context=request_context,
                )
                debug_records.append(block_debug)
            else:
                block_result = block(
                    state,
                    token_mask=token_mask,
                    positions=positions,
                    request_context=request_context,
                    cache=(
                        None
                        if cache is None
                        else cache.layers[layer_index]
                    ),
                    use_cache=use_cache,
                )
                if use_cache:
                    state, layer_cache = block_result
                    new_layer_caches.append(layer_cache)
                else:
                    assert isinstance(block_result, MixedState)
                    state = block_result
        final_signal, _ = self._fused_checkpoint_unmix(state)
        normalized = rms_norm_fp32(
            final_signal,
            self.final_norm_weight,
            self.config.rms_epsilon,
        )
        logits = F.linear(normalized, self.lm_head_weight)
        new_cache = (
            ObfuscatedLMCache(tuple(new_layer_caches))
            if use_cache
            else None
        )
        return logits, tuple(debug_records), new_cache

    def forward(
        self,
        input_ids: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        *,
        request_context: RequestContext,
        positions: Optional[torch.Tensor] = None,
        cache: Optional[ObfuscatedLMCache] = None,
        use_cache: bool = False,
    ) -> Union[
        torch.Tensor, Tuple[torch.Tensor, ObfuscatedLMCache]
    ]:
        """Return only final logits in the production API."""

        logits, _, new_cache = self._run(
            input_ids,
            token_mask=token_mask,
            request_context=request_context,
            return_debug=False,
            positions=positions,
            cache=cache,
            use_cache=use_cache,
        )
        if use_cache:
            assert new_cache is not None
            return logits, new_cache
        return logits

    def forward_debug(
        self,
        input_ids: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        *,
        request_context: RequestContext,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[ObfuscatedBlockDebug, ...]]:
        """Return per-block diagnostics only when explicitly enabled."""

        if not self.debug_enabled:
            raise PermissionError("tiny LM debug API is disabled")
        logits, debug, _ = self._run(
            input_ids,
            token_mask=token_mask,
            request_context=request_context,
            return_debug=True,
            positions=positions,
            cache=None,
            use_cache=False,
        )
        return logits, debug

    @torch.no_grad()
    def generate_greedy(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        request_context: RequestContext,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Greedy generation using the production incremental KV cache."""

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
            request_context=request_context,
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
                request_context=request_context,
                cache=cache,
                use_cache=True,
            )
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        return tokens
