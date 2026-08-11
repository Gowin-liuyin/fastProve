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
from ..layers.deployed import (
    build_deployed_attention,
    build_deployed_feed_forward,
    validate_auxiliary_budget,
)
from ..layers.rmsnorm import (
    apply_qk_orthogonal_after_rope,
    apply_rope,
    rms_norm_fp32,
)
from ..layers.swiglu import generate_swiglu_transform
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


def _make_rms_scale(basis: StructuredBasis):
    """Capture a basis for the O(n*b) RMS scale outside persistent state.

    Returns a callable ``(mixed, eps) -> [..., 1]`` in FP32. Only a scalar per
    token is produced; the signal is never reconstructed.
    """

    def rms_scale(mixed: torch.Tensor, eps: float) -> torch.Tensor:
        return basis.rms_scale(mixed, eps)

    return rms_scale


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
        self.register_buffer("kv_index", plain.kv_index.detach().clone())

        hidden = self.config.hidden_size
        heads = self.config.num_attention_heads
        kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim
        intermediate = self.config.intermediate_size
        hidden_noise = obfuscation.hidden_noise_dim
        value_noise = obfuscation.value_noise_dim_per_head

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
        self.register_buffer("common_qk", common_qk, persistent=False)

        self.register_buffer(
            "q_bias",
            plain.q_proj.bias.detach().clone()
            if plain.q_proj.bias is not None
            else torch.zeros(plain.q_proj.weight.shape[0]),
        )
        self.register_buffer(
            "k_bias",
            plain.k_proj.bias.detach().clone()
            if plain.k_proj.bias is not None
            else torch.zeros(plain.k_proj.weight.shape[0]),
        )
        self.register_buffer(
            "v_bias",
            plain.v_proj.bias.detach().clone()
            if plain.v_proj.bias is not None
            else torch.zeros(plain.v_proj.weight.shape[0]),
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

        gamma_ffn = plain.ffn_norm_weight.detach()
        swiglu_transform = generate_swiglu_transform(
            intermediate,
            seed=seed,
            domain="block-%d-swiglu" % self.layer_id,
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

        # -- deployed attention weights (offline fusion) ------------------
        # Value noise coupling must be [n, Hkv, rh] to read directly from the
        # mixed state; the legacy [d, Hkv, rh] form read from plaintext h.
        value_signal_coupling = torch.stack(
            [
                _random_matrix(
                    hidden_transform.total_dim,
                    value_noise,
                    seed=seed,
                    domain="block-%d-value-C-%d" % (self.layer_id, head),
                    scale=0.03,
                )
                for head in range(kv_heads)
            ],
            dim=1,
        )
        validate_auxiliary_budget(
            basis=hidden_transform,
            sample_signal=torch.randn(
                64, hidden, generator=make_generator(seed, "budget-probe")
            ),
            signal_noise_coupling=self.attention_noise_coupling,
            noise_propagator=self.attention_noise_propagator,
            context="block-%d-attention" % self.layer_id,
        )
        deployed_attention = build_deployed_attention(
            basis=hidden_transform,
            value_bases=value_transforms,
            gamma_attention=plain.attention_norm_weight.detach(),
            q_weight_math=plain.q_proj.weight.detach().T.contiguous(),
            k_weight_math=plain.k_proj.weight.detach().T.contiguous(),
            v_weight_math=plain.v_proj.weight.detach().T.contiguous(),
            o_weight_math=plain.o_proj.weight.detach().T.contiguous(),
            q_bias=(
                plain.q_proj.bias.detach()
                if plain.q_proj.bias is not None
                else None
            ),
            k_bias=(
                plain.k_proj.bias.detach()
                if plain.k_proj.bias is not None
                else None
            ),
            v_bias=(
                plain.v_proj.bias.detach()
                if plain.v_proj.bias is not None
                else None
            ),
            value_signal_coupling=value_signal_coupling,
            signal_noise_coupling=self.attention_noise_coupling,
            noise_propagator=self.attention_noise_propagator,
            auxiliary_to_hidden=self.attention_aux_to_hidden,
            fixed_refresh=(
                self.attention_fixed_refresh if fixed else None
            ),
            kv_index=plain.kv_index.detach(),
            head_dim=head_dim,
        )
        # Server-side deployed weights: persistent by design (B1.3 documents
        # that ``deployed_attn_noise_out`` and ``noise_read`` jointly leak the
        # noise state ``e`` up to an invertible r x r map; this is recorded in
        # docs/threat_model.md and is not fixable in this design).
        self.register_buffer(
            "deployed_q", deployed_attention.query, persistent=True
        )
        self.register_buffer(
            "deployed_k", deployed_attention.key, persistent=True
        )
        self.register_buffer(
            "deployed_v", deployed_attention.value, persistent=True
        )
        self.register_buffer(
            "deployed_q_bias", deployed_attention.query_bias, persistent=True
        )
        self.register_buffer(
            "deployed_k_bias", deployed_attention.key_bias, persistent=True
        )
        self.register_buffer(
            "deployed_v_bias", deployed_attention.value_bias, persistent=True
        )
        self.register_buffer(
            "deployed_attn_out", deployed_attention.output, persistent=True
        )
        self.register_buffer(
            "deployed_attn_noise_out",
            deployed_attention.noise_out,
            persistent=True,
        )
        self.register_buffer(
            "deployed_attn_refresh_out",
            deployed_attention.refresh_out,
            persistent=True,
        )
        # N must be online for the (c @ N) @ Wnz residual terms. B1.3 records
        # the recoverability consequence of shipping it.
        self.register_buffer(
            "noise_read",
            hidden_transform.noise_projection().to(dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "_noise_rows",
            hidden_transform.noise_rows().to(dtype=torch.float32),
            persistent=True,
        )
        # rho is a scalar per token; the basis factors stay in the closure and
        # out of state_dict, matching the existing checkpoint convention.
        self._rms_scale = _make_rms_scale(hidden_transform)
        self._debug_basis_unmix = (
            (lambda tensor: hidden_transform.unmix(tensor))
            if debug_enabled
            else None
        )

        # -- deployed feed-forward weights (offline fusion) ---------------
        validate_auxiliary_budget(
            basis=hidden_transform,
            sample_signal=torch.randn(
                64, hidden, generator=make_generator(seed, "budget-probe-ffn")
            ),
            signal_noise_coupling=self.down_noise_coupling,
            noise_propagator=self.down_noise_propagator,
            context="block-%d-ffn" % self.layer_id,
        )
        deployed_ffn = build_deployed_feed_forward(
            basis=hidden_transform,
            gamma_ffn=plain.ffn_norm_weight.detach(),
            gate_weight_math=plain.gate_proj.weight.detach().T.contiguous(),
            up_weight_math=plain.up_proj.weight.detach().T.contiguous(),
            down_weight_math=plain.down_proj.weight.detach().T.contiguous(),
            neuron_permutation=swiglu_transform.permutation,
            neuron_scale=swiglu_transform.scale,
            swiglu_noise_coupling=self.swiglu_noise_coupling,
            down_noise_coupling=self.down_noise_coupling,
            noise_propagator=self.down_noise_propagator,
            fixed_refresh=self.down_fixed_refresh if fixed else None,
        )
        self.register_buffer("deployed_gate", deployed_ffn.gate, persistent=True)
        self.register_buffer("deployed_up", deployed_ffn.up, persistent=True)
        self.register_buffer(
            "deployed_ffn_out", deployed_ffn.output, persistent=True
        )
        self.register_buffer(
            "deployed_ffn_noise_out", deployed_ffn.noise_out, persistent=True
        )
        self.register_buffer(
            "deployed_ffn_refresh_out", deployed_ffn.refresh_out, persistent=True
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

    def _debug_unmix(self, mixed: torch.Tensor) -> torch.Tensor:
        """Decode a mixed state. Debug/diagnostics only.

        Not called from the production forward path. Present so that
        ``forward_debug`` can report plaintext-referenced error metrics.
        """

        if not self.debug_enabled:
            raise PermissionError("decode is disabled outside debug mode")
        return self._debug_basis_unmix(mixed)

    def _mask_from_mixed(
        self, mixed: torch.Tensor, token_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Return a boolean [batch, sequence] validity mask."""

        if token_mask is None:
            return torch.ones(
                mixed.shape[0],
                mixed.shape[1],
                dtype=torch.bool,
                device=mixed.device,
            )
        if (
            token_mask.shape != mixed.shape[:2]
            or token_mask.dtype != torch.bool
        ):
            raise ValueError("token_mask must be boolean [batch, sequence]")
        return token_mask.to(device=mixed.device)

    def _refresh_out(
        self,
        fixed_out: torch.Tensor,
        context: RequestContext,
        domain: str,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Return the mixed-basis increment contributed by the noise refresh.

        ``fixed_debug`` uses the pre-absorbed ``diag(xi) @ M_bot`` rows summed
        to a single ``[n]`` vector. ``per_request`` samples ``xi`` from the
        request-scoped generator and maps it through the same rows.
        """

        if not self.noise_injection_enabled:
            return torch.zeros(
                fixed_out.shape[-1],
                device=reference.device,
                dtype=reference.dtype,
            )
        if self.obfuscation.refresh_mode == "fixed_debug":
            return fixed_out.sum(dim=0)
        generator = context.generator_for(
            "block", self.layer_id, domain, "refresh"
        )
        sampled = torch.randn(
            self.obfuscation.hidden_noise_dim,
            generator=generator,
            dtype=torch.float32,
        ) * (0.02 * self.refresh_noise_scale)
        rows = self._noise_rows.to(
            device=reference.device, dtype=reference.dtype
        )
        return sampled.to(device=reference.device, dtype=reference.dtype) @ rows

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
        # No decode: the signal is never reconstructed on this path.
        mixed = state.mixed
        batch, sequence, _ = mixed.shape
        if positions is None:
            positions = torch.arange(sequence, device=mixed.device)
        elif positions.shape != (sequence,):
            raise ValueError("positions must have shape [sequence]")
        positions = positions.to(device=mixed.device, dtype=torch.long)
        if torch.any(positions < 0) or torch.any(
            positions >= self.config.max_sequence_length
        ):
            raise ValueError("position is outside configured context")
        valid_tokens = self._mask_from_mixed(mixed, token_mask)

        def cast(name: str) -> torch.Tensor:
            tensor = getattr(self, name)
            return tensor.to(device=mixed.device, dtype=mixed.dtype)

        # rho carries the RMSNorm statistic; FP32 reduction per AGENTS.md.
        scale = self._rms_scale(mixed, self.config.rms_epsilon).to(
            device=mixed.device, dtype=mixed.dtype
        )
        q_flat = (mixed @ cast("deployed_q")) / scale + cast("deployed_q_bias")
        k_flat = (mixed @ cast("deployed_k")) / scale + cast("deployed_k_bias")
        q = q_flat.view(
            batch,
            sequence,
            self.config.num_attention_heads,
            self.config.head_dim,
        ).transpose(1, 2)
        k = k_flat.view(
            batch,
            sequence,
            self.config.num_key_value_heads,
            self.config.head_dim,
        ).transpose(1, 2)
        current_value_mixed = torch.einsum(
            "btn,hne->bhte", mixed, cast("deployed_v")
        ) / scale.unsqueeze(1) + cast("deployed_v_bias")[None, :, None, :]

        q_rope = apply_rope(q, positions, theta=self.config.rope_theta)
        k_rope = apply_rope(k, positions, theta=self.config.rope_theta)
        q_prime, k_prime = apply_qk_orthogonal_after_rope(
            q_rope,
            k_rope,
            self.common_qk,
            self.kv_index,
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
                or cache.key.device != mixed.device
                or cache.value_mixed.device != mixed.device
            ):
                raise ValueError("cache dtype/device mismatch")
            cached_length = cache.key.shape[2]
            if (
                cache.key_valid.shape != (batch, cached_length)
                or cache.key_valid.dtype != torch.bool
                or cache.positions.shape != (cached_length,)
                or cache.positions.dtype != torch.long
                or cache.key_valid.device != mixed.device
                or cache.positions.device != mixed.device
            ):
                raise ValueError("cache mask/position shape mismatch")
            key = torch.cat((cache.key, k_prime), dim=2)
            value_mixed = torch.cat(
                (cache.value_mixed, current_value_mixed), dim=2
            )
            key_valid = torch.cat(
                (cache.key_valid.to(mixed.device), valid_tokens), dim=1
            )
            key_positions = torch.cat(
                (cache.positions.to(mixed.device), positions), dim=0
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
        # The mixed context is consumed directly: no Value unmix, and the clean
        # attention context O is never materialized.
        post_attention_mixed = (
            mixed
            + torch.einsum(
                "bhqd,hdn->bqn", mixed_context, cast("deployed_attn_out")
            )
            + (mixed @ cast("noise_read")) @ cast("deployed_attn_noise_out")
            + self._refresh_out(
                cast("deployed_attn_refresh_out"),
                request_context,
                "attention",
                mixed,
            )
        )
        post_attention_state = MixedState(
            post_attention_mixed, self.hidden_basis
        )

        # FFN segment, also without decoding.
        post_mixed = post_attention_state.mixed
        scale_ffn = self._rms_scale(post_mixed, self.config.rms_epsilon).to(
            device=post_mixed.device, dtype=post_mixed.dtype
        )
        gate_prime = (post_mixed @ cast("deployed_gate")) / scale_ffn
        up_prime = (post_mixed @ cast("deployed_up")) / scale_ffn
        z_prime = F.silu(gate_prime) * up_prime
        final_mixed = (
            post_mixed
            + z_prime @ cast("deployed_ffn_out")
            + (post_mixed @ cast("noise_read")) @ cast("deployed_ffn_noise_out")
            + self._refresh_out(
                cast("deployed_ffn_refresh_out"),
                request_context,
                "down",
                post_mixed,
            )
        )
        final_state = MixedState(final_mixed, self.hidden_basis)

        if not return_debug:
            return final_state, None, new_cache
        assert attention_debug is not None
        # Debug path only: explicit decode for diagnostics. Never reachable from
        # forward(); gated by debug_enabled in forward_debug().
        decoded_in = self._debug_unmix(state.mixed)
        decoded_post = self._debug_unmix(post_attention_mixed)
        decoded_final = self._debug_unmix(final_mixed)
        hidden = decoded_in[..., : self.config.hidden_size]
        post_attention_signal = decoded_post[..., : self.config.hidden_size]
        attention_noise = decoded_post[..., self.config.hidden_size :]
        final_noise = decoded_final[..., self.config.hidden_size :]
        attention_output = post_attention_signal - hidden
        clean_update = torch.einsum(
            "bhqd,hdn->bqn",
            attention_debug.clean_output.float(),
            cast("deployed_attn_out"),
        )
        clean_attention_output = self._debug_unmix(clean_update)[
            ..., : self.config.hidden_size
        ]
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
            z_prime=z_prime,
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
        # P @ diag(gamma_final) @ W_head^T, in math layout [n, V]. This is a
        # separate matrix from the tied embedding table, which roughly doubles
        # the vocabulary-side peak memory (task D2 records the measurement;
        # the fused-norm-head alternative needs a fused kernel, stage E).
        projection = hidden_transform.signal_projection()
        deployed_head = (
            projection
            * plain.final_norm_weight.detach().cpu().to(torch.float64)[None, :]
        ) @ plain.lm_head.weight.detach().cpu().T.contiguous().to(torch.float64)
        self.register_buffer(
            "deployed_head",
            deployed_head.to(dtype=torch.float32),
            persistent=True,
        )
        self._basis_mix = lambda augmented: hidden_transform.mix(augmented)
        self._rms_scale = _make_rms_scale(hidden_transform)
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
        # Stage C replaces this with a pre-mixed vocabulary so the plaintext
        # embedding is never materialized. See task C2.
        state = MixedState(
            self._basis_mix(torch.cat((embedding, initial_noise), dim=-1)),
            self.hidden_basis,
        )
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
        # Final norm + head without decoding: rho from the blockwise Gram, and
        # P diag(gamma_final) folded into the head.
        scale = self._rms_scale(state.mixed, self.config.rms_epsilon).to(
            device=state.mixed.device, dtype=state.mixed.dtype
        )
        head = self.deployed_head.to(
            device=state.mixed.device, dtype=state.mixed.dtype
        )
        logits = (state.mixed @ head) / scale
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
