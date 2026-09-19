"""Offline-fusion weight builder mixed into the obfuscated decoder block."""

from __future__ import annotations

import torch

from ..layers.deployed import (
    build_deployed_attention,
    build_deployed_feed_forward,
    validate_auxiliary_budget,
)
from ..seed import make_generator
from .obfuscated_types import _random_matrix, _rms_scale_from_factors


class BlockDeploymentMixin:
    """Build the deployed server weights of :class:`ObfuscatedDecoderBlock`.

    Not an ``nn.Module``: the host block provides ``register_buffer`` and the
    converted block is the only supported mixin target.
    """

    def _build_deployed_weights(
        self,
        *,
        plain,
        hidden_transform,
        value_transforms,
        swiglu_transform,
        fixed,
        seed,
        hidden,
        head_dim,
        kv_heads,
        value_noise,
        debug_enabled,
    ) -> None:
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
        # Server-side Gram artifacts for the O(n*b) rho statistic. A_gram =
        # P P.T is block diagonal in perm_out order; the nullspace (noise
        # subspace) content is already implied by the shipped ``noise_read``
        # (B1.3), so these add no new recorded leakage.
        self.register_buffer(
            "deployed_gram_blocks",
            hidden_transform.gram_blocks.detach().clone(),
            persistent=True,
        )
        self.register_buffer(
            "deployed_gram_perm",
            hidden_transform.perm_out.detach().clone(),
            persistent=True,
        )
        # rho is a scalar per token; the factor tensors live in the deployed
        # Gram buffers, so the closure is rebuildable from the server bundle.
        self._rms_scale = _rms_scale_from_factors(
            self.config.hidden_size,
            self.deployed_gram_blocks,
            self.deployed_gram_perm,
        )
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
