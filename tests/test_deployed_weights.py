"""Deployed weights must reproduce the plaintext block exactly.

These are the aceptance tests for the Stage B non-decoding forward pass: the
deployed block is driven directly here (without the model class) so that a
failure localizes to the weight construction rather than to the module wiring.
"""

from __future__ import annotations

import math

import pytest
import torch

from fastprove.layers.deployed import (
    build_deployed_attention,
    build_deployed_feed_forward,
    validate_auxiliary_budget,
)
from fastprove.structured import generate_structured_basis

_FP64 = torch.float64
_EPS = 1e-5

#: Achievable block-identity tolerance. The bottleneck is ``rho``, which uses an
#: FP32 reduction by design (AGENTS.md requires FP32 for RMS statistics):
#: measured relative error 1.9e-7 for the FP32 Gram versus 2.4e-16 for FP64.
#: Deployed weights are FP64 in these tests so that the *algebra* is isolated;
#: what remains is the FP32 ``rho``. Do not loosen this to hide a real error.
_TOL = 1e-5


class _Setup:
    """Shared plaintext parameters and matching deployed weights."""

    def __init__(
        self,
        *,
        hidden=64,
        noise=8,
        block=8,
        heads=4,
        kv_heads=2,
        head_dim=16,
        intermediate=96,
        batch=2,
        seq=6,
        seed=17,
        with_bias=False,
        fixed_refresh=False,
    ) -> None:
        torch.manual_seed(seed)
        self.hidden = hidden
        self.noise = noise
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.intermediate = intermediate
        self.batch = batch
        self.seq = seq
        self.total = hidden + noise
        assert heads * head_dim == hidden

        self.basis = generate_structured_basis(
            hidden, noise, seed=seed, domain="block", block_size=block,
            dtype=_FP64,
        )
        self.value_bases = tuple(
            generate_structured_basis(
                head_dim, 2, seed=seed + 100 + head, domain="value",
                block_size=head_dim + 2, dtype=_FP64,
            )
            for head in range(kv_heads)
        )
        self.value_noise = 2
        repeat = heads // kv_heads
        self.kv_index = torch.arange(kv_heads).repeat_interleave(repeat)

        self.gamma_attention = torch.rand(hidden, dtype=_FP64) + 0.5
        self.gamma_ffn = torch.rand(hidden, dtype=_FP64) + 0.5
        self.w_q = torch.randn(hidden, heads * head_dim, dtype=_FP64) * 0.1
        self.w_k = torch.randn(hidden, kv_heads * head_dim, dtype=_FP64) * 0.1
        self.w_v = torch.randn(hidden, kv_heads * head_dim, dtype=_FP64) * 0.1
        self.w_o = torch.randn(heads * head_dim, hidden, dtype=_FP64) * 0.1
        self.w_gate = torch.randn(hidden, intermediate, dtype=_FP64) * 0.1
        self.w_up = torch.randn(hidden, intermediate, dtype=_FP64) * 0.1
        self.w_down = torch.randn(intermediate, hidden, dtype=_FP64) * 0.1

        if with_bias:
            self.b_q = torch.randn(heads * head_dim, dtype=_FP64) * 0.05
            self.b_k = torch.randn(kv_heads * head_dim, dtype=_FP64) * 0.05
            self.b_v = torch.randn(kv_heads * head_dim, dtype=_FP64) * 0.05
        else:
            self.b_q = self.b_k = self.b_v = None

        self.coupling_value = (
            torch.randn(self.total, kv_heads, self.value_noise, dtype=_FP64) * 0.03
        )
        self.coupling_o = torch.randn(hidden, noise, dtype=_FP64) * 0.02
        self.propagator_a = 0.5 * torch.eye(noise, dtype=_FP64)
        self.auxiliary = torch.randn(self.value_noise, noise, dtype=_FP64) * 0.08
        self.coupling_z = torch.randn(intermediate, noise, dtype=_FP64) * 0.02
        self.coupling_d = torch.randn(hidden, noise, dtype=_FP64) * 0.02
        self.propagator_f = 0.5 * torch.eye(noise, dtype=_FP64)

        generator = torch.Generator().manual_seed(seed + 7)
        self.neuron_permutation = torch.randperm(intermediate, generator=generator)
        self.neuron_scale = torch.exp(
            (2 * torch.rand(intermediate, generator=generator, dtype=_FP64) - 1) * 0.3
        )

        refresh = (
            torch.randn(noise, dtype=_FP64) * 0.02 if fixed_refresh else None
        )
        self.fixed_refresh = refresh

        self.attention = build_deployed_attention(
            basis=self.basis,
            value_bases=self.value_bases,
            gamma_attention=self.gamma_attention,
            q_weight_math=self.w_q,
            k_weight_math=self.w_k,
            v_weight_math=self.w_v,
            o_weight_math=self.w_o,
            q_bias=self.b_q,
            k_bias=self.b_k,
            v_bias=self.b_v,
            value_signal_coupling=self.coupling_value,
            signal_noise_coupling=self.coupling_o,
            noise_propagator=self.propagator_a,
            auxiliary_to_hidden=self.auxiliary,
            fixed_refresh=refresh,
            kv_index=self.kv_index,
            head_dim=head_dim,
            dtype=_FP64,
        )
        self.feed_forward = build_deployed_feed_forward(
            basis=self.basis,
            gamma_ffn=self.gamma_ffn,
            gate_weight_math=self.w_gate,
            up_weight_math=self.w_up,
            down_weight_math=self.w_down,
            neuron_permutation=self.neuron_permutation,
            neuron_scale=self.neuron_scale,
            swiglu_noise_coupling=self.coupling_z,
            down_noise_coupling=self.coupling_d,
            noise_propagator=self.propagator_f,
            fixed_refresh=refresh,
            dtype=_FP64,
        )

    # -- plaintext reference -------------------------------------------
    def _rms(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(hidden.pow(2).mean(-1, keepdim=True) + _EPS)

    def _split(self, flat: torch.Tensor, count: int) -> torch.Tensor:
        return flat.view(self.batch, self.seq, count, self.head_dim).transpose(1, 2)

    def _causal_softmax(self, scores: torch.Tensor) -> torch.Tensor:
        length = scores.shape[-1]
        mask = torch.tril(torch.ones(length, length, dtype=torch.bool))
        return torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)

    def plaintext(self, hidden: torch.Tensor) -> dict:
        normalized = hidden / self._rms(hidden) * self.gamma_attention
        q_flat = normalized @ self.w_q
        k_flat = normalized @ self.w_k
        v_flat = normalized @ self.w_v
        if self.b_q is not None:
            q_flat = q_flat + self.b_q
            k_flat = k_flat + self.b_k
            v_flat = v_flat + self.b_v
        q = self._split(q_flat, self.heads)
        k = self._split(k_flat, self.kv_heads)
        v = self._split(v_flat, self.kv_heads)
        scores = torch.einsum(
            "bhqd,bhkd->bhqk", q, k[:, self.kv_index]
        ) / math.sqrt(self.head_dim)
        probabilities = self._causal_softmax(scores)
        context = torch.einsum(
            "bhqk,bhkd->bhqd", probabilities, v[:, self.kv_index]
        )
        delta = context.transpose(1, 2).reshape(
            self.batch, self.seq, self.heads * self.head_dim
        ) @ self.w_o
        post_attention = hidden + delta
        normalized_ffn = post_attention / self._rms(post_attention) * self.gamma_ffn
        activated = torch.nn.functional.silu(normalized_ffn @ self.w_gate) * (
            normalized_ffn @ self.w_up
        )
        return {
            "scores": scores,
            "probabilities": probabilities,
            "delta": delta,
            "post_attention": post_attention,
            "activated": activated,
            "down": activated @ self.w_down,
            "output": post_attention + activated @ self.w_down,
        }

    # -- deployed forward (no decode) -----------------------------------
    def deployed(self, state: torch.Tensor) -> dict:
        attention = self.attention
        scale = self.basis.rms_scale(state, _EPS).to(dtype=_FP64)
        q = self._split(
            (state @ attention.query.to(_FP64)) / scale + attention.query_bias.to(_FP64),
            self.heads,
        )
        k = self._split(
            (state @ attention.key.to(_FP64)) / scale + attention.key_bias.to(_FP64),
            self.kv_heads,
        )
        mixed_value = torch.einsum(
            "btn,hne->bhte", state, attention.value.to(_FP64)
        ) / scale.unsqueeze(1) + attention.value_bias.to(_FP64)[None, :, None, :]
        scores = torch.einsum(
            "bhqd,bhkd->bhqk", q, k[:, self.kv_index]
        ) / math.sqrt(self.head_dim)
        probabilities = self._causal_softmax(scores)
        mixed_context = torch.einsum(
            "bhqk,bhkd->bhqd", probabilities, mixed_value[:, self.kv_index]
        )
        noise_read = self.basis.noise_projection()
        post_attention = (
            state
            + torch.einsum(
                "bhqd,hdn->bqn", mixed_context, attention.output.to(_FP64)
            )
            + (state @ noise_read) @ attention.noise_out.to(_FP64)
            + attention.refresh_out.to(_FP64).sum(0)
        )
        feed_forward = self.feed_forward
        scale_ffn = self.basis.rms_scale(post_attention, _EPS).to(dtype=_FP64)
        activated = torch.nn.functional.silu(
            (post_attention @ feed_forward.gate.to(_FP64)) / scale_ffn
        ) * ((post_attention @ feed_forward.up.to(_FP64)) / scale_ffn)
        output = (
            post_attention
            + activated @ feed_forward.output.to(_FP64)
            + (post_attention @ noise_read) @ feed_forward.noise_out.to(_FP64)
            + feed_forward.refresh_out.to(_FP64).sum(0)
        )
        return {
            "scores": scores,
            "probabilities": probabilities,
            "post_attention": post_attention,
            "activated": activated,
            "output": output,
            "mixed_context": mixed_context,
        }

    def mixed_state(self, hidden: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.basis.mix(torch.cat((hidden, noise), dim=-1))

    def sample(self):
        hidden = torch.randn(self.batch, self.seq, self.hidden, dtype=_FP64)
        noise = torch.randn(self.batch, self.seq, self.noise, dtype=_FP64) * 0.1
        return hidden, noise


# ---------------------------------------------------------------------
def test_qk_scores_match_plaintext() -> None:
    setup = _Setup()
    hidden, noise = setup.sample()
    reference = setup.plaintext(hidden)
    observed = setup.deployed(setup.mixed_state(hidden, noise))
    assert torch.allclose(reference["scores"], observed["scores"], atol=_TOL)


def test_softmax_probabilities_match_plaintext() -> None:
    setup = _Setup()
    hidden, noise = setup.sample()
    reference = setup.plaintext(hidden)
    observed = setup.deployed(setup.mixed_state(hidden, noise))
    assert torch.allclose(
        reference["probabilities"], observed["probabilities"], atol=_TOL
    )


def test_post_attention_decodes_to_the_plaintext_residual() -> None:
    setup = _Setup()
    hidden, noise = setup.sample()
    reference = setup.plaintext(hidden)
    observed = setup.deployed(setup.mixed_state(hidden, noise))
    decoded = observed["post_attention"] @ setup.basis.signal_projection()
    assert torch.allclose(decoded, reference["post_attention"], rtol=_TOL, atol=_TOL)


def test_swiglu_covariance_holds() -> None:
    setup = _Setup()
    hidden, noise = setup.sample()
    reference = setup.plaintext(hidden)
    observed = setup.deployed(setup.mixed_state(hidden, noise))
    permutation = setup.neuron_permutation
    expected = (reference["activated"] * setup.neuron_scale)[..., permutation]
    assert torch.allclose(observed["activated"], expected, rtol=_TOL, atol=_TOL)


def test_block_output_decodes_to_the_plaintext_block_output() -> None:
    setup = _Setup()
    hidden, noise = setup.sample()
    reference = setup.plaintext(hidden)
    observed = setup.deployed(setup.mixed_state(hidden, noise))
    decoded = observed["output"] @ setup.basis.signal_projection()
    assert torch.allclose(decoded, reference["output"], rtol=_TOL, atol=_TOL)


def test_block_output_is_independent_of_the_incoming_auxiliary_state() -> None:
    """The signal path must not depend on ``e``; only the auxiliary path may."""

    setup = _Setup()
    hidden, _ = setup.sample()
    first = torch.randn(setup.batch, setup.seq, setup.noise, dtype=_FP64) * 0.1
    second = torch.randn(setup.batch, setup.seq, setup.noise, dtype=_FP64) * 0.3
    projection = setup.basis.signal_projection()
    out_first = setup.deployed(setup.mixed_state(hidden, first))["output"] @ projection
    out_second = setup.deployed(setup.mixed_state(hidden, second))["output"] @ projection
    assert torch.allclose(out_first, out_second, rtol=_TOL, atol=_TOL)


def test_auxiliary_state_actually_changes_the_mixed_state() -> None:
    """Sanity: the auxiliary path must not be silently zero."""

    setup = _Setup()
    hidden, _ = setup.sample()
    first = torch.zeros(setup.batch, setup.seq, setup.noise, dtype=_FP64)
    second = torch.randn(setup.batch, setup.seq, setup.noise, dtype=_FP64) * 0.3
    out_first = setup.deployed(setup.mixed_state(hidden, first))["output"]
    out_second = setup.deployed(setup.mixed_state(hidden, second))["output"]
    assert not torch.allclose(out_first, out_second, atol=1e-6)
    noise_read = setup.basis.noise_projection()
    assert (out_first @ noise_read - out_second @ noise_read).abs().max() > 1e-6


def test_bias_terms_are_handled() -> None:
    setup = _Setup(with_bias=True)
    hidden, noise = setup.sample()
    reference = setup.plaintext(hidden)
    observed = setup.deployed(setup.mixed_state(hidden, noise))
    assert torch.allclose(reference["scores"], observed["scores"], atol=_TOL)
    decoded = observed["output"] @ setup.basis.signal_projection()
    assert torch.allclose(decoded, reference["output"], rtol=_TOL, atol=_TOL)


def test_fixed_refresh_is_absorbed_without_changing_the_signal() -> None:
    setup = _Setup(fixed_refresh=True)
    hidden, noise = setup.sample()
    reference = setup.plaintext(hidden)
    observed = setup.deployed(setup.mixed_state(hidden, noise))
    decoded = observed["output"] @ setup.basis.signal_projection()
    assert torch.allclose(decoded, reference["output"], rtol=_TOL, atol=_TOL)


def test_grouped_query_attention_mapping_is_respected() -> None:
    setup = _Setup(heads=8, kv_heads=2, head_dim=8, hidden=64)
    hidden, noise = setup.sample()
    reference = setup.plaintext(hidden)
    observed = setup.deployed(setup.mixed_state(hidden, noise))
    decoded = observed["output"] @ setup.basis.signal_projection()
    assert torch.allclose(decoded, reference["output"], rtol=_TOL, atol=_TOL)


def test_multi_head_attention_without_grouping() -> None:
    setup = _Setup(heads=4, kv_heads=4, head_dim=16, hidden=64)
    hidden, noise = setup.sample()
    reference = setup.plaintext(hidden)
    observed = setup.deployed(setup.mixed_state(hidden, noise))
    decoded = observed["output"] @ setup.basis.signal_projection()
    assert torch.allclose(decoded, reference["output"], rtol=_TOL, atol=_TOL)


def test_pytorch_layout_weight_is_rejected() -> None:
    setup = _Setup()
    with pytest.raises(ValueError, match="input dimension"):
        build_deployed_feed_forward(
            basis=setup.basis,
            gamma_ffn=setup.gamma_ffn,
            gate_weight_math=setup.w_gate.T,  # wrong layout on purpose
            up_weight_math=setup.w_up,
            down_weight_math=setup.w_down,
            neuron_permutation=setup.neuron_permutation,
            neuron_scale=setup.neuron_scale,
            swiglu_noise_coupling=setup.coupling_z,
            down_noise_coupling=setup.coupling_d,
            noise_propagator=setup.propagator_f,
            fixed_refresh=None,
        )


def test_deployed_shapes_are_as_documented() -> None:
    setup = _Setup()
    attention, feed_forward = setup.attention, setup.feed_forward
    assert attention.query.shape == (setup.total, setup.heads * setup.head_dim)
    assert attention.key.shape == (setup.total, setup.kv_heads * setup.head_dim)
    assert attention.value.shape == (
        setup.kv_heads, setup.total, setup.head_dim + setup.value_noise,
    )
    assert attention.output.shape == (
        setup.heads, setup.head_dim + setup.value_noise, setup.total,
    )
    assert attention.noise_out.shape == (setup.noise, setup.total)
    assert feed_forward.gate.shape == (setup.total, setup.intermediate)
    assert feed_forward.up.shape == (setup.total, setup.intermediate)
    assert feed_forward.output.shape == (setup.intermediate, setup.total)


def test_auxiliary_budget_validation_accepts_default_scales() -> None:
    setup = _Setup()
    hidden, _ = setup.sample()
    ratio = validate_auxiliary_budget(
        basis=setup.basis,
        sample_signal=hidden,
        signal_noise_coupling=setup.coupling_o,
        noise_propagator=setup.propagator_a,
        context="test",
    )
    assert 0.0 < ratio < 1.0


def test_auxiliary_budget_validation_rejects_a_divergent_propagator() -> None:
    setup = _Setup()
    hidden, _ = setup.sample()
    with pytest.raises(ValueError, match="not contractive"):
        validate_auxiliary_budget(
            basis=setup.basis,
            sample_signal=hidden,
            signal_noise_coupling=setup.coupling_o,
            noise_propagator=1.5 * torch.eye(setup.noise, dtype=_FP64),
            context="test",
        )


def test_auxiliary_budget_validation_rejects_an_oversized_coupling() -> None:
    setup = _Setup()
    hidden, _ = setup.sample()
    with pytest.raises(ValueError, match="exceeds the validated"):
        validate_auxiliary_budget(
            basis=setup.basis,
            sample_signal=hidden,
            signal_noise_coupling=setup.coupling_o * 5000.0,
            noise_propagator=setup.propagator_a,
            context="test",
        )


def test_rho_is_the_precision_bottleneck_not_the_algebra() -> None:
    """Pin the source of the residual block error.

    The deployed algebra is exact in FP64; the remaining error comes entirely
    from the FP32 Gram reduction used for ``rho``. If this ever inverts (FP64
    Gram no longer far more accurate), the basis construction regressed.
    """

    setup = _Setup()
    hidden, noise = setup.sample()
    state = setup.mixed_state(hidden, noise)
    exact = torch.sqrt(hidden.pow(2).mean(-1, keepdim=True) + _EPS)

    fp32_rho = setup.basis.rms_scale(state, _EPS).to(dtype=_FP64)
    fp32_error = float(((fp32_rho - exact).abs() / exact).max())

    basis = setup.basis
    gathered = state[..., basis.perm_out]
    segments = gathered.reshape(
        *state.shape[:-1], basis.block_count, basis.block_size
    )
    squared = torch.einsum(
        "...mi,mij,...mj->...", segments, basis.gram_blocks, segments
    ).clamp_min(0)
    fp64_rho = torch.sqrt(squared.unsqueeze(-1) / setup.hidden + _EPS)
    fp64_error = float(((fp64_rho - exact).abs() / exact).max())

    assert fp32_error < 1e-6
    assert fp64_error < 1e-14
    assert fp64_error < fp32_error / 1000.0
