"""Bounded RMS-scale (ρ) noise: the 折衷 deployment path.

The exact / bound=0 path must leave Gram ρ unchanged. The deployment path
adds a request-seeded field whose infinity-norm is at most the configured
bound. ξ refresh stays on the auxiliary chain and must not enter ρ.
"""

from __future__ import annotations

from pathlib import Path

import torch

from fastprove.config import load_config
from fastprove.layers.attention import AttentionMode
from fastprove.layers.rmsnorm import perturb_rms_scale
from fastprove.models.obfuscated import ObfuscatedDecoderBlock
from fastprove.models.plain import PlainDecoderBlock
from fastprove.seed import RequestContext

from tests._fixtures import (
    decoder_block_pair,
    model_config,
    obfuscation_config,
    tiny_lm_pair,
    unit_basis,
)


def _model_config():
    return model_config()


def _obfuscation(
    *,
    refresh_mode: str = "fixed_debug",
    rho_noise_bound: float = 0.0,
):
    return obfuscation_config(
        refresh_mode=refresh_mode, rho_noise_bound=rho_noise_bound
    )


def _convert_block(rho_noise_bound: float = 0.0):
    _, converted = decoder_block_pair(
        _model_config(),
        obfuscation=_obfuscation(rho_noise_bound=rho_noise_bound),
    )
    return converted


def _distinct_norm_rows(vocab: int, dim: int) -> torch.Tensor:
    """Unit-direction rows scaled so pairwise Euclidean norms are unique."""

    generator = torch.Generator().manual_seed(7)
    directions = torch.randn(vocab, dim, generator=generator)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    norms = torch.linspace(1.0, 1.04, vocab)
    return directions * norms[:, None]


def _estimated_signal_norm(rho: torch.Tensor, dim: int, eps: float) -> torch.Tensor:
    return torch.sqrt(
        (rho.to(dtype=torch.float32).squeeze(-1).square() - float(eps)).clamp_min(0)
        * float(dim)
    )


def _unique_correct_recovery(
    estimated_norms: torch.Tensor, table_norms: torch.Tensor
) -> float:
    distance = (estimated_norms[:, None] - table_norms[None, :]).abs()
    nearest = distance.argmin(dim=-1)
    ranked = distance.topk(2, dim=-1, largest=False).values
    unique = ranked[:, 1] > ranked[:, 0]
    true_ids = torch.arange(estimated_norms.shape[0])
    correct = nearest == true_ids
    return float((unique & correct).to(dtype=torch.float32).mean().item())


def test_zero_bound_is_a_noop_on_gram_rho() -> None:
    basis = unit_basis(
        32, 8, seed=11, domain="rho-zero", block_size=8, dtype=torch.float32
    )
    generator = torch.Generator().manual_seed(3)
    signal = torch.randn(2, 5, 32, generator=generator)
    noise = torch.randn(2, 5, 8, generator=generator) * 0.2
    mixed = basis.mix(torch.cat((signal, noise), dim=-1))
    eps = 1e-5
    clean = basis.rms_scale(mixed, eps)
    context = RequestContext(1, "zero-bound")
    actual = perturb_rms_scale(
        clean, 0.0, context=context, domain="block-0-attn", positions=None
    )
    torch.testing.assert_close(actual, clean)
    expected = torch.sqrt(signal.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_nonzero_bound_respects_infinity_norm_and_is_seed_deterministic() -> None:
    rho = torch.full((2, 6, 1), 0.4, dtype=torch.float32)
    bound = 0.001
    context = RequestContext(13, "rho-det")
    first = perturb_rms_scale(
        rho, bound, context=context, domain="block-0-attn", positions=None
    )
    second = perturb_rms_scale(
        rho, bound, context=context, domain="block-0-attn", positions=None
    )
    torch.testing.assert_close(first, second)
    noise = first - rho
    assert float(noise.abs().amax().item()) <= bound
    other = perturb_rms_scale(
        rho,
        bound,
        context=RequestContext(13, "rho-det-other"),
        domain="block-0-attn",
        positions=None,
    )
    assert not torch.allclose(first, other)


def test_python_float_bound_does_not_overrun_from_fp32_ulp() -> None:
    rho = torch.ones(1, 4, 1, dtype=torch.float32)
    bound = 0.001
    perturbed = perturb_rms_scale(
        rho,
        bound,
        context=RequestContext(13, "ulp-rho"),
        domain="final",
        positions=None,
    )
    assert float((perturbed - rho).abs().amax().item()) <= bound


def test_auxiliary_xi_does_not_enter_rho() -> None:
    basis = unit_basis(
        32, 8, seed=19, domain="rho-xi", block_size=8, dtype=torch.float32
    )
    generator = torch.Generator().manual_seed(4)
    signal = torch.randn(2, 4, 32, generator=generator)
    first_noise = torch.randn(2, 4, 8, generator=generator) * 0.3
    second_noise = torch.randn(2, 4, 8, generator=generator) * 0.3
    eps = 1e-5
    context = RequestContext(8, "xi-isolated")
    first = perturb_rms_scale(
        basis.rms_scale(basis.mix(torch.cat((signal, first_noise), dim=-1)), eps),
        0.01,
        context=context,
        domain="block-0-attn",
        positions=None,
    )
    second = perturb_rms_scale(
        basis.rms_scale(basis.mix(torch.cat((signal, second_noise), dim=-1)), eps),
        0.01,
        context=context,
        domain="block-0-attn",
        positions=None,
    )
    torch.testing.assert_close(first, second, rtol=1e-4, atol=1e-5)


def test_position_indexed_noise_matches_across_sequence_length() -> None:
    rho_prefill = torch.full((1, 4, 1), 0.5, dtype=torch.float32)
    rho_decode = torch.full((1, 1, 1), 0.5, dtype=torch.float32)
    bound = 0.02
    context = RequestContext(21, "pos-stable")
    prefill = perturb_rms_scale(
        rho_prefill,
        bound,
        context=context,
        domain="block-0-attn",
        positions=torch.arange(4),
    )
    decode = perturb_rms_scale(
        rho_decode,
        bound,
        context=context,
        domain="block-0-attn",
        positions=torch.tensor([2]),
    )
    torch.testing.assert_close(decode, prefill[:, 2:3])


def test_exact_block_keeps_zero_rho_noise() -> None:
    converted = _convert_block(rho_noise_bound=0.0)
    assert converted.module.obfuscation.rho_noise_bound == 0.0
    generator = torch.Generator().manual_seed(303)
    hidden = torch.randn(2, 7, 32, generator=generator)
    noise = torch.randn(2, 7, 8, generator=generator)
    token_mask = torch.ones(2, 7, dtype=torch.bool)
    state = converted.client.encode_debug(hidden, noise)
    first, _ = converted.module.forward_debug(
        state,
        token_mask=token_mask,
        request_context=RequestContext(404, "exact-a"),
    )
    second, _ = converted.module.forward_debug(
        state,
        token_mask=token_mask,
        request_context=RequestContext(405, "exact-b"),
    )
    first_signal, _ = converted.client.decode_debug(first)
    second_signal, _ = converted.client.decode_debug(second)
    torch.testing.assert_close(first_signal, second_signal, rtol=1e-5, atol=1e-5)


def test_wired_rho_bound_changes_the_exact_block_signal() -> None:
    plain = PlainDecoderBlock(
        _model_config(), seed=101, layer_id=0, debug_enabled=True
    )
    quiet = ObfuscatedDecoderBlock.from_plain(
        plain,
        obfuscation=_obfuscation(rho_noise_bound=0.0),
        mode=AttentionMode.EXACT,
        approximation=None,
        seed=202,
        debug_enabled=True,
    )
    noisy = ObfuscatedDecoderBlock.from_plain(
        plain,
        obfuscation=_obfuscation(rho_noise_bound=0.05),
        mode=AttentionMode.EXACT,
        approximation=None,
        seed=202,
        debug_enabled=True,
    )
    generator = torch.Generator().manual_seed(303)
    hidden = torch.randn(2, 7, 32, generator=generator)
    noise = torch.randn(2, 7, 8, generator=generator)
    token_mask = torch.ones(2, 7, dtype=torch.bool)
    context = RequestContext(404, "wired")
    state = quiet.client.encode_debug(hidden, noise)
    quiet_state, _ = quiet.module.forward_debug(
        state, token_mask=token_mask, request_context=context
    )
    noisy_state, _ = noisy.module.forward_debug(
        state, token_mask=token_mask, request_context=context
    )
    quiet_signal, _ = quiet.client.decode_debug(quiet_state)
    noisy_signal, _ = noisy.client.decode_debug(noisy_state)
    assert not torch.allclose(quiet_signal, noisy_signal, rtol=1e-4, atol=1e-4)


def test_distinct_norm_unique_recovery_drops_when_bound_is_on() -> None:
    dim = 32
    vocab = 48
    eps = 1e-5
    rows = _distinct_norm_rows(vocab, dim)
    basis = unit_basis(
        dim, 8, seed=23, domain="rho-recover", block_size=8, dtype=torch.float32
    )
    auxiliary = torch.zeros(vocab, 8)
    mixed = basis.mix(torch.cat((rows, auxiliary), dim=-1)).unsqueeze(0)
    clean = basis.rms_scale(mixed, eps)
    table_norms = rows.norm(dim=-1)
    clean_rate = _unique_correct_recovery(
        _estimated_signal_norm(clean.squeeze(0), dim, eps), table_norms
    )
    noisy = perturb_rms_scale(
        clean,
        0.02,
        context=RequestContext(31, "recover"),
        domain="block-0-attn",
        positions=None,
    )
    noisy_rate = _unique_correct_recovery(
        _estimated_signal_norm(noisy.squeeze(0), dim, eps), table_norms
    )
    assert clean_rate == 1.0
    assert noisy_rate < clean_rate


def test_compromise_yaml_keeps_per_request_refresh_and_nonzero_rho_bound() -> None:
    exact = load_config(Path("configs/tiny_exact.yaml"))
    approx = load_config(Path("configs/tiny_approx.yaml"))
    assert exact.obfuscation.rho_noise_bound == 0.0
    assert approx.obfuscation.refresh_mode == "per_request"
    assert approx.obfuscation.rho_noise_bound > 0.0


def test_tiny_compromise_forward_does_not_require_decode() -> None:
    _, converted = tiny_lm_pair(
        _model_config(),
        plain_seed=501,
        conversion_seed=501,
        debug_enabled=False,
        obfuscation=_obfuscation(
            refresh_mode="per_request", rho_noise_bound=1.0e-5
        ),
    )
    ids = torch.randint(0, 64, (2, 5))
    logits = converted.module(
        ids, request_context=RequestContext(9, "compromise-fwd")
    )
    assert torch.isfinite(logits).all()
    assert logits.shape == (2, 5, 64)
