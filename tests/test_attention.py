from __future__ import annotations

import math

import pytest
import torch

from fastprove.layers.attention import (
    ApproximationConfig,
    AttentionMode,
    ObfuscatedAttention,
    compute_topk_budget,
    safe_masked_softmax_fp32,
)
from fastprove.seed import RequestContext
from fastprove.transforms import generate_transform


def _tensors() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    generator = torch.Generator().manual_seed(123)
    q = torch.randn(1, 4, 5, 8, generator=generator)
    k = torch.randn(1, 2, 5, 8, generator=generator)
    value_signal = torch.randn(1, 2, 5, 8, generator=generator)
    value_noise = torch.randn(1, 2, 5, 2, generator=generator)
    transform = generate_transform(8, 2, seed=3, domain="value")
    value_mixed = torch.cat((value_signal, value_noise), dim=-1) @ transform.matrix
    valid = torch.tril(torch.ones(5, 5, dtype=torch.bool)).unsqueeze(0)
    return q, k, value_mixed, valid, transform.inverse


def test_safe_masked_softmax_keeps_invalid_logits_and_handles_all_mask() -> None:
    logits = torch.tensor(
        [[[[2.0, 1.0, -3.0], [4.0, 3.0, 2.0]]]], dtype=torch.float32
    )
    valid = torch.tensor(
        [[[[True, False, True], [False, False, False]]]]
    )
    probabilities, masked_logits = safe_masked_softmax_fp32(logits, valid)
    assert torch.isneginf(masked_logits[~valid]).all()
    assert torch.equal(probabilities[~valid], torch.zeros_like(probabilities[~valid]))
    assert probabilities[0, 0, 1].sum().item() == 0.0
    assert torch.isfinite(probabilities).all()


def test_exact_attention_matches_plain_and_value_covariance() -> None:
    q, k, value_mixed, valid, value_inverse = _tensors()
    kv_index = torch.tensor([0, 0, 1, 1])
    context = RequestContext(9, "attention-exact")
    exact = ObfuscatedAttention(
        mode=AttentionMode.EXACT,
        approximation=None,
        layer_id="attn-0",
        debug_enabled=True,
    )
    output, debug = exact.forward_debug(
        q=q,
        k=k,
        value_mixed=value_mixed,
        valid_mask=valid,
        kv_index=kv_index,
        request_context=context,
    )
    scores = torch.einsum(
        "bhqd,bhkd->bhqk", q.float(), k[:, kv_index].float()
    ) / math.sqrt(q.shape[-1])
    probabilities, _ = safe_masked_softmax_fp32(
        scores, valid.unsqueeze(1).expand_as(scores)
    )
    expected = torch.einsum(
        "bhqk,bhkd->bhqd", probabilities, value_mixed[:, kv_index].float()
    )
    torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-6)
    assert torch.equal(debug.clean_probabilities, debug.noisy_probabilities)
    assert torch.count_nonzero(debug.noise).item() == 0

    decoded = output @ value_inverse
    value_augmented = value_mixed @ value_inverse
    expected_decoded = torch.einsum(
        "bhqk,bhkd->bhqd", probabilities, value_augmented[:, kv_index]
    )
    torch.testing.assert_close(decoded, expected_decoded, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize(
    "mode", [AttentionMode.TOPK_PRESERVING, AttentionMode.FREE_BOUNDED]
)
def test_zero_tau_approximate_modes_equal_exact(mode: AttentionMode) -> None:
    q, k, value_mixed, valid, _ = _tensors()
    kv_index = torch.tensor([0, 0, 1, 1])
    context = RequestContext(11, "zero-tau")
    approximation = ApproximationConfig(
        tau_max=0.0, tau_error=0.1, alpha=0.8, preserve_top_k=2
    )
    layer = ObfuscatedAttention(
        mode=mode,
        approximation=approximation,
        layer_id="attn-zero",
        debug_enabled=True,
    )
    output, debug = layer.forward_debug(
        q=q,
        k=k,
        value_mixed=value_mixed,
        valid_mask=valid,
        kv_index=kv_index,
        request_context=context,
    )
    assert torch.count_nonzero(debug.noise).item() == 0
    torch.testing.assert_close(
        debug.noisy_probabilities, debug.clean_probabilities
    )
    assert torch.isfinite(output).all()


def test_topk_budget_is_strict_and_ties_or_short_rows_receive_zero() -> None:
    logits = torch.tensor(
        [[[[5.0, 3.0, 1.0, -2.0], [2.0, 2.0, 0.0, -1.0], [7.0, 0.0, 0.0, 0.0]]]]
    )
    valid = torch.tensor(
        [[[[True, True, True, True], [True, True, True, True], [True, False, False, False]]]]
    )
    config = ApproximationConfig(
        tau_max=10.0, tau_error=10.0, alpha=0.8, preserve_top_k=1
    )
    budget, margin = compute_topk_budget(logits, valid, config)
    assert margin[0, 0, 0, 0].item() == 2.0
    assert 2.0 * budget[0, 0, 0, 0].item() < margin[0, 0, 0, 0].item()
    assert margin[0, 0, 1, 0].item() == 0.0
    assert budget[0, 0, 1, 0].item() == 0.0
    assert budget[0, 0, 2, 0].item() == 0.0


def test_topk_noise_is_deterministic_bounded_and_preserves_set() -> None:
    q, k, value_mixed, valid, _ = _tensors()
    kv_index = torch.tensor([0, 0, 1, 1])
    context = RequestContext(12, "topk")
    approximation = ApproximationConfig(
        tau_max=0.1, tau_error=0.1, alpha=0.8, preserve_top_k=2
    )
    layer = ObfuscatedAttention(
        mode=AttentionMode.TOPK_PRESERVING,
        approximation=approximation,
        layer_id="attn-topk",
        debug_enabled=True,
    )
    _, first = layer.forward_debug(
        q=q,
        k=k,
        value_mixed=value_mixed,
        valid_mask=valid,
        kv_index=kv_index,
        request_context=context,
    )
    _, repeated = layer.forward_debug(
        q=q,
        k=k,
        value_mixed=value_mixed,
        valid_mask=valid,
        kv_index=kv_index,
        request_context=context,
    )
    torch.testing.assert_close(first.noise, repeated.noise)
    assert torch.equal(first.noise[~first.valid_mask], torch.zeros_like(first.noise[~first.valid_mask]))
    realized = first.noise.abs().amax(dim=-1, keepdim=True)
    assert torch.all(realized <= first.tau + 1e-7)

    clean_top2 = torch.sort(
        torch.topk(first.clean_logits, 2, dim=-1).indices, dim=-1
    ).values
    noisy_top2 = torch.sort(
        torch.topk(first.noisy_logits, 2, dim=-1).indices, dim=-1
    ).values
    rows = first.valid_mask.sum(dim=-1) > 2
    assert torch.equal(clean_top2[rows], noisy_top2[rows])


def test_noise_norm_respects_python_float_bound_without_fp32_ulp_overrun() -> None:
    # 0.001 is rounded upward when materialized as float32.  The sampler
    # must still satisfy the user-facing Python-float infinity-norm contract.
    q = torch.ones(1, 1, 1, 1, dtype=torch.float32)
    k = torch.tensor([[[[3.0], [1.0], [0.0]]]], dtype=torch.float32)
    value = torch.arange(3, dtype=torch.float32).reshape(1, 1, 3, 1)
    valid = torch.ones(1, 1, 3, dtype=torch.bool)
    config = ApproximationConfig(
        tau_max=0.001, tau_error=0.001, alpha=0.8, preserve_top_k=1
    )
    layer = ObfuscatedAttention(
        mode=AttentionMode.TOPK_PRESERVING,
        approximation=config,
        layer_id="ulp-bound",
        debug_enabled=True,
    )
    _, debug = layer.forward_debug(
        q=q,
        k=k,
        value_mixed=value,
        valid_mask=valid,
        kv_index=torch.tensor([0]),
        request_context=RequestContext(13, "ulp-bound"),
    )
    assert debug.tau.item() > config.tau_max
    assert debug.noise.abs().amax().item() <= config.tau_max


def test_free_bounded_uses_fixed_budget_without_margin_cap() -> None:
    logits = torch.tensor([[[[0.001, 0.0, -1.0]]]], dtype=torch.float32)
    valid = torch.ones_like(logits, dtype=torch.bool)
    q = torch.tensor([[[[1.0]]]])
    k = logits.transpose(-1, -2)
    value = torch.arange(3, dtype=torch.float32).reshape(1, 1, 3, 1)
    config = ApproximationConfig(
        tau_max=0.05, tau_error=0.04, alpha=0.8, preserve_top_k=1
    )
    layer = ObfuscatedAttention(
        mode=AttentionMode.FREE_BOUNDED,
        approximation=config,
        layer_id="free",
        debug_enabled=True,
    )
    _, debug = layer.forward_debug(
        q=q,
        k=k,
        value_mixed=value,
        valid_mask=valid[:, 0],
        kv_index=torch.tensor([0]),
        request_context=RequestContext(4, "free"),
    )
    torch.testing.assert_close(debug.tau, torch.full_like(debug.tau, 0.04))
    assert 2.0 * debug.tau.item() > 0.001
    assert debug.noise.abs().max().item() <= 0.04 + 1e-7


def test_production_api_does_not_return_probabilities() -> None:
    q, k, value_mixed, valid, _ = _tensors()
    layer = ObfuscatedAttention(
        mode=AttentionMode.EXACT,
        approximation=None,
        layer_id="production",
        debug_enabled=False,
    )
    output = layer(
        q=q,
        k=k,
        value_mixed=value_mixed,
        valid_mask=valid,
        kv_index=torch.tensor([0, 0, 1, 1]),
        request_context=RequestContext(1, "prod"),
    )
    assert isinstance(output, torch.Tensor)
    with pytest.raises(PermissionError, match="debug"):
        layer.forward_debug(
            q=q,
            k=k,
            value_mixed=value_mixed,
            valid_mask=valid,
            kv_index=torch.tensor([0, 0, 1, 1]),
            request_context=RequestContext(1, "prod"),
        )


@pytest.mark.parametrize("field", ["tau_max", "tau_error", "alpha"])
def test_approximation_config_rejects_nonfinite_values(field: str) -> None:
    values = {"tau_max": 0.1, "tau_error": 0.1, "alpha": 0.8}
    values[field] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        ApproximationConfig(**values, preserve_top_k=2)
