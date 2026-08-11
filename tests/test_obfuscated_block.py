from __future__ import annotations

import pytest
import torch

from fastprove.config import ModelConfig, ObfuscationConfig
from fastprove.layers.attention import ApproximationConfig, AttentionMode
from fastprove.models.obfuscated import ObfuscatedDecoderBlock
from fastprove.models.plain import PlainDecoderBlock
from fastprove.seed import RequestContext
from fastprove.state import MixedState


def _model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_sequence_length=32,
        rms_epsilon=1e-5,
        rope_theta=10000.0,
    )


def _obfuscation(refresh_mode: str = "fixed_debug") -> ObfuscationConfig:
    return ObfuscationConfig(
        hidden_noise_dim=8,
        value_noise_dim_per_head=2,
        max_condition_number=10.0,
        noise_propagation_gamma=0.5,
        refresh_mode=refresh_mode,
        basis_block_size=8,
    )


def _convert(
    mode: AttentionMode = AttentionMode.EXACT,
    approximation: ApproximationConfig = None,
    refresh_mode: str = "fixed_debug",
):
    plain = PlainDecoderBlock(
        _model_config(), seed=101, layer_id=0, debug_enabled=True
    )
    converted = ObfuscatedDecoderBlock.from_plain(
        plain,
        obfuscation=_obfuscation(refresh_mode),
        mode=mode,
        approximation=approximation,
        seed=202,
        debug_enabled=True,
    )
    return plain, converted


def test_exact_obfuscated_block_matches_plaintext_checkpoints() -> None:
    plain, converted = _convert()
    generator = torch.Generator().manual_seed(303)
    hidden = torch.randn(2, 7, 32, generator=generator)
    noise = torch.randn(2, 7, 8, generator=generator)
    token_mask = torch.tensor(
        [
            [True, True, True, True, True, False, False],
            [True, True, True, True, True, True, True],
        ]
    )
    plain_output, plain_debug = plain.forward_debug(hidden, token_mask)
    state = converted.client.encode_debug(hidden, noise)
    actual_state, debug = converted.module.forward_debug(
        state,
        token_mask=token_mask,
        request_context=RequestContext(404, "exact-block"),
    )
    actual_signal, actual_noise = converted.client.decode_debug(actual_state)

    assert debug.qk_score_error["max_absolute_error"] <= 5e-5
    assert debug.softmax_error["max_absolute_error"] <= 5e-6
    assert debug.valid_mask.dtype == torch.bool
    assert torch.equal(
        debug.valid_mask,
        torch.isfinite(debug.clean_logits),
    )
    torch.testing.assert_close(
        debug.clean_attention_output,
        plain_debug.attention_output,
        atol=5e-5,
        rtol=5e-5,
    )
    torch.testing.assert_close(
        debug.attention_output,
        plain_debug.attention_output,
        atol=5e-5,
        rtol=5e-5,
    )
    torch.testing.assert_close(
        debug.post_attention,
        plain_debug.post_attention,
        atol=6e-5,
        rtol=6e-5,
    )
    torch.testing.assert_close(
        actual_signal, plain_output, atol=1e-4, rtol=1e-4
    )
    assert torch.isfinite(actual_signal).all()
    assert torch.isfinite(actual_noise).all()


def test_exact_signal_is_independent_of_input_auxiliary_noise() -> None:
    _, converted = _convert()
    hidden = torch.randn(1, 5, 32)
    zero_noise = torch.zeros(1, 5, 8)
    nonzero_noise = torch.randn(1, 5, 8)
    context = RequestContext(405, "side-path")
    zero_output = converted.module(
        converted.client.encode_debug(hidden, zero_noise),
        request_context=context,
    )
    nonzero_output = converted.module(
        converted.client.encode_debug(hidden, nonzero_noise),
        request_context=context,
    )
    zero_signal, zero_aux = converted.client.decode_debug(zero_output)
    nonzero_signal, nonzero_aux = converted.client.decode_debug(nonzero_output)
    torch.testing.assert_close(
        zero_signal, nonzero_signal, atol=1e-4, rtol=1e-4
    )
    assert not torch.equal(zero_aux, nonzero_aux)
    assert torch.count_nonzero(zero_aux).item() > 0


@pytest.mark.parametrize(
    "mode", [AttentionMode.TOPK_PRESERVING, AttentionMode.FREE_BOUNDED]
)
def test_zero_tau_approximate_block_matches_exact(mode: AttentionMode) -> None:
    exact_plain, exact = _convert()
    approximation = ApproximationConfig(0.0, 0.1, 0.8, 2)
    approximate_plain, approximate = _convert(mode, approximation)
    approximate_plain.load_state_dict(exact_plain.state_dict())
    hidden = torch.randn(1, 6, 32)
    noise = torch.randn(1, 6, 8)
    context = RequestContext(406, "zero-tau-block")
    exact_output = exact.module(
        exact.client.encode_debug(hidden, noise), request_context=context
    )
    approximate_output = approximate.module(
        approximate.client.encode_debug(hidden, noise),
        request_context=context,
    )
    exact_signal, _ = exact.client.decode_debug(exact_output)
    approximate_signal, _ = approximate.client.decode_debug(
        approximate_output
    )
    torch.testing.assert_close(
        approximate_signal, exact_signal, atol=1e-4, rtol=1e-4
    )


def test_obfuscated_production_output_exposes_no_decode_or_attention_data() -> None:
    _, converted = _convert()
    converted.module.debug_enabled = False
    converted.module.attention.debug_enabled = False
    state = converted.client.encode_debug(
        torch.randn(1, 4, 32), torch.randn(1, 4, 8)
    )
    output = converted.module(
        state, request_context=RequestContext(407, "production")
    )
    assert isinstance(output, MixedState)
    assert not hasattr(output.basis, "inverse")
    assert not hasattr(output, "probabilities")
    persistent_names = tuple(converted.module.state_dict())
    buffer_names = tuple(name for name, _ in converted.module.named_buffers())
    assert all("inverse" not in name for name in persistent_names)
    assert all("checkpoint_hidden" not in name for name in persistent_names)
    assert all("checkpoint_value" not in name for name in persistent_names)
    assert all("inverse" not in name for name in buffer_names)
    with pytest.raises(PermissionError, match="debug"):
        converted.module.forward_debug(
            state, request_context=RequestContext(407, "production")
        )
    with pytest.raises(PermissionError, match="debug"):
        converted.module.attention.forward_debug(
            q=torch.randn(1, 4, 4, 8),
            k=torch.randn(1, 2, 4, 8),
            value_mixed=torch.randn(1, 2, 4, 10),
            valid_mask=torch.ones(1, 1, 4, 4, dtype=torch.bool),
            kv_index=torch.tensor([0, 0, 1, 1]),
            request_context=RequestContext(407, "production"),
        )


def test_obfuscated_block_fully_masked_rows_are_finite() -> None:
    _, converted = _convert()
    hidden = torch.randn(1, 4, 32)
    noise = torch.randn(1, 4, 8)
    state, debug = converted.module.forward_debug(
        converted.client.encode_debug(hidden, noise),
        token_mask=torch.zeros(1, 4, dtype=torch.bool),
        request_context=RequestContext(408, "all-mask"),
    )
    signal, auxiliary = converted.client.decode_debug(state)
    assert torch.isfinite(signal).all()
    assert torch.isfinite(auxiliary).all()
    assert debug.noisy_probabilities.sum().item() == 0.0


def test_exact_obfuscated_block_cached_token_matches_full_sequence() -> None:
    _, converted = _convert()
    hidden = torch.randn(1, 5, 32)
    noise = torch.randn(1, 5, 8)
    context = RequestContext(409, "exact-cache")
    full_state = converted.module(
        converted.client.encode_debug(hidden, noise),
        positions=torch.arange(5),
        request_context=context,
    )
    full_signal, _ = converted.client.decode_debug(full_state)

    prefix_state, cache = converted.module(
        converted.client.encode_debug(hidden[:, :4], noise[:, :4]),
        positions=torch.arange(4),
        request_context=context,
        use_cache=True,
    )
    assert prefix_state.mixed.shape[1] == 4
    cached_state, updated_cache = converted.module(
        converted.client.encode_debug(hidden[:, 4:], noise[:, 4:]),
        positions=torch.tensor([4]),
        request_context=context,
        cache=cache,
        use_cache=True,
    )
    cached_signal, _ = converted.client.decode_debug(cached_state)
    torch.testing.assert_close(
        cached_signal,
        full_signal[:, 4:],
        atol=1e-4,
        rtol=1e-4,
    )
    assert updated_cache.key.shape[2] == 5
    assert updated_cache.value_mixed.shape[2] == 5
    assert not hasattr(updated_cache, "inverse")


def test_obfuscated_cache_rejects_cross_request_and_cross_conversion_reuse() -> None:
    _, first = _convert()
    hidden = torch.randn(1, 3, 32)
    noise = torch.randn(1, 3, 8)
    first_context = RequestContext(410, "cache-owner")
    _, cache = first.module(
        first.client.encode_debug(hidden, noise),
        positions=torch.arange(3),
        request_context=first_context,
        use_cache=True,
    )
    with pytest.raises(ValueError, match="request identity"):
        first.module(
            first.client.encode_debug(hidden[:, :1], noise[:, :1]),
            positions=torch.tensor([3]),
            request_context=RequestContext(410, "other-request"),
            cache=cache,
            use_cache=True,
        )

    plain = PlainDecoderBlock(
        _model_config(), seed=101, layer_id=0, debug_enabled=True
    )
    second = ObfuscatedDecoderBlock.from_plain(
        plain,
        obfuscation=_obfuscation(),
        mode=AttentionMode.EXACT,
        approximation=None,
        seed=999,
        debug_enabled=True,
    )
    with pytest.raises(ValueError, match="cache identity"):
        second.module(
            second.client.encode_debug(hidden[:, :1], noise[:, :1]),
            positions=torch.tensor([3]),
            request_context=first_context,
            cache=cache,
            use_cache=True,
        )
