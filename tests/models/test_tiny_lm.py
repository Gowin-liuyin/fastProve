from __future__ import annotations

import pytest
import torch

from fastprove.layers.attention import ApproximationConfig, AttentionMode
from fastprove.models.obfuscated import ObfuscatedTinyCausalLM
from fastprove.models.plain import PlainTinyCausalLM
from fastprove.seed import RequestContext

from tests._fixtures import model_config, obfuscation_config, tiny_lm_pair


def _model_config():
    return model_config(
        vocab_size=48,
        hidden_size=24,
        intermediate_size=48,
        max_sequence_length=24,
    )


def _obfuscation():
    return obfuscation_config(refresh_mode="per_request")


def _decoded_logits(codec, logits):
    """Recover plaintext-domain logits from the column-permuted output."""

    return logits[..., codec.permutation]


def test_tiny_exact_logits_and_greedy_tokens_match_plaintext() -> None:
    plain, converted = tiny_lm_pair(
        _model_config(),
        plain_seed=501,
        conversion_seed=502,
        obfuscation=_obfuscation(),
    )
    input_ids = torch.tensor(
        [[1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]]
    )
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    plain_logits = plain(input_ids, token_mask=mask)
    codec = converted.token_codec
    exact_logits = _decoded_logits(
        codec,
        converted.module(
            codec.encode(input_ids),
            token_mask=mask,
            request_context=RequestContext(503, "tiny-exact"),
        ),
    )
    torch.testing.assert_close(
        exact_logits, plain_logits, atol=2e-4, rtol=2e-4
    )
    assert torch.equal(
        exact_logits.argmax(dim=-1), plain_logits.argmax(dim=-1)
    )
    assert torch.isfinite(exact_logits).all()

    plain_generated = plain.generate_greedy(input_ids, max_new_tokens=4)
    exact_generated = codec.decode(
        converted.module.generate_greedy(
            codec.encode(input_ids),
            max_new_tokens=4,
            request_context=RequestContext(503, "tiny-exact"),
        )
    )
    assert torch.equal(exact_generated, plain_generated)


def test_tiny_generation_preserves_explicit_prompt_mask_contract() -> None:
    plain, converted = tiny_lm_pair(
        _model_config(),
        plain_seed=518,
        conversion_seed=519,
        debug_enabled=False,
        obfuscation=_obfuscation(),
    )
    tokens = torch.tensor([[1, 2, 0, 0]])
    mask = torch.tensor([[True, True, False, False]])
    plain_generated = plain.generate_greedy(
        tokens,
        token_mask=mask,
        max_new_tokens=2,
    )
    codec = converted.token_codec
    exact_generated = codec.decode(
        converted.module.generate_greedy(
            codec.encode(tokens),
            token_mask=mask,
            max_new_tokens=2,
            request_context=RequestContext(520, "tiny-padded-generation"),
        )
    )
    assert plain_generated.shape == exact_generated.shape == (1, 6)
    assert torch.isfinite(plain_generated.float()).all()
    assert torch.equal(exact_generated, plain_generated)


def test_tiny_models_use_identical_base_weights() -> None:
    plain, converted = tiny_lm_pair(
        _model_config(),
        plain_seed=504,
        conversion_seed=505,
        obfuscation=_obfuscation(),
    )
    codec = converted.token_codec
    ids = torch.tensor([[1, 2, 3]])
    state = converted.module.embedding(codec.encode(ids))
    decoded_signal, _ = converted.client.decode_debug(state)
    torch.testing.assert_close(
        decoded_signal,
        plain.embedding.weight[ids],
        atol=1e-6,
        rtol=1e-6,
    )
    # The final norm and LM head are fused into the column-permuted
    # deployed_head ([n, V]) by the offline conversion; their plaintext
    # buffers are no longer shipped.
    assert not hasattr(converted.module, "final_norm_weight")
    assert not hasattr(converted.module, "lm_head_weight")
    expected_head = (
        plain.final_norm_weight.detach().to(torch.float64)[:, None]
        * plain.lm_head.weight.detach().T.contiguous().to(torch.float64)
    )
    projection = converted.client._transform.signal_projection()
    total = (
        converted.module.config.hidden_size
        + converted.module.obfuscation.hidden_noise_dim
    )
    assert projection.shape == (total, converted.module.config.hidden_size)
    fused = projection @ expected_head
    expected_deployed = fused[:, codec.inverse_permutation]
    torch.testing.assert_close(
        converted.module.deployed_head,
        expected_deployed.to(dtype=torch.float32),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "mode", [AttentionMode.TOPK_PRESERVING, AttentionMode.FREE_BOUNDED]
)
def test_tiny_approximate_modes_are_finite_and_reproducible(
    mode: AttentionMode,
) -> None:
    _, converted = tiny_lm_pair(
        _model_config(),
        plain_seed=506,
        conversion_seed=507,
        mode=mode,
        approximation=ApproximationConfig(0.03, 0.1, 0.8, 2),
        debug_enabled=False,
        obfuscation=_obfuscation(),
    )
    input_ids = torch.arange(1, 13).reshape(2, 6) % _model_config().vocab_size
    encoded = converted.token_codec.encode(input_ids)
    context = RequestContext(508, "tiny-approx")
    first = converted.module(encoded, request_context=context)
    repeated = converted.module(encoded, request_context=context)
    torch.testing.assert_close(first, repeated)
    assert torch.isfinite(first).all()


def test_tiny_production_api_returns_only_logits() -> None:
    _, converted = tiny_lm_pair(
        _model_config(),
        plain_seed=509,
        conversion_seed=510,
        debug_enabled=False,
        obfuscation=_obfuscation(),
    )
    output = converted.module(
        converted.token_codec.encode(torch.tensor([[1, 2, 3]])),
        request_context=RequestContext(511, "production-lm"),
    )
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 3, _model_config().vocab_size)
    persistent_names = tuple(converted.module.state_dict())
    buffer_names = tuple(name for name, _ in converted.module.named_buffers())
    assert all("inverse" not in name for name in persistent_names)
    assert all("checkpoint_hidden" not in name for name in persistent_names)
    assert all("checkpoint_value" not in name for name in persistent_names)
    assert all("inverse" not in name for name in buffer_names)
    assert all(not block.attention.debug_enabled for block in converted.module.blocks)
    with pytest.raises(PermissionError, match="debug"):
        converted.module.forward_debug(
            converted.token_codec.encode(torch.tensor([[1, 2, 3]])),
            request_context=RequestContext(511, "production-lm"),
        )


@pytest.mark.parametrize(
    "mode,approximation",
    [
        (AttentionMode.EXACT, None),
        (AttentionMode.TOPK_PRESERVING, ApproximationConfig(0.03, 0.1, 0.8, 2)),
        (AttentionMode.FREE_BOUNDED, ApproximationConfig(0.03, 0.1, 0.8, 2)),
    ],
)
def test_tiny_all_modes_cached_decode_matches_full_prefix(
    mode: AttentionMode,
    approximation: ApproximationConfig | None,
) -> None:
    _, converted = tiny_lm_pair(
        _model_config(),
        plain_seed=512,
        conversion_seed=513,
        mode=mode,
        approximation=approximation,
        debug_enabled=False,
        obfuscation=_obfuscation(),
    )
    tokens = torch.tensor([[1, 3, 5, 7, 9]])
    encoded = converted.token_codec.encode(tokens)
    context = RequestContext(514, "tiny-cache")
    full_logits = converted.module(
        encoded,
        positions=torch.arange(5),
        request_context=context,
    )
    _, cache = converted.module(
        encoded[:, :4],
        positions=torch.arange(4),
        request_context=context,
        use_cache=True,
    )
    cached_logits, updated_cache = converted.module(
        encoded[:, 4:],
        positions=torch.tensor([4]),
        request_context=context,
        cache=cache,
        use_cache=True,
    )
    torch.testing.assert_close(
        cached_logits[:, -1],
        full_logits[:, -1],
        atol=2e-4,
        rtol=2e-4,
    )
    assert len(updated_cache.layers) == _model_config().num_layers
    assert all(layer.key.shape[2] == 5 for layer in updated_cache.layers)


@pytest.mark.bf16
def test_tiny_exact_bfloat16_is_optional_but_aligned_when_supported() -> None:
    plain = PlainTinyCausalLM(
        _model_config(), seed=515, debug_enabled=False
    ).to(dtype=torch.bfloat16)
    try:
        converted = ObfuscatedTinyCausalLM.from_plain(
            plain,
            obfuscation=_obfuscation(),
            mode=AttentionMode.EXACT,
            approximation=None,
            seed=516,
            debug_enabled=False,
        )
        converted.module.to(dtype=torch.bfloat16)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        plain_logits = plain(tokens)
        exact_logits = _decoded_logits(
            converted.token_codec,
            converted.module(
                converted.token_codec.encode(tokens),
                request_context=RequestContext(517, "tiny-bfloat16"),
            ),
        )
    except RuntimeError as error:
        if "bfloat16" in str(error).lower() or "not implemented" in str(
            error
        ).lower():
            pytest.skip("BF16 kernels are unavailable on this backend")
        raise
    assert plain_logits.dtype == torch.bfloat16
    assert exact_logits.dtype == torch.bfloat16
    assert torch.isfinite(exact_logits).all()
    torch.testing.assert_close(
        exact_logits, plain_logits, atol=2e-2, rtol=2e-2
    )
    assert torch.equal(
        exact_logits.argmax(dim=-1), plain_logits.argmax(dim=-1)
    )
