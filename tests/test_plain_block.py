from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fastprove.config import ModelConfig, load_config
from fastprove.layers.attention import AttentionMode
from fastprove.models.plain import PlainDecoderBlock


def _config() -> ModelConfig:
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


def test_yaml_config_loads_and_validates_gqa() -> None:
    config = load_config(Path("configs/tiny_exact.yaml"))
    assert config.model.head_dim == 8
    assert config.model.query_heads_per_kv_head == 2
    assert config.attention.mode == AttentionMode.EXACT
    assert config.runtime.device == "cpu"
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(
            vocab_size=64,
            hidden_size=30,
            intermediate_size=64,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_sequence_length=32,
        )
    with pytest.raises(ValueError, match="attention heads"):
            ModelConfig(
                vocab_size=64,
                hidden_size=36,
                intermediate_size=64,
            num_layers=2,
            num_attention_heads=6,
            num_key_value_heads=4,
            max_sequence_length=32,
        )


def test_plain_block_is_deterministic_and_has_expected_shape() -> None:
    config = _config()
    first = PlainDecoderBlock(config, seed=17, layer_id=0, debug_enabled=True)
    second = PlainDecoderBlock(config, seed=17, layer_id=0, debug_enabled=True)
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left, right)
    x = torch.randn(2, 7, config.hidden_size)
    output = first(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_plain_block_padding_and_fully_masked_query_are_safe() -> None:
    config = _config()
    block = PlainDecoderBlock(config, seed=22, layer_id=0, debug_enabled=True)
    x = torch.randn(2, 5, config.hidden_size)
    token_mask = torch.tensor(
        [[True, True, True, False, False], [False, False, False, False, False]]
    )
    output, debug = block.forward_debug(x, token_mask=token_mask)
    assert torch.isfinite(output).all()
    assert torch.isneginf(debug.masked_logits[~debug.valid_mask]).all()
    assert debug.probabilities[1].sum().item() == 0.0
    assert debug.attention_output[1].abs().max().item() == 0.0


def test_plain_block_causal_mask_blocks_future_keys() -> None:
    config = _config()
    block = PlainDecoderBlock(config, seed=23, layer_id=0, debug_enabled=True)
    x = torch.randn(1, 6, config.hidden_size)
    _, debug = block.forward_debug(x)
    future = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
    future = future[None, None].expand_as(debug.valid_mask)
    assert not debug.valid_mask[future].any()
    assert torch.isneginf(debug.masked_logits[future]).all()
    assert torch.equal(
        debug.probabilities[future],
        torch.zeros_like(debug.probabilities[future]),
    )


def test_plain_block_cached_single_token_matches_full_sequence() -> None:
    config = _config()
    block = PlainDecoderBlock(config, seed=24, layer_id=0, debug_enabled=False)
    x = torch.randn(1, 6, config.hidden_size)
    full = block(x)
    cache = None
    pieces = []
    for position in range(x.shape[1]):
        piece, cache = block(
            x[:, position : position + 1],
            positions=torch.tensor([position]),
            cache=cache,
            use_cache=True,
        )
        pieces.append(piece)
    cached = torch.cat(pieces, dim=1)
    torch.testing.assert_close(cached, full, atol=3e-5, rtol=3e-5)
    assert cache is not None
    assert cache.key.shape[2] == x.shape[1]
    assert cache.value.shape[2] == x.shape[1]


def test_plain_production_forward_does_not_return_attention_debug() -> None:
    block = PlainDecoderBlock(
        _config(), seed=25, layer_id=0, debug_enabled=False
    )
    output = block(torch.randn(1, 3, 32))
    assert isinstance(output, torch.Tensor)
    with pytest.raises(PermissionError, match="debug"):
        block.forward_debug(torch.randn(1, 3, 32))
