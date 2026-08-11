"""Unit tests for the plaintext Llama adapter (no network, no 3B download)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fastprove.pretrained.llama import (
    _llama_mapping,
    _reject_legacy_keys,
    verify_plaintext_llama_tree,
)
from fastprove.config import ModelConfig


def test_reject_legacy_weight_keys() -> None:
    with pytest.raises(ValueError, match="legacy"):
        _reject_legacy_keys(
            [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.noise_coupling",
            ]
        )


def test_accept_standard_llama_keys() -> None:
    _reject_legacy_keys(
        [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.norm.weight",
        ]
    )


def test_verify_plaintext_tree_minimal(tmp_path: Path) -> None:
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "attention_bias": False,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    # Minimal single-file safetensors via torch if available; otherwise skip
    # structural weight_map path using index only is not enough without file.
    try:
        from safetensors.torch import save_file
    except ImportError:
        pytest.skip("safetensors not installed")

    tensors = {
        "model.embed_tokens.weight": torch.randn(128, 32),
        "model.norm.weight": torch.ones(32),
        "model.layers.0.input_layernorm.weight": torch.ones(32),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(32),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(32, 32),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(16, 32),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(16, 32),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(32, 32),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(64, 32),
        "model.layers.0.mlp.up_proj.weight": torch.randn(64, 32),
        "model.layers.0.mlp.down_proj.weight": torch.randn(32, 64),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    ok, notes = verify_plaintext_llama_tree(tmp_path)
    assert ok, notes
    assert any("NOT a legacy" in n for n in notes)


def test_verify_rejects_non_llama(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
            }
        ),
        encoding="utf-8",
    )
    ok, notes = verify_plaintext_llama_tree(tmp_path)
    assert not ok
    assert any("llama" in n.lower() for n in notes)


def test_llama_mapping_shapes_cover_layers() -> None:
    config = ModelConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_sequence_length=16,
        qkv_bias=False,
    )
    mapping = _llama_mapping(config)
    assert "model.embed_tokens.weight" in mapping
    assert "model.layers.1.mlp.down_proj.weight" in mapping
    assert mapping["model.layers.0.self_attn.q_proj.weight"] == "blocks.0.q_proj.weight"
