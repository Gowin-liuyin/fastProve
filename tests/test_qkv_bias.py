import torch

from fastprove.config import ModelConfig, ObfuscationConfig
from fastprove.layers.attention import AttentionMode
from fastprove.models.obfuscated import ObfuscatedTinyCausalLM
from fastprove.models.plain import PlainTinyCausalLM
from fastprove.seed import RequestContext


def test_qwen_style_qkv_bias_exact_path_matches_plaintext():
    config = ModelConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_sequence_length=8,
        qkv_bias=True,
    )
    plain = PlainTinyCausalLM(config, seed=91, debug_enabled=True)
    converted = ObfuscatedTinyCausalLM.from_plain(
        plain,
        obfuscation=ObfuscationConfig(
            hidden_noise_dim=4,
            value_noise_dim_per_head=2,
            max_condition_number=10.0,
            noise_propagation_gamma=0.5,
            refresh_mode="fixed_debug",
        ),
        mode=AttentionMode.EXACT,
        approximation=None,
        seed=17,
        debug_enabled=True,
    )
    obfuscated = converted.module
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.inference_mode():
        plain_logits = plain(tokens)
        obfuscated_logits = obfuscated(
            tokens,
            request_context=RequestContext(5, "qkv-bias-test"),
        )
    assert torch.isfinite(obfuscated_logits).all()
    assert torch.allclose(plain_logits, obfuscated_logits, atol=2e-5, rtol=2e-5)

