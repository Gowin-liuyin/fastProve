import torch

from fastprove.layers.attention import AttentionMode
from fastprove.seed import RequestContext

from tests._fixtures import model_config, obfuscation_config, tiny_lm_pair


def test_qwen_style_qkv_bias_exact_path_matches_plaintext():
    config = model_config(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_layers=1,
        max_sequence_length=8,
        qkv_bias=True,
    )
    plain, converted = tiny_lm_pair(
        config,
        plain_seed=91,
        conversion_seed=17,
        mode=AttentionMode.EXACT,
        obfuscation=obfuscation_config(
            hidden_noise_dim=4, basis_block_size=10
        ),
    )
    obfuscated = converted.module
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.inference_mode():
        plain_logits = plain(tokens)
        obfuscated_logits = obfuscated(
            converted.token_codec.encode(tokens),
            request_context=RequestContext(5, "qkv-bias-test"),
        )
    assert torch.isfinite(obfuscated_logits).all()
    decoded = obfuscated_logits[..., converted.token_codec.permutation]
    assert torch.allclose(plain_logits, decoded, atol=2e-5, rtol=2e-5)
