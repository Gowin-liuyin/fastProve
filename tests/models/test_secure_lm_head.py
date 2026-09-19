"""SecureLMHead: column-permuted fused head (task C3)."""

from __future__ import annotations

import torch
import pytest

from fastprove.codec import generate_token_codec
from fastprove.layers.attention import AttentionMode
from fastprove.layers.head import build_secure_lm_head
from fastprove.models.obfuscated import ObfuscatedTinyCausalLM
from fastprove.models.plain import PlainTinyCausalLM
from fastprove.seed import RequestContext
from fastprove.structured import generate_structured_basis

from tests._fixtures import model_config, obfuscation_config, tiny_lm_pair


def _model_config():
    return model_config(
        vocab_size=48,
        hidden_size=24,
        intermediate_size=48,
        max_sequence_length=32,
    )


def _obfuscation():
    return obfuscation_config()


def _converted(*, seed: int = 502):
    return tiny_lm_pair(
        _model_config(),
        plain_seed=501,
        conversion_seed=seed,
        obfuscation=_obfuscation(),
    )


def test_head_column_permutation_direction_is_consistent() -> None:
    torch.manual_seed(0)
    signal, noise, vocab = 16, 4, 32
    basis = generate_structured_basis(
        signal, noise, seed=1, domain="head", block_size=4, dtype=torch.float64
    )
    codec = generate_token_codec(vocab, seed=2, domain="head")
    gamma = torch.randn(signal, dtype=torch.float64).abs() + 0.5
    head = torch.randn(signal, vocab, dtype=torch.float64)

    permuted = build_secure_lm_head(
        basis=basis,
        gamma_final=gamma,
        head_math=head,
        token_codec=codec,
        dtype=torch.float64,
    )
    projection = basis.signal_projection()
    fused = (projection * gamma[None, :]) @ head
    # Column tau(i) holds the fused plaintext column i: permuted[:, tau] == fused.
    assert torch.allclose(permuted[:, codec.permutation], fused, atol=1e-12)
    # Equivalent gather form: permuted == fused[:, codec.inverse_permutation].
    assert torch.allclose(permuted, fused[:, codec.inverse_permutation], atol=1e-12)


def test_softmax_commutes_with_vocabulary_permutation() -> None:
    plain, converted = _converted()
    codec = converted.token_codec
    module = converted.module.eval()
    ids = torch.tensor([[1, 3, 5, 7, 9, 11]])
    with torch.no_grad():
        plain_logits = plain(ids)
        obf_logits = module(
            codec.encode(ids),
            request_context=RequestContext(1, "softmax-commute"),
        )
    plain_prob = torch.softmax(plain_logits, dim=-1)
    obf_prob = torch.softmax(obf_logits, dim=-1)
    # softmax(l Pi_voc) == softmax(l) Pi_voc
    assert torch.allclose(obf_prob[..., codec.permutation], plain_prob, atol=1e-6)


def test_greedy_token_sequence_matches_plaintext_after_decode() -> None:
    plain, converted = _converted()
    codec = converted.token_codec
    module = converted.module.eval()
    with torch.no_grad():
        for prompt_seed in range(100):
            generator = torch.Generator().manual_seed(1000 + prompt_seed)
            prompt = torch.randint(
                1,
                _model_config().vocab_size,
                (1, 6),
                generator=generator,
            )
            plain_gen = plain.generate_greedy(prompt, max_new_tokens=3)
            obf_gen = codec.decode(
                module.generate_greedy(
                    codec.encode(prompt),
                    max_new_tokens=3,
                    request_context=RequestContext(prompt_seed, "greedy-100"),
                )
            )
            assert torch.equal(
                obf_gen, plain_gen
            ), "prompt %d diverged: %s vs %s" % (
                prompt_seed,
                obf_gen.tolist(),
                plain_gen.tolist(),
            )


def test_fused_norm_head_raises_not_implemented() -> None:
    plain = PlainTinyCausalLM(_model_config(), seed=501, debug_enabled=False)
    obfuscation = obfuscation_config(
        refresh_mode="per_request", lm_head_mode="fused_norm_head"
    )
    with pytest.raises(NotImplementedError, match="fused kernel"):
        ObfuscatedTinyCausalLM.from_plain(
            plain,
            obfuscation=obfuscation,
            mode=AttentionMode.EXACT,
            approximation=None,
            seed=502,
            debug_enabled=False,
        )


def test_server_state_dict_has_no_inverse_permutation() -> None:
    _, converted = _converted()
    keys = " ".join(converted.module.state_dict().keys())
    assert "inverse" not in keys
    assert "token_codec" not in keys
    assert "tau" not in keys
