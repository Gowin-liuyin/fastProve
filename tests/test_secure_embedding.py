"""SecureEmbedding: pre-mixed vocabulary table (task C2)."""

from __future__ import annotations

import ast
import pathlib

import torch

from fastprove.codec import generate_token_codec
from fastprove.layers.embedding import SecureEmbedding, build_secure_embedding
from fastprove.structured import generate_structured_basis


def _materials(vocab: int = 32, signal: int = 16, noise: int = 4):
    basis = generate_structured_basis(
        signal, noise, seed=3, domain="embed", block_size=4, dtype=torch.float64
    )
    codec = generate_token_codec(vocab, seed=4, domain="embed")
    torch.manual_seed(5)
    embedding = torch.randn(vocab, signal, dtype=torch.float64)
    noise_table = torch.randn(vocab, noise, dtype=torch.float64) * 0.02
    return basis, codec, embedding, noise_table


def test_permutation_direction_is_consistent() -> None:
    basis, codec, embedding, noise_table = _materials()
    mixed = basis.mix(torch.cat((embedding, noise_table), dim=-1))
    via_perm = torch.empty_like(mixed)
    via_perm[codec.permutation] = mixed
    via_inverse = mixed[codec.inverse_permutation]
    assert torch.equal(via_perm, via_inverse)
    # The row at obfuscated position tau(i) is the original row i, so a
    # lookup by the encoded id recovers the original row.
    ids = torch.arange(embedding.shape[0])
    assert torch.equal(via_perm[codec.encode(ids)], mixed)


def test_secure_embedding_decodes_to_plaintext_embedding() -> None:
    basis, codec, embedding, noise_table = _materials()
    table = build_secure_embedding(
        embedding_math=embedding,
        noise_embedding=noise_table,
        basis=basis,
        token_codec=codec,
    )
    module = SecureEmbedding(
        table, basis.descriptor, vocab_size=embedding.shape[0]
    )
    ids = torch.tensor([[3, 0, 17, 17, 9]], dtype=torch.long)
    encoded = codec.encode(ids)
    state = module(encoded)
    decoded = basis.unmix(state.mixed.to(dtype=torch.float64))
    signal = decoded[..., : basis.signal_dim]
    assert torch.allclose(signal, embedding[ids], atol=1e-12)


def test_forward_contains_a_single_embedding_lookup() -> None:
    source = pathlib.Path(
        "src/fastprove/layers/embedding.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forwards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    ]
    assert len(forwards) == 1
    calls = [
        node
        for node in ast.walk(forwards[0])
        if isinstance(node, ast.Call)
    ]
    embedding_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "embedding"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "F"
    ]
    assert len(embedding_calls) == 1
    matmuls = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr in ("matmul", "mm", "einsum")
    ]
    assert matmuls == []


def test_builder_validates_shapes() -> None:
    basis, codec, embedding, noise_table = _materials()
    import pytest

    with pytest.raises(ValueError, match="signal_dim"):
        build_secure_embedding(
            embedding_math=embedding[:, :-1],
            noise_embedding=noise_table,
            basis=basis,
            token_codec=codec,
        )
    with pytest.raises(ValueError, match="noise_embedding"):
        build_secure_embedding(
            embedding_math=embedding,
            noise_embedding=noise_table[:, :-1],
            basis=basis,
            token_codec=codec,
        )
