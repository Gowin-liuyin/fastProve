"""Client-side vocabulary permutation: TokenCodec coverage (task C1)."""

from __future__ import annotations

import torch
import pytest

from fastprove.codec import TokenCodec, generate_token_codec


def _codec(seed: int = 7, domain: str = "codec-test") -> TokenCodec:
    return generate_token_codec(16, seed=seed, domain=domain)


def test_round_trip_decode_of_encode_restores_ids() -> None:
    codec = _codec()
    ids = torch.tensor([0, 5, 7, 15, 3, 3], dtype=torch.long)
    encoded = codec.encode(ids)
    assert not torch.equal(encoded, ids)  # non-trivial permutation for seed 7
    assert torch.equal(codec.decode(encoded), ids)


def test_generation_is_deterministic_for_a_seed() -> None:
    first = _codec(seed=11)
    second = _codec(seed=11)
    assert first.fingerprint == second.fingerprint
    assert torch.equal(first.permutation, second.permutation)


def test_distinct_domains_produce_distinct_permutations() -> None:
    first = _codec(domain="a")
    second = _codec(domain="b")
    assert first.fingerprint != second.fingerprint
    assert not torch.equal(first.permutation, second.permutation)


def test_non_bijective_permutation_is_rejected() -> None:
    with pytest.raises(ValueError, match="bijection"):
        TokenCodec(
            permutation=torch.tensor([0, 0, 1]),
            inverse_permutation=torch.tensor([1, 2, 0]),
            vocab_size=3,
            fingerprint="x",
        )


def test_mismatched_inverse_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not invert"):
        TokenCodec(
            permutation=torch.tensor([2, 0, 1]),
            inverse_permutation=torch.tensor([2, 1, 0]),  # wrong inverse
            vocab_size=3,
            fingerprint="x",
        )


def test_out_of_range_token_is_rejected() -> None:
    codec = _codec()
    with pytest.raises(ValueError, match="outside vocabulary"):
        codec.encode(torch.tensor([0, codec.vocab_size]))
    with pytest.raises(ValueError, match="outside vocabulary"):
        codec.decode(torch.tensor([-1]))


def test_non_integral_token_is_rejected() -> None:
    codec = _codec()
    with pytest.raises(ValueError, match="integral"):
        codec.encode(torch.tensor([0.0]))


def test_saved_client_secret_contains_inverse_material(tmp_path) -> None:
    codec = _codec(seed=21)
    path = tmp_path / "token_codec.pt"
    codec.save_client_secret(path)
    loaded = torch.load(path, weights_only=True)
    assert set(loaded.keys()) == {
        "permutation",
        "inverse_permutation",
        "vocab_size",
        "fingerprint",
    }
    assert loaded["fingerprint"] == codec.fingerprint
    assert torch.equal(loaded["inverse_permutation"], codec.inverse_permutation)
