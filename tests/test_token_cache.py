from pathlib import Path

import pytest
import torch

from fastprove.evaluation.token_cache import (
    build_token_cache,
    load_token_cache,
    save_token_cache,
)


def test_token_cache_round_trip_and_integrity(tmp_path: Path):
    cache = build_token_cache(
        sample_ids=["a", "b"],
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64),
        token_mask=torch.tensor([[True, True, True], [True, True, False]]),
        metadata={"vocab_size": 10, "source": "unit"},
    )
    path = tmp_path / "inputs.pt"
    save_token_cache(path, cache)
    loaded = load_token_cache(path, expected_vocab_size=10, expected_sequence_length=3)
    assert loaded.sample_ids == ("a", "b")
    assert loaded.content_sha256 == cache.content_sha256
    assert torch.equal(loaded.input_ids, cache.input_ids)
    assert torch.equal(loaded.token_mask, cache.token_mask)


def test_token_cache_rejects_hash_or_alignment_drift(tmp_path: Path):
    cache = build_token_cache(
        sample_ids=["a"],
        input_ids=torch.tensor([[1, 2]], dtype=torch.int64),
        token_mask=torch.tensor([[True, True]]),
    )
    path = tmp_path / "inputs.pt"
    save_token_cache(path, cache)
    payload = torch.load(path, weights_only=True)
    payload["input_ids"][0, 0] = 9
    torch.save(payload, path)
    with pytest.raises(ValueError, match="content_sha256"):
        load_token_cache(path)


def test_token_cache_rejects_single_valid_token():
    with pytest.raises(ValueError, match="at least two"):
        build_token_cache(
            sample_ids=["a"],
            input_ids=torch.tensor([[1, 2]], dtype=torch.int64),
            token_mask=torch.tensor([[True, False]]),
        )


def test_token_cache_rejects_noncontiguous_valid_tokens():
    with pytest.raises(ValueError, match="adjacent valid"):
        build_token_cache(
            sample_ids=["a"],
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.int64),
            token_mask=torch.tensor([[True, False, True]]),
        )
