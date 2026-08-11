"""Validated, model-independent token cache for aligned evaluations.

The cache intentionally stores token IDs and masks rather than raw text.  A
single cache is consumed by plaintext and every obfuscated condition so that
sample identity, ordering, truncation and padding cannot drift between runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch


TOKEN_CACHE_VERSION = "fastprove.token_cache.v1"


def _hash_payload(
    sample_ids: Sequence[str], input_ids: torch.Tensor, token_mask: torch.Tensor
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(sample_ids), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    digest.update(input_ids.detach().cpu().contiguous().numpy().tobytes())
    digest.update(token_mask.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TokenCache:
    """Validated aligned inputs and their provenance manifest."""

    sample_ids: tuple[str, ...]
    input_ids: torch.Tensor
    token_mask: torch.Tensor
    metadata: Dict[str, Any]

    @property
    def sample_count(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def content_sha256(self) -> str:
        return _hash_payload(self.sample_ids, self.input_ids, self.token_mask)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_version": TOKEN_CACHE_VERSION,
            "sample_count": self.sample_count,
            "sequence_length": self.sequence_length,
            "sample_ids": list(self.sample_ids),
            "input_ids": self.input_ids,
            "token_mask": self.token_mask,
            "metadata": dict(self.metadata),
            "content_sha256": self.content_sha256,
        }


def validate_token_cache(
    payload: Mapping[str, Any],
    *,
    expected_vocab_size: int | None = None,
    expected_sequence_length: int | None = None,
    expected_sample_count: int | None = None,
) -> TokenCache:
    """Validate a serialized cache and return CPU tensors.

    Validation is deliberately strict: malformed or stale caches fail before
    model allocation.  This is part of the fairness contract, not merely
    convenience input handling.
    """

    if payload.get("cache_version") != TOKEN_CACHE_VERSION:
        raise ValueError("unsupported token cache version")
    input_ids = payload.get("input_ids")
    token_mask = payload.get("token_mask")
    raw_ids = payload.get("sample_ids")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(token_mask, torch.Tensor):
        raise ValueError("token cache input_ids/token_mask must be tensors")
    if input_ids.ndim != 2 or token_mask.ndim != 2 or input_ids.shape != token_mask.shape:
        raise ValueError("token cache tensors must both have shape [samples, sequence]")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("token cache input_ids must be int32 or int64")
    if token_mask.dtype != torch.bool:
        raise ValueError("token cache token_mask must be boolean")
    input_ids = input_ids.detach().cpu().contiguous()
    token_mask = token_mask.detach().cpu().contiguous()
    if input_ids.shape[0] < 1 or input_ids.shape[1] < 2:
        raise ValueError("token cache needs at least one sample and two tokens")
    sample_ids = tuple(str(value) for value in raw_ids) if isinstance(raw_ids, (list, tuple)) else ()
    if len(sample_ids) != input_ids.shape[0] or any(not value for value in sample_ids):
        raise ValueError("token cache sample_ids must match samples and be non-empty")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("token cache sample_ids must be unique")
    if expected_vocab_size is not None:
        if expected_vocab_size <= 1:
            raise ValueError("expected_vocab_size must be greater than one")
        if bool(torch.any(input_ids < 0)) or bool(torch.any(input_ids >= expected_vocab_size)):
            raise ValueError("token cache contains IDs outside the model vocabulary")
    if expected_sequence_length is not None and input_ids.shape[1] != expected_sequence_length:
        raise ValueError(
            "token cache sequence length %d != expected %d"
            % (input_ids.shape[1], expected_sequence_length)
        )
    if expected_sample_count is not None and input_ids.shape[0] != expected_sample_count:
        raise ValueError(
            "token cache sample count %d != expected %d"
            % (input_ids.shape[0], expected_sample_count)
        )
    if bool(torch.any(token_mask.sum(dim=1) < 2)):
        raise ValueError("every token-cache sample needs at least two valid tokens")
    adjacent_valid = token_mask[:, :-1] & token_mask[:, 1:]
    if bool(torch.any(adjacent_valid.sum(dim=1) < 1)):
        raise ValueError(
            "every token-cache sample needs an adjacent valid next-token pair"
        )
    expected_hash = payload.get("content_sha256")
    actual_hash = _hash_payload(sample_ids, input_ids, token_mask)
    if expected_hash is not None and str(expected_hash) != actual_hash:
        raise ValueError("token cache content_sha256 does not match tensor contents")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("token cache metadata must be a mapping")
    return TokenCache(sample_ids, input_ids, token_mask, dict(metadata))


def save_token_cache(path: str | Path, cache: TokenCache) -> None:
    """Write one complete cache atomically enough for local experiments."""

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError("refusing to overwrite token cache: %s" % target)
    torch.save(cache.to_dict(), target)


def load_token_cache(
    path: str | Path,
    *,
    expected_vocab_size: int | None = None,
    expected_sequence_length: int | None = None,
    expected_sample_count: int | None = None,
) -> TokenCache:
    """Load and validate a cache produced by :func:`save_token_cache`."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError("token cache does not exist: %s" % source)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # older PyTorch fallback; cache contains tensors only
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("token cache root must be a mapping")
    return validate_token_cache(
        payload,
        expected_vocab_size=expected_vocab_size,
        expected_sequence_length=expected_sequence_length,
        expected_sample_count=expected_sample_count,
    )


def build_token_cache(
    *,
    sample_ids: Sequence[str],
    input_ids: torch.Tensor,
    token_mask: torch.Tensor,
    metadata: Mapping[str, Any] | None = None,
) -> TokenCache:
    """Construct and validate a cache from tokenizer output."""

    payload = {
        "cache_version": TOKEN_CACHE_VERSION,
        "sample_ids": list(sample_ids),
        "input_ids": input_ids,
        "token_mask": token_mask,
        "metadata": dict(metadata or {}),
    }
    return validate_token_cache(payload)
