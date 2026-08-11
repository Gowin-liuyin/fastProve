"""Client-side vocabulary permutation.

The client holds ``tau`` and ``tau^-1``. The server holds only the row-permuted
embedding table and the column-permuted head, never ``tau^-1``.

Row-vector convention. ``encode`` maps plaintext token ids to obfuscated ids;
``decode`` inverts it. Both are index gathers, not matrix products.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

from .seed import make_generator


@dataclass(frozen=True)
class TokenCodec:
    """Vocabulary permutation held by the client only."""

    permutation: torch.Tensor           # [V] int64, tau
    inverse_permutation: torch.Tensor   # [V] int64, tau^-1
    vocab_size: int
    fingerprint: str

    def __post_init__(self) -> None:
        if self.permutation.shape != (self.vocab_size,):
            raise ValueError("permutation shape must be [vocab_size]")
        if self.inverse_permutation.shape != (self.vocab_size,):
            raise ValueError("inverse shape must be [vocab_size]")
        for tensor in (self.permutation, self.inverse_permutation):
            if tensor.dtype != torch.int64:
                raise ValueError("token permutations must be int64")
            if not torch.equal(
                torch.sort(tensor.cpu()).values, torch.arange(self.vocab_size)
            ):
                raise ValueError("permutation must be a bijection on the vocab")
        if not torch.equal(
            self.permutation[self.inverse_permutation],
            torch.arange(self.vocab_size),
        ):
            raise ValueError("inverse_permutation does not invert permutation")

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map plaintext token ids to obfuscated ids. Client side only."""

        self._validate_ids(token_ids)
        return self.permutation.to(token_ids.device)[token_ids]

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map obfuscated token ids back. Client side only."""

        self._validate_ids(token_ids)
        return self.inverse_permutation.to(token_ids.device)[token_ids]

    def _validate_ids(self, token_ids: torch.Tensor) -> None:
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("token ids must be integral")
        if torch.any(token_ids < 0) or torch.any(token_ids >= self.vocab_size):
            raise ValueError("token id outside vocabulary")

    def save_client_secret(self, path: str | Path) -> None:
        """Persist client material. Must never be written to a server dir."""

        torch.save(
            {
                "permutation": self.permutation,
                "inverse_permutation": self.inverse_permutation,
                "vocab_size": self.vocab_size,
                "fingerprint": self.fingerprint,
            },
            Path(path),
        )


def generate_token_codec(vocab_size: int, *, seed: int, domain: str) -> TokenCodec:
    """Generate a deterministic vocabulary permutation."""

    if vocab_size <= 1:
        raise ValueError("vocab_size must exceed one")
    generator = make_generator(seed, domain, "vocab-permutation", vocab_size)
    permutation = torch.randperm(vocab_size, generator=generator)
    inverse = torch.argsort(permutation)
    digest = hashlib.sha256()
    digest.update(("%d:" % vocab_size).encode("ascii"))
    digest.update(permutation.numpy().tobytes())
    return TokenCodec(
        permutation=permutation,
        inverse_permutation=inverse,
        vocab_size=vocab_size,
        fingerprint=digest.hexdigest(),
    )
