"""Client-side bundle restored from the client secret directory."""

from __future__ import annotations

from pathlib import Path

import torch

from ..codec import TokenCodec
from ..models.obfuscated import ObfuscatedBlockClient
from ..structured import StructuredBasis


class ClientBundle:
    """Client-side material restored from the client secret directory."""

    def __init__(
        self,
        token_codec,
        hidden_basis: StructuredBasis,
        *,
        debug_enabled: bool = True,
    ) -> None:
        self.token_codec: TokenCodec = token_codec
        self.hidden_basis = hidden_basis
        self.client = ObfuscatedBlockClient(
            hidden_basis, debug_enabled=debug_enabled
        )

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.token_codec.encode(token_ids)

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.token_codec.decode(token_ids)

    def decode_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Recover plaintext-domain logits from column-permuted output."""

        return logits[..., self.token_codec.permutation.to(logits.device)]


def load_client(client_secret_dir: str | Path) -> ClientBundle:
    """Restore the codec and debug client from the client secret directory."""

    secret_dir = Path(client_secret_dir)
    codec_state = torch.load(secret_dir / "token_codec.pt", weights_only=True)
    token_codec = TokenCodec(
        permutation=codec_state["permutation"],
        inverse_permutation=codec_state["inverse_permutation"],
        vocab_size=codec_state["vocab_size"],
        fingerprint=codec_state["fingerprint"],
    )
    basis_state = torch.load(secret_dir / "bases.pt", weights_only=True)
    hidden = basis_state["hidden"]
    hidden_basis = StructuredBasis(**hidden)
    return ClientBundle(token_codec, hidden_basis)
