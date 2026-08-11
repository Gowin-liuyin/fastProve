"""Pre-mixed vocabulary embedding: the server lookup returns the mixed state.

Mathematics (row vectors)::

    E_tilde = Pi_voc^T [E, E_n] M_0        shape [V, n]

``Pi_voc`` is the client vocabulary permutation (row reorder of the table),
``E`` is the plaintext embedding ``[V, d]``, ``E_n`` is an independent random
noise table ``[V, r]``, and ``M_0`` is the hidden mixing basis. The server
looks up obfuscated token ids and receives ``c_0`` directly; the plaintext
embedding is never materialized.

The row-permutation direction is easy to get wrong: the obfuscated-id row
``tau(i)`` must hold the original row ``i``, so ``table[tau] = mixed`` (or
equivalently ``table = mixed[tau^-1]``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..codec import TokenCodec
from ..state import MixedState
from ..structured import StructuredBasis
from ..transforms import BasisDescriptor


def build_secure_embedding(
    *,
    embedding_math: torch.Tensor,       # [V, d]
    noise_embedding: torch.Tensor,      # [V, r]
    basis: StructuredBasis,
    token_codec: TokenCodec,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return ``Pi_voc^T [E, E_n] M_0`` in shape ``[V, n]``."""

    vocab = embedding_math.shape[0]
    if embedding_math.ndim != 2:
        raise ValueError("embedding_math must be a [V, d] matrix")
    if embedding_math.shape[1] != basis.signal_dim:
        raise ValueError("embedding width must equal basis.signal_dim")
    if noise_embedding.shape != (vocab, basis.noise_dim):
        raise ValueError("noise_embedding must be [V, r]")
    if token_codec.vocab_size != vocab:
        raise ValueError("token_codec vocab size must match the embedding")
    if dtype not in (torch.float32, torch.float64, torch.bfloat16):
        raise ValueError("unsupported embedding dtype")

    source = torch.cat((embedding_math, noise_embedding), dim=-1)
    mixed = basis.mix(source)
    table = torch.empty_like(mixed)
    # Obfuscated-id row tau(i) holds the original row i. Writing it as an
    # index gather keeps the direction explicit; tests pin both forms.
    table[token_codec.permutation] = mixed
    return table.to(dtype=dtype)


class SecureEmbedding(nn.Module):
    """Server-side embedding lookup returning the mixed state directly."""

    def __init__(
        self,
        table: torch.Tensor,
        basis_descriptor: BasisDescriptor,
        *,
        vocab_size: int,
    ) -> None:
        super().__init__()
        if table.ndim != 2 or table.shape[0] != vocab_size:
            raise ValueError("table must be [V, n]")
        if table.shape[1] != basis_descriptor.total_dim:
            raise ValueError("table width must match the basis descriptor")
        self.register_buffer("table", table.detach().clone())
        self.basis_descriptor = basis_descriptor
        self.vocab_size = int(vocab_size)

    def forward(self, input_ids: torch.Tensor) -> MixedState:
        """Return the mixed state for already-permuted token ids.

        ``input_ids`` must already be vocabulary-permuted by the client codec.
        No mixing happens here; ``M_0`` is absorbed into the table offline.
        """

        self._validate(input_ids)
        return MixedState(
            F.embedding(input_ids, self.table), self.basis_descriptor
        )

    def _validate(self, input_ids: torch.Tensor) -> None:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be integral")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.vocab_size):
            raise ValueError("input token is outside vocabulary")
