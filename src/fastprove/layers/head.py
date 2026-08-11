"""Secure LM head: final-norm absorbed and vocabulary-column permuted.

Mathematics (row vectors)::

    W_tilde_head = P diag(gamma_final) W_head Pi_voc      shape [n, V]
    logits       = (c @ W_tilde_head) / rho_final

Because Softmax is row-wise, permuting the logit columns commutes with it:
``softmax(l Pi_voc) = softmax(l) Pi_voc``, so the client recovers the
plaintext distribution after decoding the columns with ``tau^-1``.

The column-permutation direction is easy to get wrong: the obfuscated column
``tau(i)`` must hold the plaintext column ``i``, so ``permuted[:, tau] = head``.
This differs from the row-permutation used by the embedding table
(``table[tau] = mixed``); each direction has its own consistency test.
"""

from __future__ import annotations

import torch

from ..codec import TokenCodec
from ..structured import StructuredBasis


def build_secure_lm_head(
    *,
    basis: StructuredBasis,
    gamma_final: torch.Tensor,
    head_math: torch.Tensor,          # [d, V], math layout
    token_codec: TokenCodec,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return ``P diag(gamma_final) W_head Pi_voc`` in shape ``[n, V]``."""

    if head_math.ndim != 2:
        raise ValueError("head_math must be a [d, V] matrix in math layout")
    if head_math.shape[0] != basis.signal_dim:
        raise ValueError("head input dimension must equal basis.signal_dim")
    if head_math.shape[1] != token_codec.vocab_size:
        raise ValueError("head width must equal the codec vocab size")
    if gamma_final.shape != (basis.signal_dim,):
        raise ValueError("gamma_final shape must be [d]")
    if dtype not in (torch.float32, torch.float64, torch.bfloat16):
        raise ValueError("unsupported head dtype")

    projection = basis.signal_projection()
    gamma64 = gamma_final.detach().cpu().to(dtype=torch.float64)
    head64 = head_math.detach().cpu().to(dtype=torch.float64)
    fused = (projection * gamma64[None, :]) @ head64
    permuted = torch.empty_like(fused)
    # Obfuscated column tau(i) holds the plaintext column i.
    permuted[:, token_codec.permutation] = fused
    return permuted.to(dtype=dtype)
