"""Shared basis-dtype validation, descriptors, and fingerprint primitives.

Both basis families (``transforms.BasisTransform`` and
``structured.StructuredBasis``) need a dtype gate and a value-identifying
fingerprint. The algorithms differ (matrix bytes vs factor tuples), but the
canonicalization primitive is shared here so the two fingerprints stay in
lockstep.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Tuple

import torch

_FP64 = torch.float64


@dataclass(frozen=True)
class BasisDescriptor:
    """Public state metadata that contains no decoding matrix."""

    signal_dim: int
    noise_dim: int
    condition_number: float
    fingerprint: str

    @property
    def total_dim(self) -> int:
        """Augmented dimension."""

        return self.signal_dim + self.noise_dim


def validate_basis_dtype(dtype: torch.dtype, *, label: str) -> None:
    """Reject deployment dtypes other than FP32/FP64, keeping the old message."""

    if dtype not in (torch.float32, torch.float64):
        raise ValueError("%s dtype must be FP32 or FP64" % label)


def canonical_tensor_parts(
    tensor: torch.Tensor,
) -> Tuple[str, str, bytes]:
    """Return (shape, dtype, bytes) after canonical FP64 CPU conversion.

    Floating factors are canonicalized to FP64 before hashing so that the
    fingerprint identifies the *numerical values* actually stored, independent
    of whether the caller hashes before or after a dtype cast.
    """

    value = tensor.detach().to(device="cpu").contiguous()
    if value.is_floating_point():
        value = value.to(dtype=_FP64).contiguous()
    return str(tuple(value.shape)), str(value.dtype), value.numpy().tobytes()


def fingerprint_header(signal_dim: int, noise_dim: int) -> bytes:
    return ("%d:%d:" % (signal_dim, noise_dim)).encode("ascii")


def transform_fingerprint(
    matrix: torch.Tensor, signal_dim: int, noise_dim: int
) -> str:
    """Fingerprint a dense transform from its FP64 matrix bytes."""

    _, _, raw = canonical_tensor_parts(matrix)
    return hashlib.sha256(fingerprint_header(signal_dim, noise_dim) + raw).hexdigest()


def factors_fingerprint(
    perm_in: torch.Tensor,
    scales: torch.Tensor,
    blocks: torch.Tensor,
    perm_out: torch.Tensor,
    signal_dim: int,
    noise_dim: int,
) -> str:
    """Fingerprint a structured basis from its factor tensors."""

    digest = hashlib.sha256()
    digest.update(fingerprint_header(signal_dim, noise_dim))
    for tensor in (perm_in, scales, blocks, perm_out):
        shape, dtype, raw = canonical_tensor_parts(tensor)
        digest.update(shape.encode("ascii"))
        digest.update(dtype.encode("ascii"))
        digest.update(raw)
    return digest.hexdigest()
