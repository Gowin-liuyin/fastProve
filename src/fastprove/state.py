"""Opaque mixed-state container and explicitly gated debug helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

import torch

from .transforms import BasisDescriptor


@dataclass(frozen=True)
class MixedState:
    """A mixed augmented tensor.

    Production code can propagate ``mixed`` without receiving decoded signal or
    auxiliary components. The attached transform is conversion metadata, not a
    claim of cryptographic key isolation in this reference implementation.
    """

    mixed: torch.Tensor
    basis: BasisDescriptor

    def __post_init__(self) -> None:
        if self.mixed.ndim < 1:
            raise ValueError("mixed tensor must have at least one dimension")
        if self.mixed.shape[-1] != self.basis.total_dim:
            raise ValueError("mixed tensor width does not match transform")
        if not self.mixed.is_floating_point():
            raise ValueError("mixed tensor must use a real floating dtype")
        if not torch.isfinite(self.mixed).all():
            raise ValueError("mixed tensor contains NaN or Inf")

    def add(self, other: "MixedState") -> "MixedState":
        """Add residual states only when they use the identical basis."""

        if (
            self.basis.fingerprint != other.basis.fingerprint
            or self.basis.signal_dim != other.basis.signal_dim
            or self.basis.noise_dim != other.basis.noise_dim
        ):
            raise ValueError("residual states must use the identical basis")
        if self.mixed.shape != other.mixed.shape:
            raise ValueError("residual state shapes must match")
        return MixedState(self.mixed + other.mixed, self.basis)


def _require_debug(enabled: bool) -> None:
    if not enabled:
        raise PermissionError("debug encode/decode is disabled")


class _MixingBasis(Protocol):
    """Structural protocol shared by BasisTransform and StructuredBasis."""

    signal_dim: int
    noise_dim: int

    @property
    def descriptor(self) -> BasisDescriptor: ...

    def validate_integrity(self) -> None: ...


def encode_debug(
    signal: torch.Tensor,
    noise: torch.Tensor,
    transform: _MixingBasis,
    *,
    enabled: bool,
) -> MixedState:
    """Encode a state for tests/reference debugging only."""

    _require_debug(enabled)
    transform.validate_integrity()
    if signal.ndim < 1 or noise.ndim < 1:
        raise ValueError("signal and noise must have at least one dimension")
    if signal.shape[:-1] != noise.shape[:-1]:
        raise ValueError("signal and noise leading shapes must match")
    if signal.shape[-1] != transform.signal_dim:
        raise ValueError("signal width does not match transform")
    if noise.shape[-1] != transform.noise_dim:
        raise ValueError("noise width does not match transform")
    if signal.device != noise.device:
        raise ValueError("signal and noise devices must match")
    if signal.dtype != noise.dtype:
        raise ValueError("signal and noise dtypes must match")
    if not signal.is_floating_point() or not noise.is_floating_point():
        raise ValueError("signal and noise must use a floating dtype")
    augmented = torch.cat((signal, noise), dim=-1)
    if hasattr(transform, "mix"):
        mixed = transform.mix(augmented)
    else:
        matrix = transform.matrix.to(device=signal.device, dtype=signal.dtype)
        mixed = augmented @ matrix
    return MixedState(mixed, transform.descriptor)


def decode_debug(
    state: MixedState, transform: _MixingBasis, *, enabled: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode a state for tests/reference debugging only."""

    _require_debug(enabled)
    transform.validate_integrity()
    if state.basis != transform.descriptor:
        raise ValueError("debug transform basis does not match mixed state")
    if hasattr(transform, "unmix"):
        augmented = transform.unmix(state.mixed)
    else:
        inverse = transform.inverse.to(
            device=state.mixed.device, dtype=state.mixed.dtype
        )
        augmented = state.mixed @ inverse
    return (
        augmented[..., : transform.signal_dim],
        augmented[..., transform.signal_dim :],
    )
