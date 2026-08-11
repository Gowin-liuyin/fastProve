"""SwiGLU permutation/scaling covariance and auxiliary refresh."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..seed import make_generator


@dataclass(frozen=True)
class SwiGLUTransform:
    """Shared neuron permutation and nonzero Up scaling."""

    permutation: torch.Tensor
    scale: torch.Tensor

    def __post_init__(self) -> None:
        if self.permutation.ndim != 1 or self.scale.ndim != 1:
            raise ValueError("permutation and scale must be vectors")
        if self.permutation.shape != self.scale.shape:
            raise ValueError("permutation and scale lengths must match")
        size = self.permutation.numel()
        if not torch.equal(
            torch.sort(self.permutation.cpu()).values, torch.arange(size)
        ):
            raise ValueError("permutation must contain every neuron exactly once")
        if not torch.isfinite(self.scale).all() or torch.any(self.scale == 0):
            raise ValueError("SwiGLU scales must be finite and nonzero")


@dataclass(frozen=True)
class ConvertedSwiGLU:
    """Converted row-math Gate, Up, and Down weights."""

    gate_weight_math: torch.Tensor
    up_weight_math: torch.Tensor
    down_weight_math: torch.Tensor


def generate_swiglu_transform(
    intermediate_size: int,
    *,
    seed: int,
    domain: str,
    min_scale: float = 0.5,
    max_scale: float = 1.5,
) -> SwiGLUTransform:
    """Generate a deterministic shared permutation and bounded positive scale."""

    if intermediate_size <= 0:
        raise ValueError("intermediate_size must be positive")
    if min_scale <= 0 or max_scale < min_scale:
        raise ValueError("scale bounds must satisfy 0 < min <= max")
    generator = make_generator(
        seed, domain, "swiglu", intermediate_size
    )
    permutation = torch.randperm(intermediate_size, generator=generator)
    scale = min_scale + (max_scale - min_scale) * torch.rand(
        intermediate_size, generator=generator, dtype=torch.float32
    )
    return SwiGLUTransform(permutation=permutation, scale=scale)


def convert_swiglu_weights(
    *,
    hidden_rotation: torch.Tensor,
    gamma: torch.Tensor,
    gate_weight_math: torch.Tensor,
    up_weight_math: torch.Tensor,
    down_weight_math: torch.Tensor,
    transform: SwiGLUTransform,
) -> ConvertedSwiGLU:
    """Absorb RMS gamma and apply the exact SwiGLU covariance conversion.

    The conversion is

    ``Wg' = R.T Gamma Wg P``,
    ``Wu' = R.T Gamma Wu D P``, and
    ``Wd' = P.T D^-1 Wd``.

    Reference form. The deployed forward path fuses this transformation into
    ``layers/deployed.py`` (``build_deployed_feed_forward``); this function
    remains the documented general case.
    """

    if hidden_rotation.ndim != 2 or (
        hidden_rotation.shape[0] != hidden_rotation.shape[1]
    ):
        raise ValueError("hidden_rotation must be square")
    hidden_size = hidden_rotation.shape[0]
    intermediate_size = transform.permutation.numel()
    if gamma.shape != (hidden_size,):
        raise ValueError("gamma shape mismatch")
    if gate_weight_math.shape != (hidden_size, intermediate_size):
        raise ValueError("Gate weight shape mismatch")
    if up_weight_math.shape != gate_weight_math.shape:
        raise ValueError("Up weight shape mismatch")
    if down_weight_math.shape != (intermediate_size, hidden_size):
        raise ValueError("Down weight shape mismatch")

    device = gate_weight_math.device
    dtype = gate_weight_math.dtype
    rotation = hidden_rotation.to(device=device, dtype=torch.float32)
    gamma_fp32 = gamma.to(device=device, dtype=torch.float32)
    permutation = transform.permutation.to(device=device)
    scale = transform.scale.to(device=device, dtype=torch.float32)
    gate_base = rotation.T @ (
        gamma_fp32[:, None] * gate_weight_math.float()
    )
    up_base = rotation.T @ (
        gamma_fp32[:, None] * up_weight_math.float()
    )
    gate_converted = gate_base[:, permutation]
    up_converted = (up_base * scale[None, :])[:, permutation]
    down_converted = down_weight_math.float()[permutation] / scale[
        permutation, None
    ]
    return ConvertedSwiGLU(
        gate_weight_math=gate_converted.to(dtype=dtype),
        up_weight_math=up_converted.to(dtype=dtype),
        down_weight_math=down_converted.to(dtype=dtype),
    )


def refresh_swiglu_noise(
    *,
    z_prime: torch.Tensor,
    side_noise: torch.Tensor,
    coupling: torch.Tensor,
    propagator: torch.Tensor,
    refresh: torch.Tensor,
) -> torch.Tensor:
    """Compute ``e_z = z' C_z + e_side G_z + xi_z``.

    Reference form. The deployed forward path fuses the noise refresh into
    ``layers/deployed.py``; this function remains the documented general case.
    """

    if z_prime.shape[:-1] != side_noise.shape[:-1]:
        raise ValueError("signal and side-noise leading shapes must match")
    if coupling.shape[0] != z_prime.shape[-1]:
        raise ValueError("coupling input dimension mismatch")
    if propagator.shape[0] != side_noise.shape[-1]:
        raise ValueError("propagator input dimension mismatch")
    if coupling.shape[1] != propagator.shape[1]:
        raise ValueError("noise output dimensions must match")
    if refresh.shape != (coupling.shape[1],):
        raise ValueError("refresh shape mismatch")
    return z_prime @ coupling + side_noise @ propagator + refresh

