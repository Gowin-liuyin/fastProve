"""Offline row-math to PyTorch-layout conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .transforms import BasisTransform


def math_to_torch_weight(weight_math: torch.Tensor) -> torch.Tensor:
    """Convert row-vector math ``[in, out]`` to PyTorch ``[out, in]``."""

    if weight_math.ndim != 2:
        raise ValueError("mathematical weight must be rank two")
    return weight_math.T.contiguous()


def torch_to_math_weight(weight_pt: torch.Tensor) -> torch.Tensor:
    """Convert PyTorch ``[out, in]`` to row-vector math ``[in, out]``."""

    if weight_pt.ndim != 2:
        raise ValueError("PyTorch weight must be rank two")
    return weight_pt.T.contiguous()


@dataclass(frozen=True)
class ConvertedAffine:
    """Converted augmented affine parameters in deployed layout."""

    weight_pt: torch.Tensor
    bias_mixed: torch.Tensor


def convert_affine_chain(
    *,
    weight_math: torch.Tensor,
    bias: torch.Tensor,
    coupling: torch.Tensor,
    propagator: torch.Tensor,
    in_transform: BasisTransform,
    out_transform: BasisTransform,
    fixed_refresh: Optional[torch.Tensor],
) -> ConvertedAffine:
    """Convert ``[h,e]`` affine propagation completely offline.

    ``weight_math`` and all block matrices use row-vector layout. The returned
    weight is transposed to the layout consumed by ``F.linear``.
    """

    in_signal = in_transform.signal_dim
    in_noise = in_transform.noise_dim
    out_signal = out_transform.signal_dim
    out_noise = out_transform.noise_dim
    if weight_math.shape != (in_signal, out_signal):
        raise ValueError("weight_math shape mismatch")
    if bias.shape != (out_signal,):
        raise ValueError("bias shape mismatch")
    if coupling.shape != (in_signal, out_noise):
        raise ValueError("coupling shape mismatch")
    if propagator.shape != (in_noise, out_noise):
        raise ValueError("propagator shape mismatch")
    if fixed_refresh is not None and fixed_refresh.shape != (out_noise,):
        raise ValueError("fixed_refresh shape mismatch")
    parameters = [weight_math, bias, coupling, propagator]
    if fixed_refresh is not None:
        parameters.append(fixed_refresh)
    if any(not tensor.is_floating_point() for tensor in parameters):
        raise ValueError("affine conversion parameters must use real floating dtypes")
    if any(not torch.isfinite(tensor).all() for tensor in parameters):
        raise ValueError("affine conversion parameters must be finite")

    deployed_device = weight_math.device
    conversion_dtype = torch.float64
    top = torch.cat(
        (
            weight_math.detach().to(device="cpu").to(dtype=conversion_dtype),
            coupling.detach().to(device="cpu").to(dtype=conversion_dtype),
        ),
        dim=1,
    )
    bottom = torch.cat(
        (
            torch.zeros(
                in_noise,
                out_signal,
                dtype=conversion_dtype,
                device="cpu",
            ),
            propagator.detach().to(device="cpu").to(dtype=conversion_dtype),
        ),
        dim=1,
    )
    block_math = torch.cat((top, bottom), dim=0)
    in_inverse = in_transform.inverse.to(
        device="cpu", dtype=conversion_dtype
    )
    out_matrix = out_transform.matrix.to(
        device="cpu", dtype=conversion_dtype
    )
    converted_math = in_inverse @ block_math @ out_matrix

    refresh = (
        torch.zeros(out_noise, dtype=conversion_dtype, device="cpu")
        if fixed_refresh is None
        else fixed_refresh.detach().to(device="cpu").to(
            dtype=conversion_dtype
        )
    )
    augmented_bias = torch.cat(
        (
            bias.detach().to(device="cpu").to(dtype=conversion_dtype),
            refresh,
        ),
        dim=0,
    )
    bias_mixed = augmented_bias @ out_matrix
    deployed_dtype = weight_math.dtype
    return ConvertedAffine(
        weight_pt=math_to_torch_weight(converted_math).to(
            device=deployed_device, dtype=deployed_dtype
        ),
        bias_mixed=bias_mixed.to(
            device=deployed_device, dtype=deployed_dtype
        ),
    )
