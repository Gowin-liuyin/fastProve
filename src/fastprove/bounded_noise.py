"""Bit-mixing primitives for coordinate-stable bounded noise.

This is a reproducibility primitive, not a cryptographic PRF: coordinates
are derived from request-scoped seeds and absolute positions so that a cached
decode and a full-prefix evaluation see the same noise for the same logical
pair. The three call sites keep distinct, explicitly named centering
strategies:

* attention logit noise subtracts the valid-position row mean and rescales by
  the maximum absolute deviation (``layers/attention.py``);
* rho noise deliberately does **not** center, so the perturbation remains a
  bounded uniform deviation around the clean scale (``layers/rmsnorm.py``);
* router noise subtracts the full-row mean with per-coordinate request seeds
  (``layers/router.py``).
"""

from __future__ import annotations

import torch

LCG_MULTIPLIER = 6364136223846793005
LCG_INCREMENT = 1442695040888963407
MIX_MULTIPLIER = 3202034522624059733
MANTISSA_MASK = (1 << 53) - 1
MANTISSA_SCALE = float(1 << 53)


def mix_coordinates(coordinate: torch.Tensor) -> torch.Tensor:
    """Apply the documented LCG + xor-mix to integer coordinate tensors."""

    coordinate = coordinate * LCG_MULTIPLIER + LCG_INCREMENT
    coordinate = coordinate ^ (coordinate >> 33)
    coordinate = coordinate * MIX_MULTIPLIER + LCG_INCREMENT
    coordinate = coordinate ^ (coordinate >> 29)
    return coordinate


def mantissa_to_signed_unit(mantissa: torch.Tensor) -> torch.Tensor:
    """Map a 53-bit mantissa tensor to ``[-1, 1)`` in FP32."""

    return (
        2.0 * mantissa.to(dtype=torch.float64) / MANTISSA_SCALE - 1.0
    ).to(dtype=torch.float32)
