from __future__ import annotations

import torch
import torch.nn.functional as F

from fastprove.layers.swiglu import (
    convert_swiglu_weights,
    generate_swiglu_transform,
    refresh_swiglu_noise,
)
from fastprove.layers.rmsnorm import rms_norm_fp32
from fastprove.transforms import generate_orthogonal


def test_swiglu_permutation_scaling_and_down_compensation_are_exact() -> None:
    generator = torch.Generator().manual_seed(25)
    hidden, intermediate = 8, 13
    h = torch.randn(2, 4, hidden, generator=generator)
    gamma = torch.linspace(0.75, 1.25, hidden)
    gate_weight = torch.randn(hidden, intermediate, generator=generator) * 0.2
    up_weight = torch.randn(hidden, intermediate, generator=generator) * 0.2
    down_weight = torch.randn(intermediate, hidden, generator=generator) * 0.2
    rotation = generate_orthogonal(hidden, seed=8, domain="ffn-rms")
    transform = generate_swiglu_transform(
        intermediate, seed=9, domain="ffn-neurons"
    )
    converted = convert_swiglu_weights(
        hidden_rotation=rotation,
        gamma=gamma,
        gate_weight_math=gate_weight,
        up_weight_math=up_weight,
        down_weight_math=down_weight,
        transform=transform,
    )

    normalized = rms_norm_fp32(h, gamma, 1e-5)
    plain = (F.silu(normalized @ gate_weight) * (normalized @ up_weight))
    plain = plain @ down_weight

    u = h @ rotation
    rms_u = u.float() * torch.rsqrt(
        u.float().square().mean(dim=-1, keepdim=True) + 1e-5
    )
    gate_prime = rms_u @ converted.gate_weight_math
    up_prime = rms_u @ converted.up_weight_math
    obfuscated = (F.silu(gate_prime) * up_prime) @ converted.down_weight_math
    torch.testing.assert_close(obfuscated, plain, atol=3e-5, rtol=3e-5)


def test_swiglu_noise_side_path_survives_and_refresh_regenerates() -> None:
    generator = torch.Generator().manual_seed(31)
    z_prime = torch.randn(2, 3, 7, generator=generator)
    side_noise = torch.randn(2, 3, 3, generator=generator)
    coupling = torch.zeros(7, 4)
    propagator = torch.randn(3, 4, generator=generator) * 0.2
    no_refresh = torch.zeros(4)
    carried = refresh_swiglu_noise(
        z_prime=z_prime,
        side_noise=side_noise,
        coupling=coupling,
        propagator=propagator,
        refresh=no_refresh,
    )
    torch.testing.assert_close(carried, side_noise @ propagator)
    assert torch.count_nonzero(carried).item() > 0

    refresh = torch.tensor([0.1, -0.2, 0.3, -0.4])
    regenerated = refresh_swiglu_noise(
        z_prime=z_prime,
        side_noise=torch.zeros_like(side_noise),
        coupling=torch.zeros_like(coupling),
        propagator=propagator,
        refresh=refresh,
    )
    torch.testing.assert_close(
        regenerated, refresh.expand_as(regenerated)
    )
    assert torch.count_nonzero(regenerated).item() > 0


def test_swiglu_transform_is_deterministic_and_bounded() -> None:
    first = generate_swiglu_transform(
        16, seed=77, domain="bounded", min_scale=0.5, max_scale=1.5
    )
    second = generate_swiglu_transform(
        16, seed=77, domain="bounded", min_scale=0.5, max_scale=1.5
    )
    assert torch.equal(first.permutation, second.permutation)
    torch.testing.assert_close(first.scale, second.scale)
    assert torch.unique(first.permutation).numel() == 16
    assert first.scale.min().item() >= 0.5
    assert first.scale.max().item() <= 1.5
    assert torch.all(first.scale != 0)

