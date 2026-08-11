from __future__ import annotations

import torch

from fastprove.layers.rmsnorm import (
    absorbed_rms_projection,
    apply_qk_orthogonal_after_rope,
    apply_rope,
    rms_no_gamma_fp32,
    rms_norm_fp32,
)
from fastprove.transforms import generate_orthogonal


def test_rmsnorm_uses_fp32_statistics_and_preserves_activation_dtype() -> None:
    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [100.0, -50.0, 25.0, -12.5]],
        dtype=torch.bfloat16,
    )
    gamma = torch.tensor([0.8, 1.1, 0.9, 1.3], dtype=torch.bfloat16)
    actual = rms_norm_fp32(x, gamma, eps=1e-5)
    expected = (
        x.float()
        * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-5)
        * gamma.float()
    ).to(torch.bfloat16)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)


def test_absorbed_gamma_projection_matches_plain_rmsnorm() -> None:
    generator = torch.Generator().manual_seed(5)
    h = torch.randn(2, 3, 8, generator=generator)
    gamma = torch.linspace(0.7, 1.3, 8)
    weight_math = torch.randn(8, 12, generator=generator)
    rotation = generate_orthogonal(8, seed=42, domain="rms-hidden")

    plain = rms_norm_fp32(h, gamma, eps=1e-5) @ weight_math
    u = h @ rotation
    absorbed = absorbed_rms_projection(rotation, gamma, weight_math)
    converted = rms_no_gamma_fp32(u, eps=1e-5) @ absorbed
    torch.testing.assert_close(converted, plain, atol=2e-5, rtol=2e-5)


def test_rope_then_common_qk_transform_preserves_gqa_scores() -> None:
    generator = torch.Generator().manual_seed(15)
    batch, q_heads, kv_heads, seq, head_dim = 2, 4, 2, 5, 8
    q = torch.randn(
        batch, q_heads, seq, head_dim, generator=generator
    )
    k = torch.randn(
        batch, kv_heads, seq, head_dim, generator=generator
    )
    positions = torch.arange(seq)
    q_rope = apply_rope(q, positions, theta=10000.0)
    k_rope = apply_rope(k, positions, theta=10000.0)
    kv_index = torch.tensor([0, 0, 1, 1])
    common = torch.stack(
        [
            generate_orthogonal(
                head_dim, seed=100 + head, domain="qk-common"
            )
            for head in range(kv_heads)
        ]
    )
    q_prime, k_prime = apply_qk_orthogonal_after_rope(
        q_rope, k_rope, common, kv_index
    )

    plain_scores = torch.einsum(
        "bhqd,bhkd->bhqk", q_rope, k_rope[:, kv_index]
    )
    transformed_scores = torch.einsum(
        "bhqd,bhkd->bhqk", q_prime, k_prime[:, kv_index]
    )
    torch.testing.assert_close(
        transformed_scores, plain_scores, atol=3e-5, rtol=3e-5
    )


def test_applying_qk_transform_before_rope_is_not_assumed_to_commute() -> None:
    generator = torch.Generator().manual_seed(31)
    q = torch.randn(1, 1, 4, 8, generator=generator)
    positions = torch.arange(4)
    common = generate_orthogonal(8, seed=77, domain="noncommuting")
    correct = apply_rope(q, positions, theta=10000.0) @ common
    incorrect = apply_rope(q @ common, positions, theta=10000.0)
    assert torch.max(torch.abs(correct - incorrect)).item() > 1e-3

