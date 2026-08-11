"""Structured basis identities: fast application, Gram trick, conditioning."""

from __future__ import annotations

import pytest
import torch

from fastprove.structured import (
    AUXILIARY_MAGNITUDE_BOUND,
    check_auxiliary_magnitude,
    generate_structured_basis,
    structured_condition_number,
)


def _basis(signal_dim=64, noise_dim=8, block_size=8, seed=17):
    return generate_structured_basis(
        signal_dim,
        noise_dim,
        seed=seed,
        domain="test",
        block_size=block_size,
        max_condition_number=10.0,
        dtype=torch.float64,
    )


def test_mix_matches_dense_matrix_product() -> None:
    basis = _basis()
    augmented = torch.randn(3, 5, basis.total_dim, dtype=torch.float64)
    assert torch.allclose(basis.mix(augmented), augmented @ basis.dense(), atol=1e-12)


def test_unmix_inverts_mix_exactly() -> None:
    basis = _basis()
    augmented = torch.randn(3, 5, basis.total_dim, dtype=torch.float64)
    assert torch.allclose(basis.unmix(basis.mix(augmented)), augmented, atol=1e-12)


def test_dense_inverse_matches_explicit_inverse() -> None:
    basis = _basis()
    assert torch.allclose(
        basis.dense_inverse(), torch.linalg.inv(basis.dense()), atol=1e-11
    )


def test_signal_projection_recovers_signal_and_noise() -> None:
    basis = _basis()
    signal = torch.randn(2, 4, basis.signal_dim, dtype=torch.float64)
    noise = torch.randn(2, 4, basis.noise_dim, dtype=torch.float64)
    mixed = basis.mix(torch.cat((signal, noise), dim=-1))
    assert torch.allclose(mixed @ basis.signal_projection(), signal, atol=1e-12)
    assert torch.allclose(mixed @ basis.noise_projection(), noise, atol=1e-12)


def test_signal_and_noise_rows_reconstruct_the_mixed_state() -> None:
    basis = _basis()
    signal = torch.randn(2, 4, basis.signal_dim, dtype=torch.float64)
    noise = torch.randn(2, 4, basis.noise_dim, dtype=torch.float64)
    mixed = basis.mix(torch.cat((signal, noise), dim=-1))
    rebuilt = signal @ basis.signal_rows() + noise @ basis.noise_rows()
    assert torch.allclose(rebuilt, mixed, atol=1e-12)


def test_signal_norm_squared_matches_direct_norm() -> None:
    basis = _basis()
    signal = torch.randn(2, 4, basis.signal_dim, dtype=torch.float64)
    noise = torch.randn(2, 4, basis.noise_dim, dtype=torch.float64) * 0.5
    mixed = basis.mix(torch.cat((signal, noise), dim=-1))
    observed = basis.signal_norm_squared(mixed).double()
    assert torch.allclose(observed, signal.pow(2).sum(-1), rtol=1e-4, atol=1e-4)


def test_signal_norm_ignores_auxiliary_state_inside_the_validated_bound() -> None:
    """rho must depend on the signal only, for any noise inside the bound."""

    basis = _basis()
    signal = torch.randn(2, 4, basis.signal_dim, dtype=torch.float64)
    zeros = torch.zeros(2, 4, basis.noise_dim, dtype=torch.float64)
    baseline = basis.signal_norm_squared(basis.mix(torch.cat((signal, zeros), -1)))
    for magnitude in (0.1, 1.0, 3.0):
        noise = torch.randn(2, 4, basis.noise_dim, dtype=torch.float64) * magnitude
        mixed = basis.mix(torch.cat((signal, noise), dim=-1))
        assert torch.allclose(
            basis.signal_norm_squared(mixed), baseline, rtol=1e-3, atol=1e-3
        )


def test_gram_accuracy_degrades_with_noise() -> None:
    """Pin the measured FP32 cancellation behaviour that sets the bound.

    This test documents *why* AUXILIARY_MAGNITUDE_BOUND exists. If it starts
    failing, the FP32 Gram reduction changed and the bound must be re-measured,
    not widened.
    """

    basis = _basis()
    torch.manual_seed(0)
    signal = torch.randn(64, basis.signal_dim, dtype=torch.float64)
    exact = signal.pow(2).sum(-1)

    def relative_error(magnitude: float) -> float:
        noise = torch.randn(64, basis.noise_dim, dtype=torch.float64) * magnitude
        mixed = basis.mix(torch.cat((signal, noise), dim=-1))
        observed = basis.signal_norm_squared(mixed).double()
        return float(((observed - exact).abs() / exact).max())

    assert relative_error(1.0) < 1e-5
    assert relative_error(AUXILIARY_MAGNITUDE_BOUND) < 1e-4
    assert relative_error(1000.0) > 1e-4


def test_auxiliary_magnitude_check_accepts_and_rejects() -> None:
    signal = torch.ones(4, 16)
    inside = torch.ones(4, 4) * 0.1
    ratio = check_auxiliary_magnitude(signal, inside, context="test")
    assert ratio < AUXILIARY_MAGNITUDE_BOUND
    outside = torch.ones(4, 4) * 1e4
    with pytest.raises(ValueError, match="exceeds the validated"):
        check_auxiliary_magnitude(signal, outside, context="test")


def test_rms_scale_matches_plaintext_rms() -> None:
    basis = _basis()
    signal = torch.randn(2, 4, basis.signal_dim, dtype=torch.float64)
    noise = torch.randn(2, 4, basis.noise_dim, dtype=torch.float64)
    mixed = basis.mix(torch.cat((signal, noise), dim=-1))
    eps = 1e-5
    expected = torch.sqrt(signal.pow(2).mean(-1, keepdim=True) + eps)
    assert torch.allclose(
        basis.rms_scale(mixed, eps).double(), expected, rtol=1e-5, atol=1e-6
    )


def test_rms_scale_is_positive_for_a_zero_signal() -> None:
    basis = _basis()
    zeros = torch.zeros(2, 3, basis.total_dim, dtype=torch.float64)
    scale = basis.rms_scale(zeros, 1e-5)
    assert torch.all(scale > 0)
    assert torch.isfinite(scale).all()


def test_condition_number_respects_configured_bound() -> None:
    for seed in range(8):
        basis = _basis(seed=seed)
        assert basis.condition_number <= 10.0 + 1e-9
        assert structured_condition_number(basis) <= 10.0 + 1e-6


def test_blocks_are_orthogonal() -> None:
    basis = _basis()
    identity = torch.eye(basis.block_size, dtype=torch.float64)
    for index in range(basis.block_count):
        block = basis.blocks[index]
        assert torch.allclose(block @ block.T, identity, atol=1e-12)


def test_single_block_basis_is_supported_for_value_bases() -> None:
    basis = generate_structured_basis(
        8, 2, seed=3, domain="value", block_size=16, dtype=torch.float64
    )
    assert basis.block_count == 1
    assert basis.block_size == 10
    augmented = torch.randn(2, 3, 10, dtype=torch.float64)
    assert torch.allclose(basis.unmix(basis.mix(augmented)), augmented, atol=1e-12)


def test_block_size_must_divide_total_dimension() -> None:
    with pytest.raises(ValueError, match="must divide"):
        generate_structured_basis(64, 8, seed=1, domain="bad", block_size=7)


def test_generation_is_deterministic_for_a_seed() -> None:
    first, second = _basis(seed=5), _basis(seed=5)
    assert first.fingerprint == second.fingerprint
    assert torch.equal(first.perm_in, second.perm_in)
    assert torch.equal(first.blocks, second.blocks)
    assert torch.equal(first.gram_blocks, second.gram_blocks)


def test_distinct_domains_produce_distinct_bases() -> None:
    first = generate_structured_basis(
        64, 8, seed=1, domain="a", block_size=8, dtype=torch.float64
    )
    second = generate_structured_basis(
        64, 8, seed=1, domain="b", block_size=8, dtype=torch.float64
    )
    assert first.fingerprint != second.fingerprint


def test_float32_deployment_is_self_consistent() -> None:
    basis = generate_structured_basis(
        64, 8, seed=2, domain="fp32", block_size=8, dtype=torch.float32
    )
    basis.validate_integrity()
    assert basis.scales.dtype == torch.float32
    assert basis.blocks.dtype == torch.float32
    assert basis.gram_blocks.dtype == torch.float32
    augmented = torch.randn(2, 3, basis.total_dim, dtype=torch.float32)
    assert torch.allclose(
        basis.unmix(basis.mix(augmented)), augmented, atol=1e-4, rtol=1e-4
    )


def test_descriptor_carries_no_matrix_material() -> None:
    descriptor = _basis().descriptor
    for field in ("perm_in", "perm_out", "scales", "blocks", "gram_blocks"):
        assert not hasattr(descriptor, field)


def test_integrity_validation_detects_mutation() -> None:
    basis = _basis()
    basis.validate_integrity()
    basis.scales[0] += 1.0
    with pytest.raises(ValueError, match="mutated"):
        basis.validate_integrity()
