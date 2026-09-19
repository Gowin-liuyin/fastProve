"""The free noise cone ``ker(L)``: dimension, identity and accounting.

All statements here are FP64 algebraic facts recorded in
``results/raw/noise_ordering_analysis_20260919.json`` (T2-T5) and derived in
``docs/noise_placement_explained.tex``. Nothing in this file asserts that
noise inside ``ker(L)`` hides the semantic state; theorem 4 says the opposite
and the tests pin that accounting.
"""

from __future__ import annotations

import torch

from fastprove.structured import generate_structured_basis

_FP64 = torch.float64


def _basis(signal_dim=64, noise_dim=8, block_size=8, seed=17):
    return generate_structured_basis(
        signal_dim,
        noise_dim,
        seed=seed,
        domain="test-free-cone",
        block_size=block_size,
        max_condition_number=10.0,
        dtype=_FP64,
    )


def _orthonormal_rows(matrix: torch.Tensor) -> torch.Tensor:
    """Return ``Q`` with orthonormal rows spanning the row space of ``matrix``."""

    return torch.linalg.qr(matrix.T).Q.T


def _principal_angle_max(left: torch.Tensor, right: torch.Tensor) -> float:
    """Largest principal angle (radians) between two row spaces.

    ``M_bot`` rows are not orthonormal (``M = Pi1 D B Pi2``), so both bases are
    orthogonalized first; the singular values of the product of two orthonormal
    row bases are the cosines of the principal angles.
    """

    left = _orthonormal_rows(left)
    right = _orthonormal_rows(right)
    cosines = torch.linalg.svdvals(left @ right.T).clamp(-1.0, 1.0)
    return float(torch.arccos(cosines.min()))


def test_free_basis_has_expected_dimension() -> None:
    basis = _basis()
    free = basis.free_noise_basis()
    assert free.shape == (basis.noise_dim, basis.total_dim)
    gram = free @ free.T
    assert torch.allclose(gram, torch.eye(basis.noise_dim, dtype=_FP64), atol=1e-12)


def test_free_basis_annihilates_signal_projection() -> None:
    basis = _basis()
    basis.validate_integrity()
    leak = basis.free_noise_basis() @ basis.signal_projection()
    assert float(leak.abs().max()) < 1e-12


def test_noise_rows_span_equals_free_cone() -> None:
    """The recorded basis spans the same subspace as ``M_bot``.

    Tolerance note: ``arccos`` is ill-conditioned near 1, so an exact-angle
    residual of one FP64 ulp (~5e-16 in ``1 - cos``) shows up as ~3e-8 rad.
    The 1e-6 threshold is therefore a FP64 round-off tolerance, not a loose
    subspace test; tightening it below ~1e-7 would make this test flaky.
    """

    basis = _basis()
    free = basis.free_noise_basis()
    rows = basis.noise_rows()
    angle = _principal_angle_max(free, rows)
    assert angle < 1e-6, angle
    # Both spaces have the same rank, so a small angle means equality.
    assert free.shape[0] == rows.shape[0]


def test_channel_decomposition_of_pure_concat_noise() -> None:
    basis = _basis()
    noise = torch.randn(3, basis.noise_dim, dtype=_FP64)
    augmented = torch.zeros(3, basis.signal_dim, dtype=_FP64)
    concatenated = torch.cat((augmented, noise), dim=-1)
    injected = concatenated @ basis.dense()
    decomposition = basis.channel_decomposition(injected)
    assert abs(decomposition["free_fraction"] - 1.0) < 1e-12
    assert decomposition["semantic_component_norm"] < 1e-12
    assert decomposition["induced_semantic_error_norm"] < 1e-12


def test_channel_decomposition_of_generic_noise() -> None:
    basis = _basis()
    noise = torch.randn(4, basis.total_dim, dtype=_FP64)
    decomposition = basis.channel_decomposition(noise)
    assert decomposition["free_component_norm"] > 0.0
    assert decomposition["semantic_component_norm"] > 0.0
    assert 0.0 < decomposition["free_fraction"] < 1.0
    # Components are orthogonal, so the energies add up.
    total = noise.norm() ** 2
    parts = (
        decomposition["free_component_norm"] ** 2
        + decomposition["semantic_component_norm"] ** 2
    )
    assert abs(parts - float(total)) / float(total) < 1e-12


def test_induced_semantic_error_matches_n_times_L() -> None:
    """Theorem 2: the semantic effect of outside noise ``n`` is exactly ``n L``."""

    basis = _basis()
    signal = torch.randn(2, 5, basis.signal_dim, dtype=_FP64)
    noise = torch.randn(2, 5, basis.noise_dim, dtype=_FP64)
    augmented = torch.cat((signal, noise), dim=-1)
    mixed = basis.mix(augmented)

    perturbation = torch.randn(2, 5, basis.total_dim, dtype=_FP64)
    decoded_before = basis.unmix(mixed)
    decoded_after = basis.unmix(mixed + perturbation)
    observed_error = (
        decoded_after[..., : basis.signal_dim] - decoded_before[..., : basis.signal_dim]
    )
    predicted_error = perturbation @ basis.signal_projection()
    denominator = max(float(predicted_error.norm()), 1e-300)
    relative = float((observed_error - predicted_error).norm()) / denominator
    assert relative < 1e-12

    for index in (0, 1):
        decomposition = basis.channel_decomposition(perturbation[index])
        predicted = float((perturbation[index] @ basis.signal_projection()).norm())
        assert (
            abs(decomposition["induced_semantic_error_norm"] - predicted)
            / max(predicted, 1e-300)
            < 1e-12
        )
