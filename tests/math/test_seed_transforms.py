from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fastprove.seed import RequestContext, derive_seed, make_generator
from fastprove.state import MixedState, decode_debug, encode_debug
from fastprove.transforms import (
    BasisDescriptor,
    generate_orthogonal,
    generate_signed_permutation,
    generate_transform,
)


def test_domain_separated_seeds_are_stable_and_distinct() -> None:
    first = derive_seed(1234, "layer", 2, "attention")
    assert first == derive_seed(1234, "layer", 2, "attention")
    assert first != derive_seed(1234, "layer", 3, "attention")
    assert first != derive_seed(1234, "layer", 2, "refresh")
    assert 0 <= first < 2**63


def test_request_context_derives_reproducible_request_scoped_seed() -> None:
    context = RequestContext(global_seed=52, request_id="sample-7")
    assert context.seed_for("layer", 1) == context.seed_for("layer", 1)
    other = RequestContext(global_seed=52, request_id="sample-8")
    assert context.seed_for("layer", 1) != other.seed_for("layer", 1)


def test_generators_repeat_without_mutating_global_rng() -> None:
    original_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(99)
        expected_global = torch.rand(3)
        torch.manual_seed(99)
        a = torch.rand(3, generator=make_generator(7, "unit"))
        b = torch.rand(3, generator=make_generator(7, "unit"))
        actual_global = torch.rand(3)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(actual_global, expected_global)
    finally:
        torch.random.set_rng_state(original_state)


def test_seed_domains_reject_unordered_or_ambiguous_values() -> None:
    with pytest.raises(TypeError, match="domain"):
        derive_seed(1, {"unordered", "set"})
    with pytest.raises(TypeError, match="domain"):
        derive_seed(1, {"key": "value"})


def test_transform_is_well_conditioned_and_round_trips() -> None:
    transform = generate_transform(
        signal_dim=5,
        noise_dim=3,
        seed=17,
        domain="hidden",
        max_condition_number=4.0,
    )
    assert transform.matrix.shape == (8, 8)
    assert transform.inverse.shape == (8, 8)
    assert transform.condition_number <= 4.0
    torch.testing.assert_close(
        transform.matrix @ transform.inverse,
        torch.eye(8),
        atol=2e-6,
        rtol=2e-6,
    )

    signal = torch.randn(2, 4, 5)
    noise = torch.randn(2, 4, 3)
    state = encode_debug(signal, noise, transform, enabled=True)
    assert isinstance(state, MixedState)
    decoded_signal, decoded_noise = decode_debug(
        state, transform, enabled=True
    )
    torch.testing.assert_close(decoded_signal, signal, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(decoded_noise, noise, atol=3e-6, rtol=3e-6)


def test_debug_helpers_are_explicitly_gated() -> None:
    transform = generate_transform(3, 2, seed=9, domain="gated")
    signal = torch.zeros(1, 3)
    noise = torch.zeros(1, 2)
    with pytest.raises(PermissionError, match="debug"):
        encode_debug(signal, noise, transform, enabled=False)
    state = encode_debug(signal, noise, transform, enabled=True)
    with pytest.raises(PermissionError, match="debug"):
        decode_debug(state, transform, enabled=False)


def test_orthogonal_and_signed_permutation_invariants() -> None:
    orthogonal = generate_orthogonal(6, seed=8, domain="qk")
    torch.testing.assert_close(
        orthogonal @ orthogonal.T,
        torch.eye(6),
        atol=2e-6,
        rtol=2e-6,
    )
    signed = generate_signed_permutation(5, seed=11, domain="noise")
    torch.testing.assert_close(
        signed @ signed.T,
        torch.eye(5),
        atol=0,
        rtol=0,
    )
    assert torch.count_nonzero(signed, dim=0).tolist() == [1] * 5
    assert torch.count_nonzero(signed, dim=1).tolist() == [1] * 5


@pytest.mark.parametrize(
    "dtype", [torch.int64, torch.complex64, torch.float16, torch.bfloat16]
)
def test_transform_generators_reject_unsupported_deployment_dtype(
    dtype: torch.dtype,
) -> None:
    with pytest.raises(ValueError, match="FP32 or FP64"):
        generate_transform(3, 2, seed=35, domain="dtype", dtype=dtype)
    with pytest.raises(ValueError, match="FP32 or FP64"):
        generate_orthogonal(5, seed=35, domain="dtype", dtype=dtype)
    with pytest.raises(ValueError, match="FP32 or FP64"):
        generate_signed_permutation(5, seed=35, domain="dtype", dtype=dtype)


def test_mixed_state_residual_requires_identical_basis() -> None:
    left_transform = generate_transform(4, 2, seed=1, domain="left")
    right_transform = generate_transform(4, 2, seed=2, domain="right")
    left = encode_debug(
        torch.ones(1, 4), torch.ones(1, 2), left_transform, enabled=True
    )
    same_basis = encode_debug(
        torch.full((1, 4), 2.0),
        torch.full((1, 2), 3.0),
        left_transform,
        enabled=True,
    )
    result = left.add(same_basis)
    signal, noise = decode_debug(result, left_transform, enabled=True)
    torch.testing.assert_close(signal, torch.full((1, 4), 3.0))
    torch.testing.assert_close(noise, torch.full((1, 2), 4.0))
    different_basis = encode_debug(
        torch.ones(1, 4), torch.ones(1, 2), right_transform, enabled=True
    )
    with pytest.raises(ValueError, match="basis"):
        left.add(different_basis)


def test_basis_identity_includes_signal_noise_partition() -> None:
    four_plus_two = generate_transform(4, 2, seed=19, domain="partition")
    three_plus_three = generate_transform(3, 3, seed=19, domain="partition")
    assert four_plus_two.fingerprint != three_plus_three.fingerprint

    left = encode_debug(
        torch.ones(1, 4), torch.ones(1, 2), four_plus_two, enabled=True
    )
    right = encode_debug(
        torch.ones(1, 3), torch.ones(1, 3), three_plus_three, enabled=True
    )
    with pytest.raises(ValueError, match="basis"):
        left.add(right)

    shared_fingerprint = "same-fingerprint"
    left_descriptor = BasisDescriptor(4, 2, 2.0, shared_fingerprint)
    right_descriptor = BasisDescriptor(3, 3, 2.0, shared_fingerprint)
    left_state = MixedState(torch.ones(1, 6), left_descriptor)
    right_state = MixedState(torch.ones(1, 6), right_descriptor)
    with pytest.raises(ValueError, match="basis"):
        left_state.add(right_state)


def test_mixed_state_does_not_expose_inverse_transform() -> None:
    transform = generate_transform(4, 2, seed=29, domain="opaque-state")
    state = encode_debug(
        torch.ones(1, 4), torch.ones(1, 2), transform, enabled=True
    )
    assert not hasattr(state, "transform")
    assert not hasattr(state.basis, "inverse")
    with pytest.raises(ValueError, match="basis"):
        decode_debug(
            state,
            generate_transform(4, 2, seed=30, domain="other"),
            enabled=True,
        )


def test_transform_rejects_non_floating_deployment_dtype() -> None:
    with pytest.raises(ValueError, match="FP32 or FP64"):
        generate_transform(
            4, 2, seed=31, domain="integer", dtype=torch.int64
        )
    with pytest.raises(ValueError, match="FP32 or FP64"):
        generate_orthogonal(
            4, seed=31, domain="integer-orthogonal", dtype=torch.int64
        )


def test_debug_state_rejects_integer_signal_and_noise() -> None:
    transform = generate_transform(2, 1, seed=34, domain="integer-state")
    with pytest.raises(ValueError, match="floating"):
        encode_debug(
            torch.ones(1, 2, dtype=torch.int64),
            torch.ones(1, 1, dtype=torch.int64),
            transform,
            enabled=True,
        )


def test_basis_transform_rejects_corrupted_inverse() -> None:
    transform = generate_transform(4, 2, seed=32, domain="corrupt")
    with pytest.raises(ValueError, match="inverse"):
        replace(transform, inverse=torch.eye(transform.total_dim))
    with pytest.raises(ValueError, match="condition"):
        replace(transform, condition_number=1.0)


def test_basis_transform_detects_in_place_matrix_mutation() -> None:
    transform = generate_transform(4, 2, seed=36, domain="mutable")
    transform.matrix[0, 0] += 0.25
    with pytest.raises(ValueError, match="mutated"):
        encode_debug(
            torch.zeros(1, 4),
            torch.zeros(1, 2),
            transform,
            enabled=True,
        )


def test_encode_debug_rejects_scalar_inputs_cleanly() -> None:
    transform = generate_transform(1, 1, seed=33, domain="scalar")
    with pytest.raises(ValueError, match="dimension"):
        encode_debug(
            torch.tensor(1.0),
            torch.tensor(2.0),
            transform,
            enabled=True,
        )


@pytest.mark.parametrize("dtype", [torch.int64, torch.complex64])
def test_mixed_state_rejects_non_real_floating_storage(
    dtype: torch.dtype,
) -> None:
    descriptor = BasisDescriptor(2, 1, 1.0, "manual")
    with pytest.raises(ValueError, match="real floating"):
        MixedState(torch.ones(1, 3, dtype=dtype), descriptor)
