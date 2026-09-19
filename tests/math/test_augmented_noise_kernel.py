"""Phase A tests: two-sided basis, independent noise chain, linear closed loop.

Covers plan section 10, stage A acceptance items on small tensors:
condition number, inverse residual, M_n P ~= 0, the full [y, e'] output
identity, fixed-seed reproducibility, record isolation, and prefill/decode
position consistency.
"""

from __future__ import annotations

import math

import pytest
import torch

from fastprove.augmented_noise import (
    NoiseRefreshSpec,
    apply_noise_refresh,
    generate_propagator,
    noise_coverage_metrics,
    observed_noise_ratios,
    record_generator,
    sample_initial_noise,
    sample_refresh_noise,
)
from fastprove.conversion import convert_affine_chain
from fastprove.seed import RequestContext
from fastprove.state import decode_debug, encode_debug
from fastprove.transforms import (
    generate_transform,
    generate_two_sided_transform,
)


def _two_sided(signal_dim=8, noise_dim=2, seed=17, kappa=3.0, dtype=torch.float64):
    return generate_two_sided_transform(
        signal_dim,
        noise_dim,
        seed=seed,
        domain="two-sided-test",
        max_condition_number=kappa,
        dtype=dtype,
    )


def test_two_sided_transform_condition_number_matches_diagonal_bound() -> None:
    transform = _two_sided(kappa=3.0)
    assert transform.condition_number <= 3.0 + 1e-10
    singular_values = torch.linalg.svdvals(
        transform.matrix.detach().to(dtype=torch.float64)
    )
    ratio = float(singular_values.max() / singular_values.min())
    assert abs(ratio - transform.condition_number) < 1e-8


def test_two_sided_transform_round_trips_in_stored_dtype() -> None:
    transform = _two_sided(dtype=torch.float32)
    atol = 1e-5
    torch.testing.assert_close(
        transform.matrix @ transform.inverse,
        torch.eye(transform.total_dim, dtype=torch.float32),
        atol=atol,
        rtol=atol,
    )
    signal = torch.randn(3, 5, transform.signal_dim)
    noise = torch.randn(3, 5, transform.noise_dim)
    state = encode_debug(signal, noise, transform, enabled=True)
    decoded_signal, decoded_noise = decode_debug(
        state, transform, enabled=True
    )
    torch.testing.assert_close(decoded_signal, signal, atol=atol, rtol=atol)
    torch.testing.assert_close(decoded_noise, noise, atol=atol, rtol=atol)


def test_two_sided_basis_has_signal_noise_cross_row_gram() -> None:
    """The candidate's row Gram generally has nonzero cross blocks.

    This records the structural difference from the ``Pi D Q`` baseline; it is
    a randomization-structure check, not a security claim (plan 7.2).
    """

    transform = _two_sided()
    rows = transform.matrix.detach().to(dtype=torch.float64)
    cross = rows[: transform.signal_dim] @ rows[transform.signal_dim :].T
    assert float(cross.abs().max()) > 1e-6


def test_kappa_one_two_sided_transform_is_orthogonal() -> None:
    transform = _two_sided(kappa=1.0)
    torch.testing.assert_close(
        transform.matrix @ transform.matrix.T,
        torch.eye(transform.total_dim, dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )


def test_two_sided_transform_rejects_invalid_condition_target() -> None:
    with pytest.raises(ValueError, match="at least one"):
        generate_two_sided_transform(4, 2, seed=1, domain="bad", max_condition_number=0.5)


def test_legal_decode_cancels_independent_noise() -> None:
    """Same h with two independent noise draws: c P is unchanged (plan 7.3)."""

    transform = _two_sided()
    signal = torch.randn(6, transform.signal_dim, dtype=torch.float64)
    generator = torch.Generator().manual_seed(5)
    first_noise = torch.randn(6, transform.noise_dim, generator=generator, dtype=torch.float64)
    second_noise = torch.randn(6, transform.noise_dim, generator=generator, dtype=torch.float64)
    first = encode_debug(signal, first_noise, transform, enabled=True)
    second = encode_debug(signal, second_noise, transform, enabled=True)
    projection = transform.inverse[:, : transform.signal_dim]
    torch.testing.assert_close(first.mixed @ projection, signal, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(second.mixed @ projection, signal, atol=1e-10, rtol=1e-10)
    assert not torch.equal(first.mixed, second.mixed)


def test_noise_rows_are_annihilated_by_signal_projection() -> None:
    """M_n P = 0 identity (plan 7.3)."""

    transform = _two_sided()
    projection = transform.inverse[:, : transform.signal_dim]
    noise_rows = transform.matrix[transform.signal_dim :]
    torch.testing.assert_close(
        noise_rows @ projection,
        torch.zeros(transform.noise_dim, transform.signal_dim, dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )


def test_refresh_spec_validates_parameters() -> None:
    with pytest.raises(ValueError, match="gamma"):
        NoiseRefreshSpec(noise_dim=4, gamma=1.0, energy_ratio=1.0)
    with pytest.raises(ValueError, match="energy_ratio"):
        NoiseRefreshSpec(noise_dim=4, gamma=0.5, energy_ratio=float("nan"))
    spec = NoiseRefreshSpec(noise_dim=16, gamma=0.5, energy_ratio=1.0)
    assert abs(spec.component_sigma(2.0) - 2.0 / 4.0) < 1e-12


def test_component_distributions_match_plan_variances() -> None:
    spec = NoiseRefreshSpec(noise_dim=16, gamma=0.5, energy_ratio=1.0)
    sigma = spec.component_sigma(1.0)
    generator = torch.Generator().manual_seed(21)
    initial = sample_initial_noise(
        spec, calibration_rms=1.0, shape=(400, 16), generator=generator
    )
    refresh = sample_refresh_noise(
        spec, calibration_rms=1.0, shape=(400, 16), generator=generator
    )
    torch.testing.assert_close(
        initial.double().var(unbiased=True),
        torch.tensor(sigma**2, dtype=torch.float64),
        atol=0.002,
        rtol=0.05,
    )
    torch.testing.assert_close(
        refresh.double().var(unbiased=True),
        torch.tensor(sigma**2 * (1 - 0.25), dtype=torch.float64),
        atol=0.002,
        rtol=0.05,
    )
    assert float(initial.abs().max()) <= math.sqrt(3.0) * sigma + 1e-12
    assert (
        float(refresh.abs().max())
        <= math.sqrt(3.0) * sigma * math.sqrt(1 - 0.25) + 1e-12
    )


def test_noise_chain_stationary_covariance_for_linear_chain() -> None:
    """Cov(e_1) = gamma^2 Cov(e_0) + Var(xi) = sigma^2 I (plan 7.4 proposition)."""

    spec = NoiseRefreshSpec(noise_dim=16, gamma=0.5, energy_ratio=1.0)
    propagator = generate_propagator(spec, seed=9, domain="stationary")
    sigma = spec.component_sigma(1.0)
    generator = torch.Generator().manual_seed(31)
    e0 = sample_initial_noise(
        spec, calibration_rms=1.0, shape=(20000, 16), generator=generator
    )
    xi = sample_refresh_noise(
        spec, calibration_rms=1.0, shape=(20000, 16), generator=generator
    )
    e1 = apply_noise_refresh(e0, propagator, xi).to(dtype=torch.float64)
    target = sigma**2
    target_tensor = torch.tensor(target, dtype=torch.float64)
    torch.testing.assert_close(
        e0.double().var(unbiased=True), target_tensor, atol=5e-3, rtol=0.05
    )
    torch.testing.assert_close(
        e1.var(unbiased=True), target_tensor, atol=5e-3, rtol=0.05
    )


def test_record_generator_isolates_records_and_keeps_positions_stable() -> None:
    common = dict(
        global_seed=7,
        key_epoch="k0",
        request_nonce="rq1",
        sample_id="s1",
        layer_id="L0",
        operation="attn",
    )
    base = record_generator(token_position=5, head_id=0, **common)
    same = record_generator(token_position=5, head_id=0, **common)
    torch.testing.assert_close(torch.rand(4, generator=base), torch.rand(4, generator=same))

    different_token = record_generator(token_position=6, head_id=0, **common)
    assert not torch.equal(torch.rand(4, generator=base), torch.rand(4, generator=different_token))
    different_sample = record_generator(
        token_position=5, head_id=0, sample_id="s2", **{
            k: v for k, v in common.items() if k != "sample_id"
        }
    )
    assert not torch.equal(torch.rand(4, generator=base), torch.rand(4, generator=different_sample))
    different_layer = record_generator(
        token_position=5, head_id=0, layer_id="L1", **{
            k: v for k, v in common.items() if k != "layer_id"
        }
    )
    assert not torch.equal(torch.rand(4, generator=base), torch.rand(4, generator=different_layer))


def test_noise_coverage_metrics_report_effective_rank_and_zero_fraction() -> None:
    transform = _two_sided(signal_dim=64, noise_dim=16, seed=1)
    coverage = noise_coverage_metrics(transform.matrix[64:].to(dtype=torch.float64))
    assert coverage.near_zero_coordinate_fraction == 0.0
    assert coverage.effective_rank > 0.9 * 16
    assert 0.0 < coverage.row_norm_min <= coverage.row_norm_max
    with pytest.raises(ValueError, match="zero energy"):
        noise_coverage_metrics(torch.zeros(4, 8))


def test_observed_noise_ratios_record_actual_per_token_norms() -> None:
    signal = torch.randn(500, 64)
    noise = torch.randn(500, 16) * 1.0
    ratios = observed_noise_ratios(signal, noise)
    assert 0.0 < ratios["median"] < ratios["max"]
    assert ratios["p90"] >= ratios["median"]
    with pytest.raises(ValueError, match="zero-norm"):
        observed_noise_ratios(torch.zeros(2, 4), torch.randn(2, 4))


def test_augmented_linear_closed_loop_with_independent_refresh() -> None:
    """End-to-end encode -> converted linear -> legal decode (plan stage A).

    The identity checked is ``c' = [hW+b, hC+eG+xi] M_out`` with ``C = 0`` and
    the plan's uniform refresh distribution. The first plan version keeps ``r``
    fixed along the chain; non-square noise propagation is deferred.
    """

    signal_dim_in, noise_dim = 64, 16
    signal_dim_out = 32
    in_transform = generate_two_sided_transform(
        signal_dim_in,
        noise_dim,
        seed=101,
        domain="loop-in",
        max_condition_number=3.0,
        dtype=torch.float64,
    )
    out_transform = generate_two_sided_transform(
        signal_dim_out,
        noise_dim,
        seed=102,
        domain="loop-out",
        max_condition_number=3.0,
        dtype=torch.float64,
    )
    weight_math = torch.randn(signal_dim_in, signal_dim_out, dtype=torch.float64) * 0.2
    bias = torch.randn(signal_dim_out, dtype=torch.float64) * 0.1
    coupling = torch.zeros(signal_dim_in, noise_dim, dtype=torch.float64)
    spec = NoiseRefreshSpec(noise_dim=noise_dim, gamma=0.5, energy_ratio=1.0)
    propagator = generate_propagator(spec, seed=103, domain="loop-prop")

    converted = convert_affine_chain(
        weight_math=weight_math,
        bias=bias,
        coupling=coupling,
        propagator=propagator,
        in_transform=in_transform,
        out_transform=out_transform,
        fixed_refresh=None,
    )
    assert converted.weight_pt.shape == (
        signal_dim_out + noise_dim,
        signal_dim_in + noise_dim,
    )

    h = torch.randn(4, 7, signal_dim_in, dtype=torch.float64)
    e0 = sample_initial_noise(
        spec,
        calibration_rms=float(h.float().norm(dim=-1).mean() / math.sqrt(signal_dim_in)),
        shape=(4, 7, noise_dim),
        generator=torch.Generator().manual_seed(55),
    )
    state = encode_debug(h, e0, in_transform, enabled=True)
    # per_request form (plan 7.5): c' = c W~ + b~_0 + xi @ M_out,n. The static
    # converted bias carries only [b, 0] M_out; the refresh maps through the
    # noise rows of the output basis.
    xi = sample_refresh_noise(
        spec, calibration_rms=1.0, shape=(4, 7, noise_dim), generator=torch.Generator().manual_seed(56)
    )
    out_matrix = out_transform.matrix.to(dtype=torch.float64)
    output = (
        torch.nn.functional.linear(state.mixed, converted.weight_pt, converted.bias_mixed)
        + xi @ out_matrix[signal_dim_out:]
    )
    decoded_signal, decoded_noise = decode_debug(
        type(state)(output, out_transform.descriptor), out_transform, enabled=True
    )

    expected_noise = apply_noise_refresh(e0, propagator, xi)
    expected_signal = h @ weight_math + bias
    torch.testing.assert_close(decoded_signal, expected_signal, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(decoded_noise, expected_noise, atol=1e-8, rtol=1e-8)


def test_fp32_small_tensor_chain_linear_error_gate() -> None:
    """FP32 signal max-abs error <= 1e-5 gate (plan stage A / AGENTS.md)."""

    signal_dim_in, noise_dim = 8, 2
    signal_dim_out = 6
    in_transform = generate_two_sided_transform(
        signal_dim_in, noise_dim, seed=201, domain="fp32-in", dtype=torch.float32
    )
    out_transform = generate_two_sided_transform(
        signal_dim_out, noise_dim, seed=202, domain="fp32-out", dtype=torch.float32
    )
    weight_math = torch.randn(signal_dim_in, signal_dim_out) * 0.2
    bias = torch.randn(signal_dim_out) * 0.1
    coupling = torch.zeros(signal_dim_in, noise_dim)
    spec = NoiseRefreshSpec(noise_dim=noise_dim, gamma=0.5, energy_ratio=1.0)
    propagator = generate_propagator(spec, seed=203, domain="fp32-prop").to(torch.float32)
    xi = sample_refresh_noise(
        spec, calibration_rms=0.5, shape=(noise_dim,), generator=torch.Generator().manual_seed(77)
    ).to(torch.float32)

    converted = convert_affine_chain(
        weight_math=weight_math,
        bias=bias,
        coupling=coupling,
        propagator=propagator,
        in_transform=in_transform,
        out_transform=out_transform,
        fixed_refresh=xi,
    )
    h = torch.randn(3, signal_dim_in)
    e = torch.randn(3, noise_dim)
    state = encode_debug(h, e, in_transform, enabled=True)
    output = torch.nn.functional.linear(
        state.mixed, converted.weight_pt, converted.bias_mixed
    )
    decoded_signal, _ = decode_debug(
        type(state)(output, out_transform.descriptor), out_transform, enabled=True
    )
    expected_signal = h @ weight_math + bias
    max_abs_error = float((decoded_signal - expected_signal).abs().max())
    assert max_abs_error <= 1e-5


def test_fixed_seed_reproduces_identical_noise_and_conversion() -> None:
    spec = NoiseRefreshSpec(noise_dim=16, gamma=0.5, energy_ratio=1.0)
    first = generate_propagator(spec, seed=301, domain="repro")
    second = generate_propagator(spec, seed=301, domain="repro")
    torch.testing.assert_close(first, second)

    g1 = record_generator(
        7, key_epoch="k", request_nonce="r", sample_id="s", token_position=0,
        layer_id="L", operation="attn",
    )
    g2 = record_generator(
        7, key_epoch="k", request_nonce="r", sample_id="s", token_position=0,
        layer_id="L", operation="attn",
    )
    torch.testing.assert_close(
        sample_refresh_noise(spec, calibration_rms=1.0, shape=(4, 16), generator=g1),
        sample_refresh_noise(spec, calibration_rms=1.0, shape=(4, 16), generator=g2),
    )


def test_different_records_never_share_noise_streams() -> None:
    spec = NoiseRefreshSpec(noise_dim=16, gamma=0.5, energy_ratio=1.0)
    draws = []
    for position in range(4):
        generator = record_generator(
            7, key_epoch="k", request_nonce="r", sample_id="s",
            token_position=position, layer_id="L", operation="attn",
        )
        draws.append(
            sample_refresh_noise(
                spec, calibration_rms=1.0, shape=(16,), generator=generator
            )
        )
    stacked = torch.stack(draws)
    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            assert not torch.equal(stacked[i], stacked[j])


def test_prefill_and_decode_agree_at_same_absolute_position() -> None:
    spec = NoiseRefreshSpec(noise_dim=16, gamma=0.5, energy_ratio=1.0)
    kwargs = dict(
        global_seed=7, key_epoch="k", request_nonce="r", sample_id="s",
        layer_id="L", operation="attn",
    )
    prefill = sample_refresh_noise(
        spec, calibration_rms=1.0, shape=(16,),
        generator=record_generator(token_position=9, head_id=0, **kwargs),
    )
    decode = sample_refresh_noise(
        spec, calibration_rms=1.0, shape=(16,),
        generator=record_generator(token_position=9, head_id=0, **kwargs),
    )
    torch.testing.assert_close(prefill, decode)
