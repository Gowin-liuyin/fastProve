from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fastprove.conversion import (
    math_to_torch_weight,
    torch_to_math_weight,
)
from fastprove.layers.linear import ChainLinear
from fastprove.seed import RequestContext
from fastprove.state import decode_debug, encode_debug
from fastprove.transforms import generate_transform


def _fixture() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator().manual_seed(123)
    h = torch.randn(2, 3, 3, generator=generator)
    e = torch.randn(2, 3, 2, generator=generator)
    weight_math = torch.randn(3, 4, generator=generator) * 0.2
    bias = torch.randn(4, generator=generator) * 0.1
    coupling = torch.randn(3, 3, generator=generator) * 0.1
    propagator = torch.randn(2, 3, generator=generator) * 0.2
    refresh = torch.randn(3, generator=generator) * 0.05
    return h, e, weight_math, bias, coupling, propagator, refresh


def test_math_and_pytorch_weight_layouts_round_trip_non_square() -> None:
    weight_math = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    weight_pt = math_to_torch_weight(weight_math)
    assert weight_pt.shape == (4, 3)
    torch.testing.assert_close(torch_to_math_weight(weight_pt), weight_math)
    x = torch.randn(5, 3)
    bias = torch.randn(4)
    torch.testing.assert_close(
        F.linear(x, weight_pt, bias),
        x @ weight_math + bias,
    )


def test_chain_linear_matches_augmented_affine_identity() -> None:
    h, e, weight_math, bias, coupling, propagator, refresh = _fixture()
    in_transform = generate_transform(3, 2, seed=10, domain="linear-in")
    out_transform = generate_transform(4, 3, seed=20, domain="linear-out")
    layer = ChainLinear.from_math(
        weight_math=weight_math,
        bias=bias,
        coupling=coupling,
        propagator=propagator,
        in_transform=in_transform,
        out_transform=out_transform,
        refresh_mode="fixed_debug",
        fixed_refresh=refresh,
        layer_id="linear-0",
    )
    state = encode_debug(h, e, in_transform, enabled=True)
    actual = layer(state).mixed

    expected_signal = h @ weight_math + bias
    expected_noise = h @ coupling + e @ propagator + refresh
    expected_mixed = (
        torch.cat((expected_signal, expected_noise), dim=-1)
        @ out_transform.matrix
    )
    assert torch.max(torch.abs(actual - expected_mixed)).item() <= 1e-5

    decoded_signal, decoded_noise = decode_debug(
        layer(state), out_transform, enabled=True
    )
    torch.testing.assert_close(
        decoded_signal, expected_signal, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        decoded_noise, expected_noise, atol=1e-5, rtol=1e-5
    )


def test_chain_linear_forward_does_not_invert_or_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h, e, weight_math, bias, coupling, propagator, refresh = _fixture()
    in_transform = generate_transform(3, 2, seed=10, domain="linear-in")
    out_transform = generate_transform(4, 3, seed=20, domain="linear-out")
    layer = ChainLinear.from_math(
        weight_math=weight_math,
        bias=bias,
        coupling=coupling,
        propagator=propagator,
        in_transform=in_transform,
        out_transform=out_transform,
        refresh_mode="fixed_debug",
        fixed_refresh=refresh,
        layer_id="linear-0",
    )
    state = encode_debug(h, e, in_transform, enabled=True)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("inverse/solve called during forward")

    monkeypatch.setattr(torch.linalg, "inv", forbidden)
    monkeypatch.setattr(torch.linalg, "solve", forbidden)
    monkeypatch.setattr(torch.linalg, "pinv", forbidden)
    monkeypatch.setattr(torch.linalg, "inv_ex", forbidden)
    monkeypatch.setattr(torch.linalg, "solve_ex", forbidden)
    monkeypatch.setattr(torch, "inverse", forbidden)
    monkeypatch.setattr(torch, "pinverse", forbidden)
    output = layer(state)
    assert output.mixed.shape == (2, 3, 7)


def test_per_request_refresh_is_reproducible_and_signal_invariant() -> None:
    h, e, weight_math, bias, coupling, propagator, _ = _fixture()
    in_transform = generate_transform(3, 2, seed=10, domain="linear-in")
    out_transform = generate_transform(4, 3, seed=20, domain="linear-out")
    layer = ChainLinear.from_math(
        weight_math=weight_math,
        bias=bias,
        coupling=coupling,
        propagator=propagator,
        in_transform=in_transform,
        out_transform=out_transform,
        refresh_mode="per_request",
        fixed_refresh=None,
        layer_id="linear-request",
        refresh_scale=0.05,
    )
    state = encode_debug(h, e, in_transform, enabled=True)
    first_context = RequestContext(7, "request-a")
    second_context = RequestContext(7, "request-b")

    first = layer(state, request_context=first_context)
    repeated = layer(state, request_context=first_context)
    second = layer(state, request_context=second_context)
    torch.testing.assert_close(first.mixed, repeated.mixed)

    first_signal, first_noise = decode_debug(
        first, out_transform, enabled=True
    )
    second_signal, second_noise = decode_debug(
        second, out_transform, enabled=True
    )
    torch.testing.assert_close(first_signal, second_signal, atol=1e-5, rtol=1e-5)
    assert not torch.equal(first_noise, second_noise)
    refresh_generator = first_context.generator_for(
        "chain-linear", "linear-request", "refresh"
    )
    expected_refresh = (
        torch.randn(3, generator=refresh_generator) * 0.05
    )
    expected_noise = (
        h @ coupling + e @ propagator + expected_refresh
    )
    torch.testing.assert_close(
        first_noise, expected_noise, atol=1e-5, rtol=1e-5
    )


def test_per_request_refresh_requires_context() -> None:
    h, e, weight_math, bias, coupling, propagator, _ = _fixture()
    in_transform = generate_transform(3, 2, seed=10, domain="linear-in")
    out_transform = generate_transform(4, 3, seed=20, domain="linear-out")
    layer = ChainLinear.from_math(
        weight_math=weight_math,
        bias=bias,
        coupling=coupling,
        propagator=propagator,
        in_transform=in_transform,
        out_transform=out_transform,
        refresh_mode="per_request",
        fixed_refresh=None,
        layer_id="linear-request",
    )
    state = encode_debug(h, e, in_transform, enabled=True)
    with pytest.raises(ValueError, match="request_context"):
        layer(state)


def test_refresh_scale_must_be_finite() -> None:
    h, e, weight_math, bias, coupling, propagator, _ = _fixture()
    del h, e
    in_transform = generate_transform(3, 2, seed=44, domain="scale-in")
    out_transform = generate_transform(4, 3, seed=45, domain="scale-out")
    with pytest.raises(ValueError, match="finite"):
        ChainLinear.from_math(
            weight_math=weight_math,
            bias=bias,
            coupling=coupling,
            propagator=propagator,
            in_transform=in_transform,
            out_transform=out_transform,
            refresh_mode="per_request",
            fixed_refresh=None,
            layer_id="nonfinite",
            refresh_scale=float("nan"),
        )


def test_state_dict_rejects_basis_metadata_mismatch() -> None:
    _, _, weight_math, bias, coupling, propagator, refresh = _fixture()
    in_a = generate_transform(3, 2, seed=46, domain="state-in-a")
    out_a = generate_transform(4, 3, seed=47, domain="state-out-a")
    in_b = generate_transform(3, 2, seed=48, domain="state-in-b")
    out_b = generate_transform(4, 3, seed=49, domain="state-out-b")
    layer_a = ChainLinear.from_math(
        weight_math=weight_math,
        bias=bias,
        coupling=coupling,
        propagator=propagator,
        in_transform=in_a,
        out_transform=out_a,
        refresh_mode="fixed_debug",
        fixed_refresh=refresh,
        layer_id="serialized",
    )
    layer_b = ChainLinear.from_math(
        weight_math=weight_math,
        bias=bias,
        coupling=coupling,
        propagator=propagator,
        in_transform=in_b,
        out_transform=out_b,
        refresh_mode="fixed_debug",
        fixed_refresh=refresh,
        layer_id="serialized",
    )
    original_weight = layer_b.weight_pt.clone()
    original_bias = layer_b.bias_mixed.clone()
    assert not any(
        "transform_matrix" in key for key in layer_a.state_dict().keys()
    )
    assert not any(
        "transform_matrix" in name
        for name, _ in layer_a.named_buffers()
    )
    with pytest.raises(RuntimeError, match="basis metadata"):
        layer_b.load_state_dict(layer_a.state_dict())
    torch.testing.assert_close(layer_b.weight_pt, original_weight)
    torch.testing.assert_close(layer_b.bias_mixed, original_bias)

    missing_metadata = layer_a.state_dict()
    del missing_metadata["_extra_state"]
    with pytest.raises(RuntimeError, match="basis metadata"):
        layer_b.load_state_dict(missing_metadata, strict=False)
    torch.testing.assert_close(layer_b.weight_pt, original_weight)
    torch.testing.assert_close(layer_b.bias_mixed, original_bias)


def test_direct_constructor_rejects_invalid_deployed_parameters() -> None:
    in_transform = generate_transform(3, 2, seed=50, domain="direct-in")
    out_transform = generate_transform(4, 3, seed=51, domain="direct-out")
    with pytest.raises(ValueError, match="floating"):
        ChainLinear(
            weight_pt=torch.ones(7, 5, dtype=torch.int64),
            bias_mixed=torch.ones(7, dtype=torch.int64),
            in_transform=in_transform,
            out_transform=out_transform,
            refresh_mode="fixed_debug",
            layer_id="invalid-direct",
            refresh_scale=0.1,
        )
    with pytest.raises(ValueError, match="finite"):
        ChainLinear(
            weight_pt=torch.full((7, 5), float("nan")),
            bias_mixed=torch.zeros(7),
            in_transform=in_transform,
            out_transform=out_transform,
            refresh_mode="fixed_debug",
            layer_id="invalid-direct",
            refresh_scale=0.1,
        )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable"
)
def test_per_request_refresh_follows_module_device_on_mps() -> None:
    h, e, weight_math, bias, coupling, propagator, _ = _fixture()
    in_transform = generate_transform(3, 2, seed=40, domain="mps-in")
    out_transform = generate_transform(4, 3, seed=41, domain="mps-out")
    layer = ChainLinear.from_math(
        weight_math=weight_math,
        bias=bias,
        coupling=coupling,
        propagator=propagator,
        in_transform=in_transform,
        out_transform=out_transform,
        refresh_mode="per_request",
        fixed_refresh=None,
        layer_id="mps-request",
    ).to("mps")
    cpu_state = encode_debug(h, e, in_transform, enabled=True)
    state = type(cpu_state)(cpu_state.mixed.to("mps"), cpu_state.basis)
    output = layer(state, RequestContext(7, "mps"))
    assert output.mixed.device.type == "mps"


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable"
)
def test_offline_conversion_accepts_mps_source_weights() -> None:
    _, _, weight_math, bias, coupling, propagator, refresh = _fixture()
    in_transform = generate_transform(3, 2, seed=42, domain="convert-in")
    out_transform = generate_transform(4, 3, seed=43, domain="convert-out")
    layer = ChainLinear.from_math(
        weight_math=weight_math.to("mps"),
        bias=bias.to("mps"),
        coupling=coupling.to("mps"),
        propagator=propagator.to("mps"),
        in_transform=in_transform,
        out_transform=out_transform,
        refresh_mode="fixed_debug",
        fixed_refresh=refresh.to("mps"),
        layer_id="mps-conversion",
    )
    assert layer.weight_pt.device.type == "mps"
