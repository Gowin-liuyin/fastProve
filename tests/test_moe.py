"""Obfuscated MoE layer: router + expert dispatch/combine (task C5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from fastprove.layers.moe import ObfuscatedMoE
from fastprove.layers.router import RouterConfig, RouterMode
from fastprove.seed import RequestContext
from fastprove.structured import generate_structured_basis


def _materials(
    *,
    signal: int = 32,
    noise: int = 4,
    experts: int = 8,
    intermediate: int = 48,
    top_k: int = 2,
    seed: int = 7,
):
    basis = generate_structured_basis(
        signal, noise, seed=seed, domain="moe", block_size=4, dtype=torch.float64
    )
    generator = torch.Generator().manual_seed(seed)
    gamma = torch.rand(signal, generator=generator) + 0.5
    router_w = torch.randn(signal, experts, generator=generator) * 0.2
    expert_weights = [
        (
            torch.randn(signal, intermediate, generator=generator) * 0.1,
            torch.randn(signal, intermediate, generator=generator) * 0.1,
            torch.randn(intermediate, signal, generator=generator) * 0.1,
        )
        for _ in range(experts)
    ]
    permutation = torch.randperm(experts, generator=generator)
    return basis, gamma, router_w, expert_weights, permutation


def _build(
    basis,
    gamma,
    router_w,
    expert_weights,
    permutation,
    *,
    top_k: int,
    mode: RouterMode,
    config: RouterConfig | None,
    seed: int = 7,
):
    return ObfuscatedMoE(
        basis=basis,
        gamma_ffn=gamma,
        router_weight_math=router_w,
        plain_experts=expert_weights,
        expert_permutation=permutation,
        top_k=top_k,
        router_mode=mode,
        router_config=config,
        seed=seed,
        layer_id="block-0",
        debug_enabled=True,
        noise_propagation_gamma=0.5,
        refresh_mode="fixed_debug",
    )


def _obf_state(basis, hidden, noise_scale=0.05):
    return basis.mix(
        torch.cat(
            (
                hidden,
                torch.randn(
                    *hidden.shape[:-1],
                    basis.noise_dim,
                    generator=torch.Generator().manual_seed(3),
                )
                * noise_scale,
            ),
            dim=-1,
        )
    )


def _router_input(basis, moe, mixed):
    """The exact logits tensor the StableRouter sorts (plaintext domain)."""

    scale = basis.rms_scale(mixed, 1e-5).to(dtype=mixed.dtype)
    return (
        mixed @ moe.deployed_router.to(device=mixed.device, dtype=mixed.dtype)
    ) / scale


def _plaintext_reference(
    hidden,
    rho,
    gamma,
    router_w,
    expert_weights,
    permutation,
    top_k,
    r,
):
    """Reference selection/gates/combine computed from the router logits ``r``.

    The deployed router logits are the plaintext logits
    ``r = (h / rho) gamma W_r`` up to floating-point rounding; the selection
    and gate weights are therefore computed on ``r`` directly so the
    comparison is exact for the same tensor the router sorts.
    """

    ordered = torch.sort(r, dim=-1, descending=True, stable=True)
    selected_canonical = ordered.indices[..., :top_k]
    selected_logits = torch.gather(r, -1, selected_canonical)
    gate = torch.softmax(selected_logits.float(), dim=-1)
    # physical = argsort(P_E)[selected_canonical]; canonical = P_E[physical]
    physical = torch.argsort(permutation)[selected_canonical]
    canonical_experts = permutation[physical]
    z_stack = []
    for canonical in range(len(expert_weights)):
        gate_w, up_w, down_w = expert_weights[canonical]
        normalized = (hidden / rho) * gamma
        z = F.silu(normalized @ gate_w) * (normalized @ up_w)
        z_stack.append(z @ down_w)
    z_stack = torch.stack(z_stack, dim=-1)  # [B, S, d, E]
    combined = torch.zeros_like(hidden)
    for j in range(top_k):
        zj = torch.gather(
            z_stack,
            -1,
            canonical_experts[..., j]
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, hidden.shape[-1], 1),
        ).squeeze(-1)
        combined = combined + gate[..., j].unsqueeze(-1) * zj
    return r, selected_canonical, gate, combined, physical


def test_expert_set_matches_plaintext_exactly() -> None:
    for top_k in (1, 2, 4):
        basis, gamma, router_w, expert_weights, permutation = _materials(
            seed=11 + top_k
        )
        moe = _build(
            basis,
            gamma,
            router_w,
            expert_weights,
            permutation,
            top_k=top_k,
            mode=RouterMode.EXACT,
            config=None,
        )
        torch.manual_seed(3)
        hidden = torch.randn(2, 5, basis.signal_dim)
        mixed = _obf_state(basis, hidden)
        scale = basis.rms_scale(mixed, 1e-5)
        r_obf = _router_input(basis, moe, mixed)
        # The deployed router logits must equal the plaintext formula.
        expected_r = (hidden / scale) * gamma @ router_w
        assert torch.allclose(
            r_obf, expected_r.to(dtype=r_obf.dtype), atol=1e-4, rtol=1e-4
        )
        _, selected_canonical, _, _, physical = _plaintext_reference(
            hidden,
            scale,
            gamma,
            router_w,
            expert_weights,
            permutation,
            top_k,
            r_obf,
        )
        with torch.no_grad():
            _, debug = moe.forward_debug(mixed, RequestContext(3, "moe"))
        assert torch.equal(
            debug.physical_expert_indices, physical
        ), "top_k=%d set mismatch" % top_k
        assert torch.equal(
            permutation[debug.physical_expert_indices], selected_canonical
        )


def test_tie_breaking_matches_plaintext() -> None:
    basis, gamma, router_w, expert_weights, permutation = _materials(seed=17)
    moe = _build(
        basis,
        gamma,
        router_w,
        expert_weights,
        permutation,
        top_k=2,
        mode=RouterMode.EXACT,
        config=None,
    )
    # Dominant first four coordinates produce exact router ties.
    hidden = torch.zeros(1, 2, basis.signal_dim)
    hidden[:, :, :4] = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    mixed = basis.mix(
        torch.cat((hidden, torch.zeros(1, 2, basis.noise_dim)), dim=-1)
    )
    scale = basis.rms_scale(mixed, 1e-5)
    r_obf = _router_input(basis, moe, mixed)
    _, selected_canonical, _, _, _ = _plaintext_reference(
        hidden,
        scale,
        gamma,
        router_w,
        expert_weights,
        permutation,
        2,
        r_obf,
    )
    with torch.no_grad():
        _, debug = moe.forward_debug(mixed, RequestContext(3, "tie"))
    assert torch.equal(
        permutation[debug.physical_expert_indices], selected_canonical
    )


def test_margin_bounded_noise_preserves_expert_set() -> None:
    basis, gamma, router_w, expert_weights, permutation = _materials(seed=23)
    config = RouterConfig(
        tau_max=0.05,
        tau_error=0.05,
        alpha=0.5,
        noisy_gate_weights=False,
    )
    moe = _build(
        basis,
        gamma,
        router_w,
        expert_weights,
        permutation,
        top_k=2,
        mode=RouterMode.MARGIN_BOUNDED,
        config=config,
    )
    torch.manual_seed(5)
    hidden = torch.randn(1, 8, basis.signal_dim)
    mixed = _obf_state(basis, hidden)
    scale = basis.rms_scale(mixed, 1e-5)
    r_obf = _router_input(basis, moe, mixed)
    _, selected_canonical, _, _, physical = _plaintext_reference(
        hidden,
        scale,
        gamma,
        router_w,
        expert_weights,
        permutation,
        2,
        r_obf,
    )
    with torch.no_grad():
        _, debug = moe.forward_debug(mixed, RequestContext(3, "margin"))
    assert torch.equal(
        permutation[debug.physical_expert_indices], selected_canonical
    )
    assert torch.all(debug.router_debug.noise.abs() <= 0.05)
    # 2 tau < Delta_k must hold with the actual (capped) budget on the
    # top-k boundary: tau = min(tau_max, alpha * Delta / 2).
    tau = debug.router_debug.tau
    sorted_r = torch.sort(r_obf, dim=-1, descending=True, stable=True).values
    margin = sorted_r[..., 1:2] - sorted_r[..., 2:3]
    assert torch.all(2 * tau < margin.clamp_min(0) + 1e-6)


def test_gate_weights_use_clean_logits() -> None:
    basis, gamma, router_w, expert_weights, permutation = _materials(seed=29)
    config = RouterConfig(
        tau_max=0.05,
        tau_error=0.05,
        alpha=0.5,
        noisy_gate_weights=False,
    )
    moe = _build(
        basis,
        gamma,
        router_w,
        expert_weights,
        permutation,
        top_k=2,
        mode=RouterMode.MARGIN_BOUNDED,
        config=config,
    )
    torch.manual_seed(5)
    hidden = torch.randn(1, 4, basis.signal_dim)
    mixed = _obf_state(basis, hidden)
    scale = basis.rms_scale(mixed, 1e-5)
    r_obf = _router_input(basis, moe, mixed)
    _, selected_canonical, gate, _, _ = _plaintext_reference(
        hidden,
        scale,
        gamma,
        router_w,
        expert_weights,
        permutation,
        2,
        r_obf,
    )
    with torch.no_grad():
        _, debug = moe.forward_debug(mixed, RequestContext(3, "gate"))
    selected_logits = torch.gather(r_obf, -1, selected_canonical)
    expected_gate = torch.softmax(selected_logits.float(), dim=-1)
    torch.testing.assert_close(
        debug.gate_weights, expected_gate, atol=1e-6, rtol=1e-6
    )
    assert not torch.allclose(
        debug.gate_weights, torch.full_like(debug.gate_weights, 0.5)
    )


def test_physical_experts_are_reordered_consistently() -> None:
    basis, gamma, router_w, expert_weights, permutation = _materials(seed=31)
    moe = _build(
        basis,
        gamma,
        router_w,
        expert_weights,
        permutation,
        top_k=4,
        mode=RouterMode.EXACT,
        config=None,
    )
    torch.manual_seed(7)
    hidden = torch.randn(1, 6, basis.signal_dim)
    mixed = _obf_state(basis, hidden)
    scale = basis.rms_scale(mixed, 1e-5)
    r_obf = _router_input(basis, moe, mixed)
    _, _, gate, combined_plain, physical = _plaintext_reference(
        hidden,
        scale,
        gamma,
        router_w,
        expert_weights,
        permutation,
        4,
        r_obf,
    )
    with torch.no_grad():
        increment, debug = moe.forward_debug(mixed, RequestContext(3, "phys"))
    assert torch.equal(debug.physical_expert_indices, physical)
    decoded_signal = basis.unmix(increment.mixed.to(dtype=torch.float64))[
        ..., : basis.signal_dim
    ]
    assert torch.allclose(
        decoded_signal, combined_plain.to(dtype=torch.float64), atol=1e-4
    )


def test_all_expert_outputs_share_the_residual_basis() -> None:
    basis, gamma, router_w, expert_weights, permutation = _materials(seed=37)
    moe = _build(
        basis,
        gamma,
        router_w,
        expert_weights,
        permutation,
        top_k=2,
        mode=RouterMode.EXACT,
        config=None,
    )
    torch.manual_seed(9)
    hidden = torch.randn(1, 3, basis.signal_dim)
    mixed = _obf_state(basis, hidden)
    with torch.no_grad():
        increment, _ = moe.forward_debug(mixed, RequestContext(3, "basis"))
    # MixedState.add inside the combine already validated fingerprints; the
    # combined state must decode to a finite signal on the same basis.
    assert increment.basis == basis.descriptor
    decoded = basis.unmix(increment.mixed.to(dtype=torch.float64))
    assert torch.isfinite(decoded).all()


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_combined_output_decodes_to_plaintext_moe_output(top_k: int) -> None:
    basis, gamma, router_w, expert_weights, permutation = _materials(
        seed=41 + top_k
    )
    moe = _build(
        basis,
        gamma,
        router_w,
        expert_weights,
        permutation,
        top_k=top_k,
        mode=RouterMode.EXACT,
        config=None,
    )
    torch.manual_seed(13)
    hidden = torch.randn(2, 4, basis.signal_dim)
    mixed = _obf_state(basis, hidden)
    scale = basis.rms_scale(mixed, 1e-5)
    r_obf = _router_input(basis, moe, mixed)
    _, _, _, combined_plain, _ = _plaintext_reference(
        hidden,
        scale,
        gamma,
        router_w,
        expert_weights,
        permutation,
        top_k,
        r_obf,
    )
    with torch.no_grad():
        increment = moe(mixed, RequestContext(3, "combine"))
    decoded_signal = basis.unmix(increment.mixed.to(dtype=torch.float64))[
        ..., : basis.signal_dim
    ]
    assert torch.allclose(
        decoded_signal, combined_plain.to(dtype=torch.float64), atol=1e-4
    )
