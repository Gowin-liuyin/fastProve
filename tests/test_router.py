from __future__ import annotations

import torch

from fastprove.layers.router import (
    RouterConfig,
    RouterMode,
    StableRouter,
    reorder_experts,
)
from fastprove.seed import RequestContext


def test_exact_router_permutation_selects_same_canonical_experts() -> None:
    logits = torch.tensor([[5.0, 1.0, 4.0, 3.0]])
    permutation = torch.tensor([2, 0, 3, 1])
    router = StableRouter(
        permutation=permutation,
        top_k=2,
        mode=RouterMode.EXACT,
        config=None,
        debug_enabled=True,
        layer_id="router-exact",
    )
    decision, debug = router.forward_debug(
        logits, RequestContext(3, "router")
    )
    selected_canonical = permutation[decision.physical_expert_indices]
    assert torch.equal(selected_canonical, torch.tensor([[0, 2]]))
    torch.testing.assert_close(
        decision.gate_weights.sum(dim=-1), torch.ones(1)
    )
    assert torch.count_nonzero(debug.noise).item() == 0


def test_router_ties_follow_canonical_expert_order_after_permutation() -> None:
    logits = torch.tensor([[2.0, 2.0, 2.0, 1.0]])
    permutation = torch.tensor([2, 1, 0, 3])
    router = StableRouter(
        permutation=permutation,
        top_k=2,
        mode=RouterMode.EXACT,
        config=None,
        debug_enabled=False,
        layer_id="router-tie",
    )
    decision = router(logits, RequestContext(3, "tie"))
    selected_canonical = permutation[decision.physical_expert_indices]
    assert torch.equal(selected_canonical, torch.tensor([[0, 1]]))


def test_margin_bounded_router_preserves_set_and_obeys_norm() -> None:
    logits = torch.tensor([[4.0, 3.0, 1.0, -2.0], [3.0, 2.5, 2.0, 0.0]])
    permutation = torch.tensor([2, 0, 3, 1])
    config = RouterConfig(
        tau_max=0.2,
        tau_error=0.2,
        alpha=0.8,
        noisy_gate_weights=False,
    )
    router = StableRouter(
        permutation=permutation,
        top_k=2,
        mode=RouterMode.MARGIN_BOUNDED,
        config=config,
        debug_enabled=True,
        layer_id="router-margin",
    )
    decision, debug = router.forward_debug(
        logits, RequestContext(8, "margin")
    )
    selected_canonical = torch.sort(
        permutation[decision.physical_expert_indices], dim=-1
    ).values
    expected = torch.sort(torch.topk(logits, 2, dim=-1).indices, dim=-1).values
    assert torch.equal(selected_canonical, expected)
    realized = debug.noise.abs().amax(dim=-1, keepdim=True)
    assert torch.all(realized <= debug.tau + 1e-7)
    assert torch.all(2.0 * debug.tau < debug.margin)


def test_router_noise_norm_respects_python_float_bound_without_fp32_ulp_overrun() -> None:
    logits = torch.tensor([[4.0, 3.0, 1.0, -2.0]], dtype=torch.float32)
    config = RouterConfig(
        tau_max=0.001, tau_error=0.001, alpha=0.8, noisy_gate_weights=False
    )
    router = StableRouter(
        permutation=torch.tensor([2, 0, 3, 1]),
        top_k=2,
        mode=RouterMode.MARGIN_BOUNDED,
        config=config,
        debug_enabled=True,
        layer_id="router-ulp-bound",
    )
    _, debug = router.forward_debug(logits, RequestContext(14, "router-ulp"))
    assert debug.tau.item() > config.tau_max
    assert debug.noise.abs().amax().item() <= config.tau_max


def test_router_noisy_gate_weights_are_optional() -> None:
    logits = torch.tensor([[3.0, 2.0, 0.0, -1.0]])
    permutation = torch.tensor([1, 3, 0, 2])
    common = dict(
        permutation=permutation,
        top_k=2,
        mode=RouterMode.MARGIN_BOUNDED,
        debug_enabled=False,
        layer_id="router-weights",
    )
    clean_weights = StableRouter(
        config=RouterConfig(0.1, 0.1, 0.8, False), **common
    )(logits, RequestContext(9, "weights"))
    noisy_weights = StableRouter(
        config=RouterConfig(0.1, 0.1, 0.8, True), **common
    )(logits, RequestContext(9, "weights"))
    assert torch.equal(
        clean_weights.physical_expert_indices,
        noisy_weights.physical_expert_indices,
    )
    assert not torch.equal(
        clean_weights.gate_weights, noisy_weights.gate_weights
    )


def test_reorder_experts_matches_permuted_physical_layout() -> None:
    experts = ["expert-0", "expert-1", "expert-2"]
    permutation = torch.tensor([2, 0, 1])
    assert reorder_experts(experts, permutation) == [
        "expert-2",
        "expert-0",
        "expert-1",
    ]
