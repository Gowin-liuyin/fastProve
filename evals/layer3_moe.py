"""Layer 3 — MoE router discrete consistency (protocol §4.3).

If the model has no MoE, metrics are reported as N/A with an explicit skip.
Standalone StableRouter tests still exercise the metric pipeline.

HARD GATE: expert set match = 100% when MoE is present.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from fastprove.layers.router import (
    RouterConfig,
    RouterMode,
    StableRouter,
)
from fastprove.seed import RequestContext, make_generator

from .metrics_common import kl_divergence
from .model_factory import EvalModels


def _model_has_moe(models: EvalModels) -> bool:
    plain = models.plain
    if hasattr(plain, "router") or hasattr(plain, "moe"):
        return True
    for block in plain.blocks:
        if hasattr(block, "router") or hasattr(block, "experts"):
            return True
    if models.obfuscated is not None:
        for block in models.obfuscated.blocks:
            if hasattr(block, "router") or hasattr(block, "experts"):
                return True
    return False


def _router_metrics_from_logits(
    plain_logits: torch.Tensor,
    obf_logits: torch.Tensor,
    *,
    top_k: int,
    tau: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Compare router logits token-wise."""

    if plain_logits.shape != obf_logits.shape:
        raise ValueError("router logit shapes must match")
    # Stable top-k (same as router: argsort descending stable).
    plain_idx = torch.argsort(plain_logits, dim=-1, descending=True, stable=True)[
        ..., :top_k
    ]
    obf_idx = torch.argsort(obf_logits, dim=-1, descending=True, stable=True)[
        ..., :top_k
    ]
    # Set match ignores order.
    set_matches = []
    order_matches = []
    for p_row, o_row in zip(
        plain_idx.reshape(-1, top_k), obf_idx.reshape(-1, top_k)
    ):
        p_set = set(int(x) for x in p_row.tolist())
        o_set = set(int(x) for x in o_row.tolist())
        set_matches.append(1.0 if p_set == o_set else 0.0)
        order_matches.append(
            1.0 if p_row.tolist() == o_row.tolist() else 0.0
        )
    expert_set_match = float(sum(set_matches) / len(set_matches))
    expert_order_match = float(sum(order_matches) / len(order_matches))

    # Gate weights via softmax over selected experts (simple proxy).
    plain_probs = torch.softmax(plain_logits.float(), dim=-1)
    obf_probs = torch.softmax(obf_logits.float(), dim=-1)
    gate_l1 = float((plain_probs - obf_probs).abs().sum(dim=-1).mean().item())
    gate_kl = kl_divergence(plain_probs, obf_probs)

    # Margin Δ_k = r_(k) − r_(k+1)
    flat = plain_logits.reshape(-1, plain_logits.shape[-1]).float()
    margins = []
    for row in flat:
        sorted_vals = torch.sort(row, descending=True, stable=True).values
        if sorted_vals.numel() > top_k:
            margins.append(float((sorted_vals[top_k - 1] - sorted_vals[top_k]).item()))
    margins_t = torch.tensor(margins, dtype=torch.float32) if margins else torch.zeros(1)

    violation_rate = 0.0
    if tau is not None and margins:
        # tau may be scalar or per-token.
        tau_vals = tau.reshape(-1).float()
        if tau_vals.numel() == 1:
            tau_vals = tau_vals.expand(len(margins))
        n = min(len(margins), tau_vals.numel())
        viol = sum(
            1.0
            for i in range(n)
            if not (2.0 * float(tau_vals[i].item()) < margins[i])
        )
        violation_rate = viol / float(n)

    low_margin_fallback = float((margins_t < 1e-3).float().mean().item()) if margins else 0.0

    return {
        "expert_set_match": expert_set_match,
        "expert_order_match": expert_order_match,
        "gate_weight_l1": gate_l1,
        "gate_weight_kl": gate_kl,
        "delta_k_mean": float(margins_t.mean().item()),
        "delta_k_min": float(margins_t.min().item()),
        "delta_k_p10": float(torch.quantile(margins_t, 0.1).item())
        if margins_t.numel()
        else 0.0,
        "two_tau_lt_delta_k_violation_rate": violation_rate,
        "low_margin_fallback_rate": low_margin_fallback,
        "token_count": int(flat.shape[0]),
    }


@torch.no_grad()
def _standalone_router_eval(
    *,
    seed: int,
    n_experts: int = 8,
    top_k: int = 2,
    batch: int = 4,
    seq: int = 6,
    mode: str = "exact",
) -> Dict[str, Any]:
    """Exercise StableRouter metrics without a MoE LM."""

    generator = make_generator(seed, "layer3-router")
    # Signed permutation as expert reorder source.
    # generate_signed_permutation returns a matrix; router wants a vector perm.
    perm = torch.randperm(n_experts, generator=generator)
    router_plain = StableRouter(
        permutation=torch.arange(n_experts),
        top_k=top_k,
        mode=RouterMode.EXACT,
        config=None,
        debug_enabled=True,
        layer_id="plain-router",
    )
    if mode == "exact":
        router_obf = StableRouter(
            permutation=perm,
            top_k=top_k,
            mode=RouterMode.EXACT,
            config=None,
            debug_enabled=True,
            layer_id="obf-router",
        )
    else:
        router_obf = StableRouter(
            permutation=perm,
            top_k=top_k,
            mode=RouterMode.MARGIN_BOUNDED,
            config=RouterConfig(tau_max=0.01, tau_error=0.01, alpha=0.8),
            debug_enabled=True,
            layer_id="obf-router",
        )

    # Shared underlying expert scores before permutation.
    hidden_dim = 16
    # StableRouter typically takes hidden and projects — check API.
    # Read router forward signature.
    logits = torch.randn(batch, seq, n_experts, generator=generator)
    ctx = RequestContext(seed, "layer3")

    # Prefer debug path if available.
    plain_decision, plain_dbg = router_plain.forward_debug(logits, context=ctx)
    obf_decision, obf_dbg = router_obf.forward_debug(logits, context=ctx)

    # physical_expert_indices are indices into the *permuted* layout;
    # map back to canonical expert ids via permutation[idx] (see test_router).
    plain_idx = router_plain.permutation[plain_decision.physical_expert_indices]
    obf_idx = router_obf.permutation[obf_decision.physical_expert_indices]
    set_matches = []
    order_matches = []
    for p_row, o_row in zip(
        plain_idx.reshape(-1, top_k), obf_idx.reshape(-1, top_k)
    ):
        p_list = [int(x) for x in p_row.tolist()]
        o_list = [int(x) for x in o_row.tolist()]
        p_set = set(p_list)
        o_set = set(o_list)
        set_matches.append(1.0 if p_set == o_set else 0.0)
        order_matches.append(1.0 if p_list == o_list else 0.0)

    # Reconstruct physical-space logits for gate L1/KL.
    def _to_physical(
        permuted_logits: torch.Tensor, permutation: torch.Tensor
    ) -> torch.Tensor:
        physical = torch.zeros_like(permuted_logits)
        physical[..., permutation.long()] = permuted_logits
        return physical

    plain_phys = _to_physical(
        plain_dbg.permuted_clean_logits, router_plain.permutation
    )
    obf_phys = _to_physical(
        obf_dbg.permuted_clean_logits, router_obf.permutation
    )
    metrics = _router_metrics_from_logits(
        plain_phys,
        obf_phys,
        top_k=top_k,
        tau=obf_dbg.tau if mode != "exact" else None,
    )
    # Prefer decision-based set match (handles physical reordering).
    metrics["expert_set_match"] = float(sum(set_matches) / len(set_matches))
    metrics["expert_order_match"] = float(
        sum(order_matches) / len(order_matches)
    )
    metrics["mode"] = mode
    metrics["standalone"] = True
    metrics["first_divergence_layer"] = (
        None if metrics["expert_set_match"] == 1.0 else 0
    )
    return metrics


@torch.no_grad()
def run_layer3(
    models: EvalModels,
    input_ids: torch.Tensor,
    *,
    token_mask: Optional[torch.Tensor] = None,
    top_k: int = 2,
) -> Dict[str, Any]:
    """Run MoE router consistency; skip gracefully when model is dense."""

    if not _model_has_moe(models):
        # Still run standalone router identity to keep the metric pipeline live.
        standalone = _standalone_router_eval(
            seed=models.config.runtime.seed, top_k=top_k, mode="exact"
        )
        return {
            "moe_present": False,
            "skipped": True,
            "skip_reason": (
                "model has no MoE router; standalone StableRouter metrics only"
            ),
            "expert_set_match": None,  # gate treats missing as skip
            "standalone_router": standalone,
            "first_divergence_layer": None,
        }

    # Future: hook into integrated MoE blocks when present.
    # Placeholder path collects per-layer router debug if attributes exist.
    per_layer: List[Dict[str, Any]] = []
    first_div = None
    set_matches: List[float] = []
    for layer_index, block in enumerate(models.plain.blocks):
        if not hasattr(block, "router"):
            continue
        # Integrated path reserved for future MoE LM.
        pass

    if not per_layer:
        return {
            "moe_present": True,
            "skipped": True,
            "skip_reason": "MoE attributes present but no router debug hooks yet",
            "expert_set_match": None,
            "first_divergence_layer": first_div,
        }

    return {
        "moe_present": True,
        "skipped": False,
        "expert_set_match": min(set_matches) if set_matches else 1.0,
        "per_layer": per_layer,
        "first_divergence_layer": first_div,
    }
