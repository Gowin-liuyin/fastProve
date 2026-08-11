"""Obfuscated MoE: expert-permuted router and per-expert deployed FFNs.

The dense FFN deployed form (``layers/deployed.py``) is reused per expert,
each with its own domain-separated key material ``LAYER_<L>_EXPERT_<E>``.
Physical experts are reordered consistently with the router permutation
``r' = r P_E`` (``reorder_experts``), and the router logits are computed
without decoding:

    W_tilde_router = P diag(gamma_ffn) W_r P_E        shape [n, E]
    r' = (c @ W_tilde_router) / rho

Gate weights always come from the **clean** router logits (the default
``noisy_gate_weights=False``); margin-bounded noise only affects which
experts are selected. Every expert output is a residual increment in the
shared hidden basis; combining them is a gate-weighted sum validated through
``MixedState.add``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..seed import RequestContext, make_generator
from ..state import MixedState
from ..structured import StructuredBasis
from ..transforms import BasisDescriptor, generate_signed_permutation
from .deployed import (
    DeployedFeedForwardWeights,
    build_deployed_feed_forward,
    validate_auxiliary_budget,
)
from .router import (
    RouterConfig,
    RouterDebug,
    RouterMode,
    StableRouter,
    reorder_experts,
)
from .swiglu import generate_swiglu_transform


@dataclass(frozen=True)
class MoEDebug:
    """Debug-only MoE diagnostics."""

    router_debug: RouterDebug
    gate_weights: torch.Tensor
    physical_expert_indices: torch.Tensor


class _ExpertFFN(nn.Module):
    """One deployed expert FFN in the shared mixed-state basis."""

    def __init__(
        self,
        *,
        basis: StructuredBasis,
        gamma_ffn: torch.Tensor,
        gate_weight_math: torch.Tensor,
        up_weight_math: torch.Tensor,
        down_weight_math: torch.Tensor,
        domain: str,
        seed: int,
        hidden_noise_dim: int,
        intermediate: int,
        hidden_size: int,
        fixed_refresh: bool,
        noise_propagation_gamma: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.domain = domain
        self.basis_descriptor = basis.descriptor
        self.hidden_noise_dim = hidden_noise_dim
        self.noise_injection_enabled = True
        self.refresh_noise_scale = 1.0
        swiglu_transform = generate_swiglu_transform(
            intermediate,
            seed=seed,
            domain="%s-SWIGLU" % domain,
        )
        coupling_z = (
            torch.randn(intermediate, hidden_noise_dim, generator=make_generator(
                seed, "%s-Cz" % domain, intermediate, hidden_noise_dim
            ))
            * 0.02
        )
        coupling_d = (
            torch.randn(hidden_size, hidden_noise_dim, generator=make_generator(
                seed, "%s-Cd" % domain, hidden_size, hidden_noise_dim
            ))
            * 0.02
        )
        propagator = noise_propagation_gamma * generate_signed_permutation(
            hidden_noise_dim,
            seed=seed,
            domain="%s-G" % domain,
        )
        fixed = torch.randn(hidden_noise_dim, generator=make_generator(
            seed, "%s-xi" % domain, hidden_noise_dim
        )) * 0.02 if fixed_refresh else torch.zeros(hidden_noise_dim)
        validate_auxiliary_budget(
            basis=basis,
            sample_signal=torch.randn(
                64,
                hidden_size,
                generator=make_generator(seed, "%s-budget" % domain),
            ),
            signal_noise_coupling=coupling_d,
            noise_propagator=propagator,
            context=domain,
        )
        deployed = build_deployed_feed_forward(
            basis=basis,
            gamma_ffn=gamma_ffn,
            gate_weight_math=gate_weight_math,
            up_weight_math=up_weight_math,
            down_weight_math=down_weight_math,
            neuron_permutation=swiglu_transform.permutation,
            neuron_scale=swiglu_transform.scale,
            swiglu_noise_coupling=coupling_z,
            down_noise_coupling=coupling_d,
            noise_propagator=propagator,
            fixed_refresh=fixed if fixed_refresh else None,
            dtype=dtype,
        )
        self.register_buffer("deployed_gate", deployed.gate, persistent=True)
        self.register_buffer("deployed_up", deployed.up, persistent=True)
        self.register_buffer("deployed_ffn_out", deployed.output, persistent=True)
        self.register_buffer(
            "deployed_ffn_noise_out", deployed.noise_out, persistent=True
        )
        self.register_buffer(
            "deployed_ffn_refresh_out", deployed.refresh_out, persistent=True
        )
        # M_bot for the per-request refresh ``xi @ M_bot``; the same shipped
        # artifact the dense block carries (B1.3 records the consequence).
        self.register_buffer(
            "_noise_rows",
            basis.noise_rows().to(dtype=torch.float32),
            persistent=True,
        )
        # N for the ``(c @ N) @ Wnz`` residual term (B1.3 records that
        # shipping N lets an observer recover the noise state).
        self.register_buffer(
            "noise_read",
            basis.noise_projection().to(dtype=torch.float32),
            persistent=True,
        )
        self.refresh_mode = "fixed_debug" if fixed_refresh else "per_request"

    def _refresh_out(
        self, reference: torch.Tensor, context: RequestContext
    ) -> torch.Tensor:
        if not self.noise_injection_enabled:
            return torch.zeros(
                self.deployed_ffn_refresh_out.shape[-1],
                device=reference.device,
                dtype=reference.dtype,
            )
        if self.refresh_mode == "fixed_debug":
            return self.deployed_ffn_refresh_out.sum(dim=0)
        sampled = torch.randn(
            self.hidden_noise_dim,
            generator=context.generator_for("moe", self.domain, "refresh"),
            dtype=torch.float32,
        ) * (0.02 * self.refresh_noise_scale)
        rows = self._noise_rows.to(
            device=reference.device, dtype=reference.dtype
        )
        return sampled.to(device=reference.device, dtype=reference.dtype) @ rows

    def forward(
        self,
        mixed: torch.Tensor,
        scale: torch.Tensor,
        context: RequestContext,
    ) -> MixedState:
        """Return this expert's residual increment in the shared basis."""

        device, dtype = mixed.device, mixed.dtype

        def cast(name: str) -> torch.Tensor:
            return getattr(self, name).to(device=device, dtype=dtype)

        gate_prime = (mixed @ cast("deployed_gate")) / scale
        up_prime = (mixed @ cast("deployed_up")) / scale
        z_prime = F.silu(gate_prime) * up_prime
        increment = (
            z_prime @ cast("deployed_ffn_out")
            + (mixed @ cast("noise_read")) @ cast("deployed_ffn_noise_out")
            + self._refresh_out(mixed, context)
        )
        return MixedState(increment, self.basis_descriptor)

    @property
    def basis_descriptor(self) -> BasisDescriptor:
        return self._basis_descriptor

    @basis_descriptor.setter
    def basis_descriptor(self, descriptor: BasisDescriptor) -> None:
        self._basis_descriptor = descriptor


class ObfuscatedMoE(nn.Module):
    """Expert-permuted MoE layer consuming the mixed state directly."""

    def __init__(
        self,
        *,
        basis: StructuredBasis,
        gamma_ffn: torch.Tensor,
        router_weight_math: torch.Tensor,   # [d, E], math layout
        plain_experts: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        expert_permutation: torch.Tensor,   # [E] int64, P_E
        top_k: int,
        router_mode: RouterMode,
        router_config: Optional[RouterConfig],
        seed: int,
        layer_id: str,
        debug_enabled: bool,
        noise_propagation_gamma: float,
        refresh_mode: str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        hidden_size = basis.signal_dim
        expert_count = len(plain_experts)
        if router_weight_math.shape != (hidden_size, expert_count):
            raise ValueError("router_weight_math must be [d, E] in math layout")
        if expert_permutation.numel() != expert_count:
            raise ValueError("expert permutation must cover every expert")
        intermediate = plain_experts[0][0].shape[1]
        for gate, up, down in plain_experts:
            if gate.shape[1] != intermediate or up.shape != gate.shape:
                raise ValueError("expert gate/up shape mismatch")
            if down.shape != (intermediate, hidden_size):
                raise ValueError("expert down shape mismatch")

        self.basis_descriptor = basis.descriptor
        self.layer_id = layer_id
        self.debug_enabled = bool(debug_enabled)

        # Router: W_tilde = P diag(gamma) W_r, applied without decoding. The
        # expert permutation P_E is NOT folded into the weight: StableRouter
        # already carries it and selects on canonical logits, returning
        # physical positions into the reordered expert list. Folding P_E in
        # as well would double-permute and break the 100% plaintext expert
        # set match (verified against tests/test_router.py semantics).
        projection = basis.signal_projection()
        gamma64 = gamma_ffn.detach().cpu().to(dtype=torch.float64)
        router_math64 = router_weight_math.detach().cpu().to(dtype=torch.float64)
        fused = (projection * gamma64[None, :]) @ router_math64
        self.register_buffer(
            "deployed_router", fused.to(dtype=dtype), persistent=True
        )
        self.register_buffer(
            "deployed_gram_blocks",
            basis.gram_blocks.detach().clone(),
            persistent=True,
        )
        self.register_buffer(
            "deployed_gram_perm",
            basis.perm_out.detach().clone(),
            persistent=True,
        )
        self._rms_scale = _moe_rms_scale(
            hidden_size, self.deployed_gram_blocks, self.deployed_gram_perm
        )

        # Physical experts reordered consistently with r' = r P_E.
        reordered = reorder_experts(plain_experts, expert_permutation)
        fixed_refresh = refresh_mode == "fixed_debug"
        self.experts = nn.ModuleList(
            [
                _ExpertFFN(
                    basis=basis,
                    gamma_ffn=gamma_ffn,
                    gate_weight_math=gate,
                    up_weight_math=up,
                    down_weight_math=down,
                    domain="LAYER_%s_EXPERT_%d"
                    % (layer_id.replace("block-", ""), index),
                    seed=seed,
                    hidden_noise_dim=basis.noise_dim,
                    intermediate=intermediate,
                    hidden_size=hidden_size,
                    fixed_refresh=fixed_refresh,
                    noise_propagation_gamma=noise_propagation_gamma,
                    dtype=dtype,
                )
                for index, (gate, up, down) in enumerate(reordered)
            ]
        )
        self.router = StableRouter(
            permutation=expert_permutation,
            top_k=top_k,
            mode=router_mode,
            config=router_config,
            debug_enabled=debug_enabled,
            layer_id=layer_id,
        )

    def _route(
        self, mixed: torch.Tensor, context: RequestContext
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[RouterDebug]]:
        scale = self._rms_scale(mixed, 1e-5).to(
            device=mixed.device, dtype=mixed.dtype
        )
        logits = (mixed @ self.deployed_router.to(
            device=mixed.device, dtype=mixed.dtype
        )) / scale
        if self.debug_enabled:
            decision, debug = self.router.forward_debug(logits, context)
            return decision.physical_expert_indices, decision.gate_weights, debug
        decision = self.router(logits, context)
        return decision.physical_expert_indices, decision.gate_weights, None

    def forward(
        self,
        mixed: torch.Tensor,
        context: RequestContext,
    ) -> MixedState:
        """Route on the mixed state and return the gate-weighted increment.

        All expert increments share ``self.basis_descriptor``; the weighted
        combination is validated through ``MixedState.add``.
        """

        indices, gate_weights, _ = self._route(mixed, context)
        scale = self._rms_scale(mixed, 1e-5).to(
            device=mixed.device, dtype=mixed.dtype
        )
        # The router returns weights for the *selected* experts only; scatter
        # them onto the physical expert axis, then run every expert (the
        # eager reference form) and combine in the shared basis.
        weight_full = torch.zeros(
            *mixed.shape[:-1],
            len(self.experts),
            device=mixed.device,
            dtype=mixed.dtype,
        )
        for position in range(gate_weights.shape[-1]):
            weight_full.scatter_add_(
                -1,
                indices[..., position : position + 1],
                gate_weights[..., position : position + 1],
            )
        combined: Optional[MixedState] = None
        for expert_position in range(len(self.experts)):
            update = self.experts[expert_position](mixed, scale, context)
            scaled_update = MixedState(
                update.mixed * weight_full[..., expert_position : expert_position + 1],
                update.basis,
            )
            combined = (
                scaled_update
                if combined is None
                else combined.add(scaled_update)
            )
        assert combined is not None
        return combined

    def forward_debug(
        self,
        mixed: torch.Tensor,
        context: RequestContext,
    ) -> Tuple[MixedState, MoEDebug]:
        """Return router diagnostics only when explicitly enabled."""

        if not self.debug_enabled:
            raise PermissionError("MoE debug API is disabled")
        indices, gate_weights, router_debug = self._route(mixed, context)
        assert router_debug is not None
        return self.forward(mixed, context), MoEDebug(
            router_debug=router_debug,
            gate_weights=gate_weights,
            physical_expert_indices=indices,
        )


def _moe_rms_scale(
    signal_dim: int, gram_blocks: torch.Tensor, gram_perm: torch.Tensor
):
    """O(n*b) RMS scale reading the shipped Gram artifacts."""

    def rms_scale(mixed: torch.Tensor, eps: float) -> torch.Tensor:
        count, block = int(gram_blocks.shape[0]), int(gram_blocks.shape[1])
        device = mixed.device
        perm = gram_perm.to(device=device)
        gathered = mixed[..., perm].to(dtype=torch.float32)
        segments = gathered.reshape(*mixed.shape[:-1], count, block)
        gram = gram_blocks.to(device=device, dtype=torch.float32)
        contracted = torch.einsum(
            "...mi,mij,...mj->...", segments, gram, segments
        )
        return torch.sqrt(
            contracted.clamp_min(0.0) / float(signal_dim) + float(eps)
        ).unsqueeze(-1)

    return rms_scale
