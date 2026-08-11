"""Stable expert permutation and margin-bounded MoE router reference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Sequence, Tuple, TypeVar, List

import torch
import torch.nn as nn

from ..seed import RequestContext

Expert = TypeVar("Expert")


class RouterMode(str, Enum):
    """Supported router modes."""

    EXACT = "exact"
    MARGIN_BOUNDED = "margin_bounded"


@dataclass(frozen=True)
class RouterConfig:
    """Margin-bounded router-noise configuration."""

    tau_max: float
    tau_error: float
    alpha: float
    noisy_gate_weights: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.tau_max) or not math.isfinite(self.tau_error):
            raise ValueError("router tau bounds must be finite")
        if self.tau_max < 0 or self.tau_error < 0:
            raise ValueError("router tau bounds must be non-negative")
        if not math.isfinite(self.alpha) or not 0.0 < self.alpha < 1.0:
            raise ValueError("router alpha must be strictly between zero and one")


@dataclass(frozen=True)
class RouterDecision:
    """Production router result without raw router logits."""

    physical_expert_indices: torch.Tensor
    gate_weights: torch.Tensor


@dataclass(frozen=True)
class RouterDebug:
    """Debug-only router diagnostics."""

    permuted_clean_logits: torch.Tensor
    noisy_logits: torch.Tensor
    noise: torch.Tensor
    tau: torch.Tensor
    margin: torch.Tensor


def reorder_experts(
    experts: Sequence[Expert], permutation: torch.Tensor
) -> List[Expert]:
    """Reorder physical experts consistently with ``r' = r P_E``."""

    if permutation.ndim != 1 or permutation.numel() != len(experts):
        raise ValueError("expert permutation shape mismatch")
    ordered = permutation.detach().cpu().tolist()
    if sorted(int(item) for item in ordered) != list(range(len(experts))):
        raise ValueError("expert permutation is invalid")
    return [experts[int(index)] for index in ordered]


class StableRouter(nn.Module):
    """Expert-permuted router with canonical stable tie-breaking."""

    def __init__(
        self,
        *,
        permutation: torch.Tensor,
        top_k: int,
        mode: RouterMode,
        config: Optional[RouterConfig],
        debug_enabled: bool,
        layer_id: str,
    ) -> None:
        super().__init__()
        if permutation.ndim != 1:
            raise ValueError("permutation must be a vector")
        expert_count = permutation.numel()
        if not torch.equal(
            torch.sort(permutation.cpu()).values, torch.arange(expert_count)
        ):
            raise ValueError("permutation must contain every expert")
        if not 1 <= top_k < expert_count:
            raise ValueError("top_k must be in [1, expert_count)")
        self.mode = RouterMode(mode)
        if self.mode == RouterMode.MARGIN_BOUNDED and config is None:
            raise ValueError("margin-bounded mode requires config")
        if self.mode == RouterMode.EXACT and config is not None:
            raise ValueError("exact mode must not receive noise config")
        self.register_buffer("permutation", permutation.long().clone())
        self.top_k = int(top_k)
        self.config = config
        self.debug_enabled = bool(debug_enabled)
        self.layer_id = str(layer_id)

    def _select_physical(self, scores: torch.Tensor) -> torch.Tensor:
        canonical_order = torch.argsort(self.permutation, stable=True)
        ordered_scores = scores[..., canonical_order]
        order = torch.argsort(
            ordered_scores, dim=-1, descending=True, stable=True
        )
        selected_ordered = order[..., : self.top_k]
        return canonical_order[selected_ordered]

    def _margin_and_tau(
        self, canonical_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sorted_values = torch.sort(
            canonical_logits.float(),
            dim=-1,
            descending=True,
            stable=True,
        ).values
        margin = (
            sorted_values[..., self.top_k - 1 : self.top_k]
            - sorted_values[..., self.top_k : self.top_k + 1]
        )
        assert self.config is not None
        bounded = torch.minimum(
            torch.full_like(margin, self.config.tau_max),
            torch.full_like(margin, self.config.tau_error),
        )
        bounded = torch.minimum(
            bounded, self.config.alpha * margin.clamp_min(0) / 2.0
        )
        tau = torch.where(margin > 0, bounded, torch.zeros_like(bounded))
        return margin.clamp_min(0), tau

    def _noise(
        self,
        canonical_logits: torch.Tensor,
        tau: torch.Tensor,
        context: RequestContext,
    ) -> torch.Tensor:
        flat = canonical_logits.reshape(-1, canonical_logits.shape[-1])
        raw = torch.zeros_like(flat, dtype=torch.float32, device="cpu")
        for row in range(flat.shape[0]):
            for canonical_expert in range(flat.shape[1]):
                seed = context.seed_for(
                    "router-noise",
                    self.layer_id,
                    row,
                    canonical_expert,
                )
                value = 2.0 * (
                    (seed & ((1 << 53) - 1)) / float(1 << 53)
                ) - 1.0
                raw[row, canonical_expert] = value
        raw = raw.to(device=canonical_logits.device)
        centered = raw - raw.mean(dim=-1, keepdim=True)
        maximum = centered.abs().amax(dim=-1, keepdim=True)
        unit = centered / maximum.clamp_min(torch.finfo(torch.float32).tiny)
        # Keep the externally configured Python-float bound strict after
        # float32 materialization (e.g. 0.001 rounds upward by one ulp).
        tau_fp32 = tau.to(dtype=torch.float32)
        safe_tau = torch.nextafter(tau_fp32, torch.zeros_like(tau_fp32))
        canonical_noise = unit.reshape_as(canonical_logits) * safe_tau
        return canonical_noise[..., self.permutation]

    def _run(
        self, logits: torch.Tensor, context: RequestContext
    ) -> Tuple[RouterDecision, RouterDebug]:
        if logits.ndim < 2 or logits.shape[-1] != self.permutation.numel():
            raise ValueError("router logits expert dimension mismatch")
        canonical = logits.float()
        permuted_clean = canonical[..., self.permutation]
        noise = torch.zeros_like(permuted_clean)
        margin = torch.zeros_like(permuted_clean[..., :1])
        tau = torch.zeros_like(margin)
        if self.mode == RouterMode.MARGIN_BOUNDED:
            margin, tau = self._margin_and_tau(canonical)
            noise = self._noise(canonical, tau, context)
        noisy = permuted_clean + noise
        selected = self._select_physical(noisy)
        assert self.config is not None or self.mode == RouterMode.EXACT
        use_noisy_weights = (
            self.mode == RouterMode.MARGIN_BOUNDED
            and self.config is not None
            and self.config.noisy_gate_weights
        )
        weight_source = noisy if use_noisy_weights else permuted_clean
        selected_logits = torch.gather(weight_source, -1, selected)
        gate_weights = torch.softmax(selected_logits.float(), dim=-1)
        decision = RouterDecision(
            physical_expert_indices=selected,
            gate_weights=gate_weights,
        )
        debug = RouterDebug(
            permuted_clean_logits=permuted_clean,
            noisy_logits=noisy,
            noise=noise,
            tau=tau,
            margin=margin,
        )
        return decision, debug

    def forward(
        self, logits: torch.Tensor, context: RequestContext
    ) -> RouterDecision:
        """Return expert indices and gate weights without raw logits."""

        decision, _ = self._run(logits, context)
        return decision

    def forward_debug(
        self, logits: torch.Tensor, context: RequestContext
    ) -> Tuple[RouterDecision, RouterDebug]:
        """Return router diagnostics only when explicitly enabled."""

        if not self.debug_enabled:
            raise PermissionError("router debug API is disabled")
        return self._run(logits, context)
