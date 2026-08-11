"""Converted affine layer with chained auxiliary-noise refresh."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..conversion import convert_affine_chain
from ..seed import RequestContext
from ..state import MixedState
from ..transforms import BasisDescriptor, BasisTransform


class ChainLinear(nn.Module):
    """Apply an offline-converted augmented affine map.

    The conversion implements row-vector mathematics

    ``y = h @ W + b`` and ``e' = h @ C + e @ G + xi``.

    No inverse or solve occurs in :meth:`forward`.
    """

    VALID_REFRESH_MODES = ("fixed_debug", "per_request")

    def __init__(
        self,
        *,
        weight_pt: torch.Tensor,
        bias_mixed: torch.Tensor,
        in_transform: BasisTransform,
        out_transform: BasisTransform,
        refresh_mode: str,
        layer_id: str,
        refresh_scale: float,
    ) -> None:
        super().__init__()
        if refresh_mode not in self.VALID_REFRESH_MODES:
            raise ValueError("unsupported refresh_mode")
        if not math.isfinite(refresh_scale) or refresh_scale < 0:
            raise ValueError("refresh_scale must be finite and non-negative")
        if weight_pt.shape != (
            out_transform.total_dim,
            in_transform.total_dim,
        ):
            raise ValueError("converted weight shape mismatch")
        if bias_mixed.shape != (out_transform.total_dim,):
            raise ValueError("converted bias shape mismatch")
        if not weight_pt.is_floating_point() or not bias_mixed.is_floating_point():
            raise ValueError("deployed weight and bias must use real floating dtypes")
        if not torch.isfinite(weight_pt).all() or not torch.isfinite(
            bias_mixed
        ).all():
            raise ValueError("deployed weight and bias must be finite")
        self.register_buffer("weight_pt", weight_pt.detach().clone())
        self.register_buffer("bias_mixed", bias_mixed.detach().clone())
        fused_out_matrix = out_transform.matrix.detach().to("cpu").clone()

        def fused_refresh_mix(
            augmented: torch.Tensor, reference: torch.Tensor
        ) -> torch.Tensor:
            matrix = fused_out_matrix.to(
                device=reference.device, dtype=torch.float32
            )
            return (
                augmented.to(device=reference.device, dtype=torch.float32)
                @ matrix
            ).to(dtype=reference.dtype)

        self._fused_refresh_mix = fused_refresh_mix
        self.in_basis = in_transform.descriptor
        self.out_basis = out_transform.descriptor
        self.refresh_mode = refresh_mode
        self.layer_id = str(layer_id)
        self.refresh_scale = float(refresh_scale)

    @staticmethod
    def _basis_payload(basis: BasisDescriptor) -> dict:
        return {
            "signal_dim": basis.signal_dim,
            "noise_dim": basis.noise_dim,
            "condition_number": basis.condition_number,
            "fingerprint": basis.fingerprint,
        }

    def get_extra_state(self) -> dict:
        """Serialize non-tensor basis metadata with the converted buffers."""

        return {
            "in_basis": self._basis_payload(self.in_basis),
            "out_basis": self._basis_payload(self.out_basis),
            "refresh_mode": self.refresh_mode,
            "layer_id": self.layer_id,
            "refresh_scale": self.refresh_scale,
        }

    def set_extra_state(self, state: dict) -> None:
        """Reject weights converted for a different basis/configuration."""

        expected = self.get_extra_state()
        if state != expected:
            raise RuntimeError(
                "basis metadata/configuration does not match target ChainLinear"
            )

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
    ) -> None:
        extra_key = prefix + "_extra_state"
        serialized = state_dict.get(extra_key)
        if serialized is None:
            error_msgs.append(
                "%smissing basis metadata for target ChainLinear" % prefix
            )
            return
        if serialized != self.get_extra_state():
            error_msgs.append(
                "%sbasis metadata/configuration does not match target "
                "ChainLinear" % prefix
            )
            return
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @classmethod
    def from_math(
        cls,
        *,
        weight_math: torch.Tensor,
        bias: torch.Tensor,
        coupling: torch.Tensor,
        propagator: torch.Tensor,
        in_transform: BasisTransform,
        out_transform: BasisTransform,
        refresh_mode: str,
        fixed_refresh: Optional[torch.Tensor],
        layer_id: str,
        refresh_scale: float = 1.0,
    ) -> "ChainLinear":
        """Convert mathematical-layout parameters offline."""

        if refresh_mode == "fixed_debug" and fixed_refresh is None:
            raise ValueError("fixed_debug requires fixed_refresh")
        if refresh_mode == "per_request" and fixed_refresh is not None:
            raise ValueError("per_request refresh must not be fixed")
        converted = convert_affine_chain(
            weight_math=weight_math,
            bias=bias,
            coupling=coupling,
            propagator=propagator,
            in_transform=in_transform,
            out_transform=out_transform,
            fixed_refresh=fixed_refresh,
        )
        return cls(
            weight_pt=converted.weight_pt,
            bias_mixed=converted.bias_mixed,
            in_transform=in_transform,
            out_transform=out_transform,
            refresh_mode=refresh_mode,
            layer_id=layer_id,
            refresh_scale=refresh_scale,
        )

    def _request_refresh(
        self, context: RequestContext, reference: torch.Tensor
    ) -> torch.Tensor:
        generator = context.generator_for(
            "chain-linear", self.layer_id, "refresh"
        )
        refresh = torch.randn(
            self.out_basis.noise_dim,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        refresh = refresh * self.refresh_scale
        augmented = torch.cat(
            (
                torch.zeros(
                    self.out_basis.signal_dim, dtype=torch.float32
                ),
                refresh,
            )
        )
        return self._fused_refresh_mix(augmented, reference)

    def forward(
        self,
        state: MixedState,
        request_context: Optional[RequestContext] = None,
    ) -> MixedState:
        """Propagate a mixed state without decoding it."""

        if (
            state.basis.fingerprint != self.in_basis.fingerprint
            or state.basis.signal_dim != self.in_basis.signal_dim
            or state.basis.noise_dim != self.in_basis.noise_dim
        ):
            raise ValueError("input state basis does not match converted layer")
        if self.refresh_mode == "per_request" and request_context is None:
            raise ValueError("per_request refresh requires request_context")
        weight = self.weight_pt.to(
            device=state.mixed.device, dtype=state.mixed.dtype
        )
        bias = self.bias_mixed.to(
            device=state.mixed.device, dtype=state.mixed.dtype
        )
        output = F.linear(state.mixed, weight, bias)
        if self.refresh_mode == "per_request":
            assert request_context is not None
            output = output + self._request_refresh(request_context, output)
        return MixedState(output, self.out_basis)
