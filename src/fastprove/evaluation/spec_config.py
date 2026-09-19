"""Sweep-spec-to-configuration resolution."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

from ..config import AttentionConfig, PrototypeConfig
from ..layers.attention import AttentionMode
from .sweep import SweepSpec


def _config_for_spec(
    base: PrototypeConfig,
    spec: SweepSpec,
    evaluation_override: Optional[Dict[str, Any]],
    runtime_override: Optional[Dict[str, Any]] = None,
) -> PrototypeConfig:
    alpha = base.attention.alpha if spec.alpha is None else spec.alpha
    top_k = (
        base.attention.preserve_top_k
        if spec.preserve_top_k is None
        else spec.preserve_top_k
    )
    attention = AttentionConfig(
        mode=AttentionMode(spec.mode),
        tau_max=spec.tau_max,
        tau_error=spec.tau_error,
        alpha=alpha,
        preserve_top_k=top_k,
    )
    evaluation = base.evaluation
    if evaluation_override:
        allowed = set(evaluation.__dataclass_fields__)
        unknown = sorted(set(evaluation_override).difference(allowed))
        if unknown:
            raise ValueError(
                "evaluation override contains unknown fields: "
                + ", ".join(unknown)
            )
        evaluation = replace(evaluation, **evaluation_override)
    runtime = base.runtime
    if runtime_override:
        runtime = replace(runtime, **runtime_override)
    return replace(
        base,
        attention=attention,
        evaluation=evaluation,
        runtime=runtime,
    )
