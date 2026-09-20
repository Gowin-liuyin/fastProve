"""Sweep-spec-to-configuration resolution.

Base-sweep contract (P8/P9 decision, option 丙 base): every sweep point runs
with ``rho_noise_bound = 0.0`` so that each accuracy delta is attributable to
the logit-noise knobs alone. The deployment-fold RMS-scale noise
(``rho_noise_bound = 1e-5``) belongs to separate replica runs and must be
requested explicitly through ``obfuscation_override``; it is never inherited
silently from the base config. ``refresh_mode`` is intentionally left as-is
(``per_request`` in ``configs/tiny_approx.yaml``); measurements show it does
not degrade exact-mode accuracy, and the debug-only ``fixed_debug`` mode must
not be used as a privacy path.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

from ..config import AttentionConfig, PrototypeConfig
from ..layers.attention import AttentionMode
from .sweep import SweepSpec

#: Base-sweep value for the bounded RMS-scale noise. Replicas pass an explicit
#: ``obfuscation_override`` instead of changing this constant.
BASE_SWEEP_RHO_NOISE_BOUND = 0.0


def _config_for_spec(
    base: PrototypeConfig,
    spec: SweepSpec,
    evaluation_override: Optional[Dict[str, Any]],
    runtime_override: Optional[Dict[str, Any]] = None,
    obfuscation_override: Optional[Dict[str, Any]] = None,
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
    obfuscation = replace(
        base.obfuscation, rho_noise_bound=BASE_SWEEP_RHO_NOISE_BOUND
    )
    if obfuscation_override:
        allowed = set(obfuscation.__dataclass_fields__)
        unknown = sorted(set(obfuscation_override).difference(allowed))
        if unknown:
            raise ValueError(
                "obfuscation override contains unknown fields: "
                + ", ".join(unknown)
            )
        obfuscation = replace(obfuscation, **obfuscation_override)
    return replace(
        base,
        attention=attention,
        evaluation=evaluation,
        runtime=runtime,
        obfuscation=obfuscation,
    )
