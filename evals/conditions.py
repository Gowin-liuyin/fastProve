"""Precision × mode condition matrix (protocol §3).

Conditions
----------
F0  FP32 plaintext
F1  FP32 structural-only (full obfuscation graph, noise injection zeroed)
F2  FP32 full-noise (complete chain noise)
P0  BF16 plaintext
P1  BF16 structural-only
P2  BF16 full nominal (primary production result)
P3  BF16 full stress (stability boundary probe)

Degradation quantities
----------------------
Δ_struct^FP32 = Q(F0) − Q(F1)
Δ_noise^FP32  = Q(F1) − Q(F2)
Δ_struct^BF16 = Q(P0) − Q(P1)
Δ_noise^BF16  = Q(P1) − Q(P2)
Δ_prod        = Q(P0) − Q(P2)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import yaml

from fastprove.config import (
    AttentionConfig,
    EvaluationConfig,
    ModelConfig,
    ObfuscationConfig,
    PrototypeConfig,
    RuntimeConfig,
)
from fastprove.layers.attention import AttentionMode


class Precision(str, Enum):
    """Numeric activation precision."""

    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"


class ObfuscationMode(str, Enum):
    """Evaluation-mode role inside the condition matrix.

    ``structural`` keeps the full obfuscation computation graph (augmented
    dim D+R, mixed weights, Q/K transform, value mixing, KV-cache format,
    FFN permutation/scaling, LM-head vocab permutation when present) and
    only zeros noise injection terms (e_0, C_ℓ, ξ_ℓ). It is *not* a
    near-plaintext model (protocol §2 problem 2).
    """

    PLAINTEXT = "plaintext"
    STRUCTURAL = "structural"
    FULL = "full"


class ConditionId(str, Enum):
    """Stable condition identifiers from protocol §3.1."""

    F0 = "F0"
    F1 = "F1"
    F2 = "F2"
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


@dataclass(frozen=True)
class ConditionSpec:
    """One cell of the precision × mode matrix."""

    condition_id: ConditionId
    precision: Precision
    mode: ObfuscationMode
    description: str
    # Full-noise / stress knobs (only meaningful for FULL mode).
    stress: bool = False
    # Optional overrides for ablation (§5).
    hidden_noise_dim: Optional[int] = None
    value_noise_dim_per_head: Optional[int] = None
    ffn_noise_dim: Optional[int] = None
    initial_noise_scale: Optional[float] = None
    refresh_noise_scale: Optional[float] = None
    noise_propagation_gamma: Optional[float] = None
    max_condition_number: Optional[float] = None
    sequence_length: Optional[int] = None
    attention_impl: str = "reference"
    use_kv_cache: bool = True
    model_structure: str = "dense_gqa"

    @property
    def torch_dtype(self) -> torch.dtype:
        """Map precision to a torch activation dtype."""

        return {
            Precision.FP32: torch.float32,
            Precision.BF16: torch.bfloat16,
            Precision.FP16: torch.float16,
        }[self.precision]

    @property
    def activation_dtype_name(self) -> str:
        """Runtime config string for PrototypeConfig."""

        return {
            Precision.FP32: "float32",
            Precision.BF16: "bfloat16",
            Precision.FP16: "float16",
        }[self.precision]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe serialisation."""

        return {
            "condition_id": self.condition_id.value,
            "precision": self.precision.value,
            "mode": self.mode.value,
            "description": self.description,
            "stress": self.stress,
            "hidden_noise_dim": self.hidden_noise_dim,
            "value_noise_dim_per_head": self.value_noise_dim_per_head,
            "ffn_noise_dim": self.ffn_noise_dim,
            "initial_noise_scale": self.initial_noise_scale,
            "refresh_noise_scale": self.refresh_noise_scale,
            "noise_propagation_gamma": self.noise_propagation_gamma,
            "max_condition_number": self.max_condition_number,
            "sequence_length": self.sequence_length,
            "attention_impl": self.attention_impl,
            "use_kv_cache": self.use_kv_cache,
            "model_structure": self.model_structure,
        }


# Canonical matrix from protocol §3.1.
CANONICAL_CONDITIONS: Dict[ConditionId, ConditionSpec] = {
    ConditionId.F0: ConditionSpec(
        condition_id=ConditionId.F0,
        precision=Precision.FP32,
        mode=ObfuscationMode.PLAINTEXT,
        description="FP32 plaintext high-precision reference",
    ),
    ConditionId.F1: ConditionSpec(
        condition_id=ConditionId.F1,
        precision=Precision.FP32,
        mode=ObfuscationMode.STRUCTURAL,
        description=(
            "FP32 structural-only: full obfuscation graph, noise injection zeroed"
        ),
    ),
    ConditionId.F2: ConditionSpec(
        condition_id=ConditionId.F2,
        precision=Precision.FP32,
        mode=ObfuscationMode.FULL,
        description="FP32 full chained noise (signal/noise decoupling check)",
        stress=False,
    ),
    ConditionId.P0: ConditionSpec(
        condition_id=ConditionId.P0,
        precision=Precision.BF16,
        mode=ObfuscationMode.PLAINTEXT,
        description="BF16 plaintext production baseline",
    ),
    ConditionId.P1: ConditionSpec(
        condition_id=ConditionId.P1,
        precision=Precision.BF16,
        mode=ObfuscationMode.STRUCTURAL,
        description=(
            "BF16 structural-only: measures transform-induced numerical amplification"
        ),
    ),
    ConditionId.P2: ConditionSpec(
        condition_id=ConditionId.P2,
        precision=Precision.BF16,
        mode=ObfuscationMode.FULL,
        description="BF16 full nominal parameters (primary production result)",
        stress=False,
    ),
    ConditionId.P3: ConditionSpec(
        condition_id=ConditionId.P3,
        precision=Precision.BF16,
        mode=ObfuscationMode.FULL,
        description="BF16 full stress parameters (numerical stability boundary)",
        stress=True,
        # Stress defaults: larger noise dims / looser conditioning.
        hidden_noise_dim=16,
        value_noise_dim_per_head=4,
        max_condition_number=50.0,
        noise_propagation_gamma=0.9,
        initial_noise_scale=2.0,
        refresh_noise_scale=2.0,
    ),
}


def get_condition(condition_id: str | ConditionId) -> ConditionSpec:
    """Look up a canonical condition by id string or enum."""

    if isinstance(condition_id, ConditionId):
        key = condition_id
    else:
        key = ConditionId(str(condition_id))
    return CANONICAL_CONDITIONS[key]


def all_condition_ids() -> tuple[ConditionId, ...]:
    """Return condition ids in report order."""

    return (
        ConditionId.F0,
        ConditionId.F1,
        ConditionId.F2,
        ConditionId.P0,
        ConditionId.P1,
        ConditionId.P2,
        ConditionId.P3,
    )


@dataclass(frozen=True)
class DegradationReport:
    """Protocol §3.2 degradation quantities.

    For higher-is-better metrics (accuracy), values are percentage-point drops
    plain − obfuscated. For lower-is-better metrics (PPL), callers should use
    relative increase helpers instead; this dataclass stores raw Q differences
    Q(a) − Q(b) and documents units in the report tables.
    """

    delta_struct_fp32: Optional[float]
    delta_noise_fp32: Optional[float]
    delta_struct_bf16: Optional[float]
    delta_noise_bf16: Optional[float]
    delta_prod: Optional[float]
    metric_name: str
    unit: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "unit": self.unit,
            "delta_struct_fp32": self.delta_struct_fp32,
            "delta_noise_fp32": self.delta_noise_fp32,
            "delta_struct_bf16": self.delta_struct_bf16,
            "delta_noise_bf16": self.delta_noise_bf16,
            "delta_prod": self.delta_prod,
        }


def compute_degradations(
    quality_by_condition: Mapping[str, float],
    *,
    metric_name: str,
    unit: str,
    higher_is_better: bool = True,
) -> DegradationReport:
    """Compute Δ quantities from a {condition_id: Q} map.

    Parameters
    ----------
    quality_by_condition:
        Map of condition id → quality score.
    higher_is_better:
        If True, Δ = Q(plain-ish) − Q(obf-ish) so positive means loss.
        If False (PPL/NLL), still report Q(a) − Q(b); relative increase is
        computed separately by the report layer.
    """

    def _get(cid: ConditionId) -> Optional[float]:
        value = quality_by_condition.get(cid.value)
        if value is None:
            return None
        return float(value)

    def _diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        # For higher-is-better: plain − obf (positive = drop).
        # For lower-is-better: still plain − obf (negative = improvement);
        # report layer converts to relative increase for PPL.
        return a - b if higher_is_better else b - a

    return DegradationReport(
        delta_struct_fp32=_diff(_get(ConditionId.F0), _get(ConditionId.F1)),
        delta_noise_fp32=_diff(_get(ConditionId.F1), _get(ConditionId.F2)),
        delta_struct_bf16=_diff(_get(ConditionId.P0), _get(ConditionId.P1)),
        delta_noise_bf16=_diff(_get(ConditionId.P1), _get(ConditionId.P2)),
        delta_prod=_diff(_get(ConditionId.P0), _get(ConditionId.P2)),
        metric_name=metric_name,
        unit=unit,
    )


def load_condition_config(path: Path) -> Dict[str, Any]:
    """Load a YAML condition config (F0–P3 files under evals/configs)."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("condition config root must be a mapping")
    return raw


def condition_from_config(raw: Mapping[str, Any]) -> ConditionSpec:
    """Build a ConditionSpec from a loaded YAML mapping."""

    base_id = ConditionId(str(raw.get("condition_id", raw.get("id", "P2"))))
    base = CANONICAL_CONDITIONS[base_id]
    overrides = raw.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError("overrides must be a mapping")
    precision = Precision(str(raw.get("precision", base.precision.value)))
    mode = ObfuscationMode(str(raw.get("mode", base.mode.value)))
    return replace(
        base,
        condition_id=base_id,
        precision=precision,
        mode=mode,
        description=str(raw.get("description", base.description)),
        stress=bool(raw.get("stress", base.stress)),
        hidden_noise_dim=overrides.get(
            "hidden_noise_dim", raw.get("hidden_noise_dim", base.hidden_noise_dim)
        ),
        value_noise_dim_per_head=overrides.get(
            "value_noise_dim_per_head",
            raw.get("value_noise_dim_per_head", base.value_noise_dim_per_head),
        ),
        ffn_noise_dim=overrides.get(
            "ffn_noise_dim", raw.get("ffn_noise_dim", base.ffn_noise_dim)
        ),
        initial_noise_scale=overrides.get(
            "initial_noise_scale",
            raw.get("initial_noise_scale", base.initial_noise_scale),
        ),
        refresh_noise_scale=overrides.get(
            "refresh_noise_scale",
            raw.get("refresh_noise_scale", base.refresh_noise_scale),
        ),
        noise_propagation_gamma=overrides.get(
            "noise_propagation_gamma",
            raw.get("noise_propagation_gamma", base.noise_propagation_gamma),
        ),
        max_condition_number=overrides.get(
            "max_condition_number",
            raw.get("max_condition_number", base.max_condition_number),
        ),
        sequence_length=overrides.get(
            "sequence_length", raw.get("sequence_length", base.sequence_length)
        ),
        attention_impl=str(
            overrides.get(
                "attention_impl", raw.get("attention_impl", base.attention_impl)
            )
        ),
        use_kv_cache=bool(
            overrides.get("use_kv_cache", raw.get("use_kv_cache", base.use_kv_cache))
        ),
        model_structure=str(
            overrides.get(
                "model_structure",
                raw.get("model_structure", base.model_structure),
            )
        ),
    )


def resolve_prototype_config(
    base: PrototypeConfig,
    condition: ConditionSpec,
    *,
    seed: Optional[int] = None,
    request_id: Optional[str] = None,
    debug_enabled: bool = True,
) -> PrototypeConfig:
    """Apply a condition to a base PrototypeConfig (dtype, noise dims, etc.)."""

    obfuscation = base.obfuscation
    if condition.hidden_noise_dim is not None:
        obfuscation = replace(
            obfuscation, hidden_noise_dim=int(condition.hidden_noise_dim)
        )
    if condition.value_noise_dim_per_head is not None:
        obfuscation = replace(
            obfuscation,
            value_noise_dim_per_head=int(condition.value_noise_dim_per_head),
        )
    if condition.noise_propagation_gamma is not None:
        obfuscation = replace(
            obfuscation,
            noise_propagation_gamma=float(condition.noise_propagation_gamma),
        )
    if condition.max_condition_number is not None:
        obfuscation = replace(
            obfuscation,
            max_condition_number=float(condition.max_condition_number),
        )

    # Structural and full modes use exact attention for hard-correctness layers;
    # approximate Softmax noise is a separate ablation (not F1/P1 structural).
    if condition.mode == ObfuscationMode.PLAINTEXT:
        attention = AttentionConfig(
            mode=AttentionMode.PLAINTEXT,
            tau_max=0.0,
            tau_error=0.0,
            alpha=0.8,
            preserve_top_k=4,
        )
    else:
        attention = AttentionConfig(
            mode=AttentionMode.EXACT,
            tau_max=0.0,
            tau_error=0.0,
            alpha=0.8,
            preserve_top_k=base.attention.preserve_top_k,
        )

    activation = condition.activation_dtype_name
    # PrototypeConfig currently allows float32/bfloat16 only.
    if activation == "float16":
        activation = "float32"  # fallback documented; FP16 via cast at runtime

    evaluation = base.evaluation
    if condition.sequence_length is not None:
        evaluation = replace(
            evaluation, sequence_length=int(condition.sequence_length)
        )

    runtime = RuntimeConfig(
        seed=int(seed if seed is not None else base.runtime.seed),
        request_id=str(
            request_id if request_id is not None else base.runtime.request_id
        ),
        device=base.runtime.device,
        activation_dtype=activation if activation in ("float32", "bfloat16") else "float32",
        debug_enabled=bool(debug_enabled),
    )
    return PrototypeConfig(
        model=base.model,
        obfuscation=obfuscation,
        attention=attention,
        runtime=runtime,
        evaluation=evaluation,
    )


def default_base_prototype_config() -> PrototypeConfig:
    """Tiny deterministic model used when no base YAML is provided."""

    return PrototypeConfig(
        model=ModelConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_sequence_length=64,
            rms_epsilon=1e-5,
            rope_theta=10000.0,
        ),
        obfuscation=ObfuscationConfig(
            hidden_noise_dim=8,
            value_noise_dim_per_head=2,
            max_condition_number=10.0,
            noise_propagation_gamma=0.5,
            refresh_mode="fixed_debug",
            basis_block_size=8,
        ),
        attention=AttentionConfig(
            mode=AttentionMode.EXACT,
            tau_max=0.0,
            tau_error=0.0,
            alpha=0.8,
            preserve_top_k=4,
        ),
        runtime=RuntimeConfig(
            seed=20260731,
            request_id="eval-suite",
            device="cpu",
            activation_dtype="float32",
            debug_enabled=True,
        ),
        evaluation=EvaluationConfig(
            batch_size=2,
            sequence_length=16,
            sample_count=8,
            generation_tokens=4,
            warmup_runs=0,
            timed_runs=1,
        ),
    )
