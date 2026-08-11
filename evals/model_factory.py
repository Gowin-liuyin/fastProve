"""Build plaintext / structural / full-noise models for a condition.

Structural mode (protocol §2 problem 2) **retains** the full obfuscation
computation graph and only zeros noise-injection terms (e_0, C_ℓ, ξ_ℓ):

* augmented dimension D+R and mixing matrices M_ℓ
* Q/K orthogonal transforms after RoPE
* mixed Value layout and KV-cache format
* FFN permutation / scaling
* LM-head vocab permutation when present

It does **not** create a near-plaintext model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from fastprove.config import PrototypeConfig
from fastprove.layers.attention import ApproximationConfig, AttentionMode
from fastprove.models.obfuscated import (
    ConvertedObfuscatedLM,
    ObfuscatedTinyCausalLM,
)
from fastprove.models.plain import PlainTinyCausalLM
from fastprove.seed import RequestContext
from fastprove.pretrained.qwen2 import (
    Qwen2Artifact,
    load_qwen2_artifact,
    load_qwen2_plain,
)

from .conditions import ConditionSpec, ObfuscationMode, resolve_prototype_config
from .keys import MasterKey


# Buffer name substrings that implement noise *injection* (C_ℓ, ξ_ℓ, e_0 sources).
# Propagators G and mixing transforms M are intentionally kept.
_NOISE_INJECTION_BUFFER_MARKERS: Tuple[str, ...] = (
    "noise_coupling",  # C_ℓ signal→noise
    "fixed_refresh",  # ξ_ℓ
    "initial_noise_coupling",  # e_0 via h C_0
    "value_signal_coupling",  # C on value path
    "value_fixed_refresh",
    "attention_aux_to_hidden",  # auxiliary re-injection into hidden noise
)


@dataclass
class EvalModels:
    """Paired models for one (condition, key) evaluation cell."""

    plain: PlainTinyCausalLM
    obfuscated: Optional[ObfuscatedTinyCausalLM]
    client: Any
    config: PrototypeConfig
    condition: ConditionSpec
    key: Optional[MasterKey]
    conversion_time_seconds: float
    device: torch.device
    dtype: torch.dtype
    structural_noise_zeroed: bool
    vocab_permutation: Optional[torch.Tensor]  # Π_voc, or None = identity
    notes: List[str]
    model_manifest: Optional[Dict[str, Any]] = None
    obfuscation_manifest: Optional[Dict[str, Any]] = None

    def request_context(self, request_id: str = "eval") -> RequestContext:
        seed = self.config.runtime.seed
        if self.key is not None:
            seed = self.key.request_seed(request_id)
        return RequestContext(seed, request_id)


def _dtype_from_config(config: PrototypeConfig) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }.get(config.runtime.activation_dtype, torch.float32)


def _device_from_config(config: PrototypeConfig) -> torch.device:
    name = config.runtime.device
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent fallback")
    if name == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS requested but unavailable; no silent fallback")
    if name == "mps" and os.environ.get(
        "PYTORCH_ENABLE_MPS_FALLBACK", ""
    ).lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            "PYTORCH_ENABLE_MPS_FALLBACK is enabled; refusing a run that may "
            "silently execute operators on CPU"
        )
    return torch.device(name)


def zero_noise_injection_(module: nn.Module) -> List[str]:
    """In-place zero of noise-injection buffers; keep structure (protocol §2).

    Returns the list of zeroed buffer names for auditability.
    """

    zeroed: List[str] = []
    for name, buf in module.named_buffers():
        short = name.rsplit(".", 1)[-1]
        if any(marker in short for marker in _NOISE_INJECTION_BUFFER_MARKERS):
            buf.zero_()
            zeroed.append(name)
        # Also catch full-path markers.
        elif any(marker in name for marker in _NOISE_INJECTION_BUFFER_MARKERS):
            if name not in zeroed:
                buf.zero_()
                zeroed.append(name)
    # ``per_request`` refreshes are generated dynamically and are not buffers;
    # disable those terms explicitly for the structural-only control cell.
    dynamic_disabled = 0
    for child in module.modules():
        if hasattr(child, "noise_injection_enabled"):
            setattr(child, "noise_injection_enabled", False)
            dynamic_disabled += 1
    if dynamic_disabled:
        zeroed.append("<dynamic-refresh:%d>" % dynamic_disabled)
    return zeroed


def apply_noise_scale_(
    module: nn.Module,
    scale: float = 1.0,
    *,
    initial_scale: Optional[float] = None,
    refresh_scale: Optional[float] = None,
) -> None:
    """Scale initial and refresh noise independently for an ablation."""

    if initial_scale is None:
        initial_scale = scale
    if refresh_scale is None:
        refresh_scale = scale
    if initial_scale == 1.0 and refresh_scale == 1.0:
        return
    for child in module.modules():
        if hasattr(child, "initial_refresh_noise_scale"):
            setattr(child, "initial_refresh_noise_scale", float(initial_scale))
        if hasattr(child, "refresh_noise_scale"):
            setattr(child, "refresh_noise_scale", float(refresh_scale))
    for name, buf in module.named_buffers():
        short = name.rsplit(".", 1)[-1]
        if "initial_" in name:
            selected_scale = initial_scale
        else:
            selected_scale = refresh_scale
        if any(
            marker in short
            for marker in (
                "noise_coupling",
                "fixed_refresh",
                "initial_noise_coupling",
                "value_signal_coupling",
                "value_fixed_refresh",
            )
        ):
            buf.mul_(selected_scale)


def _obfuscation_manifest(
    module: Optional[ObfuscatedTinyCausalLM],
) -> Optional[Dict[str, Any]]:
    """Record public conditioning metadata without serializing basis material."""

    if module is None:
        return None
    hidden = module.hidden_basis
    return {
        "max_condition_number": float(module.obfuscation.max_condition_number),
        "hidden_basis": {
            "signal_dim": int(hidden.signal_dim),
            "noise_dim": int(hidden.noise_dim),
            "condition_number": float(hidden.condition_number),
            "fingerprint": hidden.fingerprint,
        },
        "value_bases": [
            {
                "condition_number": float(condition_number),
                "fingerprint": fingerprint,
            }
            for layer_conditions, layer_fingerprints in zip(
                module.value_basis_condition_numbers,
                module.value_basis_fingerprints,
            )
            for condition_number, fingerprint in zip(
                layer_conditions, layer_fingerprints
            )
        ],
    }


def build_models(
    base_config: PrototypeConfig,
    condition: ConditionSpec,
    *,
    key: Optional[MasterKey] = None,
    model_seed: Optional[int] = None,
    debug_enabled: bool = True,
    pretrained_path: Optional[str | Path] = None,
    pretrained_artifact: Optional[Qwen2Artifact] = None,
    pretrained_plain: Optional[PlainTinyCausalLM] = None,
    model_max_sequence_length: Optional[int] = None,
) -> EvalModels:
    """Construct plain (+ optional obfuscated) models for one condition cell.

    When ``pretrained_path`` is supplied, the local Qwen2 adapter is used and
    no random weights are created.  ``pretrained_plain`` lets a multi-key run
    reuse one loaded checkpoint while converting independent obfuscation keys.
    """

    artifact = pretrained_artifact
    if pretrained_path is not None and artifact is None:
        requested_context = model_max_sequence_length or (
            (condition.sequence_length or base_config.evaluation.sequence_length)
            + base_config.evaluation.generation_tokens
        )
        artifact = load_qwen2_artifact(
            pretrained_path,
            max_sequence_length=int(requested_context),
            compute_hashes=True,
        )
    if artifact is not None:
        base_config = replace(base_config, model=artifact.model_config)

    config = resolve_prototype_config(
        base_config,
        condition,
        seed=model_seed if model_seed is not None else base_config.runtime.seed,
        debug_enabled=debug_enabled,
    )
    device = _device_from_config(config)
    dtype = condition.torch_dtype
    if dtype == torch.float16:
        # Prototype runtime config lacks fp16; cast activations after float32 weights.
        dtype = torch.float16

    plain_seed = config.runtime.seed
    if pretrained_plain is not None:
        if pretrained_plain.config != config.model:
            raise ValueError("pretrained_plain configuration does not match condition")
        plain = pretrained_plain
    elif artifact is not None:
        plain = load_qwen2_plain(
            artifact,
            device="cpu",
            dtype=dtype,
            seed=plain_seed,
            debug_enabled=debug_enabled,
        )
    else:
        plain = PlainTinyCausalLM(
            config.model, seed=plain_seed, debug_enabled=debug_enabled
        )
    plain.eval()

    notes: List[str] = []
    obfuscated: Optional[ObfuscatedTinyCausalLM] = None
    client = None
    conversion_time = 0.0
    structural_zeroed = False
    vocab_perm: Optional[torch.Tensor] = None

    if condition.mode != ObfuscationMode.PLAINTEXT:
        conversion_seed = (
            key.conversion_seed() if key is not None else config.runtime.seed
        )
        approx = None
        if config.attention.mode in (
            AttentionMode.TOPK_PRESERVING,
            AttentionMode.FREE_BOUNDED,
        ):
            approx = ApproximationConfig(
                tau_max=config.attention.tau_max,
                tau_error=config.attention.tau_error,
                alpha=config.attention.alpha,
                preserve_top_k=config.attention.preserve_top_k,
            )
        converted: ConvertedObfuscatedLM = ObfuscatedTinyCausalLM.from_plain(
            plain,
            obfuscation=config.obfuscation,
            mode=config.attention.mode,
            approximation=approx,
            seed=conversion_seed,
            debug_enabled=debug_enabled,
        )
        obfuscated = converted.module
        client = converted.client
        conversion_time = converted.conversion_time_seconds
        obfuscated.eval()

        # Optional vocab permutation buffer (identity if absent).
        if hasattr(obfuscated, "vocab_permutation"):
            vocab_perm = getattr(obfuscated, "vocab_permutation")
            notes.append("using model vocab_permutation buffer")
        else:
            notes.append(
                "LM-head vocab permutation Π_voc not present in prototype; "
                "identity used for inverse-align"
            )

        if condition.mode == ObfuscationMode.STRUCTURAL:
            zeroed = zero_noise_injection_(obfuscated)
            structural_zeroed = True
            notes.append(
                "structural mode: zeroed %d noise-injection buffers; "
                "full obfuscation graph retained" % len(zeroed)
            )
        elif condition.mode == ObfuscationMode.FULL:
            initial_scale = (
                float(condition.initial_noise_scale)
                if condition.initial_noise_scale is not None
                else 1.0
            )
            refresh_scale = (
                float(condition.refresh_noise_scale)
                if condition.refresh_noise_scale is not None
                else 1.0
            )
            if condition.stress and initial_scale == 1.0 and refresh_scale == 1.0:
                initial_scale = refresh_scale = 2.0
            if initial_scale != 1.0 or refresh_scale != 1.0:
                apply_noise_scale_(
                    obfuscated,
                    initial_scale=initial_scale,
                    refresh_scale=refresh_scale,
                )
                notes.append(
                    "full mode noise scale factors: initial=%s refresh=%s"
                    % (initial_scale, refresh_scale)
                )

    # The condition dtype is part of the fairness contract.  Cast parameters,
    # registered conversion buffers, and activations together; merely labelling
    # an FP32 model as BF16 would make P0--P3 incomparable.
    plain.to(device=device, dtype=dtype)
    if obfuscated is not None:
        obfuscated.to(device=device, dtype=dtype)

    return EvalModels(
        plain=plain,
        obfuscated=obfuscated,
        client=client,
        config=config,
        condition=condition,
        key=key,
        conversion_time_seconds=conversion_time,
        device=device,
        dtype=dtype,
        structural_noise_zeroed=structural_zeroed,
        vocab_permutation=vocab_perm,
        notes=notes,
        model_manifest=artifact.to_dict() if artifact is not None else None,
        obfuscation_manifest=_obfuscation_manifest(obfuscated),
    )


def inverse_align_logits(
    logits: torch.Tensor,
    vocab_permutation: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply ℓ̂ = ℓ̃ · Π_voc^T (protocol §4 Layer 4).

    Convention: ``vocab_permutation[k]`` is the obfuscated-logit index that
    corresponds to plaintext vocabulary item ``k``. Then
    ``aligned[..., k] = logits[..., perm[k]]``.

    If ``vocab_permutation`` is None, the permutation is identity.
    """

    if vocab_permutation is None:
        return logits
    perm = vocab_permutation.to(device=logits.device, dtype=torch.long)
    if perm.ndim != 1 or perm.numel() != logits.shape[-1]:
        raise ValueError("vocab_permutation must have shape [vocab_size]")
    return logits[..., perm]


def cast_activations(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast floating activations to the condition dtype."""

    if tensor.is_floating_point() and tensor.dtype != dtype:
        return tensor.to(dtype=dtype)
    return tensor
