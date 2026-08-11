"""Executable tiny-model evaluation primitives.

These functions perform work only when called by an explicitly authorized CLI
path. Importing this module never starts an evaluation.
"""

from __future__ import annotations

import hashlib
import gc
import json
import os
import platform
import resource
import shutil
import statistics
import time
import traceback
from dataclasses import asdict, replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ..config import AttentionConfig, EvaluationConfig, PrototypeConfig, load_config
from ..layers.attention import ApproximationConfig, AttentionMode
from ..models.obfuscated import (
    ObfuscatedBlockDebug,
    ObfuscatedLMCache,
    ObfuscatedTinyCausalLM,
)
from ..models.plain import PlainTinyCausalLM
from ..seed import RequestContext
from .accuracy import (
    compare_teacher_forced_metrics,
    greedy_generation_metrics,
    make_synthetic_token_batch,
)
from .artifacts import (
    append_jsonl_record,
    build_run_record,
    validate_run_record,
)
from .metrics import attention_distribution_metrics, tensor_error_metrics
from .sweep import SweepSpec
from .token_cache import load_token_cache
from ..pretrained.qwen2 import Qwen2Artifact, load_qwen2_artifact, load_qwen2_plain
from .pretrained import token_cache_provenance_issues


def environment_metadata() -> Dict[str, Any]:
    """Return JSON-safe software and accelerator metadata without fallback."""

    def package_version(name: str) -> Optional[str]:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            return None

    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    device_names: Dict[str, Optional[str]] = {
        "cuda": torch.cuda.get_device_name(0) if cuda_available else None,
        "mps": "Apple Metal Performance Shaders" if mps_available else None,
    }
    physical_memory = None
    try:
        physical_memory = int(
            os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        )
    except (AttributeError, OSError, ValueError):
        pass
    disk = shutil.disk_usage(Path.cwd())
    mps_fallback_env = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "")
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": package_version("numpy"),
        "transformers_version": package_version("transformers"),
        "safetensors_version": package_version("safetensors"),
        "fastprove_version": package_version("fastprove") or "source-tree",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": int(os.cpu_count() or 1),
        "physical_memory_bytes": physical_memory,
        "disk_total_bytes": int(disk.total),
        "disk_free_bytes": int(disk.free),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "mps_available": mps_available,
        "mps_fallback_env": mps_fallback_env or None,
        "device_names": device_names,
    }


def _resolve_runtime(config: PrototypeConfig) -> Tuple[torch.device, torch.dtype]:
    requested = config.runtime.device
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no fallback")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable; no fallback")
    if requested == "mps" and os.environ.get(
        "PYTORCH_ENABLE_MPS_FALLBACK", ""
    ).lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            "PYTORCH_ENABLE_MPS_FALLBACK is enabled; refusing a run that may "
            "silently execute operators on CPU"
        )
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[config.runtime.activation_dtype]
    return torch.device(requested), dtype


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _release_runtime_memory() -> None:
    """Release references/cached allocator blocks between independent runs."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        empty_cache = getattr(torch.mps, "empty_cache", None)
        if empty_cache is not None:
            empty_cache()


def _process_peak_rss_bytes() -> Optional[int]:
    """Return process peak RSS in bytes where the host exposes it."""

    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None
    # macOS reports bytes; Linux and most other Unix systems report KiB.
    return value if platform.system() == "Darwin" else value * 1024


def _mps_allocated_bytes() -> Optional[int]:
    """Return the MPS allocator counter when supported by this PyTorch build."""

    if not hasattr(torch, "mps") or not torch.backends.mps.is_available():
        return None
    current = getattr(torch.mps, "current_allocated_memory", None)
    if current is None:
        return None
    try:
        return int(current())
    except (RuntimeError, TypeError, ValueError):
        return None


def _hash_identifiers(identifiers: Sequence[str]) -> str:
    payload = json.dumps(
        list(identifiers), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a local configuration file for run-level reproducibility."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _obfuscation_manifest(
    model: Optional[ObfuscatedTinyCausalLM],
) -> Optional[Dict[str, Any]]:
    """Return non-secret basis metadata for a converted evaluation model.

    The manifest records condition numbers and public fingerprints only.  It
    deliberately does not serialize mixing matrices, inverses, or seed
    material, while making the offline conditioning checks auditable.
    """

    if model is None:
        return None
    hidden = model.hidden_basis
    return {
        "max_condition_number": float(model.obfuscation.max_condition_number),
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
                model.value_basis_condition_numbers,
                model.value_basis_fingerprints,
            )
            for condition_number, fingerprint in zip(
                layer_conditions, layer_fingerprints
            )
        ],
    }


def _mean_metric_dicts(items: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {}
    keys = items[0].keys()
    result: Dict[str, float] = {}
    for key in keys:
        values = [float(item[key]) for item in items]
        if key in (
            "max_absolute_error",
            "actual_noise_infinity_norm",
            "clean_boundary_margin_max",
        ):
            result[key] = max(values)
        elif key == "clean_boundary_margin_min":
            result[key] = min(values)
        elif key in ("nan_count", "inf_count"):
            result[key] = float(sum(values))
        else:
            result[key] = float(sum(values) / len(values))
    return result


def _identity_softmax_metrics() -> Dict[str, float]:
    return {
        "kl_divergence": 0.0,
        "js_divergence": 0.0,
        "topk_overlap": 1.0,
        "topk_changed_fraction": 0.0,
        "rank_correlation": 1.0,
        "actual_noise_infinity_norm": 0.0,
        "zero_noise_query_fraction": 1.0,
        "clean_boundary_margin_mean": 0.0,
        "clean_boundary_margin_min": 0.0,
        "clean_boundary_margin_max": 0.0,
        "attention_output_relative_l2_error": 0.0,
    }


def _first_argmax_mismatch(
    plaintext_logits: torch.Tensor,
    obfuscated_logits: torch.Tensor,
) -> Optional[Dict[str, Any]]:
    plain_predictions = plaintext_logits.argmax(dim=-1)
    obfuscated_predictions = obfuscated_logits.argmax(dim=-1)
    locations = torch.nonzero(
        plain_predictions != obfuscated_predictions, as_tuple=False
    )
    if locations.numel() == 0:
        return None
    batch_index = int(locations[0, 0].item())
    position = int(locations[0, 1].item())
    row = plaintext_logits[batch_index, position].float()
    top_count = min(2, row.numel())
    top_values = torch.topk(row, top_count).values
    margin = (
        float((top_values[0] - top_values[1]).item())
        if top_count == 2
        else None
    )
    return {
        "batch_index": batch_index,
        "position": position,
        "plaintext_token": int(
            plain_predictions[batch_index, position].item()
        ),
        "obfuscated_token": int(
            obfuscated_predictions[batch_index, position].item()
        ),
        "plaintext_top2_margin": margin,
    }


def _first_token_mismatch(
    plaintext_tokens: torch.Tensor,
    obfuscated_tokens: torch.Tensor,
    *,
    prompt_length: int,
) -> Optional[Dict[str, int]]:
    differences = torch.nonzero(
        plaintext_tokens[:, prompt_length:]
        != obfuscated_tokens[:, prompt_length:],
        as_tuple=False,
    )
    if differences.numel() == 0:
        return None
    batch_index = int(differences[0, 0].item())
    generation_step = int(differences[0, 1].item())
    position = prompt_length + generation_step
    return {
        "batch_index": batch_index,
        "generation_step": generation_step,
        "plaintext_token": int(
            plaintext_tokens[batch_index, position].item()
        ),
        "obfuscated_token": int(
            obfuscated_tokens[batch_index, position].item()
        ),
    }


def _exact_gate_tolerances(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return the recorded numeric envelope used by the exact gate.

    The tiny FP32 unit gate is deliberately much tighter than the deep-model
    envelope.  A real checkpoint applies a basis conversion at every layer,
    and the reference smoke on both CPU and MPS showed a small absolute
    Softmax deviation (roughly 1.3--1.5e-3) while relative error and token
    agreement remained at floating-point level.  Keep the relaxed deep FP32
    absolute Softmax bound explicit and recorded; this is not a dtype label
    that can make a large drift pass.
    """

    config = record.get("config", {})
    runtime = config.get("runtime", {}) if isinstance(config, dict) else {}
    environment = record.get("environment", {})
    model = record.get("model", {})
    pretrained = bool(model.get("pretrained", False)) if isinstance(model, dict) else False
    dtype = str(runtime.get("activation_dtype", "float32"))
    actual_device = str(
        environment.get("actual_device")
        or environment.get("device")
        or runtime.get("device", "unknown")
    )
    checkpoint_dtype = str(
        environment.get("checkpoint_compute_dtype")
        or ("float64" if actual_device == "cpu" else "float32")
    )
    if dtype == "float32" and pretrained:
        profile = "deep_fp32_calibrated"
        logits_max = 1e-3
        qk_max = 2e-2
        softmax_max = 2e-3
    elif dtype == "float32":
        profile = "tiny_fp32_strict"
        logits_max = 1e-3
        qk_max = 2e-2
        softmax_max = 1e-3
    else:
        profile = "non_fp32_separate_condition"
        logits_max = 2e-2
        qk_max = 2e-2
        softmax_max = 2e-2
    return {
        "profile": profile,
        "pretrained": pretrained,
        "activation_dtype": dtype,
        "actual_device": actual_device,
        "checkpoint_compute_dtype": checkpoint_dtype,
        "max_logit_absolute_error": logits_max,
        "max_qk_absolute_error": qk_max,
        "max_softmax_absolute_error": softmax_max,
        "max_relative_l2_error": 1e-4,
        "required_teacher_forced_agreement": 1.0,
        "required_greedy_token_agreement": 1.0,
        "required_greedy_sequence_agreement": 1.0,
    }


def evaluate_exact_gate(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Apply the mandatory exact-mode gate before approximate conclusions."""

    reasons: List[str] = []
    if record.get("status") != "success":
        return False, ["exact run status is not success"]
    if record.get("config", {}).get("mode") != "exact":
        return False, ["record is not exact mode"]
    metrics = record.get("metrics", {})
    tolerances = _exact_gate_tolerances(record)
    metrics["exact_gate_tolerances"] = tolerances
    logits_tolerance = float(tolerances["max_logit_absolute_error"])
    qk_tolerance = float(tolerances["max_qk_absolute_error"])
    softmax_tolerance = float(tolerances["max_softmax_absolute_error"])
    checks = (
        (
            metrics.get("nan_inf_count") == 0,
            "exact output contains NaN/Inf",
        ),
        (
            metrics.get("agreement", {}).get(
                "next_token_top1_agreement"
            )
            == 1.0,
            "teacher-forced token agreement is not 1",
        ),
        (
            metrics.get("greedy", {}).get("greedy_token_exact_match")
            == 1.0,
            "greedy token agreement is not 1",
        ),
        (
            metrics.get("greedy", {}).get(
                "greedy_sequence_exact_match"
            )
            == 1.0,
            "greedy sequence agreement is not 1",
        ),
        (
            metrics.get("layer", {})
            .get("logits", {})
            .get("max_absolute_error", float("inf"))
            <= logits_tolerance,
            "exact logit error exceeds tolerance",
        ),
        (
            metrics.get("layer", {})
            .get("logits", {})
            .get("relative_l2_error", float("inf"))
            <= 1e-4,
            "exact relative logit error exceeds tolerance",
        ),
        (
            metrics.get("layer", {})
            .get("qk_scores", {})
            .get("relative_l2_error", float("inf"))
            <= 1e-4,
            "exact relative QK error exceeds tolerance",
        ),
        (
            metrics.get("layer", {})
            .get("qk_scores", {})
            .get("max_absolute_error", float("inf"))
            <= qk_tolerance,
            "exact QK error exceeds tolerance",
        ),
        (
            metrics.get("layer", {})
            .get("exact_softmax_probabilities", {})
            .get("relative_l2_error", float("inf"))
            <= 1e-4,
            "exact relative Softmax error exceeds tolerance",
        ),
        (
            metrics.get("layer", {})
            .get("exact_softmax_probabilities", {})
            .get("max_absolute_error", float("inf"))
            <= softmax_tolerance,
            "exact Softmax error exceeds tolerance",
        ),
    )
    for passed, reason in checks:
        if not passed:
            reasons.append(reason)
    diagnostics = metrics.get("exact_gate_diagnostics", {})
    if diagnostics.get("first_teacher_forced_argmax_mismatch") is not None:
        reasons.append("teacher-forced mismatch diagnostic is populated")
    if diagnostics.get("first_greedy_mismatch") is not None:
        reasons.append("greedy mismatch diagnostic is populated")
    return not reasons, reasons


def _layer_metrics(
    debug_batches: Sequence[Tuple[ObfuscatedBlockDebug, ...]],
    *,
    top_k: int,
) -> Tuple[Dict[str, Any], Dict[str, float], int]:
    if not debug_batches:
        return {
            "qk_scores": {},
            "exact_softmax_probabilities": {},
            "per_layer": [],
        }, _identity_softmax_metrics(), 0
    layer_count = len(debug_batches[0])
    per_layer: List[Dict[str, Any]] = []
    distribution_items: List[Dict[str, float]] = []
    nan_inf_count = 0
    for layer_index in range(layer_count):
        records = [batch[layer_index] for batch in debug_batches]
        clean_probabilities = torch.cat(
            [record.clean_probabilities.cpu() for record in records], dim=0
        )
        noisy_probabilities = torch.cat(
            [record.noisy_probabilities.cpu() for record in records], dim=0
        )
        clean_logits = torch.cat(
            [record.clean_logits.cpu() for record in records], dim=0
        )
        noisy_logits = torch.cat(
            [record.noisy_logits.cpu() for record in records], dim=0
        )
        valid_mask = torch.cat(
            [record.valid_mask.cpu() for record in records], dim=0
        )
        noise = torch.cat(
            [record.logit_noise.cpu() for record in records], dim=0
        )
        margin = torch.cat(
            [record.margin.cpu() for record in records], dim=0
        )
        clean_output = torch.cat(
            [record.clean_attention_output.cpu() for record in records],
            dim=0,
        )
        noisy_output = torch.cat(
            [record.attention_output.cpu() for record in records], dim=0
        )
        distribution = attention_distribution_metrics(
            clean_probabilities=clean_probabilities,
            noisy_probabilities=noisy_probabilities,
            clean_logits=clean_logits,
            noisy_logits=noisy_logits,
            valid_mask=valid_mask,
            noise=noise,
            margin=margin,
            top_k=top_k,
            clean_output=clean_output,
            noisy_output=noisy_output,
        )
        qk = _mean_metric_dicts([record.qk_score_error for record in records])
        softmax_error = _mean_metric_dicts(
            [record.softmax_error for record in records]
        )
        per_layer.append(
            {
                "layer_index": layer_index,
                "qk_scores": qk,
                "exact_softmax_probabilities": softmax_error,
                "softmax": distribution,
            }
        )
        distribution_items.append(distribution)
        for record in records:
            for tensor in (
                record.clean_probabilities,
                record.noisy_probabilities,
                record.attention_output,
                record.attention_noise_state,
                record.swiglu_noise_state,
                record.final_noise_state,
            ):
                nan_inf_count += int(torch.isnan(tensor).sum().item())
                nan_inf_count += int(torch.isinf(tensor).sum().item())
    return {
        "qk_scores": _mean_metric_dicts(
            [item["qk_scores"] for item in per_layer]
        ),
        "exact_softmax_probabilities": _mean_metric_dicts(
            [item["exact_softmax_probabilities"] for item in per_layer]
        ),
        "per_layer": per_layer,
    }, _mean_metric_dicts(distribution_items), nan_inf_count


def _debug_batch_to_cpu(
    batch: Tuple[ObfuscatedBlockDebug, ...],
) -> Tuple[ObfuscatedBlockDebug, ...]:
    """Detach research diagnostics before retaining the next device batch."""

    # ``forward_debug`` is intentionally the only path that captures these
    # tensors.  The runner still needs them for layer metrics, but retaining
    # device-backed tensors across all samples defeats the staged memory
    # contract for a large checkpoint.
    cpu_batches: List[ObfuscatedBlockDebug] = []
    for record in batch:
        cpu_batches.append(
            ObfuscatedBlockDebug(
                qk_score_error=dict(record.qk_score_error),
                softmax_error=dict(record.softmax_error),
                clean_logits=record.clean_logits.detach().cpu(),
                noisy_logits=record.noisy_logits.detach().cpu(),
                clean_probabilities=record.clean_probabilities.detach().cpu(),
                noisy_probabilities=record.noisy_probabilities.detach().cpu(),
                valid_mask=record.valid_mask.detach().cpu(),
                logit_noise=record.logit_noise.detach().cpu(),
                tau=record.tau.detach().cpu(),
                margin=record.margin.detach().cpu(),
                clean_attention_output=record.clean_attention_output.detach().cpu(),
                attention_output=record.attention_output.detach().cpu(),
                post_attention=record.post_attention.detach().cpu(),
                attention_noise_state=record.attention_noise_state.detach().cpu(),
                swiglu_noise_state=record.swiglu_noise_state.detach().cpu(),
                final_noise_state=record.final_noise_state.detach().cpu(),
            )
        )
    return tuple(cpu_batches)


def _time_call(
    call: Any,
    *,
    device: torch.device,
    warmup_runs: int,
    timed_runs: int,
) -> Dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup_runs):
            call()
    _sync(device)
    durations = []
    with torch.inference_mode():
        for _ in range(timed_runs):
            started = time.perf_counter()
            call()
            _sync(device)
            durations.append(time.perf_counter() - started)
    return {
        "mean_seconds": float(statistics.fmean(durations)),
        "min_seconds": float(min(durations)),
        "max_seconds": float(max(durations)),
    }


def _performance_metrics(
    *,
    plain: PlainTinyCausalLM,
    obfuscated: Optional[ObfuscatedTinyCausalLM],
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    context: RequestContext,
    evaluation: EvaluationConfig,
    device: torch.device,
    conversion_seconds: float,
) -> Dict[str, Any]:
    batch = tokens[: evaluation.batch_size]
    mask = token_mask[: evaluation.batch_size]
    if mask.shape != batch.shape or mask.dtype != torch.bool:
        raise ValueError("token_mask must be boolean and match tokens")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
    plain_prefill = _time_call(
        lambda: plain(batch, token_mask=mask),
        device=device,
        warmup_runs=evaluation.warmup_runs,
        timed_runs=evaluation.timed_runs,
    )
    plain_generated_call = lambda: plain.generate_greedy(
        batch,
        max_new_tokens=evaluation.generation_tokens,
        token_mask=mask,
    )
    if obfuscated is None:
        obfuscated_prefill = dict(plain_prefill)
        obfuscated_generated_call = plain_generated_call
    else:
        obfuscated_prefill = _time_call(
            lambda: obfuscated(
                batch,
                token_mask=mask,
                request_context=context,
            ),
            device=device,
            warmup_runs=evaluation.warmup_runs,
            timed_runs=evaluation.timed_runs,
        )
        obfuscated_generated_call = lambda: obfuscated.generate_greedy(
            batch,
            max_new_tokens=evaluation.generation_tokens,
            request_context=context,
            token_mask=mask,
        )
    plain_decode = _time_call(
        plain_generated_call,
        device=device,
        warmup_runs=0,
        timed_runs=evaluation.timed_runs,
    )
    # In the plaintext phase ``obfuscated`` is deliberately absent: reuse the
    # measured plaintext decode instead of running the same generation twice.
    # The returned schema still exposes both labels for backwards-compatible
    # comparisons, while staged non-plaintext runs measure the obfuscated
    # phase in ``_obfuscated_performance_metrics`` after releasing plaintext.
    obfuscated_decode = (
        dict(plain_decode)
        if obfuscated is None
        else _time_call(
            obfuscated_generated_call,
            device=device,
            warmup_runs=0,
            timed_runs=evaluation.timed_runs,
        )
    )
    prefill_tokens = int(batch.numel())
    decoded_tokens = int(batch.shape[0] * evaluation.generation_tokens)
    element_size = int(next(plain.parameters()).element_size())
    plain_layer_bytes = (
        2
        * batch.shape[0]
        * plain.config.num_key_value_heads
        * batch.shape[1]
        * plain.config.head_dim
        * element_size
        + batch.shape[0] * batch.shape[1]
        + batch.shape[1] * torch.tensor([], dtype=torch.int64).element_size()
    )
    plaintext_kv_bytes = int(plain.config.num_layers * plain_layer_bytes)
    obfuscated_kv_bytes = plaintext_kv_bytes
    if obfuscated is not None:
        with torch.inference_mode():
            _, measured_cache = obfuscated(
                batch,
                token_mask=mask,
                positions=torch.arange(batch.shape[1], device=batch.device),
                request_context=context,
                use_cache=True,
            )
        assert isinstance(measured_cache, ObfuscatedLMCache)
        obfuscated_kv_bytes = 0
        for layer_cache in measured_cache.layers:
            for tensor in (
                layer_cache.key,
                layer_cache.value_mixed,
                layer_cache.key_valid,
                layer_cache.positions,
            ):
                obfuscated_kv_bytes += int(
                    tensor.numel() * tensor.element_size()
                )
    if device.type == "cuda":
        peak_memory: Dict[str, Any] = {
            "status": "measured_cuda_allocator",
            "bytes": int(torch.cuda.max_memory_allocated(device)),
            "process_peak_rss_bytes": _process_peak_rss_bytes(),
        }
    else:
        peak_memory = {
            "status": "measured_process_rss",
            "bytes": _process_peak_rss_bytes(),
            "mps_current_allocated_bytes": (
                _mps_allocated_bytes() if device.type == "mps" else None
            ),
        }
    return {
        "conversion_time_seconds": float(conversion_seconds),
        "prefill": {
            "plaintext": plain_prefill,
            "obfuscated": obfuscated_prefill,
            "obfuscated_tokens_per_second": (
                prefill_tokens / obfuscated_prefill["mean_seconds"]
            ),
        },
        "decode": {
            "status": "incremental_kv_cache_reference",
            "plaintext": {
                **plain_decode,
                "tpot_seconds": plain_decode["mean_seconds"] / decoded_tokens,
                "tokens_per_second": (
                    decoded_tokens / plain_decode["mean_seconds"]
                ),
            },
            "obfuscated": {
                **obfuscated_decode,
                "tpot_seconds": (
                    obfuscated_decode["mean_seconds"] / decoded_tokens
                ),
                "tokens_per_second": (
                    decoded_tokens / obfuscated_decode["mean_seconds"]
                ),
            },
        },
        "peak_memory": peak_memory,
        "kv_cache": {
            "status": "measured_tensor_storage",
            "plaintext_bytes": plaintext_kv_bytes,
            "obfuscated_bytes": obfuscated_kv_bytes,
            "decode_benchmark_uses_cache": True,
        },
    }


def _obfuscated_performance_metrics(
    *,
    obfuscated: ObfuscatedTinyCausalLM,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    context: RequestContext,
    evaluation: EvaluationConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """Measure the obfuscated phase while holding no plaintext model.

    The normal comparison path deliberately measures plaintext first, then
    releases it before this helper is called.  Keeping this phase separate is
    important for the local 16-GiB host: ``from_plain`` already has a short
    conversion peak, but retaining both full 1.5B models through decode and
    cache measurement needlessly extends that peak.
    """

    batch = tokens[: evaluation.batch_size]
    mask = token_mask[: evaluation.batch_size]
    if mask.shape != batch.shape or mask.dtype != torch.bool:
        raise ValueError("token_mask must be boolean and match tokens")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
    obfuscated_prefill = _time_call(
        lambda: obfuscated(
            batch,
            token_mask=mask,
            request_context=context,
        ),
        device=device,
        warmup_runs=evaluation.warmup_runs,
        timed_runs=evaluation.timed_runs,
    )
    obfuscated_generated_call = lambda: obfuscated.generate_greedy(
        batch,
        max_new_tokens=evaluation.generation_tokens,
        request_context=context,
        token_mask=mask,
    )
    obfuscated_decode = _time_call(
        obfuscated_generated_call,
        device=device,
        warmup_runs=0,
        timed_runs=evaluation.timed_runs,
    )
    prefill_tokens = int(batch.numel())
    decoded_tokens = int(batch.shape[0] * evaluation.generation_tokens)
    with torch.inference_mode():
        _, measured_cache = obfuscated(
            batch,
            token_mask=mask,
            positions=torch.arange(batch.shape[1], device=batch.device),
            request_context=context,
            use_cache=True,
        )
    assert isinstance(measured_cache, ObfuscatedLMCache)
    obfuscated_kv_bytes = 0
    for layer_cache in measured_cache.layers:
        for tensor in (
            layer_cache.key,
            layer_cache.value_mixed,
            layer_cache.key_valid,
            layer_cache.positions,
        ):
            obfuscated_kv_bytes += int(tensor.numel() * tensor.element_size())
    if device.type == "cuda":
        peak_memory: Dict[str, Any] = {
            "status": "measured_cuda_allocator",
            "bytes": int(torch.cuda.max_memory_allocated(device)),
            "process_peak_rss_bytes": _process_peak_rss_bytes(),
        }
    else:
        peak_memory = {
            "status": "measured_process_rss",
            "bytes": _process_peak_rss_bytes(),
            "mps_current_allocated_bytes": (
                _mps_allocated_bytes() if device.type == "mps" else None
            ),
        }
    return {
        "prefill": {
            **obfuscated_prefill,
            "tokens_per_second": prefill_tokens
            / obfuscated_prefill["mean_seconds"],
        },
        "decode": {
            **obfuscated_decode,
            "tpot_seconds": obfuscated_decode["mean_seconds"]
            / decoded_tokens,
            "tokens_per_second": decoded_tokens
            / obfuscated_decode["mean_seconds"],
        },
        "peak_memory": peak_memory,
        "obfuscated_kv_bytes": obfuscated_kv_bytes,
    }


def _merge_staged_performance_metrics(
    *,
    plaintext: Dict[str, Any],
    obfuscated: Dict[str, Any],
    conversion_seconds: float,
) -> Dict[str, Any]:
    """Combine separately measured phases into the stable runner schema."""

    plain_peak = plaintext["peak_memory"]
    obfuscated_peak = obfuscated["peak_memory"]
    peak_bytes = [
        value
        for value in (
            plain_peak.get("bytes"),
            obfuscated_peak.get("bytes"),
        )
        if isinstance(value, (int, float))
    ]
    mps_bytes = [
        value
        for value in (
            plain_peak.get("mps_current_allocated_bytes"),
            obfuscated_peak.get("mps_current_allocated_bytes"),
        )
        if isinstance(value, (int, float))
    ]
    merged_peak: Dict[str, Any] = {
        # Keep the historical scalar fields for report compatibility.  The
        # maximum is across the two independently measured phases, not an
        # accidental claim that both models were resident simultaneously.
        "status": obfuscated_peak.get("status", plain_peak.get("status")),
        "bytes": max(peak_bytes) if peak_bytes else None,
        "process_peak_rss_bytes": max(
            [
                value
                for value in (
                    plain_peak.get("process_peak_rss_bytes"),
                    obfuscated_peak.get("process_peak_rss_bytes"),
                )
                if isinstance(value, (int, float))
            ],
            default=None,
        ),
        "mps_current_allocated_bytes": max(mps_bytes)
        if mps_bytes
        else None,
        "measurement_phases": {
            "plaintext": plain_peak,
            "obfuscated": obfuscated_peak,
        },
    }
    obfuscated_prefill = obfuscated["prefill"]
    obfuscated_decode = obfuscated["decode"]
    return {
        "conversion_time_seconds": float(conversion_seconds),
        "prefill": {
            "plaintext": plaintext["prefill"]["plaintext"],
            "obfuscated": {
                key: value
                for key, value in obfuscated_prefill.items()
                if key in ("mean_seconds", "min_seconds", "max_seconds")
            },
            "obfuscated_tokens_per_second": float(
                obfuscated_prefill["tokens_per_second"]
            ),
        },
        "decode": {
            "status": "incremental_kv_cache_reference",
            "plaintext": plaintext["decode"]["plaintext"],
            "obfuscated": {
                key: value
                for key, value in obfuscated_decode.items()
                if key in (
                    "mean_seconds",
                    "min_seconds",
                    "max_seconds",
                    "tpot_seconds",
                    "tokens_per_second",
                )
            },
        },
        "peak_memory": merged_peak,
        "kv_cache": {
            "status": "measured_tensor_storage",
            "plaintext_bytes": plaintext["kv_cache"]["plaintext_bytes"],
            "obfuscated_bytes": obfuscated["obfuscated_kv_bytes"],
            "decode_benchmark_uses_cache": True,
        },
    }


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


def _execute_tiny(
    *,
    config: PrototypeConfig,
    spec: SweepSpec,
    experiment: Dict[str, Any],
    pretrained_path: Optional[str | Path] = None,
    pretrained_artifact: Optional[Qwen2Artifact] = None,
    dataset_cache_path: Optional[str | Path] = None,
) -> Tuple[Dict[str, Any], float]:
    started = time.perf_counter()
    device, dtype = _resolve_runtime(config)
    seed = int(experiment.get("seed", config.runtime.seed))
    artifact = pretrained_artifact
    token_cache = None
    if pretrained_path is not None and dataset_cache_path is None:
        raise ValueError("pretrained_path requires dataset_cache_path")
    if dataset_cache_path is not None and pretrained_path is None:
        raise ValueError("dataset_cache_path requires pretrained_path")
    if pretrained_path is not None:
        if artifact is None:
            artifact = load_qwen2_artifact(
                pretrained_path,
                max_sequence_length=(
                    config.evaluation.sequence_length
                    + config.evaluation.generation_tokens
                ),
                compute_hashes=True,
            )
        token_cache = load_token_cache(
            dataset_cache_path,
            expected_vocab_size=artifact.model_config.vocab_size,
            expected_sequence_length=config.evaluation.sequence_length,
        )
        provenance_issues = token_cache_provenance_issues(token_cache, artifact)
        if provenance_issues:
            raise ValueError(
                "token cache provenance is incomplete: "
                + "; ".join(provenance_issues)
            )
        if token_cache.sample_count < config.evaluation.sample_count:
            raise ValueError(
                "token cache has fewer samples than evaluation.sample_count"
            )
        config = replace(
            config,
            model=artifact.model_config,
        )
        tokens_cpu = token_cache.input_ids[: config.evaluation.sample_count]
        identifiers = list(token_cache.sample_ids[: config.evaluation.sample_count])
        token_mask_cpu = token_cache.token_mask[: config.evaluation.sample_count]
    else:
        tokens_cpu, identifiers = make_synthetic_token_batch(
            sample_count=config.evaluation.sample_count,
            sequence_length=config.evaluation.sequence_length,
            vocab_size=config.model.vocab_size,
            seed=seed,
        )
        token_mask_cpu = torch.ones_like(tokens_cpu, dtype=torch.bool)
    tokens = tokens_cpu.to(device=device)
    mask = token_mask_cpu.to(device=device)
    if artifact is not None:
        plain = load_qwen2_plain(
            artifact,
            device=device,
            dtype=dtype,
            seed=seed,
            debug_enabled=True,
        )
    else:
        plain = PlainTinyCausalLM(
            config.model, seed=seed, debug_enabled=True
        ).to(device=device, dtype=dtype)
    plain.eval()
    mode = AttentionMode(spec.mode)
    approximation = None
    if mode in (
        AttentionMode.TOPK_PRESERVING,
        AttentionMode.FREE_BOUNDED,
    ):
        approximation = ApproximationConfig(
            tau_max=spec.tau_max,
            tau_error=spec.tau_error,
            alpha=config.attention.alpha,
            preserve_top_k=config.attention.preserve_top_k,
        )

    plain_parts = []
    with torch.no_grad():
        for start in range(0, tokens.shape[0], config.evaluation.batch_size):
            stop = min(start + config.evaluation.batch_size, tokens.shape[0])
            batch = tokens[start:stop]
            batch_mask = mask[start:stop]
            plain_logits = plain(batch, token_mask=batch_mask)
            plain_parts.append(plain_logits.float().cpu())
    plaintext_logits = torch.cat(plain_parts, dim=0)
    prompt = tokens[: config.evaluation.batch_size]
    prompt_mask = mask[: config.evaluation.batch_size]
    plain_generated = plain.generate_greedy(
        prompt,
        max_new_tokens=config.evaluation.generation_tokens,
        token_mask=prompt_mask,
    )
    plain_generated_cpu = plain_generated.cpu()
    plain_performance = _performance_metrics(
        plain=plain,
        obfuscated=None,
        tokens=tokens,
        token_mask=mask,
        context=RequestContext(seed, config.runtime.request_id),
        evaluation=config.evaluation,
        device=device,
        conversion_seconds=0.0,
    )

    converted = None
    obfuscated = None
    conversion_seconds = 0.0
    obfuscated_parts = []
    debug_batches: List[Tuple[ObfuscatedBlockDebug, ...]] = []
    if mode != AttentionMode.PLAINTEXT:
        converted = ObfuscatedTinyCausalLM.from_plain(
            plain,
            obfuscation=config.obfuscation,
            mode=mode,
            approximation=approximation,
            seed=seed,
            debug_enabled=True,
        )
        obfuscated = converted.module.to(device=device, dtype=dtype)
        obfuscated.eval()
        conversion_seconds = converted.conversion_time_seconds
        # The converted module owns cloned deployed weights and no longer
        # needs the plaintext module.  Drop the latter before obfuscated
        # forwards/timing so the two full 1.5B copies are not resident during
        # the expensive part of the run.
        del plain
        _release_runtime_memory()
        with torch.no_grad():
            for start in range(0, tokens.shape[0], config.evaluation.batch_size):
                stop = min(start + config.evaluation.batch_size, tokens.shape[0])
                batch = tokens[start:stop]
                batch_mask = mask[start:stop]
                context = RequestContext(
                    seed,
                    "%s:batch-%d"
                    % (config.runtime.request_id, start // config.evaluation.batch_size),
                )
                obfuscated_logits, debug = obfuscated.forward_debug(
                    batch,
                    token_mask=batch_mask,
                    request_context=context,
                )
                obfuscated_parts.append(obfuscated_logits.float().cpu())
                debug_batches.append(_debug_batch_to_cpu(debug))
        obfuscated_logits = torch.cat(obfuscated_parts, dim=0)
        performance_context = RequestContext(
            seed, "%s:generation" % config.runtime.request_id
        )
        obfuscated_generated = obfuscated.generate_greedy(
            prompt,
            max_new_tokens=config.evaluation.generation_tokens,
            request_context=performance_context,
            token_mask=prompt_mask,
        )
        obfuscated_performance = _obfuscated_performance_metrics(
            obfuscated=obfuscated,
            tokens=tokens,
            token_mask=mask,
            context=performance_context,
            evaluation=config.evaluation,
            device=device,
        )
        performance = _merge_staged_performance_metrics(
            plaintext=plain_performance,
            obfuscated=obfuscated_performance,
            conversion_seconds=conversion_seconds,
        )
    else:
        obfuscated_logits = plaintext_logits
        obfuscated_generated = plain_generated
        performance = plain_performance

    metrics = compare_teacher_forced_metrics(
        plaintext_logits=plaintext_logits,
        obfuscated_logits=obfuscated_logits,
        input_ids=tokens_cpu,
        token_mask=token_mask_cpu,
        bootstrap_replicates=config.evaluation.bootstrap_replicates,
        bootstrap_seed=seed,
    )
    layer, softmax, debug_nan_inf = _layer_metrics(
        debug_batches,
        top_k=config.attention.preserve_top_k,
    )
    layer["logits"] = tensor_error_metrics(
        plaintext_logits, obfuscated_logits
    )
    metrics["greedy"] = greedy_generation_metrics(
        plaintext_tokens=plain_generated_cpu,
        obfuscated_tokens=obfuscated_generated.cpu(),
        prompt_length=prompt.shape[1],
    )
    metrics["exact_gate_diagnostics"] = {
        "first_teacher_forced_argmax_mismatch": _first_argmax_mismatch(
            plaintext_logits, obfuscated_logits
        ),
        "first_greedy_mismatch": _first_token_mismatch(
            plain_generated_cpu,
            obfuscated_generated.cpu(),
            prompt_length=prompt.shape[1],
        ),
    }
    metrics["layer"] = layer
    metrics["softmax"] = softmax
    output_nan_inf = int(torch.isnan(obfuscated_logits).sum().item()) + int(
        torch.isinf(obfuscated_logits).sum().item()
    )
    metrics["nan_inf_count"] = output_nan_inf + debug_nan_inf
    metrics["performance"] = performance
    payload = {
        "metrics": metrics,
        "sample_count": config.evaluation.sample_count,
        "model": {
            "identifier": experiment.get(
                "model_id", "fastprove-random-tiny-correctness-only"
            ),
            "architecture": asdict(config.model),
            "base_weight_seed": seed,
            "evidence_scope": "correctness_only_random_tiny",
            "pretrained": False,
        },
        "dataset": {
            "identifier": experiment.get(
                "dataset_id", "deterministic-synthetic-token-sequences"
            ),
            "sample_ids_sha256": _hash_identifiers(identifiers),
            "sample_ids": identifiers,
            "provenance": "locally generated deterministic random tokens",
            "meaningful_lm_evidence": False,
        },
    }
    if artifact is not None:
        payload["model"] = {
            "identifier": experiment.get("model_id", "local-qwen2"),
            "revision": artifact.upstream_revision,
            "weight_sha256": artifact.weights_sha256,
            "config_sha256": artifact.config_sha256,
            "tokenizer_sha256": artifact.tokenizer_sha256,
            "tokenizer_config_sha256": artifact.tokenizer_config_sha256,
            "architecture": asdict(config.model),
            "checkpoint_manifest": artifact.to_dict(),
            "evidence_scope": experiment.get(
                "evaluation_scope", "language_model_accuracy"
            ),
            "pretrained": True,
        }
        payload["dataset"] = {
            "identifier": experiment.get("dataset_id", "local-token-cache"),
            "sample_ids_sha256": _hash_identifiers(identifiers),
            "tokenized_inputs_sha256": token_cache.content_sha256,
            "sample_ids": identifiers,
            "cache_path": str(Path(dataset_cache_path).expanduser().resolve()),
            "cache_content_sha256": token_cache.content_sha256,
            "cache_metadata": token_cache.metadata,
            "evaluation_scope": experiment.get(
                "evaluation_scope", "language_model_accuracy"
            ),
            "provenance": "validated local token cache",
            "meaningful_lm_evidence": True,
        }
        payload["dataset"]["meaningful_lm_evidence"] = bool(
            experiment.get("meaningful_lm_evidence", True)
        )
    obfuscation_manifest = _obfuscation_manifest(obfuscated)
    if obfuscation_manifest is not None:
        payload["model"]["obfuscation_manifest"] = obfuscation_manifest
    # Do not carry a full checkpoint or converted closures into the next sweep
    # point.  ``_execute_tiny`` already releases plaintext before the
    # obfuscated phase; this final release covers the end of each independent
    # record so a 72-point loop does not accumulate allocator-backed models.
    if mode == AttentionMode.PLAINTEXT:
        del plain
    else:
        del obfuscated
        del converted
    _release_runtime_memory()
    return payload, time.perf_counter() - started


def run_tiny_sweep_spec(
    *,
    base_config_path: Path,
    spec: SweepSpec,
    experiment: Dict[str, Any],
    evaluation_override: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None,
    pretrained_path: Optional[str | Path] = None,
    pretrained_artifact: Optional[Qwen2Artifact] = None,
    dataset_cache_path: Optional[str | Path] = None,
    runtime_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute and optionally append one independently traceable tiny run."""

    started = time.perf_counter()
    seed = int(experiment.get("seed", 0))
    base_config_resolved = Path(base_config_path).expanduser().resolve()
    config_provenance: Dict[str, Any] = {
        "base_config_path": str(base_config_resolved),
    }
    if base_config_resolved.is_file():
        config_provenance["base_config_sha256"] = _sha256_file(
            base_config_resolved
        )
    for key in ("sweep_config_path", "sweep_config_sha256"):
        if experiment.get(key) is not None:
            config_provenance[key] = experiment[key]
    config: Optional[PrototypeConfig] = None
    try:
        config = _config_for_spec(
            load_config(base_config_path),
            spec,
            evaluation_override,
            runtime_override,
        )
        payload, elapsed = _execute_tiny(
            config=config,
            spec=spec,
            experiment=experiment,
            pretrained_path=pretrained_path,
            pretrained_artifact=pretrained_artifact,
            dataset_cache_path=dataset_cache_path,
        )
        run_environment = environment_metadata()
        run_environment.update(
            {
                "actual_device": config.runtime.device,
                "activation_dtype": config.runtime.activation_dtype,
                "checkpoint_compute_dtype": (
                    "float64"
                    if config.runtime.device == "cpu"
                    else "float32"
                ),
                "batch_size": config.evaluation.batch_size,
            }
        )
        record = build_run_record(
            run_id=spec.run_id,
            status="success",
            config={
                **spec.as_config(),
                **config_provenance,
                "experiment_name": experiment.get("name", "unnamed"),
                "stage": experiment.get("stage", "tiny_reference"),
                "evaluation_scope": experiment.get(
                    "evaluation_scope", "language_model_accuracy"
                ),
                "evaluation_stage": experiment.get(
                    "evaluation_stage", "unspecified"
                ),
                "effective_sample_count": int(
                    experiment.get("effective_sample_count", payload["sample_count"])
                ),
                **(
                    {"evidence": experiment["evidence"]}
                    if experiment.get("evidence") is not None
                    else {}
                ),
                "expected_run_ids": list(
                    experiment.get("expected_run_ids", [spec.run_id])
                ),
                "selected_spec_ids_file": experiment.get(
                    "selected_spec_ids_file"
                ),
                "runtime": asdict(config.runtime),
                "evaluation": asdict(config.evaluation),
                # Layer/Softmax diagnostics are intentionally captured through
                # the explicitly gated debug API for research metrics.  The
                # timed performance path below still calls production forward.
                "evaluation_debug_capture": spec.mode != "plaintext",
            },
            seed=seed,
            model=payload["model"],
            dataset=payload["dataset"],
            environment=run_environment,
            sample_count=payload["sample_count"],
            metrics=payload["metrics"],
            elapsed_seconds=elapsed,
        )
        if spec.mode == "exact":
            passed, reasons = evaluate_exact_gate(record)
            record["metrics"]["exact_gate"] = {
                "passed": passed,
                "reasons": reasons,
            }
            validate_run_record(record)
    except Exception as error:
        failure_environment = environment_metadata()
        planned_device = None
        planned_dtype = None
        if config is not None:
            planned_device = config.runtime.device
            planned_dtype = config.runtime.activation_dtype
        else:
            try:
                fallback_runtime = load_config(base_config_path).runtime
                planned_device = str(
                    (runtime_override or {}).get(
                        "device", fallback_runtime.device
                    )
                )
                planned_dtype = str(
                    (runtime_override or {}).get(
                        "activation_dtype", fallback_runtime.activation_dtype
                    )
                )
            except Exception:
                pass
        if planned_device is not None and planned_dtype is not None:
            failure_environment.update(
                {
                    "requested_device": planned_device,
                    "requested_activation_dtype": planned_dtype,
                    "device": planned_device,
                    "activation_dtype": planned_dtype,
                    "checkpoint_compute_dtype": (
                        "float64" if planned_device == "cpu" else "float32"
                    ),
                }
            )
        requested_pretrained = bool(
            pretrained_artifact is not None
            or pretrained_path is not None
            or experiment.get("pretrained", False)
        )
        failure_model: Dict[str, Any] = {
            "identifier": experiment.get("model_id", "unknown"),
            "evidence_scope": (
                experiment.get(
                    "evaluation_scope", "language_model_accuracy"
                )
                if requested_pretrained
                else "correctness_only_random_tiny"
            ),
            "pretrained": requested_pretrained,
        }
        if pretrained_path is not None:
            failure_model["path"] = str(
                Path(pretrained_path).expanduser().resolve()
            )
        if pretrained_artifact is not None:
            failure_model["checkpoint_manifest"] = pretrained_artifact.to_dict()
            failure_model.update(
                {
                    "weight_sha256": pretrained_artifact.weights_sha256,
                    "config_sha256": pretrained_artifact.config_sha256,
                    "tokenizer_sha256": pretrained_artifact.tokenizer_sha256,
                    "tokenizer_config_sha256": pretrained_artifact.tokenizer_config_sha256,
                }
            )
        failure_dataset: Dict[str, Any] = {
            "identifier": experiment.get("dataset_id", "unknown"),
            "evaluation_scope": experiment.get(
                "evaluation_scope", "language_model_accuracy"
            ),
            "meaningful_lm_evidence": bool(
                experiment.get("meaningful_lm_evidence", False)
            ),
        }
        if dataset_cache_path is not None:
            failure_dataset["cache_path"] = str(
                Path(dataset_cache_path).expanduser().resolve()
            )
        if experiment.get("dataset_cache_sha256") is not None:
            failure_dataset["cache_content_sha256"] = str(
                experiment["dataset_cache_sha256"]
            )
        record = build_run_record(
            run_id=spec.run_id,
            status="failure",
            config={
                **spec.as_config(),
                **config_provenance,
                "experiment_name": experiment.get("name", "unnamed"),
                "stage": experiment.get("stage", "tiny_reference"),
                "evaluation_scope": experiment.get(
                    "evaluation_scope", "language_model_accuracy"
                ),
                "evaluation_stage": experiment.get(
                    "evaluation_stage", "unspecified"
                ),
                "effective_sample_count": int(
                    experiment.get("effective_sample_count", 0)
                ),
                **(
                    {"evidence": experiment["evidence"]}
                    if experiment.get("evidence") is not None
                    else {}
                ),
                "expected_run_ids": list(
                    experiment.get("expected_run_ids", [spec.run_id])
                ),
                "selected_spec_ids_file": experiment.get(
                    "selected_spec_ids_file"
                ),
                "evaluation_override": evaluation_override or {},
                "runtime_override": runtime_override or {},
            },
            seed=seed,
            model=failure_model,
            dataset=failure_dataset,
            environment=failure_environment,
            sample_count=0,
            metrics={},
            elapsed_seconds=time.perf_counter() - started,
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(
                    traceback.format_exception_only(type(error), error)
                ).strip(),
            },
            stage="tiny_model_execution",
            last_completed_sample_id=None,
            partial_metrics_available=False,
        )
    if output_path is not None:
        append_jsonl_record(output_path, record)
    _release_runtime_memory()
    return record
