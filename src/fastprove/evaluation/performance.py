"""Latency, throughput, memory, and KV-cache measurements."""

from __future__ import annotations

import statistics
import time
from typing import Any, Dict, Optional

import torch

from ..codec import TokenCodec
from ..config import EvaluationConfig
from ..models.obfuscated import ObfuscatedLMCache, ObfuscatedTinyCausalLM
from ..models.plain import PlainTinyCausalLM
from ..seed import RequestContext
from .runtime import _mps_allocated_bytes, _process_peak_rss_bytes, _sync


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
    token_codec: Optional[TokenCodec] = None,
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
        if token_codec is None:
            raise ValueError("obfuscated performance needs the token codec")
        encoded = token_codec.encode(batch).to(device=device)
        with torch.inference_mode():
            _, measured_cache = obfuscated(
                encoded,
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
    token_codec: TokenCodec,
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
    encoded = token_codec.encode(batch).to(device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
    obfuscated_prefill = _time_call(
        lambda: obfuscated(
            encoded,
            token_mask=mask,
            request_context=context,
        ),
        device=device,
        warmup_runs=evaluation.warmup_runs,
        timed_runs=evaluation.timed_runs,
    )
    obfuscated_generated_call = lambda: obfuscated.generate_greedy(
        encoded,
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
            encoded,
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
