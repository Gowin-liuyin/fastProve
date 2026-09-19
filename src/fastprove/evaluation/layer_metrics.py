"""Layer-level attention and Softmax error metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch

from ..models.obfuscated import ObfuscatedBlockDebug
from .metrics import attention_distribution_metrics


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
                record.z_prime,
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
                z_prime=record.z_prime.detach().cpu(),
                final_noise_state=record.final_noise_state.detach().cpu(),
            )
        )
    return tuple(cpu_batches)
