"""Aligned teacher-forced and greedy accuracy helpers."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..seed import make_generator


def make_synthetic_token_batch(
    *,
    sample_count: int,
    sequence_length: int,
    vocab_size: int,
    seed: int,
) -> Tuple[torch.Tensor, List[str]]:
    """Create deterministic correctness-only token inputs and stable IDs."""

    if sample_count <= 0 or sequence_length <= 1 or vocab_size <= 1:
        raise ValueError("synthetic batch dimensions are invalid")
    generator = make_generator(
        seed, "synthetic-token-batch", sample_count, sequence_length, vocab_size
    )
    tokens = torch.randint(
        0,
        vocab_size,
        (sample_count, sequence_length),
        generator=generator,
        dtype=torch.int64,
    )
    identifiers = [
        "synthetic-%d-%04d" % (seed, index)
        for index in range(sample_count)
    ]
    return tokens, identifiers


def _single_model_metrics(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    target_valid: torch.Tensor,
) -> Dict[str, float]:
    prediction_logits = logits[:, :-1].float()
    labels = input_ids[:, 1:]
    flat_logits = prediction_logits[target_valid]
    flat_labels = labels[target_valid]
    if flat_labels.numel() == 0:
        raise ValueError("teacher-forced evaluation has no valid targets")
    nll = F.cross_entropy(flat_logits, flat_labels, reduction="mean")
    top1 = flat_logits.argmax(dim=-1)
    top1_accuracy = (top1 == flat_labels).float().mean()
    top_count = min(5, flat_logits.shape[-1])
    top5 = torch.topk(flat_logits, top_count, dim=-1).indices
    top5_accuracy = (top5 == flat_labels[:, None]).any(dim=-1).float().mean()
    return {
        "negative_log_likelihood": float(nll.item()),
        "perplexity": float(math.exp(float(nll.item()))),
        "next_token_top1_accuracy": float(top1_accuracy.item()),
        "next_token_top5_accuracy": float(top5_accuracy.item()),
    }


def _relative_drop(reference: float, actual: float) -> Optional[float]:
    if reference == 0:
        return None
    return (reference - actual) / reference


def _sample_level_metrics(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    target_valid: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return per-sample statistics for sample-unit bootstrap resampling."""

    prediction_logits = logits[:, :-1].float()
    labels = input_ids[:, 1:]
    if target_valid.ndim != 2 or target_valid.shape != labels.shape:
        raise ValueError("target_valid must match next-token label shape")
    counts = target_valid.sum(dim=1)
    if bool(torch.any(counts == 0)):
        raise ValueError(
            "every sample must contain an adjacent valid next-token pair"
        )
    log_probabilities = F.log_softmax(prediction_logits, dim=-1)
    token_nll = -log_probabilities.gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    top1 = prediction_logits.argmax(dim=-1)
    top5 = torch.topk(
        prediction_logits, min(5, prediction_logits.shape[-1]), dim=-1
    ).indices
    valid_float = target_valid.to(dtype=torch.float32)
    denominator = counts.to(dtype=torch.float32)
    return {
        "nll": (token_nll * valid_float).sum(dim=1) / denominator,
        "top1": (
            ((top1 == labels) & target_valid).to(dtype=torch.float32).sum(dim=1)
            / denominator
        ),
        "top5": (
            (
                ((top5 == labels.unsqueeze(-1)).any(dim=-1) & target_valid)
                .to(dtype=torch.float32)
                .sum(dim=1)
            )
            / denominator
        ),
    }


def _bootstrap_interval(values: torch.Tensor) -> Dict[str, float]:
    """Return a deterministic percentile-95 interval from finite samples."""

    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("bootstrap values must be a non-empty vector")
    finite = values.float()
    if not bool(torch.isfinite(finite).all()):
        raise ValueError("bootstrap values must be finite")
    quantiles = torch.quantile(finite, torch.tensor([0.025, 0.975]))
    return {
        "low": float(quantiles[0].item()),
        "high": float(quantiles[1].item()),
    }


def _paired_sample_bootstrap(
    *,
    plaintext_logits: torch.Tensor,
    obfuscated_logits: torch.Tensor,
    input_ids: torch.Tensor,
    target_valid: torch.Tensor,
    replicates: int,
    seed: int,
) -> Dict[str, object]:
    """Compute paired sample-unit 95% bootstrap intervals.

    Token-level NLL/accuracy point estimates remain token-weighted in the main
    metrics.  Bootstrap resampling samples whole examples, so repeated tokens
    inside one sequence are never treated as independent observations.
    """

    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    plain = _sample_level_metrics(plaintext_logits, input_ids, target_valid)
    obfuscated = _sample_level_metrics(
        obfuscated_logits, input_ids, target_valid
    )
    sample_count = int(input_ids.shape[0])
    generator = make_generator(seed, "teacher-forced-bootstrap")
    indices = torch.randint(
        0,
        sample_count,
        (replicates, sample_count),
        generator=generator,
        dtype=torch.int64,
    )
    plain_nll = plain["nll"][indices].mean(dim=1)
    obfuscated_nll = obfuscated["nll"][indices].mean(dim=1)
    plain_top1 = plain["top1"][indices].mean(dim=1)
    obfuscated_top1 = obfuscated["top1"][indices].mean(dim=1)
    plain_top5 = plain["top5"][indices].mean(dim=1)
    obfuscated_top5 = obfuscated["top5"][indices].mean(dim=1)
    agreement = (
        (
            plaintext_logits[:, :-1].argmax(dim=-1)
            == obfuscated_logits[:, :-1].argmax(dim=-1)
        )
        & target_valid
    ).to(dtype=torch.float32)
    agreement_per_sample = agreement.sum(dim=1) / target_valid.sum(dim=1)
    agreement_bootstrap = agreement_per_sample[indices].mean(dim=1)
    plain_ppl = torch.exp(plain_nll)
    obfuscated_ppl = torch.exp(obfuscated_nll)
    ppl_increase = obfuscated_ppl - plain_ppl
    ppl_relative = ppl_increase / plain_ppl
    return {
        "unit": "sample",
        "replicates": int(replicates),
        "seed": int(seed),
        "confidence_level": 0.95,
        "metrics": {
            "nll_absolute_increase": _bootstrap_interval(
                obfuscated_nll - plain_nll
            ),
            "perplexity_absolute_increase": _bootstrap_interval(ppl_increase),
            "perplexity_relative_increase": _bootstrap_interval(ppl_relative),
            "top1_absolute_drop": _bootstrap_interval(
                plain_top1 - obfuscated_top1
            ),
            "top5_absolute_drop": _bootstrap_interval(
                plain_top5 - obfuscated_top5
            ),
            "next_token_top1_agreement": _bootstrap_interval(
                agreement_bootstrap
            ),
        },
    }


def compare_teacher_forced_metrics(
    *,
    plaintext_logits: torch.Tensor,
    obfuscated_logits: torch.Tensor,
    input_ids: torch.Tensor,
    token_mask: torch.Tensor,
    bootstrap_replicates: int = 0,
    bootstrap_seed: int = 0,
) -> Dict[str, object]:
    """Compare aligned next-token metrics with absolute/relative degradation."""

    if plaintext_logits.shape != obfuscated_logits.shape:
        raise ValueError("plaintext and obfuscated logits must align")
    if plaintext_logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and token IDs must align")
    if token_mask.shape != input_ids.shape or token_mask.dtype != torch.bool:
        raise ValueError("token_mask must be boolean and match input IDs")
    target_valid = token_mask[:, :-1] & token_mask[:, 1:]
    if not bool(target_valid.any()):
        raise ValueError(
            "token_mask must contain at least one adjacent valid next-token pair"
        )
    plain = _single_model_metrics(
        plaintext_logits, input_ids, target_valid
    )
    obfuscated = _single_model_metrics(
        obfuscated_logits, input_ids, target_valid
    )
    plain_predictions = plaintext_logits[:, :-1].argmax(dim=-1)
    obfuscated_predictions = obfuscated_logits[:, :-1].argmax(dim=-1)
    agreement = (
        plain_predictions[target_valid]
        == obfuscated_predictions[target_valid]
    ).float().mean()
    perplexity_increase = (
        obfuscated["perplexity"] - plain["perplexity"]
    )
    nll_increase = (
        obfuscated["negative_log_likelihood"]
        - plain["negative_log_likelihood"]
    )
    nll_relative = (
        nll_increase / plain["negative_log_likelihood"]
        if plain["negative_log_likelihood"] != 0
        else None
    )
    perplexity_relative = (
        perplexity_increase / plain["perplexity"]
        if plain["perplexity"] != 0
        else math.inf
    )
    result: Dict[str, object] = {
        "plaintext": plain,
        "obfuscated": obfuscated,
        "agreement": {
            "next_token_top1_agreement": float(agreement.item())
        },
        "degradation": {
            "nll_absolute_increase": nll_increase,
            "nll_relative_increase": nll_relative,
            "perplexity_absolute_increase": perplexity_increase,
            "perplexity_relative_increase": perplexity_relative,
            "top1_absolute_drop": plain["next_token_top1_accuracy"]
            - obfuscated["next_token_top1_accuracy"],
            "top1_relative_drop": _relative_drop(
                plain["next_token_top1_accuracy"],
                obfuscated["next_token_top1_accuracy"],
            ),
            "top5_absolute_drop": plain["next_token_top5_accuracy"]
            - obfuscated["next_token_top5_accuracy"],
            "top5_relative_drop": _relative_drop(
                plain["next_token_top5_accuracy"],
                obfuscated["next_token_top5_accuracy"],
            ),
        },
        "token_count": int(target_valid.sum().item()),
    }
    if bootstrap_replicates:
        result["bootstrap"] = _paired_sample_bootstrap(
            plaintext_logits=plaintext_logits,
            obfuscated_logits=obfuscated_logits,
            input_ids=input_ids,
            target_valid=target_valid,
            replicates=int(bootstrap_replicates),
            seed=int(bootstrap_seed),
        )
    elif bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be non-negative")
    return result


def greedy_generation_metrics(
    *,
    plaintext_tokens: torch.Tensor,
    obfuscated_tokens: torch.Tensor,
    prompt_length: int,
) -> Dict[str, float]:
    """Compare aligned generated suffix tokens and complete sequences."""

    if plaintext_tokens.shape != obfuscated_tokens.shape:
        raise ValueError("generated token tensors must align")
    if not 0 <= prompt_length < plaintext_tokens.shape[1]:
        raise ValueError("prompt_length must leave a generated suffix")
    plain_suffix = plaintext_tokens[:, prompt_length:]
    obfuscated_suffix = obfuscated_tokens[:, prompt_length:]
    token_match = (plain_suffix == obfuscated_suffix).float().mean()
    sequence_match = (plain_suffix == obfuscated_suffix).all(dim=-1)
    return {
        "greedy_token_exact_match": float(token_match.item()),
        "greedy_sequence_exact_match": float(
            sequence_match.float().mean().item()
        ),
    }
