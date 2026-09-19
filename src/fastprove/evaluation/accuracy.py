"""Aligned teacher-forced and greedy accuracy helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import torch
import torch.nn.functional as F

from ..seed import RequestContext, make_generator

if TYPE_CHECKING:
    from ..codec import TokenCodec
    from ..models.obfuscated import ObfuscatedTinyCausalLM
    from ..models.plain import PlainTinyCausalLM


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


def _build_token_batch(
    *,
    model_path: Optional[Path],
    sample_count: int,
    sequence_length: int,
    vocab_size: int,
    seed: int,
    prompts: Optional[Sequence[str]],
    sample_id_prefix: str = "scenario",
) -> tuple[torch.Tensor, List[str], torch.Tensor]:
    from ..pretrained.llama import load_llama_tokenizer

    if model_path is not None and prompts:
        tokenizer = load_llama_tokenizer(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        texts = list(prompts[:sample_count])
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=sequence_length,
            padding="max_length",
        )
        tokens = encoded["input_ids"].to(dtype=torch.long)
        if "attention_mask" in encoded:
            mask = encoded["attention_mask"].to(dtype=torch.bool)
        else:
            pad_id = int(tokenizer.pad_token_id)
            mask = tokens.ne(pad_id)
        ids = ["%s-%05d" % (sample_id_prefix, index) for index in range(tokens.shape[0])]
        return tokens, ids, mask

    tokens, ids = make_synthetic_token_batch(
        sample_count=sample_count,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        seed=seed,
    )
    mask = torch.ones_like(tokens, dtype=torch.bool)
    return tokens, ids, mask


@torch.no_grad()
def greedy_unpadded_pair(
    *,
    plain: PlainTinyCausalLM,
    obfuscated: ObfuscatedTinyCausalLM,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    sample_ids: Sequence[str],
    generation_tokens: int,
    request_context: RequestContext,
    codec: Optional[TokenCodec] = None,
) -> Dict[str, Any]:
    """Greedy continuation on **every** sample after stripping pad tokens.

    ``generate_greedy`` always reads ``logits[:, -1]``. Right-padded prompts
    would therefore decode from a pad position. This helper feeds only the
    valid (non-pad) prefix per row so the last index is the last real token.
    All ``sample_ids`` are evaluated (not a batch_size slice).

    When ``codec`` is given, the obfuscated model runs in the obfuscated
    token domain: inputs are encoded and generated tokens are decoded back.
    """

    if tokens.shape[0] != len(sample_ids):
        raise ValueError("sample_ids length must match batch dimension")
    if tokens.shape != mask.shape:
        raise ValueError("tokens and mask shapes must match")

    seq_matches: List[float] = []
    token_matches: List[float] = []
    prompt_lengths: List[int] = []
    first_div: List[Optional[int]] = []
    evaluated_ids: List[str] = []

    n_total = tokens.shape[0]
    progress_every = max(1, n_total // 20)
    for row in range(n_total):
        valid = mask[row]
        if not bool(valid.any().item()):
            raise ValueError("sample %s has no valid tokens" % sample_ids[row])
        prompt = tokens[row, valid].unsqueeze(0)
        prompt_len = int(prompt.shape[1])
        if prompt_len + generation_tokens > plain.config.max_sequence_length:
            raise ValueError(
                "prompt+generation exceeds context for sample %s" % sample_ids[row]
            )
        plain_gen = plain.generate_greedy(
            prompt, max_new_tokens=generation_tokens
        )
        enc_prompt = (
            prompt if codec is None else codec.encode(prompt).to(prompt.device)
        )
        obf_gen = obfuscated.generate_greedy(
            enc_prompt,
            max_new_tokens=generation_tokens,
            request_context=request_context,
        )
        if codec is not None:
            obf_gen = codec.decode(obf_gen)
        plain_suf = plain_gen[0, prompt_len:]
        obf_suf = obf_gen[0, prompt_len:]
        if plain_suf.numel() != generation_tokens or obf_suf.numel() != generation_tokens:
            raise RuntimeError("generation length mismatch for %s" % sample_ids[row])
        seq_eq = bool(torch.equal(plain_suf, obf_suf))
        seq_matches.append(1.0 if seq_eq else 0.0)
        token_matches.append(float((plain_suf == obf_suf).float().mean().item()))
        prompt_lengths.append(prompt_len)
        evaluated_ids.append(str(sample_ids[row]))
        if seq_eq:
            first_div.append(None)
        else:
            diff = plain_suf != obf_suf
            first_div.append(int(torch.nonzero(diff, as_tuple=False)[0].item()))
        if (row + 1) % progress_every == 0 or (row + 1) == n_total:
            print(
                "    greedy progress %d/%d  running_seq_match=%.4f"
                % (
                    row + 1,
                    n_total,
                    float(sum(seq_matches) / len(seq_matches)),
                ),
                flush=True,
            )

    n = len(seq_matches)
    return {
        "greedy_sequence_exact_match": float(sum(seq_matches) / n),
        "greedy_token_exact_match": float(sum(token_matches) / n),
        "greedy_n_samples": n,
        "greedy_sample_ids": evaluated_ids,
        "greedy_prompt_lengths": prompt_lengths,
        "greedy_first_divergence_step": first_div,
        "greedy_padding_stripped": True,
    }


@torch.no_grad()
def _run_pair(
    *,
    plain: PlainTinyCausalLM,
    obfuscated: ObfuscatedTinyCausalLM,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    sample_ids: Sequence[str],
    mode_name: str,
    key_meta: Dict[str, Any],
    conversion_seconds: float,
    structural_noise_zeroed: bool,
    request_seed: int,
    generation_tokens: int,
    batch_size: int,
    device: torch.device,
    run_greedy: bool = True,
    codec: Optional[TokenCodec] = None,
) -> Dict[str, Any]:
    """Stream teacher-forced metrics per batch (no full-logit materialization).

    Storing 1500×64×128k logits twice exceeds host RAM; accumulate agreement,
    NLL, and max logit error online instead.

    When ``codec`` is given, the obfuscated model runs in the obfuscated
    token domain: inputs are encoded and logit columns are decoded before any
    comparison with the plaintext reference.
    """

    ctx = RequestContext(request_seed, "pretrained-compare-%s" % mode_name)
    n_batches = (tokens.shape[0] + batch_size - 1) // batch_size
    n_agree = 0
    n_valid = 0
    sum_nll_plain = 0.0
    sum_nll_obf = 0.0
    n_ce = 0
    top1_plain_hits = 0
    top1_obf_hits = 0
    e2e_max_abs = 0.0
    e2e_sum_abs = 0.0
    e2e_count = 0

    for bi, start in enumerate(range(0, tokens.shape[0], batch_size)):
        stop = min(start + batch_size, tokens.shape[0])
        batch = tokens[start:stop]
        batch_mask = mask[start:stop]
        plain_logits = plain(batch, token_mask=batch_mask).float()
        enc_batch = (
            batch if codec is None else codec.encode(batch).to(device)
        )
        obf_logits = obfuscated(
            enc_batch, token_mask=batch_mask, request_context=ctx
        ).float()
        if codec is not None:
            obf_logits = obf_logits[..., codec.permutation.to(obf_logits.device)]

        # Next-token targets on positions 0..T-2 predicting 1..T-1.
        valid = batch_mask[:, :-1] & batch_mask[:, 1:]
        if valid.any():
            p_pred = plain_logits[:, :-1]
            o_pred = obf_logits[:, :-1]
            labels = batch[:, 1:]
            # Agreement plain vs obf.
            agree = (p_pred.argmax(dim=-1) == o_pred.argmax(dim=-1)) & valid
            n_agree += int(agree.sum().item())
            n_valid += int(valid.sum().item())
            # Accuracy vs labels.
            top1_plain_hits += int(
                ((p_pred.argmax(dim=-1) == labels) & valid).sum().item()
            )
            top1_obf_hits += int(
                ((o_pred.argmax(dim=-1) == labels) & valid).sum().item()
            )
            # NLL (mean later over valid positions).
            flat_p = p_pred[valid]
            flat_o = o_pred[valid]
            flat_y = labels[valid]
            sum_nll_plain += float(
                F.cross_entropy(flat_p, flat_y, reduction="sum").item()
            )
            sum_nll_obf += float(
                F.cross_entropy(flat_o, flat_y, reduction="sum").item()
            )
            n_ce += int(flat_y.numel())
            # E2E logit error on valid next-token positions only.
            diff = (flat_p - flat_o).abs()
            e2e_max_abs = max(e2e_max_abs, float(diff.max().item()))
            e2e_sum_abs += float(diff.sum().item())
            e2e_count += int(diff.numel())

        del plain_logits, obf_logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (bi + 1) % max(1, n_batches // 10) == 0 or (bi + 1) == n_batches:
            print(
                "    teacher-forced batch %d/%d  running_agree=%.4f"
                % (
                    bi + 1,
                    n_batches,
                    (n_agree / max(n_valid, 1)),
                ),
                flush=True,
            )

    nll_plain = sum_nll_plain / max(n_ce, 1)
    nll_obf = sum_nll_obf / max(n_ce, 1)
    import math

    from evals.stats import relative_increase, wilson_interval

    ppl_plain = float(math.exp(min(nll_plain, 80.0)))
    ppl_obf = float(math.exp(min(nll_obf, 80.0)))
    ppl_rel = relative_increase(ppl_plain, ppl_obf)
    top1_plain = top1_plain_hits / max(n_ce, 1)
    top1_obf = top1_obf_hits / max(n_ce, 1)
    top1_drop = (top1_plain - top1_obf) * 100.0
    agree_rate = n_agree / max(n_valid, 1)
    wilson = wilson_interval(n_agree, max(n_valid, 1), level=0.95)
    e2e_mean_abs = e2e_sum_abs / max(e2e_count, 1)
    e2e_logit_err = {
        "max_absolute_error": e2e_max_abs,
        "mean_absolute_error": e2e_mean_abs,
        "n_elements": e2e_count,
        "note": "streamed over valid next-token positions only",
    }
    tf = {
        "plaintext": {
            "negative_log_likelihood": nll_plain,
            "perplexity": ppl_plain,
            "next_token_top1_accuracy": top1_plain,
        },
        "obfuscated": {
            "negative_log_likelihood": nll_obf,
            "perplexity": ppl_obf,
            "next_token_top1_accuracy": top1_obf,
        },
        "agreement": {"next_token_top1_agreement": agree_rate},
        "degradation": {
            "perplexity_relative_increase": ppl_rel,
            "top1_absolute_drop": top1_plain - top1_obf,
        },
        "token_count": n_ce,
        "streaming": True,
    }

    if run_greedy and generation_tokens > 0:
        greedy = greedy_unpadded_pair(
            plain=plain,
            obfuscated=obfuscated,
            tokens=tokens,
            mask=mask,
            sample_ids=sample_ids,
            generation_tokens=generation_tokens,
            request_context=ctx,
            codec=codec,
        )
        if greedy["greedy_n_samples"] != len(sample_ids):
            raise RuntimeError(
                "greedy evaluated %d samples but sample_ids has %d"
                % (greedy["greedy_n_samples"], len(sample_ids))
            )
        if greedy["greedy_sample_ids"] != [str(x) for x in sample_ids]:
            raise RuntimeError("greedy sample_ids order/content mismatch")
        # Shrink per-pair payload for large runs (ids already in evaluation).
        if len(sample_ids) > 32:
            greedy = {
                **greedy,
                "greedy_sample_ids": [
                    greedy["greedy_sample_ids"][0],
                    "...",
                    greedy["greedy_sample_ids"][-1],
                ],
                "greedy_sample_ids_truncated": True,
                "greedy_first_divergence_step": None,
            }
    else:
        greedy = {
            "greedy_sequence_exact_match": None,
            "greedy_token_exact_match": None,
            "greedy_n_samples": 0,
            "greedy_sample_ids": [],
            "greedy_prompt_lengths": [],
            "greedy_first_divergence_step": [],
            "greedy_padding_stripped": True,
            "skipped": True,
        }

    return {
        "mode": mode_name,
        "legacy_modelsplit_obfuscation": False,
        "scheme": "current_fastprove_covariant",
        "structural_noise_zeroed": structural_noise_zeroed,
        "key": key_meta,
        "conversion_time_seconds": conversion_seconds,
        # Avoid duplicating 1500 ids in every pair; full list lives under evaluation.
        "sample_ids": list(sample_ids) if len(sample_ids) <= 32 else None,
        "sample_id_count": len(sample_ids),
        "teacher_forced": tf,
        "greedy": greedy,
        "logits_e2e": e2e_logit_err,
        "e2e_logit_max_absolute_error": float(e2e_max_abs),
        "top1_token_agreement": agree_rate,
        "top1_token_agreement_ci95": wilson.to_dict(),
        "lm_head_argmax_match": agree_rate,
        "greedy_sequence_exact_match": (
            None
            if greedy.get("greedy_sequence_exact_match") is None
            else float(greedy["greedy_sequence_exact_match"])
        ),
        "greedy_n_samples": int(greedy["greedy_n_samples"]),
        "ppl_relative_increase": (
            float(ppl_rel) if ppl_rel is not None else None
        ),
        "top1_absolute_drop_pp": float(top1_drop),
        "n_valid_token_decisions": n_valid,
    }
