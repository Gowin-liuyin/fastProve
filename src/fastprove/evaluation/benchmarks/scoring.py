"""Likelihood scoring and evaluation loops for OSNIP-style benchmarks."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ...seed import RequestContext
from ..stats import wilson_interval
from .tasks import MCExample, apply_chat_template_if_available


def _wilson_interval(correct: int, n: int) -> Tuple[float, float]:
    """Wilson 95% interval bounds.

    Single implementation: ``fastprove.evaluation.stats.wilson_interval``
    (docs/PROBLEMS.md §3.4). The ``n <= 0`` sentinel is kept because the
    caller may pass an empty task record.
    """

    if n <= 0:
        return (float("nan"), float("nan"))
    estimate = wilson_interval(correct, n, level=0.95)
    return (estimate.ci_low, estimate.ci_high)


def _annotate_chance_floor(res: Dict[str, Any]) -> Dict[str, Any]:
    """Mark tasks whose score is statistically indistinguishable from chance.

    A task sitting on its chance floor cannot register degradation: its
    retention ratio is pinned near 100% regardless of what the obfuscation
    does.  Averaging such tasks into a headline retention number inflates it,
    so the flag is recorded per task and the summary reports the average both
    with and without them.
    """

    n = int(res.get("n") or 0)
    correct = int(res.get("correct") or 0)
    if n <= 0 or res.get("accuracy") is None:
        return res
    n_choices = int(res.get("n_choices_max") or 0)
    chance = 1.0 / n_choices if n_choices > 0 else None
    lo, hi = _wilson_interval(correct, n)
    res["accuracy_ci95"] = [lo, hi]
    res["chance_level"] = chance
    if chance is not None:
        # "At chance" = the 95% interval does not exclude the floor.
        res["at_chance_level"] = bool(lo <= chance)
    return res


def _degenerate_label_reason(examples: Sequence[MCExample]) -> Optional[str]:
    """Return a reason string when gold labels carry no discriminative signal.

    A corpus whose gold labels are constant cannot measure accuracy: the score
    it produces is just the rate at which the model happens to pick that fixed
    option, and it is identical across plaintext and obfuscated modes, so it
    contributes a hard-coded 100% retention to any average.  This is exactly
    the state ``piqa_val.jsonl`` is in locally (all 1838 rows have label 0),
    which is why the check is applied to every task rather than one.
    """

    if not examples:
        return None
    distinct = {int(ex.label) for ex in examples}
    if len(distinct) <= 1:
        return (
            "degenerate_labels: all %d gold labels equal %d; corpus cannot "
            "measure accuracy and must be rebuilt from an authoritative source"
            % (len(examples), next(iter(distinct)))
        )
    n_choices = {len(ex.choices) for ex in examples}
    bad = [ex for ex in examples if not 0 <= int(ex.label) < len(ex.choices)]
    if bad:
        return (
            "label_out_of_range: %d/%d rows have a gold label outside their "
            "choice list (choice counts seen: %s)"
            % (len(bad), len(examples), sorted(n_choices))
        )
    return None


# ---------------------------------------------------------------------------
# Likelihood scoring
# ---------------------------------------------------------------------------


def _encode_pair(
    tokenizer,
    context: str,
    choice: str,
    *,
    max_length: int,
) -> Tuple[List[int], int]:
    """Return full token ids and the index where choice tokens begin.

    Choice scoring uses tokens [ctx_len, full_len) predicted from previous
    positions. Context keeps the tokenizer BOS if present; choice does not
    add an extra BOS.
    """

    # Encode full string once for consistent BPE merges at the boundary.
    full_ids = tokenizer.encode(context + choice, add_special_tokens=True)
    ctx_ids = tokenizer.encode(context, add_special_tokens=True)
    # Truncate from the left on context if needed, keeping the end + choice.
    if len(full_ids) > max_length:
        # Keep as much of the choice as possible.
        choice_ids_approx = full_ids[len(ctx_ids) :]
        keep_choice = min(len(choice_ids_approx), max(1, max_length // 4))
        keep_ctx = max_length - keep_choice
        # Re-slice from the full sequence: last keep_ctx of context + choice head.
        ctx_part = full_ids[: len(ctx_ids)][-keep_ctx:]
        choice_part = full_ids[len(ctx_ids) :][:keep_choice]
        full_ids = ctx_part + choice_part
        ctx_len = len(ctx_part)
    else:
        ctx_len = len(ctx_ids)
        # If encode(context+choice) diverges from encode(context)+encode(choice)
        # due to BPE, prefer boundary from length of context encoding capped.
        if ctx_len > len(full_ids):
            ctx_len = max(1, len(full_ids) - 1)
    if ctx_len < 1:
        ctx_len = 1
    if ctx_len >= len(full_ids):
        # Degenerate: no choice tokens — force at least last token as target.
        ctx_len = max(1, len(full_ids) - 1)
    return full_ids, ctx_len


@torch.no_grad()
def score_continuation_logprob(
    *,
    model: Any,
    tokenizer,
    context: str,
    choice: str,
    device: torch.device,
    max_length: int,
    request_context: Optional[RequestContext],
    is_obfuscated: bool,
) -> Tuple[float, int]:
    """Return (sum log-prob of choice tokens, number of scored tokens).

    The caller decides whether to compare raw sums (``acc``) or per-token means
    (``acc_norm``).  Unnormalized sums systematically favour short options, so
    tasks whose candidates differ in length (PIQA, HellaSwag, ARC) need the
    normalized variant to be comparable with published numbers.
    """

    full_ids, ctx_len = _encode_pair(
        tokenizer, context, choice, max_length=max_length
    )
    tokens = torch.tensor([full_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    if is_obfuscated:
        if request_context is None:
            raise ValueError("obfuscated scoring requires request_context")
        logits = model(
            tokens, token_mask=mask, request_context=request_context
        )
    else:
        logits = model(tokens, token_mask=mask)
    # logits[t] predicts token t+1
    log_probs = F.log_softmax(logits[0].float(), dim=-1)
    total = 0.0
    n = 0
    for pos in range(ctx_len - 1, len(full_ids) - 1):
        target = full_ids[pos + 1]
        total += float(log_probs[pos, target].item())
        n += 1
    if n == 0:
        return float("-inf"), 0
    return total, n


@torch.no_grad()
def evaluate_mc_task(
    *,
    model: Any,
    tokenizer,
    examples: Sequence[MCExample],
    device: torch.device,
    max_length: int,
    request_context: Optional[RequestContext],
    is_obfuscated: bool,
    task_name: str,
    length_normalize: bool = True,
    use_chat_template: bool = True,
    progress_every: int = 50,
) -> Dict[str, Any]:
    if not examples:
        return {
            "task": task_name,
            "n": 0,
            "accuracy": None,
            "skipped": True,
            "reason": "no_examples",
        }
    correct = 0
    correct_raw = 0
    correct_norm = 0
    n = len(examples)
    preds: List[int] = []
    t0 = time.perf_counter()
    letter_count = sum(1 for ex in examples if ex.scoring_mode == "letter")
    for i, ex in enumerate(examples):
        # Instruct chat template wraps the stem; choices are scored as
        # continuations of the assistant turn (generation prompt).
        context = apply_chat_template_if_available(
            tokenizer, ex.context, enabled=use_chat_template
        )
        # Letter MC / fixed-label NLI: do not length-normalize (letters are
        # equal length; NLI labels like "entailment" vs "neutral" are not).
        use_norm = length_normalize and ex.scoring_mode not in (
            "letter",
            "continuation_raw",
        )
        sums: List[float] = []
        means: List[float] = []
        for c in ex.choices:
            total, n_tok = score_continuation_logprob(
                model=model,
                tokenizer=tokenizer,
                context=context,
                choice=c,
                device=device,
                max_length=max_length,
                request_context=request_context,
                is_obfuscated=is_obfuscated,
            )
            sums.append(total)
            means.append(total / n_tok if n_tok > 0 else float("-inf"))
        pred_raw = int(max(range(len(sums)), key=lambda j: sums[j]))
        pred_norm = int(max(range(len(means)), key=lambda j: means[j]))
        pred = pred_norm if use_norm else pred_raw
        preds.append(pred)
        gold = int(ex.label)
        if pred_raw == gold:
            correct_raw += 1
        if pred_norm == gold:
            correct_norm += 1
        if pred == gold:
            correct += 1
        if (i + 1) % progress_every == 0 or (i + 1) == n:
            print(
                "    %s %d/%d acc=%.4f (raw=%.4f norm=%.4f)"
                % (
                    task_name,
                    i + 1,
                    n,
                    correct / (i + 1),
                    correct_raw / (i + 1),
                    correct_norm / (i + 1),
                ),
                flush=True,
            )
    acc = correct / n
    raw_count = sum(1 for ex in examples if ex.scoring_mode == "continuation_raw")
    if letter_count == n:
        primary_rule = "letter"
    elif raw_count == n:
        primary_rule = "acc"
    else:
        primary_rule = "acc_norm" if length_normalize else "acc"
    return {
        "task": task_name,
        "n": n,
        "correct": correct,
        "accuracy": acc,
        # Both scoring rules are always recorded so that switching the primary
        # rule is visible in the artifact rather than a silent change of
        # measurement.  "acc" = argmax of summed logprob (length-biased),
        # "acc_norm" = argmax of per-token mean logprob (lm-eval-harness style),
        # "letter" = single-letter labels after options listed in stem.
        "scoring_rule": primary_rule,
        "accuracy_raw_sum": correct_raw / n,
        "accuracy_length_normalized": correct_norm / n,
        "n_letter_examples": letter_count,
        "use_chat_template": bool(use_chat_template),
        "n_choices_min": min(len(ex.choices) for ex in examples),
        "n_choices_max": max(len(ex.choices) for ex in examples),
        "seconds": time.perf_counter() - t0,
        "skipped": False,
    }


@torch.no_grad()
def evaluate_wikitext_ppl(
    *,
    model: Any,
    tokenizer,
    text_path: Path,
    device: torch.device,
    max_length: int,
    stride: int,
    max_tokens: Optional[int],
    request_context: Optional[RequestContext],
    is_obfuscated: bool,
) -> Dict[str, Any]:
    if not text_path.is_file():
        return {
            "task": "wikitext2",
            "n_tokens": 0,
            "perplexity": None,
            "skipped": True,
            "reason": "missing_file",
        }
    text = text_path.read_text(encoding="utf-8")
    # Tokenize only what we need. Full WikiText-2 test is > tokenizer max length
    # if encoded at once; a character budget keeps encode() bounded.
    if max_tokens is not None:
        # ~4 chars/token upper bound for English; add margin.
        text = text[: max(4096, int(max_tokens) * 8)]
    all_ids = tokenizer.encode(text, add_special_tokens=True)
    if max_tokens is not None and len(all_ids) > max_tokens:
        all_ids = all_ids[:max_tokens]
    if len(all_ids) < 2:
        return {
            "task": "wikitext2",
            "n_tokens": 0,
            "perplexity": None,
            "skipped": True,
            "reason": "too_short",
        }

    nll_sum = 0.0
    n_tok = 0
    t0 = time.perf_counter()
    # Sliding window (HF-style).
    for begin in range(0, len(all_ids) - 1, stride):
        end = min(begin + max_length, len(all_ids))
        chunk = all_ids[begin:end]
        if len(chunk) < 2:
            break
        # Tokens to score: for non-first window, only the new tail after overlap.
        if begin == 0:
            score_from = 0
        else:
            score_from = max(0, len(chunk) - stride)
        tokens = torch.tensor([chunk], dtype=torch.long, device=device)
        mask = torch.ones_like(tokens, dtype=torch.bool)
        if is_obfuscated:
            logits = model(
                tokens, token_mask=mask, request_context=request_context
            )
        else:
            logits = model(tokens, token_mask=mask)
        log_probs = F.log_softmax(logits[0].float(), dim=-1)
        for pos in range(score_from, len(chunk) - 1):
            target = chunk[pos + 1]
            nll_sum += -float(log_probs[pos, target].item())
            n_tok += 1
        if end >= len(all_ids):
            break
        if n_tok > 0 and (begin // stride) % 20 == 0:
            print(
                "    wikitext tokens_scored=%d running_ppl=%.3f"
                % (n_tok, math.exp(min(nll_sum / n_tok, 80.0))),
                flush=True,
            )

    mean_nll = nll_sum / max(n_tok, 1)
    ppl = float(math.exp(min(mean_nll, 80.0)))
    return {
        "task": "wikitext2",
        "n_tokens": n_tok,
        "negative_log_likelihood": mean_nll,
        "perplexity": ppl,
        "max_length": max_length,
        "stride": stride,
        "seconds": time.perf_counter() - t0,
        "skipped": False,
    }
