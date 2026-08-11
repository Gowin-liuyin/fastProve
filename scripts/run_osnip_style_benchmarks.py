#!/usr/bin/env python3
"""OSNIP-style utility benchmarks for plain vs fastProve (structural/full).

Tasks (aligned with OSNIP Table 1 + WikiText-2 PPL):
  closed-ended accuracy via log-likelihood MC:
    ARC-Easy, PIQA, MNLI, SST-2, ANLI-R1/R2/R3, WiC, HellaSwag, MMLU
  language modeling:
    WikiText-2 test perplexity

Does **not** run privacy ASR / KNN attacks (out of current scope).
Summarization (CNN/DM ROUGE) is optional and disabled by default.

Examples
--------
PYTHONPATH=src:. python scripts/run_osnip_style_benchmarks.py \\
  --model-path /path/to/Llama-3.2-3B-Instruct \\
  --data-dir results/raw/osnip_bench_data \\
  --device cuda --dtype bfloat16 \\
  --modes plain,structural,full --keys 1 \\
  --max-examples 500 \\
  --output results/raw/osnip_style_bench.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from fastprove.config import ObfuscationConfig
from fastprove.layers.attention import AttentionMode
from fastprove.models.obfuscated import ObfuscatedTinyCausalLM
from fastprove.models.plain import PlainTinyCausalLM
from fastprove.pretrained.llama import (
    load_llama_artifact,
    load_llama_plain,
    load_llama_tokenizer,
    verify_plaintext_llama_tree,
)
from fastprove.seed import RequestContext, derive_seed

from evals.keys import MasterKey, generate_master_keys
from evals.model_factory import zero_noise_injection_
from evals.stats import relative_increase


# OSNIP Table 1 numbers for Llama-3.2-3B-Instruct (paper reference only).
OSNIP_LLAMA3B_TABLE1: Dict[str, Dict[str, float]] = {
    "non_private": {
        "arc_easy": 0.712,
        "piqa": 0.768,
        "mnli": 0.542,
        "sst2": 0.867,
        "anli_r1": 0.440,
        "anli_r2": 0.401,
        "anli_r3": 0.425,
        "wic": 0.497,
        "hellaswag": 0.716,
        "mmlu": 0.606,
        "avg": 0.597,
        "rp": 100.0,
    },
    "osnip": {
        "arc_easy": 0.707,
        "piqa": 0.766,
        "mnli": 0.535,
        "sst2": 0.825,
        "anli_r1": 0.431,
        "anli_r2": 0.407,
        "anli_r3": 0.410,
        "wic": 0.513,
        "hellaswag": 0.709,
        "mmlu": 0.572,
        "avg": 0.588,
        "rp": 98.49,
    },
}


def _env_meta(device: torch.device) -> Dict[str, Any]:
    cuda = torch.cuda.is_available()
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_memory_total_bytes": (
            int(torch.cuda.get_device_properties(0).total_memory) if cuda else None
        ),
    }


def _read_jsonl(
    path: Path,
    max_examples: Optional[int] = None,
    *,
    sample_seed: Optional[int] = None,
    stratify_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Read a JSONL corpus, optionally subsampling reproducibly.

    Historical behaviour was to truncate to the first ``max_examples`` lines.
    That is only sound when the file is already shuffled; several corpora in
    ``results/raw/osnip_bench_data`` are not (``mmlu_test.jsonl`` is sorted by
    subject, ``anli_r3_test.jsonl`` is sorted by label), so head-truncation
    silently changed the construct being measured.  Selection is therefore now
    an explicit seeded sample.

    ``stratify_key`` draws proportionally from each value of that field (used
    for MMLU so all 57 subjects are represented) and falls back to a plain
    sample when the field is absent.
    """

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if max_examples is None or len(rows) <= max_examples:
        return rows
    if sample_seed is None:
        # Explicit opt-out: preserve legacy head-truncation for callers that
        # deliberately want a prefix. Never reached by the benchmark CLI.
        return rows[:max_examples]

    rng = random.Random(sample_seed)
    indexed = list(range(len(rows)))

    if stratify_key is not None and all(stratify_key in r for r in rows):
        groups: Dict[Any, List[int]] = {}
        for i in indexed:
            groups.setdefault(rows[i][stratify_key], []).append(i)
        ordered_keys = sorted(groups, key=str)
        for key in ordered_keys:
            rng.shuffle(groups[key])
        # Largest-remainder allocation so the quota is met exactly and the
        # per-group counts do not depend on dict iteration order.
        total = len(rows)
        exact = {k: max_examples * len(groups[k]) / total for k in ordered_keys}
        quota = {k: int(exact[k]) for k in ordered_keys}
        remaining = max_examples - sum(quota.values())
        by_frac = sorted(
            ordered_keys, key=lambda k: (-(exact[k] - int(exact[k])), str(k))
        )
        for k in by_frac[:remaining]:
            quota[k] += 1
        chosen: List[int] = []
        for key in ordered_keys:
            chosen.extend(groups[key][: quota[key]])
    else:
        chosen = rng.sample(indexed, max_examples)

    chosen.sort()
    return [rows[i] for i in chosen]


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


@dataclass
class MCExample:
    """One multiple-choice / verbalizer classification item.

    ``scoring_mode``:
      * ``continuation`` — score full choice strings (HellaSwag endings, etc.)
      * ``letter`` — options already listed in ``context``; score only short
        labels such as ``" A"`` / ``" B"`` (MMLU / ARC style).  Letter mode
        uses raw sum logprob (single-token); length-norm is a no-op.
    """

    context: str
    choices: List[str]
    label: int
    meta: Optional[Dict[str, Any]] = None
    scoring_mode: str = "continuation"


def _letter_labels(n: int) -> List[str]:
    letters = "ABCDEFGHIJ"
    if n > len(letters):
        return [" %d" % (i + 1) for i in range(n)]
    return [" %s" % letters[i] for i in range(n)]


def _format_mc_block(question: str, choice_texts: Sequence[str], *, preamble: str = "") -> str:
    """Build a letter-scored MC prompt body (options listed; answer is a letter)."""

    letters = "ABCDEFGHIJ"
    lines: List[str] = []
    if preamble:
        lines.append(preamble.rstrip())
        lines.append("")
    lines.append(question.strip())
    for i, text in enumerate(choice_texts):
        letter = letters[i] if i < len(letters) else str(i + 1)
        lines.append("%s. %s" % (letter, text.strip()))
    lines.append("Answer:")
    return "\n".join(lines)


def apply_chat_template_if_available(tokenizer, user_text: str, *, enabled: bool) -> str:
    """Wrap a user prompt for Instruct models when a chat template exists."""

    if not enabled:
        return user_text
    chat_template = getattr(tokenizer, "chat_template", None)
    apply = getattr(tokenizer, "apply_chat_template", None)
    if not chat_template or apply is None:
        return user_text
    try:
        return apply(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return user_text


# ---------------------------------------------------------------------------
# Task loaders → MCExample
# ---------------------------------------------------------------------------


def load_arc_easy(
    data_dir: Path, max_examples: Optional[int], sample_seed: Optional[int] = None
) -> List[MCExample]:
    path = data_dir / "arc_easy_test.jsonl"
    if not path.is_file():
        return []
    out: List[MCExample] = []
    for row in _read_jsonl(path, max_examples, sample_seed=sample_seed):
        q = str(row["question"]).strip()
        choice_texts = [str(c).strip() for c in row["choices"]]
        # Letter scoring: list options in context, score only " A"/" B"/...
        ctx = _format_mc_block(
            "Question: %s" % q,
            choice_texts,
            preamble="The following is a multiple-choice science question.",
        )
        out.append(
            MCExample(
                context=ctx,
                choices=_letter_labels(len(choice_texts)),
                label=int(row["label"]),
                scoring_mode="letter",
                meta={"prompt_style": "letter_mc"},
            )
        )
    return out


def load_hellaswag(
    data_dir: Path, max_examples: Optional[int], sample_seed: Optional[int] = None
) -> List[MCExample]:
    path = data_dir / "hellaswag_val.jsonl"
    if not path.is_file():
        return []
    out: List[MCExample] = []
    for row in _read_jsonl(path, max_examples, sample_seed=sample_seed):
        ctx = str(row["ctx"]).rstrip()
        endings = [str(e) for e in row["endings"]]
        # Continuation-style: score each ending given context.
        # Prefer a leading space so BPE tokenization is stable.
        choices = [(" " + e.lstrip()) if not e.startswith(" ") else e for e in endings]
        out.append(
            MCExample(
                context=ctx,
                choices=choices,
                label=int(row["label"]),
            )
        )
    return out


def load_piqa(
    data_dir: Path, max_examples: Optional[int], sample_seed: Optional[int] = None
) -> List[MCExample]:
    path = data_dir / "piqa_val.jsonl"
    if not path.is_file():
        return []
    out: List[MCExample] = []
    for row in _read_jsonl(path, max_examples, sample_seed=sample_seed):
        goal = str(row["goal"]).strip()
        s1 = str(row["sol1"]).strip()
        s2 = str(row["sol2"]).strip()
        ctx = "Goal: %s\nSolution:" % goal
        out.append(
            MCExample(
                context=ctx,
                choices=[" " + s1, " " + s2],
                label=int(row["label"]),
            )
        )
    return out


def _wilson_interval(
    correct: int, n: int, z: float = 1.959963985
) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if n <= 0:
        return (float("nan"), float("nan"))
    phat = correct / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (
        z
        * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))
        / denom
    )
    return (max(0.0, center - half), min(1.0, center + half))


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


def load_sst2(
    data_dir: Path, max_examples: Optional[int], sample_seed: Optional[int] = None
) -> List[MCExample]:
    path = data_dir / "sst2_val.jsonl"
    if not path.is_file():
        return []
    out: List[MCExample] = []
    for row in _read_jsonl(path, max_examples, sample_seed=sample_seed):
        sent = str(row["sentence"]).strip()
        ctx = 'Review: "%s"\nSentiment:' % sent
        # label 0 negative, 1 positive
        out.append(
            MCExample(
                context=ctx,
                choices=[" negative", " positive"],
                label=int(row["label"]),
            )
        )
    return out


def _nli_example(premise: str, hypothesis: str, label: int) -> MCExample:
    """0-shot NLI with explicit three-way labels (entailment/neutral/contradiction).

    Label order matches GLUE/ANLI: 0=entailment, 1=neutral, 2=contradiction.
    """

    ctx = (
        "Please identify whether the premise entails the hypothesis.\n"
        "Answer with exactly one of: entailment, neutral, contradiction.\n\n"
        "Premise: %s\n"
        "Hypothesis: %s\n"
        "Answer:"
    ) % (premise.strip(), hypothesis.strip())
    return MCExample(
        context=ctx,
        choices=[" entailment", " neutral", " contradiction"],
        label=int(label),
        # Unequal label lengths ("entailment" vs "neutral") make per-token
        # means length-biased; primary rule must be raw sum (continuation_raw).
        scoring_mode="continuation_raw",
        meta={"prompt_style": "nli_three_way"},
    )


def load_mnli(
    data_dir: Path, max_examples: Optional[int], sample_seed: Optional[int] = None
) -> List[MCExample]:
    path = data_dir / "mnli_val.jsonl"
    if not path.is_file():
        return []
    out: List[MCExample] = []
    for row in _read_jsonl(path, max_examples, sample_seed=sample_seed):
        out.append(
            _nli_example(str(row["premise"]), str(row["hypothesis"]), int(row["label"]))
        )
    return out


def load_anli(
    data_dir: Path,
    round_name: str,
    max_examples: Optional[int],
    sample_seed: Optional[int] = None,
) -> List[MCExample]:
    path = data_dir / ("%s_test.jsonl" % round_name)
    if not path.is_file():
        return []
    out: List[MCExample] = []
    for row in _read_jsonl(path, max_examples, sample_seed=sample_seed):
        out.append(
            _nli_example(str(row["premise"]), str(row["hypothesis"]), int(row["label"]))
        )
    return out


def load_wic(
    data_dir: Path, max_examples: Optional[int], sample_seed: Optional[int] = None
) -> List[MCExample]:
    path = data_dir / "wic_val.jsonl"
    if not path.is_file():
        return []
    out: List[MCExample] = []
    for row in _read_jsonl(path, max_examples, sample_seed=sample_seed):
        word = str(row["word"]).strip()
        s1 = str(row["sentence1"]).strip()
        s2 = str(row["sentence2"]).strip()
        ctx = (
            "Word: %s\nSentence 1: %s\nSentence 2: %s\n"
            "Does the word have the same meaning in both sentences?"
        ) % (word, s1, s2)
        # 0 false, 1 true
        out.append(
            MCExample(
                context=ctx,
                choices=[" No", " Yes"],
                label=int(row["label"]),
            )
        )
    return out


def load_mmlu(
    data_dir: Path, max_examples: Optional[int], sample_seed: Optional[int] = None
) -> List[MCExample]:
    path = data_dir / "mmlu_test.jsonl"
    if not path.is_file():
        return []
    out: List[MCExample] = []
    # mmlu_test.jsonl is a sorted concatenation of 57 per-subject corpora, so
    # it must be stratified: an unstratified sample of 200 from 14042 rows
    # would still cover the subjects unevenly.
    for row in _read_jsonl(
        path, max_examples, sample_seed=sample_seed, stratify_key="subject"
    ):
        q = str(row["question"]).strip()
        choice_texts = [str(c).strip() for c in row["choices"]]
        subject = str(row.get("subject") or "")
        subject_pretty = subject.replace("_", " ") if subject else "general knowledge"
        # Letter scoring (lm-eval / MMLU standard): options in stem, score " A"/" B"/...
        preamble = (
            "The following are multiple choice questions (with answers) about %s."
            % subject_pretty
        )
        ctx = _format_mc_block(q, choice_texts, preamble=preamble)
        out.append(
            MCExample(
                context=ctx,
                choices=_letter_labels(len(choice_texts)),
                label=int(row["label"]),
                scoring_mode="letter",
                meta={"subject": subject, "prompt_style": "letter_mc"},
            )
        )
    return out


TASK_LOADERS: Dict[
    str, Callable[[Path, Optional[int], Optional[int]], List[MCExample]]
] = {
    "arc_easy": load_arc_easy,
    "hellaswag": load_hellaswag,
    "piqa": load_piqa,
    "sst2": load_sst2,
    "mnli": load_mnli,
    # Third arg is sample_seed (per-task); must accept it so adding/removing
    # tasks does not break callers that pass three arguments.
    "anli_r1": lambda d, m, s=None: load_anli(d, "anli_r1", m, s),
    "anli_r2": lambda d, m, s=None: load_anli(d, "anli_r2", m, s),
    "anli_r3": lambda d, m, s=None: load_anli(d, "anli_r3", m, s),
    "wic": load_wic,
    "mmlu": load_mmlu,
}

CLOSED_ENDED_ORDER = (
    "arc_easy",
    "piqa",
    "mnli",
    "sst2",
    "anli_r1",
    "anli_r2",
    "anli_r3",
    "wic",
    "hellaswag",
    "mmlu",
)


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


def _build_model_bundle(
    *,
    plain: PlainTinyCausalLM,
    mode: str,
    key: Optional[MasterKey],
    obfuscation: ObfuscationConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Tuple[Any, Optional[RequestContext], float, bool]:
    """Return (model, request_ctx, conversion_seconds, structural_flag)."""

    if mode == "plain":
        return plain, None, 0.0, False

    if key is None:
        raise ValueError("structural/full require a master key")
    t0 = time.perf_counter()
    # fixed_debug refresh requires debug_enabled (production path uses per_request).
    converted = ObfuscatedTinyCausalLM.from_plain(
        plain,
        obfuscation=obfuscation,
        mode=AttentionMode.EXACT,
        approximation=None,
        seed=key.conversion_seed(),
        debug_enabled=True,
    )
    module = converted.module.to(device=device, dtype=dtype)
    module.eval()
    structural = mode == "structural"
    if structural:
        zeroed = zero_noise_injection_(module)
        print("  structural: zeroed %d noise-injection buffers" % len(zeroed))
    conversion_seconds = time.perf_counter() - t0
    ctx = RequestContext(key.request_seed("osnip-bench"), "osnip-bench")
    return module, ctx, conversion_seconds, structural


def run_suite_for_model(
    *,
    model: Any,
    tokenizer,
    data_dir: Path,
    device: torch.device,
    max_length: int,
    max_examples: Optional[int],
    request_context: Optional[RequestContext],
    is_obfuscated: bool,
    tasks: Sequence[str],
    wikitext_max_tokens: Optional[int],
    wikitext_stride: int,
    sample_seed: int,
    length_normalize: bool,
    use_chat_template: bool = True,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "tasks": {},
        "closed_ended": {},
        "use_chat_template": bool(use_chat_template),
    }
    for task in tasks:
        if task == "wikitext2":
            print("  [wikitext2]", flush=True)
            results["tasks"]["wikitext2"] = evaluate_wikitext_ppl(
                model=model,
                tokenizer=tokenizer,
                text_path=data_dir / "wikitext2_test.txt",
                device=device,
                max_length=max_length,
                stride=wikitext_stride,
                max_tokens=wikitext_max_tokens,
                request_context=request_context,
                is_obfuscated=is_obfuscated,
            )
            continue
        loader = TASK_LOADERS.get(task)
        if loader is None:
            results["tasks"][task] = {
                "task": task,
                "skipped": True,
                "reason": "unknown_task",
            }
            continue
        # Per-task seed so that adding/removing a task does not reshuffle the
        # sample drawn for the others.
        task_seed = derive_seed(sample_seed, "osnip-bench-sample", task)
        examples = loader(data_dir, max_examples, task_seed)
        print(
            "  [%s] n=%d" % (task, len(examples)),
            flush=True,
        )
        if not examples:
            results["tasks"][task] = {
                "task": task,
                "n": 0,
                "accuracy": None,
                "skipped": True,
                "reason": "missing_data",
            }
            continue
        degenerate = _degenerate_label_reason(examples)
        if degenerate is not None:
            print("    SKIP %s: %s" % (task, degenerate), flush=True)
            results["tasks"][task] = {
                "task": task,
                "n": len(examples),
                "accuracy": None,
                "skipped": True,
                "reason": degenerate,
            }
            continue
        res = evaluate_mc_task(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            device=device,
            max_length=max_length,
            request_context=request_context,
            is_obfuscated=is_obfuscated,
            task_name=task,
            length_normalize=length_normalize,
            use_chat_template=use_chat_template,
        )
        results["tasks"][task] = _annotate_chance_floor(res)
        if task in CLOSED_ENDED_ORDER and res.get("accuracy") is not None:
            results["closed_ended"][task] = res["accuracy"]

    # Average over available closed-ended tasks (same order as OSNIP when present).
    accs = [
        results["closed_ended"][t]
        for t in CLOSED_ENDED_ORDER
        if t in results["closed_ended"]
    ]
    results["avg_closed_ended"] = (
        float(sum(accs) / len(accs)) if accs else None
    )
    results["n_closed_ended_tasks"] = len(accs)

    # Signal-bearing subset: tasks whose accuracy is significantly above the
    # chance floor.  Retention computed over floored tasks is uninformative,
    # so the honest headline is the one restricted to this subset.
    signal_tasks = [
        t
        for t in CLOSED_ENDED_ORDER
        if t in results["closed_ended"]
        and not results["tasks"].get(t, {}).get("at_chance_level", False)
    ]
    signal_accs = [results["closed_ended"][t] for t in signal_tasks]
    results["signal_bearing_tasks"] = signal_tasks
    results["at_chance_tasks"] = [
        t
        for t in CLOSED_ENDED_ORDER
        if t in results["closed_ended"] and t not in signal_tasks
    ]
    results["avg_closed_ended_signal_only"] = (
        float(sum(signal_accs) / len(signal_accs)) if signal_accs else None
    )
    results["skipped_tasks"] = {
        t: results["tasks"][t].get("reason")
        for t in results["tasks"]
        if results["tasks"][t].get("skipped")
    }
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--data-dir",
        type=str,
        default="results/raw/osnip_bench_data",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16"],
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="plain,structural,full",
        help="Comma list among: plain, structural, full",
    )
    parser.add_argument("--keys", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Cap examples per closed-ended task (default: all available)",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--tasks",
        type=str,
        default="arc_easy,piqa,mnli,sst2,anli_r1,anli_r2,anli_r3,wic,hellaswag,mmlu,wikitext2",
    )
    parser.add_argument(
        "--wikitext-max-tokens",
        type=int,
        default=None,
        help="Optional cap on WikiText tokenized length for faster PPL",
    )
    parser.add_argument("--wikitext-stride", type=int, default=256)
    parser.add_argument("--hidden-noise-dim", type=int, default=8)
    parser.add_argument("--value-noise-dim", type=int, default=2)
    parser.add_argument("--max-condition-number", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument(
        "--output",
        type=str,
        default="results/raw/osnip_style_bench.json",
    )
    parser.add_argument("--skip-hash", action="store_true")
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=20260802,
        help=(
            "Seed for per-task example sampling. Selection is a seeded sample "
            "(stratified by subject for MMLU), never a head-truncation of the "
            "corpus, because several local corpora are sorted by label/subject."
        ),
    )
    parser.add_argument(
        "--no-length-normalize",
        action="store_true",
        help=(
            "Score multiple choice by summed logprob instead of per-token mean. "
            "The summed rule favours short candidates and is the historical "
            "behaviour; both are always recorded in the artifact."
        ),
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help=(
            "Do not wrap MC stems with the tokenizer chat template. "
            "Default is to use apply_chat_template for Instruct models "
            "(required to approach published Llama-3.2-Instruct scores)."
        ),
    )
    args = parser.parse_args(argv)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in ("plain", "structural", "full"):
            print("unsupported mode %r" % m, file=sys.stderr)
            return 2
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    model_path = Path(args.model_path).expanduser().resolve()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = _REPO / data_dir
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = _REPO / out_path

    ok, notes = verify_plaintext_llama_tree(model_path)
    if not ok:
        print("PLAINTEXT VERIFICATION FAILED:", "; ".join(notes), file=sys.stderr)
        return 3
    print("Plaintext verification OK:", notes)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable", file=sys.stderr)
        return 4
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    artifact = load_llama_artifact(
        model_path,
        max_sequence_length=max(args.max_length, 2048),
        compute_hashes=not args.skip_hash,
    )
    print("Loading plaintext Llama...", flush=True)
    t0 = time.perf_counter()
    plain = load_llama_plain(
        artifact,
        device=device,
        dtype=dtype,
        seed=args.seed,
        debug_enabled=False,
    )
    plain.eval()
    load_seconds = time.perf_counter() - t0
    print("Loaded in %.1fs" % load_seconds, flush=True)
    tokenizer = load_llama_tokenizer(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    obfuscation = ObfuscationConfig(
        hidden_noise_dim=args.hidden_noise_dim,
        value_noise_dim_per_head=args.value_noise_dim,
        max_condition_number=args.max_condition_number,
        noise_propagation_gamma=args.gamma,
        refresh_mode="fixed_debug",
    )
    keys = generate_master_keys(count=max(args.keys, 1), base_seed=args.seed)

    payload: Dict[str, Any] = {
        "protocol_note": (
            "OSNIP-style utility metrics on plain vs current fastProve "
            "(structural / full). Not a privacy ASR evaluation. "
            "MC scoring: continuation logprob (acc_norm for multi-token "
            "choices; letter labels for MMLU/ARC). Instruct chat template "
            "applied by default."
        ),
        "osnip_paper_reference_llama3b": OSNIP_LLAMA3B_TABLE1,
        "environment": _env_meta(device),
        "plaintext_base": artifact.to_dict(),
        "load_seconds": load_seconds,
        "config": {
            "model_path": str(model_path),
            "data_dir": str(data_dir),
            "modes": modes,
            "tasks": tasks,
            "max_examples": args.max_examples,
            "max_length": args.max_length,
            "dtype": args.dtype,
            "device": str(device),
            "seed": args.seed,
            "sample_seed": args.sample_seed,
            "example_selection": "seeded_sample_stratified_mmlu",
            "mc_scoring_rule": (
                "acc" if args.no_length_normalize else "acc_norm"
            ),
            "mc_letter_tasks": ["arc_easy", "mmlu"],
            "use_chat_template": not args.no_chat_template,
            "n_keys": args.keys,
            "obfuscation": asdict(obfuscation),
            "wikitext_max_tokens": args.wikitext_max_tokens,
            "wikitext_stride": args.wikitext_stride,
        },
        "runs": [],
        "status": "running",
    }
    _write_json(out_path, payload)

    for mode in modes:
        if mode == "plain":
            key_iter: List[Optional[MasterKey]] = [None]
        else:
            key_iter = list(keys[: args.keys])

        for key in key_iter:
            label = "plain" if key is None else "%s/%s" % (mode, key.label)
            print("=== Run %s ===" % label, flush=True)
            model, req_ctx, conv_s, structural = _build_model_bundle(
                plain=plain,
                mode=mode,
                key=key,
                obfuscation=obfuscation,
                device=device,
                dtype=dtype,
                seed=args.seed,
            )
            suite = run_suite_for_model(
                model=model,
                tokenizer=tokenizer,
                data_dir=data_dir,
                device=device,
                max_length=args.max_length,
                max_examples=args.max_examples,
                request_context=req_ctx,
                is_obfuscated=(mode != "plain"),
                tasks=tasks,
                wikitext_max_tokens=args.wikitext_max_tokens,
                wikitext_stride=args.wikitext_stride,
                sample_seed=args.sample_seed,
                length_normalize=not args.no_length_normalize,
                use_chat_template=not args.no_chat_template,
            )
            run_rec: Dict[str, Any] = {
                "mode": mode,
                "key": None if key is None else key.to_dict(),
                "structural_noise_zeroed": structural,
                "conversion_seconds": conv_s,
                "suite": suite,
            }
            # Retention vs plain (filled after all plain metrics known).
            payload["runs"].append(run_rec)
            payload["status"] = "partial"
            _write_json(out_path, payload)
            print(
                "  avg_closed_ended=%s"
                % suite.get("avg_closed_ended"),
                flush=True,
            )
            if mode != "plain":
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    # Compute RP relative to our plain run (not OSNIP's non-private).
    plain_runs = [r for r in payload["runs"] if r["mode"] == "plain"]
    plain_closed = (
        plain_runs[0]["suite"].get("closed_ended", {}) if plain_runs else {}
    )
    plain_avg = plain_runs[0]["suite"].get("avg_closed_ended") if plain_runs else None
    plain_ppl = None
    if plain_runs:
        wt = plain_runs[0]["suite"]["tasks"].get("wikitext2") or {}
        plain_ppl = wt.get("perplexity")

    for run in payload["runs"]:
        closed = run["suite"].get("closed_ended") or {}
        per_task_rp = {}
        for t, acc in closed.items():
            base = plain_closed.get(t)
            if base is not None and base > 0 and acc is not None:
                per_task_rp[t] = 100.0 * float(acc) / float(base)
        run["retention_vs_plain_pct"] = per_task_rp
        avg = run["suite"].get("avg_closed_ended")
        if plain_avg is not None and plain_avg > 0 and avg is not None:
            run["avg_rp_vs_plain_pct"] = 100.0 * float(avg) / float(plain_avg)
        else:
            run["avg_rp_vs_plain_pct"] = None
        wt = run["suite"]["tasks"].get("wikitext2") or {}
        ppl = wt.get("perplexity")
        if plain_ppl is not None and ppl is not None:
            run["ppl_relative_increase_vs_plain"] = relative_increase(
                float(plain_ppl), float(ppl)
            )
        else:
            run["ppl_relative_increase_vs_plain"] = None

    # Compact summary table for report.
    summary_rows = []
    for run in payload["runs"]:
        row = {
            "mode": run["mode"],
            "key_label": None
            if run["key"] is None
            else run["key"].get("label"),
            "avg_closed_ended": run["suite"].get("avg_closed_ended"),
            "avg_rp_vs_plain_pct": run.get("avg_rp_vs_plain_pct"),
            "ppl": (run["suite"]["tasks"].get("wikitext2") or {}).get(
                "perplexity"
            ),
            "ppl_rel_inc": run.get("ppl_relative_increase_vs_plain"),
        }
        for t in CLOSED_ENDED_ORDER:
            row[t] = (run["suite"].get("closed_ended") or {}).get(t)
        summary_rows.append(row)
    payload["summary"] = summary_rows
    payload["status"] = "complete"
    _write_json(out_path, payload)

    print("\n=== Summary ===", flush=True)
    for row in summary_rows:
        print(
            "mode=%s key=%s avg=%.4f rp=%s ppl=%s"
            % (
                row["mode"],
                row["key_label"],
                row["avg_closed_ended"] or float("nan"),
                (
                    "%.2f%%" % row["avg_rp_vs_plain_pct"]
                    if row["avg_rp_vs_plain_pct"] is not None
                    else "n/a"
                ),
                (
                    "%.3f" % row["ppl"]
                    if row["ppl"] is not None
                    else "n/a"
                ),
            ),
            flush=True,
        )
    print("Wrote", out_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
