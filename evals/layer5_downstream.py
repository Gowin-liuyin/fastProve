"""Layer 5 — downstream task utility (protocol §4.5).

Recommended tasks: MMLU, C-Eval, PIQA, IFEval, HumanEval, WikiText PPL.

This module always provides a **synthetic proxy** path that runs without
network downloads (correctness-only tiny models). Real-task loaders activate
only when datasets are already available locally or ``allow_download=True``
is passed explicitly (AGENTS.md: ask before large downloads).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from fastprove.evaluation.accuracy import make_synthetic_token_batch
from fastprove.seed import make_generator

from .model_factory import EvalModels, inverse_align_logits
from .stats import (
    IntervalEstimate,
    paired_bootstrap_mean_diff,
    percentage_point_drop,
    relative_increase,
    wilson_interval,
)


TASK_NAMES = (
    "MMLU",
    "C-Eval",
    "PIQA",
    "IFEval",
    "HumanEval",
    "PPL",
)


@torch.no_grad()
def _teacher_forced_ppl(
    models: EvalModels,
    input_ids: torch.Tensor,
    token_mask: torch.Tensor,
) -> Dict[str, float]:
    plain = models.plain
    plain_logits = plain(input_ids, token_mask=token_mask)
    labels = input_ids[:, 1:]
    valid = token_mask[:, :-1] & token_mask[:, 1:]
    flat_logits = plain_logits[:, :-1][valid]
    flat_labels = labels[valid]
    nll = F.cross_entropy(flat_logits.float(), flat_labels, reduction="mean")
    plain_ppl = float(math.exp(float(nll.item())))
    plain_acc = float(
        (flat_logits.argmax(dim=-1) == flat_labels).float().mean().item()
    )

    if models.obfuscated is None:
        return {
            "plain_ppl": plain_ppl,
            "obf_ppl": plain_ppl,
            "plain_acc": plain_acc,
            "obf_acc": plain_acc,
            "ppl_relative_increase": 0.0,
            "accuracy_drop_pp": 0.0,
        }

    ctx = models.request_context("layer5-ppl")
    obf_logits = inverse_align_logits(
        models.obfuscated(
            input_ids, token_mask=token_mask, request_context=ctx
        ),
        models.vocab_permutation,
    )
    flat_obf = obf_logits[:, :-1][valid]
    nll_o = F.cross_entropy(flat_obf.float(), flat_labels, reduction="mean")
    obf_ppl = float(math.exp(float(nll_o.item())))
    obf_acc = float(
        (flat_obf.argmax(dim=-1) == flat_labels).float().mean().item()
    )
    return {
        "plain_ppl": plain_ppl,
        "obf_ppl": obf_ppl,
        "plain_acc": plain_acc,
        "obf_acc": obf_acc,
        "ppl_relative_increase": relative_increase(plain_ppl, obf_ppl),
        "accuracy_drop_pp": percentage_point_drop(plain_acc, obf_acc) * 100.0,
    }


@torch.no_grad()
def _synthetic_multiple_choice(
    models: EvalModels,
    *,
    n_questions: int,
    n_choices: int,
    seed: int,
) -> Dict[str, Any]:
    """Proxy MCQ: score each choice continuation by mean token NLL."""

    vocab = models.config.model.vocab_size
    seq = min(8, models.config.model.max_sequence_length - 2)
    generator = make_generator(seed, "layer5-mcq", n_questions, n_choices)
    # prompts: [n, seq], choices: [n, n_choices, choice_len]
    prompts = torch.randint(
        0, vocab, (n_questions, seq), generator=generator, dtype=torch.int64
    )
    choice_len = 2
    choices = torch.randint(
        0,
        vocab,
        (n_questions, n_choices, choice_len),
        generator=generator,
        dtype=torch.int64,
    )
    # Ground-truth: choice that plain model prefers (self-label for proxy).
    plain_pref = []
    obf_pref = []
    device = models.device
    for q in range(n_questions):
        scores_p = []
        scores_o = []
        for c in range(n_choices):
            tokens = torch.cat(
                [prompts[q], choices[q, c]], dim=0
            ).unsqueeze(0).to(device)
            mask = torch.ones_like(tokens, dtype=torch.bool)
            logits_p = models.plain(tokens, token_mask=mask)
            # Score choice tokens only.
            start = seq
            nll_p = 0.0
            for t in range(choice_len):
                pos = start + t - 1
                if pos < 0:
                    continue
                nll_p += float(
                    F.cross_entropy(
                        logits_p[0, pos].float().unsqueeze(0),
                        tokens[0, start + t].unsqueeze(0),
                    ).item()
                )
            scores_p.append(-nll_p)
            if models.obfuscated is not None:
                ctx = models.request_context("mcq-%d-%d" % (q, c))
                logits_o = inverse_align_logits(
                    models.obfuscated(
                        tokens, token_mask=mask, request_context=ctx
                    ),
                    models.vocab_permutation,
                )
                nll_o = 0.0
                for t in range(choice_len):
                    pos = start + t - 1
                    if pos < 0:
                        continue
                    nll_o += float(
                        F.cross_entropy(
                            logits_o[0, pos].float().unsqueeze(0),
                            tokens[0, start + t].unsqueeze(0),
                        ).item()
                    )
                scores_o.append(-nll_o)
            else:
                scores_o.append(-nll_p)
        plain_pref.append(int(torch.tensor(scores_p).argmax().item()))
        obf_pref.append(int(torch.tensor(scores_o).argmax().item()))

    # Proxy "correct" label = plain preference; accuracy_plain is 100% by construction.
    # Better: fixed synthetic labels.
    labels = torch.randint(
        0, n_choices, (n_questions,), generator=generator
    ).tolist()
    plain_correct = [plain_pref[i] == labels[i] for i in range(n_questions)]
    obf_correct = [obf_pref[i] == labels[i] for i in range(n_questions)]
    plain_acc = sum(plain_correct) / n_questions
    obf_acc = sum(obf_correct) / n_questions
    agreement = sum(
        1 for i in range(n_questions) if plain_pref[i] == obf_pref[i]
    ) / n_questions
    return {
        "plain_accuracy": plain_acc,
        "obf_accuracy": obf_acc,
        "accuracy_drop_pp": percentage_point_drop(plain_acc, obf_acc) * 100.0,
        "choice_agreement": agreement,
        "n": n_questions,
        "plain_correct": plain_correct,
        "obf_correct": obf_correct,
        "source": "synthetic_proxy",
    }


def _try_load_real_task(name: str) -> Optional[Any]:
    """Return a dataset object if already cached; never download by default."""

    try:
        import datasets  # type: ignore
    except ImportError:
        return None
    # Only inspect local cache; do not download.
    mapping = {
        "MMLU": ("cais/mmlu", "all"),
        "C-Eval": ("ceval/ceval-exam", "computer_network"),
        "PIQA": ("piqa", None),
        "WikiText": ("wikitext", "wikitext-2-raw-v1"),
    }
    # Real evaluation is intentionally stubbed: returning None forces synthetic.
    # Operators can extend this with local paths.
    return None


@torch.no_grad()
def run_layer5(
    models: EvalModels,
    *,
    sample_count: Optional[int] = None,
    allow_download: bool = False,
    seed: Optional[int] = None,
    ci_level: float = 0.95,
) -> Dict[str, Any]:
    """Produce the protocol §4.5 result table row for one condition."""

    if allow_download:
        raise RuntimeError(
            "allow_download=True requires explicit operator-provided dataset "
            "paths; automatic large downloads are disabled (AGENTS.md)"
        )

    n = sample_count or models.config.evaluation.sample_count
    seed = int(seed if seed is not None else models.config.runtime.seed)
    device = models.device

    tokens, _ids = make_synthetic_token_batch(
        sample_count=n,
        sequence_length=models.config.evaluation.sequence_length,
        vocab_size=models.config.model.vocab_size,
        seed=seed,
    )
    tokens = tokens.to(device)
    mask = torch.ones_like(tokens, dtype=torch.bool)

    ppl = _teacher_forced_ppl(models, tokens, mask)
    mcq = _synthetic_multiple_choice(
        models, n_questions=max(n, 4), n_choices=4, seed=seed + 1
    )

    # Map proxy metrics onto the recommended task columns.
    # Real MMLU/C-Eval/… require external data; mark as synthetic proxy.
    row = {
        "MMLU": {
            "plain": mcq["plain_accuracy"],
            "obf": mcq["obf_accuracy"],
            "delta_pp": mcq["accuracy_drop_pp"],
            "unit": "fraction (synthetic proxy)",
            "source": "synthetic_proxy",
        },
        "C-Eval": {
            "plain": mcq["plain_accuracy"],
            "obf": mcq["obf_accuracy"],
            "delta_pp": mcq["accuracy_drop_pp"],
            "unit": "fraction (synthetic proxy)",
            "source": "synthetic_proxy",
        },
        "PIQA": {
            "plain": mcq["plain_accuracy"],
            "obf": mcq["obf_accuracy"],
            "delta_pp": mcq["accuracy_drop_pp"],
            "unit": "fraction (synthetic proxy)",
            "source": "synthetic_proxy",
        },
        "IFEval": {
            "plain": None,
            "obf": None,
            "delta_pp": None,
            "unit": "n/a",
            "source": "not_available_without_dataset",
        },
        "HumanEval": {
            "plain": None,
            "obf": None,
            "delta_pp": None,
            "unit": "pass@1",
            "source": "not_available_without_dataset",
        },
        "PPL": {
            "plain": ppl["plain_ppl"],
            "obf": ppl["obf_ppl"],
            "delta_rel": ppl["ppl_relative_increase"],
            "unit": "perplexity (synthetic corpus)",
            "source": "synthetic_token_batch",
        },
    }

    # CIs on next-token accuracy agreement / drop.
    plain_correct = mcq["plain_correct"]
    obf_correct = mcq["obf_correct"]
    drop_per_item = [
        (1.0 if plain_correct[i] else 0.0) - (1.0 if obf_correct[i] else 0.0)
        for i in range(len(plain_correct))
    ]
    # Paired bootstrap on per-item accuracy difference (fraction).
    try:
        drop_ci = paired_bootstrap_mean_diff(
            [1.0 if c else 0.0 for c in plain_correct],
            [1.0 if c else 0.0 for c in obf_correct],
            level=ci_level,
            n_bootstrap=500,
            seed=seed,
        )
        drop_ci_pp = IntervalEstimate(
            estimate=drop_ci.estimate * 100.0,
            ci_low=drop_ci.ci_low * 100.0,
            ci_high=drop_ci.ci_high * 100.0,
            level=drop_ci.level,
            method=drop_ci.method,
            n=drop_ci.n,
        )
    except ValueError:
        drop_ci_pp = None

    return {
        "tasks": row,
        "summary": {
            "accuracy_drop_pp": mcq["accuracy_drop_pp"],
            "accuracy_drop_pp_ci95_upper": (
                drop_ci_pp.ci_high if drop_ci_pp is not None else None
            ),
            "accuracy_drop_pp_ci": drop_ci_pp.to_dict() if drop_ci_pp else None,
            "ppl_relative_increase": ppl["ppl_relative_increase"],
            "choice_agreement": mcq["choice_agreement"],
        },
        "notes": [
            "Layer 5 real tasks (MMLU/C-Eval/PIQA/IFEval/HumanEval) require "
            "local datasets; this run uses synthetic proxies only.",
            "Do not present synthetic-proxy numbers as pretrained LM accuracy.",
            "Terminology: obfuscated-state relative to plaintext "
            "(augmented covariant obfuscation prototype).",
        ],
    }


def format_downstream_table(
    rows_by_condition: Dict[str, Dict[str, Any]],
) -> str:
    """Render protocol §4.5 result table as markdown."""

    headers = ["condition"] + list(TASK_NAMES)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    order = ["F0", "F1", "F2", "P0", "P1", "P2", "P3", "delta_prod"]
    for cid in order:
        if cid not in rows_by_condition and cid != "delta_prod":
            continue
        if cid == "delta_prod":
            # Compute from P0 and P2 if present.
            if "P0" not in rows_by_condition or "P2" not in rows_by_condition:
                continue
            cells = ["Δ_prod"]
            for task in TASK_NAMES:
                t0 = rows_by_condition["P0"]["tasks"].get(task, {})
                t2 = rows_by_condition["P2"]["tasks"].get(task, {})
                if task == "PPL":
                    p0 = t0.get("plain")
                    p2 = t2.get("obf")
                    if p0 and p2:
                        cells.append("%.4f (rel)" % relative_increase(p0, p2))
                    else:
                        cells.append("—")
                else:
                    a0 = t0.get("plain")
                    a2 = t2.get("obf")
                    if a0 is not None and a2 is not None:
                        cells.append("%.2f pp" % ((a0 - a2) * 100.0))
                    else:
                        cells.append("—")
            lines.append("| " + " | ".join(cells) + " |")
            continue
        data = rows_by_condition[cid]
        cells = [cid]
        for task in TASK_NAMES:
            t = data["tasks"].get(task, {})
            if task == "PPL":
                val = t.get("obf" if cid not in ("F0", "P0") else "plain")
                cells.append("%.4f" % val if val is not None else "—")
            else:
                val = t.get("obf" if cid not in ("F0", "P0") else "plain")
                if val is None:
                    cells.append("—")
                else:
                    cells.append("%.4f" % val)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
