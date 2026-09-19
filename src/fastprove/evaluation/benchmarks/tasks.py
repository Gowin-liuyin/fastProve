"""Benchmark task loaders for OSNIP-style utility evaluation.

Dataset loaders return :class:`MCExample` items for closed-ended tasks and
share the seeded, stratified JSONL sampling in :func:`_read_jsonl`.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence


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
