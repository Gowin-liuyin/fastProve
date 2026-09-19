"""Model bundle construction and per-model suite execution for benchmarks."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple

import torch

from ...config import ObfuscationConfig
from ...layers.attention import AttentionMode
from ...models.obfuscated import ObfuscatedTinyCausalLM
from ...models.plain import PlainTinyCausalLM
from ...seed import RequestContext, derive_seed
from .scoring import (
    _annotate_chance_floor,
    _degenerate_label_reason,
    evaluate_mc_task,
    evaluate_wikitext_ppl,
)
from .tasks import CLOSED_ENDED_ORDER, TASK_LOADERS

if TYPE_CHECKING:
    from evals.keys import MasterKey


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

    from evals.model_factory import zero_noise_injection_

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
