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
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from fastprove.config import ObfuscationConfig
from fastprove.evaluation.benchmarks.bundle import (
    _build_model_bundle,
    run_suite_for_model,
)
from fastprove.evaluation.benchmarks.tasks import CLOSED_ENDED_ORDER
from fastprove.evaluation.environment import environment_meta
from fastprove.io import write_json_file
from fastprove.pretrained.llama import (
    load_llama_artifact,
    load_llama_plain,
    load_llama_tokenizer,
    verify_plaintext_llama_tree,
)

_REPO = Path(__file__).resolve().parents[1]


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

    from evals.keys import MasterKey, generate_master_keys
    from evals.stats import relative_increase

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
        "environment": environment_meta(device),
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
    write_json_file(payload, out_path)

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
            write_json_file(payload, out_path)
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
    write_json_file(payload, out_path)

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
