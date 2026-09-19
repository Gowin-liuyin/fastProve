#!/usr/bin/env python3
"""Plaintext vs current-fastProve obfuscated comparison for Llama (or tiny).

Protocol-aligned pair evaluation only:
  - plaintext base (verified)
  - current-repo conversion (structural and/or full), independent master keys

Does **not** compare against legacy ModelSplit obfuscation.

Examples
--------
PYTHONPATH=src:. python scripts/run_pretrained_compare.py \\
  --model-path /path/to/Llama-3.2-3B-Instruct \\
  --device cuda --dtype bfloat16 \\
  --modes structural,full --keys 3 \\
  --sample-count 4 --seq-len 32 --gen-tokens 8 \\
  --check-gates --output results/raw/llama_compare.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch

_REPO = Path(__file__).resolve().parents[1]

from fastprove.config import ObfuscationConfig
from fastprove.evaluation.accuracy import (
    _build_token_batch,
    _run_pair,
    greedy_unpadded_pair,
)
from fastprove.evaluation.environment import environment_meta
from fastprove.evaluation.prompts import load_prompt_texts
from fastprove.io import write_json_file
from fastprove.layers.attention import AttentionMode
from fastprove.models.obfuscated import ObfuscatedTinyCausalLM
from fastprove.pretrained.llama import (
    load_llama_artifact,
    load_llama_plain,
    verify_plaintext_llama_tree,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to plaintext Llama-3.2-3B-Instruct directory",
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
        default="structural,full",
        help="Comma list: structural and/or full (current scheme only)",
    )
    parser.add_argument("--keys", type=int, default=3)
    parser.add_argument(
        "--sample-count",
        type=int,
        default=1500,
        help="Number of prompts per key (default 1500 for large-scale real-scenario)",
    )
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--gen-tokens", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-noise-dim", type=int, default=8)
    parser.add_argument("--value-noise-dim", type=int, default=2)
    parser.add_argument("--max-condition-number", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--check-gates", action="store_true")
    parser.add_argument("--output", type=str, default="results/raw/llama_compare.json")
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Skip full weight hashing (still verifies structure)",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Optional layer cap for smoke (loads full config but only first N layers)",
    )
    parser.add_argument(
        "--real-scenario",
        action="store_true",
        default=True,
        help="Use real-scenario prompt bank (≥sample-count prompts)",
    )
    parser.add_argument(
        "--no-real-scenario",
        action="store_true",
        help="Disable real-scenario bank (tiny DEFAULT_PROMPTS / synthetic)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Optional JSONL/JSON/txt prompt file (overrides builder when set)",
    )
    parser.add_argument(
        "--skip-greedy",
        action="store_true",
        help="Teacher-forced only (faster; not default for large-scale report)",
    )
    args = parser.parse_args(argv)
    if args.no_real_scenario:
        args.real_scenario = False

    from evals.keys import generate_master_keys
    from evals.model_factory import zero_noise_injection_
    from evals.thresholds import evaluate_gates

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in modes:
        if mode not in ("structural", "full"):
            print("unsupported mode %r (allowed: structural, full)" % mode, file=sys.stderr)
            return 2
    if "legacy" in args.modes.lower() or "modelsplit" in args.modes.lower():
        print("refusing legacy ModelSplit comparison arm", file=sys.stderr)
        return 2

    model_path = Path(args.model_path).expanduser().resolve()
    ok, notes = verify_plaintext_llama_tree(model_path)
    if not ok:
        print("PLAINTEXT VERIFICATION FAILED:", "; ".join(notes), file=sys.stderr)
        return 3
    print("Plaintext verification OK:", notes)

    max_seq = args.seq_len + args.gen_tokens
    artifact = load_llama_artifact(
        model_path,
        max_sequence_length=max_seq,
        compute_hashes=not args.skip_hash,
    )
    # Optional smoke: truncate layers by rewriting model_config via replace.
    from dataclasses import replace

    model_config = artifact.model_config
    if args.max_layers is not None:
        if args.max_layers < 1 or args.max_layers > model_config.num_layers:
            print("invalid --max-layers", file=sys.stderr)
            return 2
        model_config = replace(model_config, num_layers=int(args.max_layers))
        # Rebuild artifact-like config only for load size control — still
        # streams only first N layer weights via mapping.
        artifact = replace(artifact, model_config=model_config)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable", file=sys.stderr)
        return 4
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    print("Loading plaintext Llama into fastProve reference module...")
    t0 = time.perf_counter()
    plain = load_llama_plain(
        artifact,
        device=device,
        dtype=dtype,
        seed=args.seed,
        debug_enabled=True,
    )
    load_seconds = time.perf_counter() - t0
    print("Loaded in %.1fs" % load_seconds)

    prompt_file = (
        Path(args.prompt_file).expanduser().resolve()
        if args.prompt_file
        else None
    )
    prompt_texts = load_prompt_texts(
        prompt_file=prompt_file,
        sample_count=args.sample_count,
        seed=args.seed,
        real_scenario=bool(args.real_scenario),
    )
    print(
        "Prompts ready: count=%d real_scenario=%s prompt_file=%s"
        % (len(prompt_texts), args.real_scenario, prompt_file)
    )
    if len(prompt_texts) < args.sample_count:
        print(
            "ERROR: only %d prompts for requested sample_count=%d"
            % (len(prompt_texts), args.sample_count),
            file=sys.stderr,
        )
        return 5

    tokens, sample_ids, mask = _build_token_batch(
        model_path=model_path,
        sample_count=args.sample_count,
        sequence_length=args.seq_len,
        vocab_size=artifact.model_config.vocab_size,
        seed=args.seed,
        prompts=prompt_texts,
        sample_id_prefix="scenario",
    )
    tokens = tokens.to(device)
    mask = mask.to(device)
    print(
        "Tokenized batch: shape=%s valid_frac=%.4f"
        % (tuple(tokens.shape), float(mask.float().mean().item()))
    )

    obfuscation = ObfuscationConfig(
        hidden_noise_dim=args.hidden_noise_dim,
        value_noise_dim_per_head=args.value_noise_dim,
        max_condition_number=args.max_condition_number,
        noise_propagation_gamma=args.gamma,
        refresh_mode="fixed_debug",
    )
    keys = generate_master_keys(count=args.keys, base_seed=args.seed)

    results: Dict[str, Any] = {
        "protocol_note": (
            "plaintext vs current fastProve only; "
            "legacy ModelSplit obfuscation was NOT used as baseline or result"
        ),
        "terminology": (
            "obfuscated-state relative to plaintext; real-valued augmented "
            "covariant obfuscation. NOT LWE-keyed and NOT LWE-inspired: the "
            "map h -> c is exactly invertible over the reals, so h is "
            "recoverable from c regardless of noise magnitude or refresh mode "
            "(see docs/threat_model.md section 5bis)"
        ),
        "environment": environment_meta(device),
        "plaintext_base": artifact.to_dict(),
        "load_seconds": load_seconds,
        "evaluation": {
            "sample_count": int(tokens.shape[0]),
            "sequence_length": args.seq_len,
            "generation_tokens": args.gen_tokens,
            "sample_ids": sample_ids,
            "sample_id_count": len(sample_ids),
            "prompts_per_key": int(tokens.shape[0]),
            "real_scenario": bool(args.real_scenario),
            "prompt_file": str(prompt_file) if prompt_file else None,
            "dtype": args.dtype,
            "device": str(device),
            "batch_size": args.batch_size,
            "skip_greedy": bool(args.skip_greedy),
            "obfuscation": asdict(obfuscation),
            "n_keys": args.keys,
            "modes": modes,
            "seed": args.seed,
        },
        "pairs": [],
        "legacy_comparison_arm": False,
        "status": "running",
    }

    for mode_name in modes:
        for key in keys:
            print(
                "Converting current-scheme mode=%s key=%s ..."
                % (mode_name, key.label)
            )
            t1 = time.perf_counter()
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
            structural = mode_name == "structural"
            if structural:
                zeroed = zero_noise_injection_(module)
                print("  structural: zeroed %d noise-injection buffers" % len(zeroed))
            conversion_seconds = time.perf_counter() - t1
            print("  conversion %.1fs" % conversion_seconds)

            if args.skip_greedy:
                # Teacher-forced only path for emergency time budgets.
                pair = _run_pair(
                    plain=plain,
                    obfuscated=module,
                    tokens=tokens,
                    mask=mask,
                    sample_ids=sample_ids,
                    mode_name=mode_name,
                    key_meta=key.to_dict(),
                    conversion_seconds=conversion_seconds,
                    structural_noise_zeroed=structural,
                    request_seed=key.request_seed("compare"),
                    generation_tokens=0,
                    batch_size=args.batch_size,
                    device=device,
                    run_greedy=False,
                    codec=converted.token_codec,
                )
            else:
                pair = _run_pair(
                    plain=plain,
                    obfuscated=module,
                    tokens=tokens,
                    mask=mask,
                    sample_ids=sample_ids,
                    mode_name=mode_name,
                    key_meta=key.to_dict(),
                    conversion_seconds=conversion_seconds,
                    structural_noise_zeroed=structural,
                    request_seed=key.request_seed("compare"),
                    generation_tokens=args.gen_tokens,
                    batch_size=args.batch_size,
                    device=device,
                    run_greedy=True,
                    codec=converted.token_codec,
                )
            results["pairs"].append(pair)
            # Checkpoint after every key×mode so partial runs remain usable.
            results["status"] = "partial"
            write_json_file(results, args.output, base_dir=_REPO)
            print(
                "  top1_agree=%.6g greedy_seq=%.6g greedy_n=%s ppl_rel=%s"
                % (
                    pair["top1_token_agreement"],
                    pair["greedy_sequence_exact_match"],
                    pair.get("greedy_n_samples"),
                    pair["ppl_relative_increase"],
                ),
                flush=True,
            )
            # Free converted module before next key.
            del module, converted
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Aggregate gate metrics from first full-mode key (or structural if only).
    primary = None
    for pair in results["pairs"]:
        if pair["mode"] == "full":
            primary = pair
            break
    if primary is None and results["pairs"]:
        primary = results["pairs"][0]

    # Two DISTINCT error sources (must never be swapped):
    # 1) FP32 ChainLinear unit identity (protocol §7.1 hard math gate)
    # 2) End-to-end activation-dtype logit max abs (diagnostic only, not ChainLinear)
    from evals.layer1_operator_equiv import _chain_linear_identity_test

    chain_gate = _chain_linear_identity_test(seed=args.seed)
    e2e_logit_max = (
        float(primary["e2e_logit_max_absolute_error"]) if primary else None
    )

    gate_metrics: Dict[str, Any] = {
        "layer1": {
            "chain_linear": {
                "max_absolute_error": float(chain_gate["max_absolute_error"]),
                "gate_passed": bool(chain_gate["gate_passed"]),
                "source": "fp32_unit_identity_test",
            }
        },
        "layer2": {
            "rank_flip_rate": None,
            "top1_match": None,
            "topk_overlap": None,
            "causal_mask_match": None,
        },
        "layer3": {"expert_set_match": None},
        "layer4": {
            "lm_head_argmax_match": (
                primary["lm_head_argmax_match"] if primary else None
            ),
            "greedy_sequence_exact_match": (
                primary["greedy_sequence_exact_match"] if primary else None
            ),
            "ppl_relative_increase": (
                primary["ppl_relative_increase"] if primary else None
            ),
            "top1_absolute_drop_pp": (
                primary["top1_absolute_drop_pp"] if primary else None
            ),
            "greedy_n_samples": (
                primary.get("greedy_n_samples") if primary else None
            ),
        },
        "cache": {"cache_vs_nocache_identical": None},
        "utility": {
            "accuracy_drop_pp": (
                primary["top1_absolute_drop_pp"] if primary else None
            ),
            "ppl_relative_increase": (
                primary["ppl_relative_increase"] if primary else None
            ),
        },
        "diagnostics": {
            "e2e_logit_max_absolute_error": e2e_logit_max,
            "e2e_logit_note": (
                "BF16/activation end-to-end logit max|obf-plain|; "
                "NOT used as ChainLinear hard gate"
            ),
            "chain_linear_unit_max_absolute_error": float(
                chain_gate["max_absolute_error"]
            ),
            "chain_linear_unit_source": "fp32_unit_identity_test",
            "chain_linear_scope_caveat": (
                "This gate exercises the standalone ChainLinear module, which "
                "the evaluated Llama path does NOT instantiate: "
                "models/obfuscated.py applies converted weights directly "
                "(normalized @ self.*_weight_math) and references ChainLinear "
                "nowhere. A PASS here therefore validates the affine-refresh "
                "identity in isolation, not the affine algebra actually "
                "executed by the model under test."
            ),
        },
    }
    results["chain_linear_unit_gate"] = chain_gate
    results["e2e_logit_max_absolute_error_primary"] = e2e_logit_max
    results["primary_pair_mode"] = primary["mode"] if primary else None
    results["aggregated_metrics"] = gate_metrics
    results["metric_source_notes"] = {
        "hard_gate_chain_linear": (
            "FP32 unit identity only (evals.layer1_operator_equiv."
            "_chain_linear_identity_test); never BF16 e2e logits"
        ),
        "e2e_logit_max_absolute_error": (
            "diagnostic from teacher-forced plain vs obf logits on the eval set"
        ),
        "greedy": (
            "all sample_ids; pad stripped so generate_greedy last index is "
            "last real token"
        ),
    }

    if args.check_gates:
        summary = evaluate_gates(gate_metrics)
        results["gates"] = summary.to_dict()
        print("")
        print("=== Metric sources (do not confuse) ===")
        print(
            "ChainLinear FP32 unit max|err| = %.6g  (hard gate source)"
            % float(chain_gate["max_absolute_error"])
        )
        print(
            "E2E logit max|err| (primary pair) = %s  (diagnostic only)"
            % (
                ("%.6g" % e2e_logit_max)
                if e2e_logit_max is not None
                else "n/a"
            )
        )
        print("")
        print(summary.format_table())
        if not summary.hard_passed:
            results["exit_hard_fail"] = True

    # Short markdown report body.
    report_lines = [
        "# Llama plaintext vs current fastProve",
        "",
        results["protocol_note"],
        "",
        "## Environment",
        "```json",
        json.dumps(results["environment"], indent=2),
        "```",
        "",
        "## Plaintext base",
        "- path: `%s`" % artifact.root,
        "- weights_sha256: `%s`" % artifact.weights_sha256,
        "- config_sha256: `%s`" % artifact.config_sha256,
        "- is_plaintext_verified: %s" % artifact.is_plaintext_verified,
        "- legacy ModelSplit baseline: **not used**",
        "",
        "## Metric sources",
        "- ChainLinear hard gate: FP32 unit identity only "
        "(not e2e BF16 logit max).",
        "- E2E logit max abs (primary): %s"
        % (
            ("%.6g" % e2e_logit_max)
            if e2e_logit_max is not None
            else "n/a"
        ),
        "- Greedy: all sample_ids; padding stripped before generate_greedy.",
        "",
        "## Pairs (current scheme only)",
    ]
    for pair in results["pairs"]:
        report_lines.append(
            "- mode=`%s` key=`%s`: top1_agree=%.6g, greedy_seq=%.6g "
            "(n=%s), ΔPPL_rel=%s, top1_drop_pp=%.4g, e2e_logit_max=%.6g"
            % (
                pair["mode"],
                pair["key"]["label"],
                pair["top1_token_agreement"],
                pair["greedy_sequence_exact_match"],
                pair.get("greedy_n_samples"),
                pair["ppl_relative_increase"],
                pair["top1_absolute_drop_pp"],
                pair["e2e_logit_max_absolute_error"],
            )
        )
    results["report_markdown"] = "\n".join(report_lines) + "\n"

    results["status"] = "complete"
    out_path = write_json_file(results, args.output, base_dir=_REPO)
    report_path = out_path.with_suffix(".md")
    report_path.write_text(results["report_markdown"], encoding="utf-8")
    print("Wrote %s" % out_path)
    print("Wrote %s" % report_path)

    if results.get("exit_hard_fail") and args.check_gates:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
