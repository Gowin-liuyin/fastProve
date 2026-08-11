"""Aggregate five-layer evaluation and emit protocol §4 + §7 tables.

Usage (from repository root)::

    PYTHONPATH=src:. python -m evals.report \\
        --config evals/configs/P2.yaml \\
        --keys 5 --ci 0.95 --check-gates

Terminology (protocol §8.2): results are *obfuscated-state relative to
plaintext*. This is an augmented covariant obfuscation prototype with bounded
auxiliary noise, not a cryptographic ciphertext or fully encrypted inference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import yaml

from fastprove.config import load_config
from fastprove.evaluation.accuracy import make_synthetic_token_batch
from fastprove.evaluation.token_cache import load_token_cache
from fastprove.pretrained.qwen2 import Qwen2Artifact, load_qwen2_artifact, load_qwen2_plain
from fastprove.evaluation.pretrained import token_cache_provenance_issues

from .conditions import (
    CANONICAL_CONDITIONS,
    ConditionId,
    ConditionSpec,
    ObfuscationMode,
    compute_degradations,
    condition_from_config,
    default_base_prototype_config,
    get_condition,
    load_condition_config,
)
from .keys import MasterKey, generate_master_keys
from .layer1_operator_equiv import run_layer1
from .layer2_attention import run_layer2
from .layer3_moe import run_layer3
from .layer4_output_logit import run_layer4
from .layer5_downstream import format_downstream_table, run_layer5
from .model_factory import EvalModels, build_models
from .stats import stratified_key_bootstrap
from .thresholds import evaluate_gates


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_config_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    # Relative to evals/
    alt = Path(__file__).resolve().parent / path
    if alt.is_file():
        return alt
    # Relative to evals/configs/
    alt2 = Path(__file__).resolve().parent / "configs" / path
    if alt2.is_file():
        return alt2
    # Relative to project root
    alt3 = _project_root() / path
    if alt3.is_file():
        return alt3
    raise FileNotFoundError("config not found: %s" % path)


def _load_base_config(path: Optional[str]):
    if path is None:
        # Prefer project tiny_exact.yaml if present.
        default = _project_root() / "configs" / "tiny_exact.yaml"
        if default.is_file():
            return load_config(default)
        return default_base_prototype_config()
    return load_config(_resolve_config_path(path))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "value"):  # Enum
        return obj.value
    raise TypeError("not JSON serialisable: %r" % type(obj))


def run_single_key(
    models: EvalModels,
    *,
    input_ids: torch.Tensor,
    token_mask: torch.Tensor,
    layers: Sequence[int],
    ci_level: float,
    top_k_attn: int,
) -> Dict[str, Any]:
    """Run selected layers for one (condition, key) pair."""

    result: Dict[str, Any] = {
        "condition": models.condition.to_dict(),
        "key": models.key.to_dict() if models.key else None,
        "notes": list(models.notes),
        "executed_dtype": str(models.dtype),
        "device": str(models.device),
        "plaintext_embedding_dtype": str(models.plain.embedding.weight.dtype),
        "obfuscated_embedding_dtype": (
            str(models.obfuscated.embedding_weight.dtype)
            if models.obfuscated is not None
            else None
        ),
        "conversion_time_seconds": models.conversion_time_seconds,
        "structural_noise_zeroed": models.structural_noise_zeroed,
        "model_manifest": models.model_manifest,
        "obfuscation_manifest": models.obfuscation_manifest,
    }
    layer_set = set(layers)

    if 1 in layer_set:
        result["layer1"] = run_layer1(models, input_ids, token_mask=token_mask)
    if 2 in layer_set:
        result["layer2"] = run_layer2(
            models, input_ids, token_mask=token_mask, top_k=top_k_attn
        )
    if 3 in layer_set:
        result["layer3"] = run_layer3(
            models, input_ids, token_mask=token_mask
        )
    if 4 in layer_set:
        result["layer4"] = run_layer4(
            models, input_ids, token_mask=token_mask
        )
    if 5 in layer_set:
        result["layer5"] = run_layer5(models, ci_level=ci_level)

    return result


def _aggregate_for_gates(key_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten multi-key results into the metric paths expected by thresholds."""

    if not key_results:
        return {}

    def _mean_or_max(path: Sequence[str], reduce: str = "mean") -> Optional[float]:
        values = []
        for rec in key_results:
            node: Any = rec
            ok = True
            for p in path:
                if not isinstance(node, dict) or p not in node:
                    ok = False
                    break
                node = node[p]
            if ok and node is not None and isinstance(node, (int, float)):
                values.append(float(node))
        if not values:
            return None
        if reduce == "max":
            return max(values)
        if reduce == "min":
            return min(values)
        return sum(values) / len(values)

    layer1_chain = _mean_or_max(
        ("layer1", "chain_linear", "max_absolute_error"), reduce="max"
    )
    # Also expose nested path for thresholds helper.
    metrics: Dict[str, Any] = {
        "layer1": {
            "chain_linear": {
                "max_absolute_error": layer1_chain
                if layer1_chain is not None
                else _mean_or_max(
                    ("layer1", "modules", "chain_linear", "max_absolute_error"),
                    reduce="max",
                )
            }
        },
        "layer2": {
            "rank_flip_rate": _mean_or_max(
                ("layer2", "rank_flip_rate"), reduce="max"
            ),
            "top1_match": _mean_or_max(("layer2", "top1_match"), reduce="min"),
            "topk_overlap": _mean_or_max(
                ("layer2", "topk_overlap"), reduce="min"
            ),
            "causal_mask_match": _mean_or_max(
                ("layer2", "causal_mask_match"), reduce="min"
            ),
        },
        "layer3": {
            "expert_set_match": _mean_or_max(
                ("layer3", "expert_set_match"), reduce="min"
            ),
        },
        "layer4": {
            "lm_head_argmax_match": _mean_or_max(
                ("layer4", "lm_head_argmax_match"), reduce="min"
            ),
            "greedy_sequence_exact_match": _mean_or_max(
                ("layer4", "greedy_sequence_exact_match"), reduce="min"
            ),
            "ppl_relative_increase": _mean_or_max(
                ("layer4", "ppl_relative_increase"), reduce="max"
            ),
            "top1_absolute_drop_pp": _mean_or_max(
                ("layer4", "top1_absolute_drop_pp"), reduce="max"
            ),
        },
        "cache": {
            "cache_vs_nocache_identical": _mean_or_max(
                ("layer4", "cache_vs_nocache_identical"), reduce="min"
            ),
        },
        "utility": {
            "accuracy_drop_pp": _mean_or_max(
                ("layer5", "summary", "accuracy_drop_pp"), reduce="max"
            ),
            "ppl_relative_increase": _mean_or_max(
                ("layer5", "summary", "ppl_relative_increase"), reduce="max"
            ),
            "accuracy_drop_pp_ci95_upper": _mean_or_max(
                ("layer5", "summary", "accuracy_drop_pp_ci95_upper"),
                reduce="max",
            ),
        },
    }
    return metrics


def run_condition(
    *,
    condition: ConditionSpec,
    base_config,
    keys: Sequence[MasterKey],
    layers: Sequence[int],
    ci_level: float,
    top_k_attn: int,
    sample_count: Optional[int] = None,
    pretrained_path: Optional[str | Path] = None,
    dataset_cache_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Full multi-key evaluation for one condition."""

    cfg = base_config
    n = sample_count or cfg.evaluation.sample_count
    seq = condition.sequence_length or cfg.evaluation.sequence_length
    artifact: Optional[Qwen2Artifact] = None
    token_cache = None
    if pretrained_path is not None and dataset_cache_path is None:
        raise ValueError("--model-path requires --dataset-cache for meaningful evaluation")
    if dataset_cache_path is not None and pretrained_path is None:
        raise ValueError("--dataset-cache requires --model-path so tokenizer/model identity can be checked")
    if pretrained_path is not None:
        # The artifact context includes the generation suffix; this validates
        # context limits before allocating the 1.5B checkpoint.
        artifact = load_qwen2_artifact(
            pretrained_path,
            max_sequence_length=seq + cfg.evaluation.generation_tokens,
            compute_hashes=True,
        )
        token_cache = load_token_cache(
            dataset_cache_path,
            expected_vocab_size=artifact.model_config.vocab_size,
            expected_sequence_length=seq,
        )
        provenance_issues = token_cache_provenance_issues(token_cache, artifact)
        if provenance_issues:
            raise ValueError(
                "token cache provenance is incomplete: "
                + "; ".join(provenance_issues)
            )
        if sample_count is not None and token_cache.sample_count < sample_count:
            raise ValueError("token cache has fewer samples than --sample-count")
        n = sample_count or token_cache.sample_count
        tokens = token_cache.input_ids[:n]
        mask = token_cache.token_mask[:n]
        identifiers = list(token_cache.sample_ids[:n])
        cfg = replace(
            cfg,
            model=artifact.model_config,
            evaluation=replace(cfg.evaluation, sample_count=n, sequence_length=seq),
        )
        # Load the checkpoint once and reuse it for all independent keys.
        shared_plain = load_qwen2_plain(
            artifact,
            device="cpu",
            dtype=condition.torch_dtype,
            seed=cfg.runtime.seed,
            debug_enabled=True,
        )
    else:
        tokens, identifiers = make_synthetic_token_batch(
            sample_count=n,
            sequence_length=seq,
            vocab_size=cfg.model.vocab_size,
            seed=cfg.runtime.seed,
        )
        mask = torch.ones_like(tokens, dtype=torch.bool)
        shared_plain = None

    key_results: List[Dict[str, Any]] = []
    # Plaintext conditions ignore keys (single pass).
    key_list: Sequence[Optional[MasterKey]]
    if condition.mode == ObfuscationMode.PLAINTEXT:
        key_list = [None]
    else:
        key_list = list(keys)

    started = time.perf_counter()
    for key in key_list:
        models = build_models(
            cfg,
            condition,
            key=key,
            debug_enabled=True,
            pretrained_path=pretrained_path,
            pretrained_artifact=artifact,
            pretrained_plain=shared_plain,
            model_max_sequence_length=seq + cfg.evaluation.generation_tokens,
        )
        rec = run_single_key(
            models,
            input_ids=tokens,
            token_mask=mask,
            layers=layers,
            ci_level=ci_level,
            top_k_attn=top_k_attn,
        )
        key_results.append(rec)

    elapsed = time.perf_counter() - started
    gate_metrics = _aggregate_for_gates(key_results)

    # Stratified bootstrap over keys for a few headline metrics.
    multi_key_stats: Dict[str, Any] = {}
    if len(key_results) >= 2 and 4 in layers:
        for name, path in (
            ("top1_agreement", ("layer4", "top1_token_agreement")),
            ("greedy_seq_match", ("layer4", "greedy_sequence_exact_match")),
        ):
            strata = []
            for rec in key_results:
                node: Any = rec
                for p in path:
                    node = node.get(p, {}) if isinstance(node, dict) else {}
                if isinstance(node, (int, float)):
                    # One scalar per key → treat as single-sample stratum.
                    strata.append([float(node)])
            if strata:
                try:
                    multi_key_stats[name] = stratified_key_bootstrap(
                        strata, level=ci_level, n_bootstrap=500, seed=cfg.runtime.seed
                    ).to_dict()
                except ValueError:
                    pass

    return {
        "condition_id": condition.condition_id.value,
        "condition": condition.to_dict(),
        "sample_identifiers": identifiers,
        "dataset_cache": (
            {
                "path": str(Path(dataset_cache_path).expanduser().resolve()),
                "content_sha256": token_cache.content_sha256,
                "metadata": token_cache.metadata,
            }
            if token_cache is not None
            else None
        ),
        "model_manifest": artifact.to_dict() if artifact is not None else None,
        "n_keys": len(key_list),
        "elapsed_seconds": elapsed,
        "per_key": key_results,
        "aggregated_metrics": gate_metrics,
        "multi_key_stats": multi_key_stats,
        "terminology": (
            "obfuscated-state relative to plaintext; "
            "augmented covariant obfuscation prototype"
        ),
    }


def format_layer_summary(report: Dict[str, Any]) -> str:
    """Human-readable five-layer summary for one condition."""

    lines = [
        "=== fastProve evaluation report ===",
        "Condition: %s" % report.get("condition_id"),
        "Keys: %d" % report.get("n_keys", 0),
        "Terminology: %s" % report.get("terminology", ""),
        "",
    ]
    m = report.get("aggregated_metrics", {})
    l1 = m.get("layer1", {}).get("chain_linear", {})
    l2 = m.get("layer2", {})
    l3 = m.get("layer3", {})
    l4 = m.get("layer4", {})
    util = m.get("utility", {})
    cache = m.get("cache", {})

    lines.append("--- Layer 1: operator equivalence ---")
    lines.append(
        "  ChainLinear max|ĥ−h|: %s  (gate ≤ 1e-4)"
        % _fmt(l1.get("max_absolute_error"))
    )
    lines.append("--- Layer 2: attention ranking (MOST IMPORTANT) ---")
    lines.append("  rank-flip rate: %s  (HARD gate = 0)" % _fmt(l2.get("rank_flip_rate")))
    lines.append("  top-1 match:    %s" % _fmt(l2.get("top1_match")))
    lines.append("  top-k overlap:  %s" % _fmt(l2.get("topk_overlap")))
    lines.append("  causal mask:    %s" % _fmt(l2.get("causal_mask_match")))
    lines.append("--- Layer 3: MoE router ---")
    lines.append(
        "  expert set match: %s  (HARD gate = 100%% when MoE present)"
        % _fmt(l3.get("expert_set_match"))
    )
    lines.append("--- Layer 4: output logit / token trajectory ---")
    lines.append("  LM argmax match:     %s" % _fmt(l4.get("lm_head_argmax_match")))
    lines.append(
        "  greedy seq match:    %s" % _fmt(l4.get("greedy_sequence_exact_match"))
    )
    lines.append(
        "  PPL rel increase:    %s" % _fmt(l4.get("ppl_relative_increase"))
    )
    lines.append(
        "  cache vs no-cache:   %s" % _fmt(cache.get("cache_vs_nocache_identical"))
    )
    lines.append("--- Layer 5: downstream utility ---")
    lines.append(
        "  accuracy drop (pp):  %s" % _fmt(util.get("accuracy_drop_pp"))
    )
    lines.append(
        "  PPL rel increase:    %s" % _fmt(util.get("ppl_relative_increase"))
    )
    if report.get("multi_key_stats"):
        lines.append("--- Multi-key 95% CI (stratified bootstrap) ---")
        for name, est in report["multi_key_stats"].items():
            lines.append(
                "  %s: %.6g [%.6g, %.6g]"
                % (name, est["estimate"], est["ci_low"], est["ci_high"])
            )
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return "%.6g" % v
    return str(v)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m evals.report",
        description=(
            "fastProve five-layer evaluation suite "
            "(protocol evaluation_protocol_fastProve.md)"
        ),
    )
    p.add_argument(
        "--config",
        type=str,
        default="evals/configs/P2.yaml",
        help="Condition YAML (F0–P3). Default: evals/configs/P2.yaml",
    )
    p.add_argument(
        "--base-config",
        type=str,
        default=None,
        help="Base PrototypeConfig YAML (default: configs/tiny_exact.yaml)",
    )
    p.add_argument(
        "--keys",
        type=int,
        default=5,
        help="Number of independent obfuscation master keys (≥3 recommended)",
    )
    p.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Confidence level for intervals (default 0.95)",
    )
    p.add_argument(
        "--check-gates",
        action="store_true",
        help="Print PASS/FAIL for protocol §7 hard and soft gates",
    )
    p.add_argument(
        "--layers",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated layer numbers to run (default all)",
    )
    p.add_argument(
        "--top-k-attn",
        type=int,
        default=4,
        help="Top-k for attention overlap metrics",
    )
    p.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Override evaluation sample count",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write full JSON report to this path",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local Qwen2 artifact directory; requires --dataset-cache",
    )
    p.add_argument(
        "--dataset-cache",
        type=str,
        default=None,
        help="Validated .pt token cache shared by every condition",
    )
    p.add_argument(
        "--all-conditions",
        action="store_true",
        help="Run the full F0–P3 matrix (slow)",
    )
    p.add_argument(
        "--min-keys-protocol",
        type=int,
        default=1,
        help=(
            "Minimum keys required (protocol recommends 3; use 1 for smoke). "
            "Default 1 so --keys 1 works; set 3 for release runs."
        ),
    )
    # Ablation knobs (protocol §5) — override condition when set.
    p.add_argument("--R", type=int, default=None, help="Main noise dim R")
    p.add_argument("--R-h", type=int, default=None, dest="R_h", help="Value noise dim R_h")
    p.add_argument("--R-ff", type=int, default=None, dest="R_ff", help="FFN noise dim R_ff")
    p.add_argument("--gamma", type=float, default=None, help="Noise decay γ")
    p.add_argument("--max-kappa", type=float, default=None, help="max κ(M_ℓ)")
    p.add_argument("--seq-len", type=int, default=None, help="Sequence length")
    p.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["fp32", "bf16", "fp16"],
        help="Override precision",
    )
    p.add_argument(
        "--attention-impl",
        type=str,
        default=None,
        choices=["reference", "sdpa", "flash"],
        help="Attention implementation (reference only in prototype)",
    )
    p.add_argument(
        "--kv-cache",
        type=str,
        default=None,
        choices=["on", "off"],
        help="KV cache on/off",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override base seed for keys and synthetic data",
    )
    return p


def _apply_cli_overrides(condition: ConditionSpec, args: argparse.Namespace) -> ConditionSpec:
    from dataclasses import replace
    from .conditions import Precision

    updates = {}
    if args.R is not None:
        updates["hidden_noise_dim"] = args.R
    if args.R_h is not None:
        updates["value_noise_dim_per_head"] = args.R_h
    if args.R_ff is not None:
        updates["ffn_noise_dim"] = args.R_ff
    if args.gamma is not None:
        updates["noise_propagation_gamma"] = args.gamma
    if args.max_kappa is not None:
        updates["max_condition_number"] = args.max_kappa
    if args.seq_len is not None:
        updates["sequence_length"] = args.seq_len
    if args.precision is not None:
        updates["precision"] = Precision(args.precision)
    if args.attention_impl is not None:
        updates["attention_impl"] = args.attention_impl
    if args.kv_cache is not None:
        updates["use_kv_cache"] = args.kv_cache == "on"
    if updates:
        return replace(condition, **updates)
    return condition


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    if any(layer < 1 or layer > 5 for layer in layers):
        print("layers must be in 1..5", file=sys.stderr)
        return 2
    if args.keys < args.min_keys_protocol:
        print(
            "error: --keys %d < --min-keys-protocol %d"
            % (args.keys, args.min_keys_protocol),
            file=sys.stderr,
        )
        return 2

    base_config = _load_base_config(args.base_config)
    if (args.model_path is None) != (args.dataset_cache is None):
        print("--model-path and --dataset-cache must be supplied together", file=sys.stderr)
        return 2
    if args.seed is not None:
        from dataclasses import replace as dc_replace
        from fastprove.config import RuntimeConfig

        base_config = dc_replace(
            base_config,
            runtime=dc_replace(base_config.runtime, seed=int(args.seed)),
        )

    keys = generate_master_keys(
        count=args.keys, base_seed=base_config.runtime.seed
    )

    if args.all_conditions:
        conditions = [
            get_condition(cid) for cid in ConditionId
        ]
    else:
        raw = load_condition_config(_resolve_config_path(args.config))
        conditions = [_apply_cli_overrides(condition_from_config(raw), args)]

    all_reports: Dict[str, Any] = {
        "protocol": "docs/literature-review/reports/evaluation_protocol_fastProve.md",
        "ci_level": args.ci,
        "n_keys": args.keys,
        "layers": layers,
        "conditions": {},
    }

    for condition in conditions:
        print(
            "Running condition %s (%s) with %d key(s)..."
            % (condition.condition_id.value, condition.mode.value, args.keys),
            flush=True,
        )
        report = run_condition(
            condition=condition,
            base_config=base_config,
            keys=keys,
            layers=layers,
            ci_level=args.ci,
            top_k_attn=args.top_k_attn,
            sample_count=args.sample_count,
            pretrained_path=args.model_path,
            dataset_cache_path=args.dataset_cache,
        )
        all_reports["conditions"][condition.condition_id.value] = report
        print(format_layer_summary(report))
        print("")

        if args.check_gates:
            summary = evaluate_gates(report["aggregated_metrics"])
            report["gates"] = summary.to_dict()
            print(summary.format_table())
            print("")
            if not summary.hard_passed:
                all_reports["exit_hard_fail"] = True

    # Degradation table if enough conditions present.
    if len(all_reports["conditions"]) >= 2:
        # Use layer4 top1 agreement as a quality proxy when available.
        q = {}
        for cid, rep in all_reports["conditions"].items():
            m = rep.get("aggregated_metrics", {})
            val = m.get("layer4", {}).get("lm_head_argmax_match")
            if val is not None:
                q[cid] = val
        if q:
            deg = compute_degradations(
                q, metric_name="lm_head_argmax_match", unit="fraction"
            )
            all_reports["degradations"] = deg.to_dict()
            print("--- Protocol §3.2 degradations (argmax match, fraction) ---")
            print(json.dumps(deg.to_dict(), indent=2))

    # Downstream markdown table when layer 5 present.
    layer5_rows = {}
    for cid, rep in all_reports["conditions"].items():
        per_key = rep.get("per_key") or []
        if per_key and "layer5" in per_key[0]:
            layer5_rows[cid] = per_key[0]["layer5"]
    if layer5_rows:
        print("")
        print("--- Protocol §4.5 downstream table ---")
        print(format_downstream_table(layer5_rows))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(all_reports, indent=2, default=_json_default),
            encoding="utf-8",
        )
        print("Wrote %s" % out)

    if all_reports.get("exit_hard_fail") and args.check_gates:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
