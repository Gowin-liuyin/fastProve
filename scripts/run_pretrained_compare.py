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
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

# Ensure repo roots on path when invoked as a script.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from fastprove.codec import TokenCodec
from fastprove.config import ObfuscationConfig
from fastprove.evaluation.accuracy import (
    compare_teacher_forced_metrics,
    make_synthetic_token_batch,
)
from fastprove.evaluation.metrics import tensor_error_metrics
from fastprove.layers.attention import AttentionMode
from fastprove.models.obfuscated import ObfuscatedTinyCausalLM
from fastprove.models.plain import PlainTinyCausalLM
from fastprove.pretrained.llama import (
    load_llama_artifact,
    load_llama_plain,
    load_llama_tokenizer,
    verify_plaintext_llama_tree,
)
from fastprove.seed import RequestContext, make_generator

from evals.keys import generate_master_keys
from evals.model_factory import zero_noise_injection_
from evals.stats import relative_increase, wilson_interval
from evals.thresholds import evaluate_gates


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


def load_prompt_texts(
    *,
    prompt_file: Optional[Path],
    sample_count: int,
    seed: int,
    real_scenario: bool,
) -> List[str]:
    """Load or synthesize prompt strings (at least sample_count)."""

    texts: List[str] = []
    if prompt_file is not None:
        path = Path(prompt_file)
        if not path.is_file():
            raise FileNotFoundError("prompt file not found: %s" % path)
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                texts.append(str(obj.get("text") or obj.get("prompt") or obj))
        elif path.suffix == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict):
                        texts.append(str(item.get("text") or item.get("prompt")))
            else:
                raise ValueError("JSON prompt file must be a list")
        else:
            texts = [
                line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
    def _build_real(n: int, s: int) -> List[str]:
        import importlib.util

        helper = Path(__file__).resolve().parent / "build_real_scenario_prompts.py"
        spec = importlib.util.spec_from_file_location("real_prompts", helper)
        if spec is None or spec.loader is None:
            raise ImportError("cannot load build_real_scenario_prompts.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return list(mod.build_real_scenario_prompts(n, seed=s))

    if not texts and real_scenario:
        texts = _build_real(sample_count, seed)
    if not texts:
        texts = list(DEFAULT_PROMPTS)
    if len(texts) < sample_count and real_scenario:
        texts = _build_real(sample_count, seed)
    if len(texts) < sample_count:
        # Repeat with index tags rather than silent synthetic token IDs.
        base = list(texts)
        while len(texts) < sample_count:
            texts.append(base[len(texts) % len(base)] + " [dup-%d]" % len(texts))
    return texts[:sample_count]


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


DEFAULT_PROMPTS = (
    "The capital of France is",
    "In mathematics, the derivative of x squared is",
    "def fibonacci(n):",
    "Once upon a time in a small village,",
)


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


def _write_results_checkpoint(results: Dict[str, Any], output: str | Path) -> Path:
    """Persist partial/final results so long runs remain recoverable."""

    out_path = Path(output)
    if not out_path.is_absolute():
        out_path = _REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    return out_path


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
        "environment": _env_meta(device),
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
            _write_results_checkpoint(results, args.output)
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
    out_path = _write_results_checkpoint(results, args.output)
    report_path = out_path.with_suffix(".md")
    report_path.write_text(results["report_markdown"], encoding="utf-8")
    print("Wrote %s" % out_path)
    print("Wrote %s" % report_path)

    if results.get("exit_hard_fail") and args.check_gates:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
