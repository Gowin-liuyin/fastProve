"""Exact-mode verification on a real pretrained checkpoint (docs/PROBLEMS.md 4.1).

Closes the evidence gap recorded in ``docs/PROBLEMS.md`` section 4.1: every
accuracy conclusion so far was produced on ``fastprove-random-tiny-correctness-only``
with ``meaningful_lm_evidence: False``. That weakens *both* directions of the
project's findings:

* the positive one -- that the covariant construction is mathematically exact --
  had no real-weight measurement;
* the negative one -- that this class of construction leaks -- was open to the
  objection "your implementation is simply wrong".

This script measures both on real pretrained weights:

1. **Exact-mode agreement.** Plaintext reference versus obfuscated conversion on
   identical tokens, seeds, dtype and device. Judged in FP32, because
   ``AGENTS.md`` R5 requires FP32 for RMS statistics and the exact gate is an
   FP32 gate; BF16 is recorded as a separate condition, never as the gate.
2. **The layer-0 norm leak on the same weights.** ``rho`` is read from the real
   model's first RMSNorm and used to recover the input token ids, so the leak is
   demonstrated on the very checkpoint whose exactness was just verified.

**Layer truncation.** A 16 GB host cannot hold the plaintext and obfuscated
copies of all 28 FP32 layers (a full-depth attempt peaked at 44.5 GB and was
killed). ``--layers`` truncates the model to its first ``k`` transformer blocks.
The weights remain the real pretrained weights; only the depth is reduced. Every
record states the depth, and no record may be described as a full-model result.

Usage:
    python scripts/verify_pretrained_exact_mode.py \\
        --model-path "$FASTPROVE_MODEL_ROOT/deepseek-r1-distill-qwen-1.5b/base_model" \\
        --layers 4 \\
        --output results/raw/pretrained_exact_mode_qwen_L4.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import resource
import time
from pathlib import Path
from typing import Dict, List

import torch

from fastprove.config import ModelConfig, ObfuscationConfig
from fastprove.layers.attention import AttentionMode
from fastprove.models.obfuscated import ObfuscatedTinyCausalLM
from fastprove.pretrained.qwen2 import (
    load_qwen2_artifact,
    load_qwen2_plain,
    load_qwen2_tokenizer,
)
from fastprove.seed import RequestContext

_FP64 = torch.float64


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _load_pair(
    model_path: Path,
    *,
    layers: int,
    sequence_length: int,
    dtype: torch.dtype,
    seed: int,
    noise_dim: int,
    value_noise_dim: int,
    block_size: int,
):
    """Load the plaintext reference and its obfuscated conversion."""

    artifact = load_qwen2_artifact(
        model_path, max_sequence_length=sequence_length, compute_hashes=True
    )
    full_layers = artifact.model_config.num_layers
    truncated_config = dataclasses.replace(
        artifact.model_config,
        num_layers=layers,
        max_sequence_length=sequence_length,
    )
    artifact = dataclasses.replace(artifact, model_config=truncated_config)

    plain = load_qwen2_plain(
        artifact, dtype=dtype, seed=seed, debug_enabled=False
    ).eval()

    obfuscation = ObfuscationConfig(
        hidden_noise_dim=noise_dim,
        value_noise_dim_per_head=value_noise_dim,
        max_condition_number=10.0,
        noise_propagation_gamma=0.5,
        refresh_mode="per_request",
        basis_block_size=block_size,
    )
    started = time.perf_counter()
    converted = ObfuscatedTinyCausalLM.from_plain(
        plain,
        obfuscation=obfuscation,
        mode=AttentionMode.EXACT,
        approximation=None,
        seed=seed,
        debug_enabled=False,
    )
    conversion_seconds = time.perf_counter() - started
    return (
        artifact,
        plain,
        converted.module.eval(),
        converted.token_codec,
        conversion_seconds,
        full_layers,
    )


def _exact_metrics(
    plain, obfuscated, codec, input_ids: torch.Tensor, *, seed: int
) -> Dict[str, object]:
    """Teacher-forced plaintext versus obfuscated logits on identical inputs.

    The obfuscated model lives in the *obfuscated domain*: it consumes
    vocabulary-permuted token ids and returns vocabulary-permuted logits
    (``softmax(l) Pi_voc``). The client therefore encodes the input and
    inverse-permutes the logits before any comparison. Comparing plaintext ids
    or un-permuted logits directly is a harness error, not a conversion error.
    """

    context = RequestContext(global_seed=seed, request_id="pretrained-exact")
    with torch.no_grad():
        reference = plain(input_ids)
        permuted = obfuscated(codec.encode(input_ids), request_context=context)
        observed = permuted[..., codec.permutation]
    difference = (reference - observed).to(_FP64)
    reference64 = reference.to(_FP64)
    agreement = (reference.argmax(-1) == observed.argmax(-1)).to(_FP64)
    return {
        "comparison_domain": "plaintext (obfuscated logits inverse-permuted)",
        "logit_max_absolute_error": float(difference.abs().max()),
        "logit_relative_l2_error": float(
            difference.norm() / reference64.norm()
        ),
        "teacher_forced_argmax_agreement": float(agreement.mean()),
        "nan_count": int(torch.isnan(observed).sum()),
        "inf_count": int(torch.isinf(observed).sum()),
        "token_positions_compared": int(agreement.numel()),
    }


def _greedy_metrics(
    plain, obfuscated, codec, input_ids: torch.Tensor, *, new_tokens: int, seed: int
) -> Dict[str, object]:
    """Greedy continuation agreement using each path's own cache.

    The obfuscated path generates in the obfuscated domain throughout, so the
    client decodes the whole returned sequence once at the boundary.
    """

    context = RequestContext(global_seed=seed, request_id="pretrained-greedy")
    with torch.no_grad():
        reference = plain.generate_greedy(input_ids, max_new_tokens=new_tokens)
        observed = codec.decode(
            obfuscated.generate_greedy(
                codec.encode(input_ids),
                max_new_tokens=new_tokens,
                request_context=context,
            )
        )
    prompt = input_ids.shape[1]
    reference_tail = reference[:, prompt:]
    observed_tail = observed[:, prompt:]
    per_token = (reference_tail == observed_tail).to(_FP64)
    sequence = (reference_tail == observed_tail).all(dim=-1).to(_FP64)
    return {
        "generated_tokens": int(reference_tail.shape[1]),
        "greedy_token_exact_match": float(per_token.mean()),
        "greedy_sequence_exact_match": float(sequence.mean()),
    }


def _layer0_leak(
    plain, input_ids: torch.Tensor, *, epsilon: float
) -> Dict[str, object]:
    """Recover input token ids from the real model's first-layer rho."""

    embedding = plain.embedding.weight.detach().to(_FP64)
    hidden_dim = embedding.shape[1]
    norms = embedding.norm(dim=-1)
    sorted_norms, order = torch.sort(norms)

    flat = input_ids.reshape(-1)
    results = {}
    for name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        unique_correct = 0
        counts: List[int] = []
        for token in flat.tolist():
            exact_rho = torch.sqrt(embedding[token].pow(2).mean() + epsilon)
            stored = exact_rho.to(dtype)
            value = stored.to(_FP64)
            resolution = float(
                torch.nextafter(
                    stored.clone(), torch.tensor(float("inf"), dtype=dtype)
                ).to(_FP64)
                - value
            )
            recovered = torch.sqrt(
                ((value * value - epsilon) * hidden_dim).clamp_min(0)
            )
            window = max(
                resolution * hidden_dim * float(value) / max(float(recovered), 1e-12),
                1e-13,
            )
            low = int(torch.searchsorted(sorted_norms, recovered - window))
            high = int(
                torch.searchsorted(sorted_norms, recovered + window, right=True)
            )
            counts.append(high - low)
            if high - low == 1 and int(order[low]) == token:
                unique_correct += 1
        total = len(counts)
        results[name] = {
            "unique_and_correct_fraction": unique_correct / total,
            "mean_candidate_count": sum(counts) / total,
            "tokens_attacked": total,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--layers", type=int, required=True, help="truncate to first k blocks"
    )
    parser.add_argument("--prompt-file", default="results/raw/real_scenario_prompts_1500.jsonl")
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--gen-tokens", type=int, default=4)
    parser.add_argument("--hidden-noise-dim", type=int, default=16)
    parser.add_argument("--value-noise-dim", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16"), default="float32"
    )
    arguments = parser.parse_args()

    dtype = torch.float32 if arguments.dtype == "float32" else torch.bfloat16
    model_path = Path(arguments.model_path).expanduser()
    context_length = arguments.seq_len + arguments.gen_tokens + 8

    artifact, plain, obfuscated, codec, conversion_seconds, full_layers = _load_pair(
        model_path,
        layers=arguments.layers,
        sequence_length=context_length,
        dtype=dtype,
        seed=arguments.seed,
        noise_dim=arguments.hidden_noise_dim,
        value_noise_dim=arguments.value_noise_dim,
        block_size=arguments.block_size,
    )

    tokenizer = load_qwen2_tokenizer(model_path)
    prompts = [
        json.loads(line)["text"]
        for line in Path(arguments.prompt_file)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    encoded: List[List[int]] = []
    for text in prompts:
        # ``load_qwen2_tokenizer`` returns a HuggingFace tokenizer whose
        # ``encode`` yields a plain list of ids, not a tokenizers.Encoding.
        ids = tokenizer.encode(text, add_special_tokens=True)
        if len(ids) >= arguments.seq_len:
            encoded.append(list(ids[: arguments.seq_len]))
        if len(encoded) == arguments.sample_count:
            break
    if len(encoded) < arguments.sample_count:
        raise RuntimeError(
            "only %d prompts reached sequence length %d"
            % (len(encoded), arguments.seq_len)
        )
    input_ids = torch.tensor(encoded, dtype=torch.long)

    exact = _exact_metrics(plain, obfuscated, codec, input_ids, seed=arguments.seed)
    greedy = _greedy_metrics(
        plain,
        obfuscated,
        codec,
        input_ids,
        new_tokens=arguments.gen_tokens,
        seed=arguments.seed,
    )
    leak = _layer0_leak(
        plain, input_ids, epsilon=float(artifact.model_config.rms_epsilon)
    )

    # FP32 exact-gate thresholds mirror evaluation/runner.py tiny_fp32_strict.
    gate_applicable = arguments.dtype == "float32"
    gates = {
        "applicable": gate_applicable,
        "note": (
            "gate is judged in FP32 only; a bfloat16 run is a separate recorded "
            "condition and must never be presented as the exact gate"
        ),
        "max_logit_absolute_error_threshold": 1e-3,
        "required_teacher_forced_argmax_agreement": 1.0,
        "required_greedy_sequence_exact_match": 1.0,
        "results": [],
    }
    if gate_applicable:
        checks = (
            (
                "logit max absolute error",
                exact["logit_max_absolute_error"] <= 1e-3,
                exact["logit_max_absolute_error"],
                1e-3,
            ),
            (
                "teacher-forced argmax agreement",
                exact["teacher_forced_argmax_agreement"] >= 1.0,
                exact["teacher_forced_argmax_agreement"],
                1.0,
            ),
            (
                "greedy sequence exact match",
                greedy["greedy_sequence_exact_match"] >= 1.0,
                greedy["greedy_sequence_exact_match"],
                1.0,
            ),
            ("no NaN", exact["nan_count"] == 0, exact["nan_count"], 0),
            ("no Inf", exact["inf_count"] == 0, exact["inf_count"], 0),
        )
        gates["results"] = [
            {
                "name": name,
                "passed": bool(passed),
                "observed": observed,
                "threshold": threshold,
            }
            for name, passed, observed, threshold in checks
        ]
        gates["all_passed"] = all(item["passed"] for item in gates["results"])

    record = {
        "purpose": "docs/PROBLEMS.md section 4.1 -- real-weight evidence",
        "evidence_scope": (
            "real pretrained weights, depth-truncated; NOT a full-model result"
        ),
        "model": {
            "identifier": "local-qwen2-1.5b-deepseek-r1-distill",
            "root": str(model_path),
            "architecture": artifact.architecture,
            "pretrained": True,
            "config_sha256": artifact.config_sha256,
            "tokenizer_sha256": artifact.tokenizer_sha256,
            "weights_sha256": artifact.weights_sha256,
            "weights_bytes": artifact.weights_bytes,
            "full_num_hidden_layers": full_layers,
            "evaluated_num_hidden_layers": arguments.layers,
            "hidden_size": artifact.model_config.hidden_size,
            "num_attention_heads": artifact.model_config.num_attention_heads,
            "num_key_value_heads": artifact.model_config.num_key_value_heads,
            "vocab_size": artifact.model_config.vocab_size,
            "rms_epsilon": artifact.model_config.rms_epsilon,
            "tie_word_embeddings": False,
        },
        "obfuscation": {
            "hidden_noise_dim": arguments.hidden_noise_dim,
            "value_noise_dim_per_head": arguments.value_noise_dim,
            "basis_block_size": arguments.block_size,
            "mode": "exact",
            "refresh_mode": "per_request",
        },
        "data": {
            "prompt_file": arguments.prompt_file,
            "sample_count": len(encoded),
            "sequence_length": arguments.seq_len,
            "generation_tokens": arguments.gen_tokens,
            "identical_samples_for_both_paths": True,
        },
        "runtime": {
            "requested_dtype": arguments.dtype,
            "device": "cpu",
            "seed": arguments.seed,
            "conversion_seconds": conversion_seconds,
            "peak_rss_bytes": _peak_rss_bytes(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
        },
        "exact_mode": exact,
        "greedy": greedy,
        "layer0_norm_leak_on_the_same_weights": {
            "claim": (
                "the exactness verified above is what forces rho to be correct, "
                "and a correct rho recovers the input tokens"
            ),
            "by_rho_precision": leak,
        },
        "gates": gates,
    }
    path = Path(arguments.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(
        "model: %s  depth %d/%d  dtype %s"
        % (
            record["model"]["identifier"],
            arguments.layers,
            full_layers,
            arguments.dtype,
        )
    )
    print(
        "exact: logit max|err| %.3e  rel L2 %.3e  argmax %.4f"
        % (
            exact["logit_max_absolute_error"],
            exact["logit_relative_l2_error"],
            exact["teacher_forced_argmax_agreement"],
        )
    )
    print(
        "greedy: token %.4f  sequence %.4f"
        % (greedy["greedy_token_exact_match"], greedy["greedy_sequence_exact_match"])
    )
    for name, values in leak.items():
        print(
            "leak(rho %s): unique&correct %.1f%%  mean candidates %.2f"
            % (name, 100 * values["unique_and_correct_fraction"],
               values["mean_candidate_count"])
        )
    if gate_applicable:
        for item in gates["results"]:
            print(
                "gate %-34s %-4s observed=%s threshold=%s"
                % (
                    item["name"],
                    "PASS" if item["passed"] else "FAIL",
                    item["observed"],
                    item["threshold"],
                )
            )
        print("ALL: %s" % ("PASS" if gates["all_passed"] else "FAIL"))
    print("peak RSS %.2f GB" % (record["runtime"]["peak_rss_bytes"] / 1e9))


if __name__ == "__main__":
    main()
