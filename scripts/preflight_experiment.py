#!/usr/bin/env python3
"""Fail-fast preflight for the real-model fastProve experiment.

This command allocates no model weights and runs no inference.  It validates
the local checkpoint, tokenizer, aligned token cache, context budget and host
resources, then writes a manifest that can be attached to the first sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastprove.evaluation.runner import environment_metadata  # noqa: E402
from fastprove.evaluation.pretrained import (  # noqa: E402
    token_cache_evidence_scope,
    token_cache_provenance_issues,
)
from fastprove.evaluation.token_cache import load_token_cache  # noqa: E402
from fastprove.pretrained.qwen2 import load_qwen2_artifact, load_qwen2_tokenizer  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate all inputs before a real fastProve sweep.")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--dataset-cache", required=True, type=Path)
    parser.add_argument("--generation-tokens", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        required=True,
        help="Device that the subsequent sweep will request; no implicit fallback",
    )
    parser.add_argument("--activation-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--calibration-count", type=int, default=8)
    parser.add_argument("--full-count", type=int, default=64)
    parser.add_argument(
        "--accept-nonstandard-caption",
        action="store_true",
        help=(
            "Explicitly accept a Flickr/caption-only token cache as a limited "
            "non-standard LM comparison scope"
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional preflight JSON manifest")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def _accepted_evaluation_scope(
    scope: str, *, accept_nonstandard_caption: bool
) -> str:
    """Require an explicit opt-in for the caption-only evaluation scope."""

    if (
        scope == "caption_only_nonstandard_lm_candidate"
        and not accept_nonstandard_caption
    ):
        raise ValueError(
            "token cache is a non-standard caption-only candidate; "
            "rerun with --accept-nonstandard-caption after explicitly "
            "choosing the limited evaluation scope"
        )
    return scope


def _runtime_capability_check(
    *, device_name: str, activation_dtype: str
) -> Dict[str, Any]:
    """Smoke-test the requested device/dtype without allocating model weights."""

    fallback_value = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch reports it unavailable")
    if device_name == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but PyTorch reports it unavailable")
    if device_name == "mps" and fallback_value.lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise RuntimeError(
            "PYTORCH_ENABLE_MPS_FALLBACK is enabled; refusing a run that may "
            "silently execute operators on CPU"
        )
    dtype = torch.float32 if activation_dtype == "fp32" else torch.bfloat16
    device = torch.device(device_name)
    left = torch.ones((2, 8), device=device, dtype=dtype)
    right = torch.eye(8, device=device, dtype=dtype)
    product = left @ right
    probabilities = torch.softmax(
        product.reshape(2, 2, 4), dim=-1
    )
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
    if product.device.type != device_name or probabilities.device.type != device_name:
        raise RuntimeError(
            "runtime smoke landed on %s instead of requested %s"
            % (product.device.type, device_name)
        )
    if product.dtype != dtype or probabilities.dtype != dtype:
        raise RuntimeError(
            "runtime smoke returned dtype %s/%s instead of %s"
            % (product.dtype, probabilities.dtype, dtype)
        )
    return {
        "requested_device": device_name,
        "requested_activation_dtype": activation_dtype,
        "actual_device": product.device.type,
        "actual_activation_dtype": str(product.dtype).removeprefix("torch."),
        "checkpoint_compute_dtype": reduction_compute_dtype_name(),
        "mps_fallback_env": fallback_value or None,
        "matmul_and_softmax": "pass",
    }


def _estimate_memory(artifact: Any, *, activation_dtype: str) -> Dict[str, int]:
    # model.safetensors is a useful lower bound, but conversion keeps a plain
    # copy and obfuscated buffers concurrently.  This is an estimate, not a
    # reservation or a performance guarantee.
    weights = int(artifact.weights_bytes)
    working_factor = 4.8 if activation_dtype == "fp32" else 3.0
    return {
        "checkpoint_bytes": weights,
        "estimated_plain_activation_bytes": int(weights * (2.0 if activation_dtype == "fp32" else 1.0)),
        "estimated_plain_plus_obfuscated_activation_bytes": int(weights * working_factor * 0.8),
        "estimated_working_set_activation_bytes": int(weights * working_factor),
        "activation_dtype": activation_dtype,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # The runner always records decode/TPOT metrics, so a zero-token request
    # would pass this asset preflight and then fail later in the performance
    # path (division by zero/config validation).  Keep the preflight contract
    # aligned with ``EvaluationConfig`` and require at least one token.
    if args.generation_tokens < 1 or args.calibration_count < 1 or args.full_count < 1:
        raise SystemExit("generation and sample counts must be positive")
    if args.calibration_count > args.full_count:
        raise SystemExit("calibration count cannot exceed full count")
    model_path = args.model_path.expanduser().resolve()
    cache_path = args.dataset_cache.expanduser().resolve()
    result: Dict[str, Any] = {
        "status": "failure",
        "checks": [],
        "environment": environment_metadata(),
        "requested": {
            "model_path": str(model_path),
            "dataset_cache": str(cache_path),
            "generation_tokens": args.generation_tokens,
            "device": args.device,
            "activation_dtype": args.activation_dtype,
            "calibration_count": args.calibration_count,
            "full_count": args.full_count,
            "accept_nonstandard_caption": bool(args.accept_nonstandard_caption),
        },
    }
    try:
        result["runtime"] = _runtime_capability_check(
            device_name=args.device,
            activation_dtype=args.activation_dtype,
        )
        result["checks"].append({"name": "requested_runtime", "status": "pass"})
        artifact = load_qwen2_artifact(
            model_path,
            max_sequence_length=2048,
            compute_hashes=True,
        )
        result["model"] = artifact.to_dict()
        result["checks"].append({"name": "qwen2_artifact", "status": "pass"})
        tokenizer = load_qwen2_tokenizer(model_path)
        tokenizer_vocab = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
        tokenizer_max_id = max(int(value) for value in tokenizer.get_vocab().values())
        if tokenizer_max_id >= artifact.model_config.vocab_size:
            raise ValueError(
                "tokenizer max ID %d is outside model vocab %d"
                % (tokenizer_max_id, artifact.model_config.vocab_size)
            )
        result["checks"].append({
            "name": "tokenizer_vocab",
            "status": "pass",
            "tokenizer_base_vocab_size": tokenizer_vocab,
            "tokenizer_length_with_added_tokens": int(len(tokenizer)),
            "model_vocab_size": artifact.model_config.vocab_size,
            "padded_embedding_rows": artifact.model_config.vocab_size - int(len(tokenizer)),
        })
        cache = load_token_cache(cache_path, expected_vocab_size=artifact.model_config.vocab_size)
        provenance_issues = token_cache_provenance_issues(cache, artifact)
        if provenance_issues:
            raise ValueError(
                "token cache provenance is incomplete: "
                + "; ".join(provenance_issues)
            )
        if cache.sample_count < args.full_count:
            raise ValueError("token cache has %d samples, need at least %d" % (cache.sample_count, args.full_count))
        context_required = cache.sequence_length + args.generation_tokens
        if context_required > artifact.max_position_embeddings:
            raise ValueError(
                "cache sequence + generation (%d) exceeds context %d"
                % (context_required, artifact.max_position_embeddings)
            )
        evaluation_scope = _accepted_evaluation_scope(
            token_cache_evidence_scope(cache),
            accept_nonstandard_caption=args.accept_nonstandard_caption,
        )
        result["dataset"] = {
            "path": str(cache_path),
            "content_sha256": cache.content_sha256,
            "sample_count": cache.sample_count,
            "sequence_length": cache.sequence_length,
            "evaluation_scope": evaluation_scope,
            "scope_acceptance": (
                "explicit_nonstandard_caption"
                if evaluation_scope == "caption_only_nonstandard_lm_candidate"
                else "standard_or_validated"
            ),
            "metadata": cache.metadata,
        }
        result["checks"].append({
            "name": "aligned_token_cache",
            "status": "pass",
            "calibration_count": args.calibration_count,
            "full_count": args.full_count,
        })
        if cache.metadata.get("model_config_sha256") not in (None, artifact.config_sha256):
            raise ValueError("token cache was produced from a different model config hash")
        if cache.metadata.get("tokenizer_sha256") not in (None, artifact.tokenizer_sha256):
            raise ValueError("token cache was produced from a different tokenizer hash")
        result["checks"].append({"name": "model_tokenizer_cache_identity", "status": "pass"})
        result["memory_estimate"] = _estimate_memory(
            artifact, activation_dtype=args.activation_dtype
        )
        physical = result["environment"].get("physical_memory_bytes")
        estimate = result["memory_estimate"]["estimated_working_set_activation_bytes"]
        result["resource_warning"] = (
            "estimated working set is near/above physical RAM; reduce batch or run on CUDA"
            if physical is not None and estimate > int(physical * 0.8)
            else "estimate fits within the host RAM envelope; monitor peak memory"
        )
        result["status"] = "ready_for_experiment"
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}

    serialized = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    print(serialized)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if output.exists() and not args.allow_overwrite:
            print("refusing to overwrite preflight output: %s" % output, file=sys.stderr)
            return 2
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
        print("wrote %s" % output)
    return 0 if result["status"] == "ready_for_experiment" else 1


if __name__ == "__main__":
    raise SystemExit(main())
