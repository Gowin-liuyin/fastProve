"""Record exact-mode tolerance for the current checkpoint arithmetic dtype.

Usage:
    python scripts/record_exact_tolerance.py --output results/raw/exact_tolerance_A4.json

The recorded numbers are the evidence for the FP64 -> FP32 checkpoint change of
task A4 and for the Stage B non-decoding rewrite. They are produced by a run,
never written by hand.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from fastprove.config import ModelConfig, ObfuscationConfig
from fastprove.layers.attention import AttentionMode
from fastprove.models.obfuscated import ObfuscatedTinyCausalLM
from fastprove.models.plain import PlainTinyCausalLM
from fastprove.seed import RequestContext


def _checkpoint_dtype_label() -> str:
    """Report the checkpoint arithmetic dtype without hard-coding it."""

    try:
        from fastprove.models.obfuscated import _checkpoint_compute_dtype
    except ImportError:
        return "absent (Stage B removed the checkpoint arithmetic)"
    return str(_checkpoint_compute_dtype("cpu"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()

    config = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_sequence_length=32,
    )
    obfuscation = ObfuscationConfig(
        hidden_noise_dim=8,
        value_noise_dim_per_head=2,
        max_condition_number=10.0,
        noise_propagation_gamma=0.5,
        refresh_mode="per_request",
        basis_block_size=8,
    )
    plain = PlainTinyCausalLM(
        config, seed=arguments.seed, debug_enabled=False
    ).eval()
    obfuscated = ObfuscatedTinyCausalLM.from_plain(
        plain,
        obfuscation=obfuscation,
        mode=AttentionMode.EXACT,
        approximation=None,
        seed=arguments.seed,
        debug_enabled=False,
    ).module.eval()

    torch.manual_seed(arguments.seed)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    context = RequestContext(global_seed=arguments.seed, request_id="tolerance")
    with torch.no_grad():
        reference = plain(input_ids)
        observed = obfuscated(input_ids, request_context=context)
    difference = reference - observed

    record = {
        "checkpoint_compute_dtype": _checkpoint_dtype_label(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "seed": arguments.seed,
        "model": {
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "hidden_noise_dim": obfuscation.hidden_noise_dim,
            "basis_block_size": obfuscation.basis_block_size,
        },
        "exact_max_absolute_error": float(difference.abs().max()),
        "exact_relative_l2_error": float(difference.norm() / reference.norm()),
        "nan_count": int(torch.isnan(observed).sum()),
        "inf_count": int(torch.isinf(observed).sum()),
    }
    path = Path(arguments.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
