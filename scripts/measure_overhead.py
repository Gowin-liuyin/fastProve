"""Measure reference-implementation overhead versus the plaintext model.

Usage:
    python scripts/measure_overhead.py --output results/raw/overhead_B7.json

Records prefill and decode timings for the plaintext and obfuscated reference
modules on identical inputs, seeds, dtype and device. This measures the *eager
reference* implementation. It is not a fused-kernel result and must never be
reported as one.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from fastprove.config import ModelConfig, ObfuscationConfig
from fastprove.layers.attention import AttentionMode
from fastprove.models.obfuscated import ObfuscatedTinyCausalLM
from fastprove.models.plain import PlainTinyCausalLM
from fastprove.seed import RequestContext


def _time(function, repeats: int, warmup: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            function()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            function()
            samples.append(time.perf_counter() - start)
    return min(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()

    torch.set_num_threads(arguments.threads)
    hidden = arguments.hidden_size
    config = ModelConfig(
        vocab_size=2048,
        hidden_size=hidden,
        intermediate_size=int(hidden * 2.6875),
        num_layers=arguments.layers,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_sequence_length=arguments.sequence_length + arguments.decode_tokens + 8,
    )
    obfuscation = ObfuscationConfig(
        hidden_noise_dim=16,
        value_noise_dim_per_head=2,
        max_condition_number=10.0,
        noise_propagation_gamma=0.5,
        refresh_mode="per_request",
        basis_block_size=16,
    )
    plain = PlainTinyCausalLM(
        config, seed=arguments.seed, debug_enabled=False
    ).eval()
    started = time.perf_counter()
    converted = ObfuscatedTinyCausalLM.from_plain(
        plain,
        obfuscation=obfuscation,
        mode=AttentionMode.EXACT,
        approximation=None,
        seed=arguments.seed,
        debug_enabled=False,
    )
    conversion_seconds = time.perf_counter() - started
    obfuscated = converted.module.eval()
    context = RequestContext(global_seed=arguments.seed, request_id="overhead")

    torch.manual_seed(arguments.seed)
    input_ids = torch.randint(
        0, config.vocab_size, (1, arguments.sequence_length)
    )

    prefill_plain = _time(
        lambda: plain(input_ids), arguments.repeats, arguments.warmup
    )
    prefill_obfuscated = _time(
        lambda: obfuscated(input_ids, request_context=context),
        arguments.repeats,
        arguments.warmup,
    )

    def decode_plain() -> None:
        logits, cache = plain(input_ids, use_cache=True)
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        for step in range(arguments.decode_tokens):
            position = torch.tensor([arguments.sequence_length + step])
            logits, cache = plain(
                token, positions=position, cache=cache, use_cache=True
            )
            token = logits[:, -1].argmax(dim=-1, keepdim=True)

    def decode_obfuscated() -> None:
        logits, cache = obfuscated(
            input_ids, request_context=context, use_cache=True
        )
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        for step in range(arguments.decode_tokens):
            position = torch.tensor([arguments.sequence_length + step])
            logits, cache = obfuscated(
                token,
                positions=position,
                request_context=context,
                cache=cache,
                use_cache=True,
            )
            token = logits[:, -1].argmax(dim=-1, keepdim=True)

    decode_plain_seconds = _time(decode_plain, arguments.repeats, arguments.warmup)
    decode_obfuscated_seconds = _time(
        decode_obfuscated, arguments.repeats, arguments.warmup
    )

    tokens = arguments.decode_tokens
    record = {
        "implementation": "eager reference, not a fused kernel",
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "threads": arguments.threads,
        "device": "cpu",
        "activation_dtype": "float32",
        "model": {
            "hidden_size": hidden,
            "intermediate_size": config.intermediate_size,
            "num_layers": config.num_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "sequence_length": arguments.sequence_length,
            "decode_tokens": tokens,
        },
        "obfuscation": {
            "hidden_noise_dim": obfuscation.hidden_noise_dim,
            "value_noise_dim_per_head": obfuscation.value_noise_dim_per_head,
            "basis_block_size": obfuscation.basis_block_size,
        },
        "conversion_seconds": conversion_seconds,
        "prefill_plaintext_seconds": prefill_plain,
        "prefill_obfuscated_seconds": prefill_obfuscated,
        "prefill_overhead_fraction": prefill_obfuscated / prefill_plain - 1.0,
        "decode_plaintext_seconds": decode_plain_seconds,
        "decode_obfuscated_seconds": decode_obfuscated_seconds,
        "decode_overhead_fraction": (
            decode_obfuscated_seconds / decode_plain_seconds - 1.0
        ),
        "decode_plaintext_tpot_seconds": decode_plain_seconds / tokens,
        "decode_obfuscated_tpot_seconds": decode_obfuscated_seconds / tokens,
    }
    path = Path(arguments.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
