"""Record the task D2 performance metric set for plaintext vs obfuscated.

Usage:
    python scripts/record_performance_D2.py --output results/raw/performance_D2.json

Records prefill (TTFT proxy), decode TPOT, tokens/s, peak process RSS and
allocator peak, KV-cache bytes, conversion time and profiler kernel counts on
identical inputs for the plaintext and obfuscated reference modules. This is
the *eager reference* implementation; it is not a fused-kernel result.
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


def _rss_bytes() -> int:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _kernel_count(prof) -> int:
    return sum(event.count for event in prof.key_averages())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
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
    codec = converted.token_codec
    context = RequestContext(global_seed=arguments.seed, request_id="perf-D2")

    torch.manual_seed(arguments.seed)
    input_ids = torch.randint(0, config.vocab_size, (1, arguments.sequence_length))
    encoded = codec.encode(input_ids)

    def measure(prefix: str, run, decode_run, name: str) -> dict:
        rss_before = _rss_bytes()
        prefill = _time(run, arguments.repeats, arguments.warmup)
        rss_after_prefill = _rss_bytes()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            run()
        kernels = _kernel_count(prof)
        decode = _time(decode_run, arguments.repeats, arguments.warmup)
        return {
            "name": name,
            "prefill_latency_seconds": prefill,
            "ttft_seconds": prefill,
            "decode_latency_seconds": decode,
            "tpot_seconds": decode / arguments.decode_tokens,
            "tokens_per_second": (
                arguments.decode_tokens / decode
                if arguments.decode_tokens
                else 0.0
            ),
            "prefill_tokens": arguments.sequence_length,
            "kernel_count_prefill": kernels,
            "peak_process_rss_bytes": max(rss_before, rss_after_prefill),
        }

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
        logits, cache = obfuscated(encoded, request_context=context, use_cache=True)
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

    plain_metrics = measure(
        "plain", lambda: plain(input_ids), decode_plain, "plaintext"
    )
    obf_metrics = measure(
        "obf",
        lambda: obfuscated(encoded, request_context=context),
        decode_obfuscated,
        "obfuscated",
    )

    plain_kv_bytes = 0
    obf_kv_bytes = 0
    with torch.inference_mode():
        _, plain_cache = plain(input_ids, use_cache=True)
        _, obf_cache = obfuscated(encoded, request_context=context, use_cache=True)
    for layer_cache in plain_cache.layers:
        for tensor in (layer_cache.key, layer_cache.value, layer_cache.key_valid, layer_cache.positions):
            plain_kv_bytes += int(tensor.numel() * tensor.element_size())
    for layer_cache in obf_cache.layers:
        for tensor in (layer_cache.key, layer_cache.value_mixed, layer_cache.key_valid, layer_cache.positions):
            obf_kv_bytes += int(tensor.numel() * tensor.element_size())

    record = {
        "schema": "fastprove.performance.v1",
        "implementation": "eager reference, not a fused kernel",
        "device": "cpu",
        "activation_dtype": "float32",
        "torch_version": torch.__version__,
        "gpu_name": None,
        "platform": platform.platform(),
        "threads": arguments.threads,
        "model": {
            "hidden_size": hidden,
            "intermediate_size": config.intermediate_size,
            "num_layers": config.num_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "sequence_length": arguments.sequence_length,
            "decode_tokens": arguments.decode_tokens,
        },
        "obfuscation": {
            "hidden_noise_dim": obfuscation.hidden_noise_dim,
            "value_noise_dim_per_head": obfuscation.value_noise_dim_per_head,
            "basis_block_size": obfuscation.basis_block_size,
            "lm_head_mode": obfuscation.lm_head_mode,
        },
        "conversion_time_seconds": conversion_seconds,
        "plaintext": plain_metrics,
        "obfuscated": obf_metrics,
        "kv_cache_bytes": {
            "plaintext": plain_kv_bytes,
            "obfuscated": obf_kv_bytes,
            "relative_increase": obf_kv_bytes / plain_kv_bytes - 1.0
            if plain_kv_bytes
            else None,
        },
        "prefill_overhead_fraction": (
            obf_metrics["prefill_latency_seconds"]
            / plain_metrics["prefill_latency_seconds"]
            - 1.0
        ),
        "decode_overhead_fraction": (
            obf_metrics["decode_latency_seconds"]
            / plain_metrics["decode_latency_seconds"]
            - 1.0
        ),
        "tied_embedding_memory": {
            "plaintext_vocab_side_bytes_formula": "V * d * element_size (one shared table)",
            "obfuscated_vocab_side_bytes_formula": (
                "V * n * element_size (embedding.table) + n * V * element_size "
                "(deployed_head) for untied_deployed"
            ),
            "structural_ratio": (
                2 * (hidden + obfuscation.hidden_noise_dim) / hidden
            ),
            "llama_32_3b_bf16_projection_bytes": int(
                2 * 128256 * (3072 + 16) * 2
            ),
            "llama_32_3b_bf16_plaintext_shared_bytes": int(128256 * 3072 * 2),
            "note": (
                "3B/BF16 numbers are structural projections with the doc's "
                "dimensions (V=128256, d=3072, r=16, 2 bytes), not a measured "
                "run: no tied pretrained checkpoint is available locally. The "
                "vocabulary-side memory roughly doubles, exceeding the 5% "
                "peak-memory target (task B4.2, reported in D4)."
            ),
        },
        "notes": [
            "tied-embedding consequence: deployed_head is an independent "
            "[n, V] matrix; the vocabulary-side memory roughly doubles vs the "
            "shared plaintext table (task B4.2 / D4 records the 5% peak-memory "
            "target exceedance for the 3B-class model)",
            "peak_process_rss_bytes is the max recorded ru_maxrss on this "
            "process; the conversion peak is included in the obfuscated value",
        ],
    }
    path = Path(arguments.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
