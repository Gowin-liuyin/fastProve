"""S1 bit-exact equivalence recorder for the noise-scale configuration refactor.

S1 turns the hard-coded deployed noise scales into configuration fields
(``free_noise_energy_ratio`` for the refresh ``xi`` and
``signal_noise_coupling_scale`` for the linear coupling ``C``). The refactor is
accepted only if every recorded output is *bit-identical* before and after.
"Close enough" is not accepted: the exact-mode gates are compared against
recorded artifacts.

This script hashes, for one config and one deterministic input:

* the FP32 logits tensor produced by the production forward path;
* every tensor in the converted module's ``state_dict`` (all deployed weights
  and buffers), name by name.

Usage:
    PYTHONPATH=src python scripts/record_noise_scale_equivalence.py \
        --config configs/tiny_exact.yaml \
        --output results/raw/pre_s1_baseline.json
    # ... apply the refactor, then:
    PYTHONPATH=src python scripts/record_noise_scale_equivalence.py \
        --config configs/tiny_exact.yaml \
        --output results/raw/post_s1_tiny_exact.json
    PYTHONPATH=src python scripts/record_noise_scale_equivalence.py \
        --config configs/tiny_exact.yaml \
        --compare results/raw/pre_s1_baseline.json \
        --output results/raw/post_s1_tiny_exact.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from fastprove.config import load_config
from fastprove.layers.attention import ApproximationConfig, AttentionMode
from fastprove.models.obfuscated import ObfuscatedTinyCausalLM
from fastprove.models.plain import PlainTinyCausalLM
from fastprove.seed import RequestContext


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _state_digest(state: dict[str, torch.Tensor]) -> tuple[str, dict[str, str]]:
    per_tensor = {name: _tensor_digest(state[name]) for name in sorted(state)}
    digest = hashlib.sha256()
    for name in sorted(per_tensor):
        digest.update(name.encode("ascii"))
        digest.update(per_tensor[name].encode("ascii"))
    return digest.hexdigest(), per_tensor


def _run(config_path: Path) -> dict:
    config = load_config(config_path)
    plain = PlainTinyCausalLM(
        config.model, seed=config.runtime.seed, debug_enabled=False
    )
    approximation = None
    if config.attention.mode in (
        AttentionMode.TOPK_PRESERVING,
        AttentionMode.FREE_BOUNDED,
    ):
        approximation = ApproximationConfig(
            tau_max=config.attention.tau_max,
            tau_error=config.attention.tau_error,
            alpha=config.attention.alpha,
            preserve_top_k=config.attention.preserve_top_k,
        )
    converted = ObfuscatedTinyCausalLM.from_plain(
        plain,
        obfuscation=config.obfuscation,
        mode=config.attention.mode,
        approximation=approximation,
        seed=config.runtime.seed,
        debug_enabled=config.runtime.debug_enabled,
    )
    module = converted.module.eval()
    codec = converted.token_codec

    generator = torch.Generator().manual_seed(config.runtime.seed)
    input_ids = torch.randint(
        0, config.model.vocab_size, (2, config.model.max_sequence_length), generator=generator
    )
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    with torch.no_grad():
        logits = module(
            codec.encode(input_ids),
            token_mask=mask,
            request_context=RequestContext(config.runtime.seed, "s1-equivalence"),
        )
    weights_digest, per_tensor = _state_digest(module.state_dict())
    return {
        "config": str(config_path),
        "attention_mode": config.attention.mode.value,
        "obfuscation": {
            "hidden_noise_dim": config.obfuscation.hidden_noise_dim,
            "value_noise_dim_per_head": config.obfuscation.value_noise_dim_per_head,
            "refresh_mode": config.obfuscation.refresh_mode,
            "free_noise_energy_ratio": getattr(
                config.obfuscation, "free_noise_energy_ratio", None
            ),
            "signal_noise_coupling_scale": getattr(
                config.obfuscation, "signal_noise_coupling_scale", None
            ),
        },
        "torch_version": torch.__version__,
        "seed": config.runtime.seed,
        "input_ids_sha256": hashlib.sha256(input_ids.numpy().tobytes()).hexdigest(),
        "logits_sha256": _tensor_digest(logits),
        "state_dict_sha256": weights_digest,
        "state_dict_per_tensor": per_tensor,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compare", type=Path, default=None)
    arguments = parser.parse_args()

    record = _run(arguments.config)
    if arguments.compare is not None:
        baseline = json.loads(arguments.compare.read_text(encoding="utf-8"))
        mismatched = [
            name
            for name in sorted(record["state_dict_per_tensor"])
            if baseline["state_dict_per_tensor"].get(name)
            != record["state_dict_per_tensor"][name]
        ]
        comparison = {
            "baseline": str(arguments.compare),
            "logits_bit_identical": (
                baseline["logits_sha256"] == record["logits_sha256"]
            ),
            "state_dict_bit_identical": (
                baseline["state_dict_sha256"] == record["state_dict_sha256"]
            ),
            "mismatched_tensors": mismatched,
        }
        record["comparison"] = comparison
        print(json.dumps(comparison, indent=2))
        if not (
            comparison["logits_bit_identical"]
            and comparison["state_dict_bit_identical"]
        ):
            raise SystemExit("S1 equivalence regression FAILED")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("wrote %s" % arguments.output)


if __name__ == "__main__":
    main()
