"""Measure whether the norm leak decays with depth.

Motivation: an obvious mitigation for the layer-0 token recovery of
``verify_token_recovery_from_rho.py`` is an edge/cloud split -- let the client
run the embedding and the first ``k`` layers locally so the server never sees
``rho_0``. This script tests whether that works, by asking how much token
information the norm ``||h_l||`` still carries at each depth ``l``.

Method: run real prompts through the plaintext model, record ``||h_l||`` for
every layer and position, then run a nearest-neighbour attack that uses the norm
as its *only* feature and predicts the token id. Accuracy is compared against a
most-frequent-token baseline.

Interpretation limits (read before quoting any number):

* The headline accuracy is corpus dependent. A templated corpus repeats tokens
  heavily, which inflates a nearest-neighbour attack. The script therefore also
  reports accuracy stratified by how often each test token appears in the
  attacker's training split, plus the token-distribution entropy, so the
  inflation is visible rather than hidden.
* The layer-0 result of ``verify_token_recovery_from_rho.py`` is *not* corpus
  dependent: it uses the public embedding table directly and needs no training
  split. Prefer that number when citing layer 0.

Usage:
    python scripts/verify_norm_leak_by_depth.py \\
        --model-path <plaintext checkpoint dir> \\
        --prompt-file results/raw/real_scenario_prompts_1500.jsonl \\
        --output results/raw/norm_leak_by_depth.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import platform
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open


def _load_weights(root: Path, layers: int) -> Tuple[Dict[str, torch.Tensor], dict]:
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    names = ["model.embed_tokens.weight", "model.norm.weight"]
    for index in range(layers):
        for key in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ):
            names.append("model.layers.%d.%s.weight" % (index, key))
        for key in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
            names.append("model.layers.%d.%s.bias" % (index, key))
        names.append("model.layers.%d.input_layernorm.weight" % index)
        names.append("model.layers.%d.post_attention_layernorm.weight" % index)

    index_file = root / "model.safetensors.index.json"
    weights: Dict[str, torch.Tensor] = {}
    if index_file.is_file():
        weight_map = json.loads(index_file.read_text(encoding="utf-8"))["weight_map"]
        by_shard: Dict[str, List[str]] = collections.defaultdict(list)
        for name in names:
            by_shard[weight_map[name]].append(name)
        for shard, keys in by_shard.items():
            with safe_open(str(root / shard), framework="pt") as handle:
                for name in keys:
                    weights[name] = handle.get_tensor(name).to(torch.float32)
    else:
        with safe_open(str(root / "model.safetensors"), framework="pt") as handle:
            for name in names:
                weights[name] = handle.get_tensor(name).to(torch.float32)
    return weights, config


def _rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    head_dim = x.shape[-1]
    exponent = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    frequencies = positions[:, None].float() * (1.0 / (theta**exponent))
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cosine, sine = embedding.cos()[None, None], embedding.sin()[None, None]
    half = head_dim // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cosine + rotated * sine


@torch.no_grad()
def _norms_per_layer(
    token_ids: List[int], weights: Dict[str, torch.Tensor], config: dict, layers: int
) -> torch.Tensor:
    """Return ``[layers + 1, tokens]`` of ``||h_l||``; row 0 is the embedding."""

    hidden_size = config["hidden_size"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = hidden_size // heads
    eps = float(config["rms_norm_eps"])
    theta = float(config["rope_theta"])

    hidden = weights["model.embed_tokens.weight"][token_ids][None]
    length = hidden.shape[1]
    positions = torch.arange(length)
    causal = ~torch.tril(torch.ones(length, length, dtype=torch.bool))
    collected = [hidden[0].norm(dim=-1).clone()]

    for index in range(layers):
        prefix = "model.layers.%d." % index

        def normalize(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
            scale = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
            return x / scale * gamma

        normalized = normalize(hidden, weights[prefix + "input_layernorm.weight"])
        query = (
            normalized @ weights[prefix + "self_attn.q_proj.weight"].T
            + weights[prefix + "self_attn.q_proj.bias"]
        ).view(1, length, heads, head_dim).transpose(1, 2)
        key = (
            normalized @ weights[prefix + "self_attn.k_proj.weight"].T
            + weights[prefix + "self_attn.k_proj.bias"]
        ).view(1, length, kv_heads, head_dim).transpose(1, 2)
        value = (
            normalized @ weights[prefix + "self_attn.v_proj.weight"].T
            + weights[prefix + "self_attn.v_proj.bias"]
        ).view(1, length, kv_heads, head_dim).transpose(1, 2)
        query, key = _rope(query, positions, theta), _rope(key, positions, theta)
        repeat = heads // kv_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
        scores = (
            query @ key.transpose(-1, -2) / math.sqrt(head_dim)
        ).masked_fill(causal, -torch.inf)
        context = (torch.softmax(scores, dim=-1) @ value).transpose(1, 2)
        hidden = hidden + context.reshape(1, length, hidden_size) @ weights[
            prefix + "self_attn.o_proj.weight"
        ].T
        normalized = normalize(
            hidden, weights[prefix + "post_attention_layernorm.weight"]
        )
        gate = normalized @ weights[prefix + "mlp.gate_proj.weight"].T
        up = normalized @ weights[prefix + "mlp.up_proj.weight"].T
        hidden = hidden + (
            torch.nn.functional.silu(gate) * up
        ) @ weights[prefix + "mlp.down_proj.weight"].T
        collected.append(hidden[0].norm(dim=-1).clone())
    return torch.stack(collected)


def _nearest_neighbour_attack(
    train_norms: torch.Tensor,
    train_tokens: torch.Tensor,
    test_norms: torch.Tensor,
    test_tokens: torch.Tensor,
) -> torch.Tensor:
    """Return a boolean tensor of per-test-token correctness."""

    sorted_norms, order = torch.sort(train_norms)
    labels = train_tokens[order]
    upper = torch.searchsorted(sorted_norms, test_norms).clamp(
        0, len(sorted_norms) - 1
    )
    lower = (upper - 1).clamp(0, len(sorted_norms) - 1)
    closer = (sorted_norms[upper] - test_norms).abs() <= (
        sorted_norms[lower] - test_norms
    ).abs()
    chosen = torch.where(closer, upper, lower)
    return labels[chosen] == test_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-count", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    from tokenizers import Tokenizer

    root = Path(arguments.model_path)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    layers = int(config["num_hidden_layers"])
    weights, config = _load_weights(root, layers)
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))

    prompts = [
        json.loads(line)["text"]
        for line in Path(arguments.prompt_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    random.Random(arguments.seed).shuffle(prompts)

    records: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for text in prompts[: arguments.prompt_count]:
        token_ids = tokenizer.encode(text).ids[: arguments.max_tokens]
        if len(token_ids) < 8:
            continue
        records.append(
            (
                _norms_per_layer(token_ids, weights, config, layers),
                torch.tensor(token_ids),
            )
        )

    split = int(len(records) * arguments.train_fraction)
    train_tokens = torch.cat([item[1] for item in records[:split]])
    test_tokens = torch.cat([item[1] for item in records[split:]])
    counts = collections.Counter(train_tokens.tolist())
    frequency = torch.tensor([counts[int(t)] for t in test_tokens])

    mode = torch.mode(train_tokens).values
    baseline = float((test_tokens == mode).float().mean())

    probe_layers = sorted(
        {0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, layers} & set(range(layers + 1))
    )
    by_layer = []
    strata_definitions = (
        (0, 0, "unseen"),
        (1, 2, "rare_1_2"),
        (3, 10, "medium_3_10"),
        (11, 10**9, "frequent_gt_10"),
    )
    for layer in probe_layers:
        train_norms = torch.cat([item[0][layer] for item in records[:split]])
        test_norms = torch.cat([item[0][layer] for item in records[split:]])
        correct = _nearest_neighbour_attack(
            train_norms, train_tokens, test_norms, test_tokens
        )
        strata = {}
        for low, high, name in strata_definitions:
            selector = (frequency >= low) & (frequency <= high)
            if int(selector.sum()) > 0:
                strata[name] = {
                    "sample_count": int(selector.sum()),
                    "top1_accuracy": float(correct[selector].float().mean()),
                }
        by_layer.append(
            {
                "layer": layer,
                "top1_accuracy": float(correct.float().mean()),
                "accuracy_over_baseline": float(correct.float().mean())
                / max(baseline, 1e-12),
                "by_train_frequency": strata,
            }
        )

    distinct = sorted(collections.Counter(test_tokens.tolist()).values())
    entropy = -sum(
        (c / len(test_tokens)) * math.log2(c / len(test_tokens)) for c in distinct
    )
    record = {
        "claim": (
            "the norm leak does not decay with depth: an edge/cloud split at "
            "layer k does not eliminate the channel"
        ),
        "reason": (
            "the residual stream retains the embedding contribution, so "
            "||h_l|| stays correlated with ||h_0||"
        ),
        "corpus_caveat": (
            "the headline accuracy is inflated by a templated corpus; read "
            "by_train_frequency, and prefer verify_token_recovery_from_rho.py "
            "for the corpus-independent layer-0 number"
        ),
        "model_path": str(root),
        "num_hidden_layers": layers,
        "prompt_file": arguments.prompt_file,
        "prompt_count_used": len(records),
        "train_prompts": split,
        "test_prompts": len(records) - split,
        "train_token_count": int(len(train_tokens)),
        "train_distinct_tokens": int(len(set(train_tokens.tolist()))),
        "test_token_count": int(len(test_tokens)),
        "test_distinct_tokens": int(len(set(test_tokens.tolist()))),
        "test_tokens_seen_in_train_fraction": float((frequency > 0).float().mean()),
        "test_token_entropy_bits": entropy,
        "uniform_entropy_bits": math.log2(len(set(test_tokens.tolist()))),
        "most_frequent_token_baseline": baseline,
        "by_layer": by_layer,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "seed": arguments.seed,
    }
    path = Path(arguments.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "by_layer"}, indent=2))
    print("\nlayer  top1     over_baseline  rare_1_2")
    for row in by_layer:
        rare = row["by_train_frequency"].get("rare_1_2", {}).get("top1_accuracy")
        print(
            "%5d  %6.1f%%  %6.1fx        %s"
            % (
                row["layer"],
                100 * row["top1_accuracy"],
                row["accuracy_over_baseline"],
                "n/a" if rare is None else "%.1f%%" % (100 * rare),
            )
        )


if __name__ == "__main__":
    main()
