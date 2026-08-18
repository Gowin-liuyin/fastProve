"""Recover input tokens from the RMSNorm scale alone (layer 0).

This is the concrete consequence of the correctness-forced norm leak recorded in
``docs/threat_model.md`` 5bis.9 and ``docs/PROBLEMS.md`` section 1.1. It is not a
hypothetical channel: it recovers the plaintext input token ids that the
vocabulary permutation of task C1/C2 is supposed to hide.

Why it works
------------
1. The deployed server must compute ``rho = sqrt(||h||^2 / d + eps)`` correctly,
   otherwise RMSNorm and therefore the model output are wrong (constraint C1).
2. ``rho`` algebraically determines ``||h||``: ``||h||^2 = (rho^2 - eps) * d``.
3. At layer 0, ``h_0 = E[token]`` is a row of the embedding table, so
   ``||h_0||`` is a per-token constant.
4. **Row norms are invariant under a row permutation.** The vocabulary
   permutation ``tau`` reorders embedding rows but does not change any row's
   norm, so it provides no protection on this channel.
5. The base checkpoint is public, so the attacker can build the
   ``norm -> token`` table offline with no known-plaintext pairs and no
   knowledge of the mixing basis ``M``.

The dominant factor is the precision at which ``rho`` is materialized. AGENTS.md
R5 *requires* FP32 or better for RMS statistics, and the exact-mode gate is
judged in FP32; that requirement is exactly what makes the attack succeed.

Usage:
    python scripts/verify_token_recovery_from_rho.py \\
        --model-path <plaintext checkpoint dir> \\
        --output results/raw/token_recovery_from_rho.json
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open

_FP64 = torch.float64


def _load_embedding(root: Path) -> torch.Tensor:
    """Load the plaintext embedding table from a local safetensors checkpoint."""

    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        target = next(
            key for key in weight_map if key.endswith("embed_tokens.weight")
        )
        shard = root / weight_map[target]
        with safe_open(str(shard), framework="pt") as handle:
            return handle.get_tensor(target).to(dtype=_FP64)
    single = root / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError("no safetensors weights under %s" % root)
    with safe_open(str(single), framework="pt") as handle:
        target = next(
            key for key in handle.keys() if key.endswith("embed_tokens.weight")
        )
        return handle.get_tensor(target).to(dtype=_FP64)


def _rms_epsilon(root: Path) -> float:
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    return float(config.get("rms_norm_eps", 1e-5))


def _quantize(value: torch.Tensor, dtype_name: str) -> Tuple[torch.Tensor, float]:
    """Return ``rho`` as stored in ``dtype_name`` plus its absolute resolution."""

    if dtype_name == "float64":
        stored = value
    elif dtype_name == "float32":
        stored = value.to(torch.float32)
    elif dtype_name == "bfloat16":
        stored = value.to(torch.bfloat16)
    else:
        raise ValueError("unsupported dtype %s" % dtype_name)
    exact = stored.to(_FP64)
    upper = torch.nextafter(
        stored.clone(), torch.tensor(float("inf"), dtype=stored.dtype)
    ).to(_FP64)
    return exact, float(upper - exact)


def _attack(
    embedding: torch.Tensor,
    epsilon: float,
    token_ids: List[int],
    dtype_name: str,
) -> Dict[str, object]:
    """Recover tokens from ``rho`` at a given storage precision."""

    hidden_dim = embedding.shape[1]
    norms = embedding.norm(dim=-1)
    sorted_norms, order = torch.sort(norms)

    candidate_counts: List[int] = []
    unique = 0
    unique_and_correct = 0
    for token in token_ids:
        exact_rho = torch.sqrt(
            embedding[token].pow(2).mean() + epsilon
        )
        stored_rho, resolution = _quantize(exact_rho, dtype_name)
        recovered = torch.sqrt(
            ((stored_rho * stored_rho - epsilon) * hidden_dim).clamp_min(0)
        )
        # Propagate one ulp of rho to the norm: d(||h||) = rho * d / ||h|| * d(rho)
        window = max(
            resolution * hidden_dim * float(stored_rho) / max(float(recovered), 1e-12),
            1e-13,
        )
        low = int(torch.searchsorted(sorted_norms, recovered - window))
        high = int(
            torch.searchsorted(sorted_norms, recovered + window, right=True)
        )
        count = high - low
        candidate_counts.append(count)
        if count == 1:
            unique += 1
            if int(order[low]) == token:
                unique_and_correct += 1

    total = len(token_ids)
    counts = torch.tensor(candidate_counts, dtype=_FP64)
    return {
        "rho_storage_dtype": dtype_name,
        "mean_candidate_count": float(counts.mean()),
        "median_candidate_count": float(counts.median()),
        "uniquely_determined_fraction": unique / total,
        "unique_and_correct_fraction": unique_and_correct / total,
        "sample_count": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-count", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()

    root = Path(arguments.model_path)
    embedding = _load_embedding(root)
    epsilon = _rms_epsilon(root)
    vocab_size, hidden_dim = embedding.shape
    norms = embedding.norm(dim=-1)

    generator = torch.Generator().manual_seed(arguments.seed)
    token_ids = torch.randint(
        0, vocab_size, (arguments.sample_count,), generator=generator
    ).tolist()

    results = [
        _attack(embedding, epsilon, token_ids, name)
        for name in ("float64", "float32", "bfloat16")
    ]

    record = {
        "claim": (
            "the RMSNorm scale rho alone recovers layer-0 input token ids; the "
            "vocabulary permutation provides no protection because row norms "
            "are invariant under a row permutation"
        ),
        "requires_known_plaintext_pairs": False,
        "requires_knowledge_of_the_mixing_basis": False,
        "requires_public_base_checkpoint": True,
        "forced_by": (
            "constraint C1 (exact output): the server must compute rho "
            "correctly, and AGENTS.md R5 requires FP32 or better for RMS "
            "statistics, which is the precision that makes this succeed"
        ),
        "model_path": str(root),
        "vocab_size": int(vocab_size),
        "hidden_size": int(hidden_dim),
        "rms_norm_eps": epsilon,
        "embedding_norm_statistics": {
            "min": float(norms.min()),
            "max": float(norms.max()),
            "mean": float(norms.mean()),
            "std": float(norms.std()),
            "distinct_fraction": float(
                len(torch.unique(norms)) / vocab_size
            ),
        },
        "by_rho_precision": results,
        "mitigation_note": (
            "storing rho in bfloat16 blocks this channel but violates "
            "AGENTS.md R5 and exceeds the exact-mode logit tolerance, so it "
            "trades the privacy leak for a correctness failure rather than "
            "resolving the tension"
        ),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "seed": arguments.seed,
    }
    path = Path(arguments.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
