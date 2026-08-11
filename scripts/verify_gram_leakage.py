"""Quantify what the server-side Gram blocks reveal about the hidden state.

The deployed forward pass computes the RMSNorm scale ``rho`` from the blockwise
Gram of ``A = P P^T`` (``P = M^-1[:, :d]``). The Gram blocks and ``perm_out``
must therefore live in the *server* bundle. This script measures the
consequence, which is stronger than the condition-number bound recorded in
``docs/threat_model.md`` 5bis.3.

Result: ``A`` determines ``P`` up to a right orthogonal factor, so an observer
holding only the server bundle recovers ``h`` up to one global orthogonal map.
Norms, pairwise inner products and pairwise distances of ``h`` are exposed
*exactly*, not merely bounded by ``kappa``.

Usage:
    python scripts/verify_gram_leakage.py --output results/raw/gram_leakage.json
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from fastprove.structured import generate_structured_basis

_FP64 = torch.float64


def _recover_up_to_orthogonal(
    mixed: torch.Tensor, gram: torch.Tensor, signal_dim: int
) -> torch.Tensor:
    """Return ``h Q`` for some orthogonal ``Q``, using only ``A = P P^T``.

    Any factorization ``A = R R^T`` with ``R`` of shape ``[n, d]`` satisfies
    ``R = P Q``, so ``c @ R = h P^-1 P Q = h Q``. The symmetric eigenbasis is
    one such factorization and requires no knowledge of ``M``.
    """

    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)[:signal_dim]
    factor = eigenvectors[:, order] * eigenvalues[order].clamp_min(0).sqrt()
    return mixed @ factor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--signal-dim", type=int, default=256)
    parser.add_argument("--noise-dim", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    signal_dim = arguments.signal_dim
    basis = generate_structured_basis(
        signal_dim,
        arguments.noise_dim,
        seed=arguments.seed,
        domain="gram-leakage",
        block_size=arguments.block_size,
        dtype=_FP64,
    )
    projection = basis.signal_projection()
    gram = projection @ projection.T

    signal = torch.randn(arguments.samples, signal_dim, dtype=_FP64)
    noise = torch.randn(arguments.samples, arguments.noise_dim, dtype=_FP64)
    mixed = basis.mix(torch.cat((signal, noise), dim=-1))

    recovered = _recover_up_to_orthogonal(mixed, gram, signal_dim)

    norm_error = float(
        (recovered.norm(dim=-1) - signal.norm(dim=-1)).abs().max()
    )
    inner_error = float((recovered @ recovered.T - signal @ signal.T).abs().max())
    distance_error = float(
        (torch.cdist(recovered, recovered) - torch.cdist(signal, signal))
        .abs()
        .max()
    )
    # A recovery that ignored the Gram would do no better than chance on norms;
    # record that contrast so the number above is interpretable.
    baseline_norm_error = float(
        (mixed.norm(dim=-1) - signal.norm(dim=-1)).abs().max()
    )

    record = {
        "claim": (
            "server-side Gram blocks expose the exact Euclidean geometry of h; "
            "h is recoverable up to one global orthogonal map without any "
            "known-plaintext pair and without knowledge of M"
        ),
        "supersedes": (
            "threat_model 5bis.3 bounded distance distortion by kappa; the "
            "distances are in fact exact"
        ),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "seed": arguments.seed,
        "signal_dim": signal_dim,
        "noise_dim": arguments.noise_dim,
        "block_size": arguments.block_size,
        "samples": arguments.samples,
        "arithmetic_dtype": "float64",
        "max_absolute_error_of_recovered_norms": norm_error,
        "max_absolute_error_of_pairwise_inner_products": inner_error,
        "max_absolute_error_of_pairwise_distances": distance_error,
        "baseline_norm_error_without_using_the_gram": baseline_norm_error,
        "interpretation": (
            "errors at FP64 round-off level mean the geometry is exposed "
            "exactly, not approximately"
        ),
    }
    path = Path(arguments.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
