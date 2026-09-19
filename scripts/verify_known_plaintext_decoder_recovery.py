"""Tier-0 attack: closed-form semantic-decoder recovery from known plaintext.

Implements the T5 attack of ``docs/noise_placement_explained.tex`` against the
actual production basis generator (``generate_structured_basis``, the same
object the deployed bundle ships) in FP64 torch, and measures:

* the sample threshold: ``T >= d + r`` realises the rank condition
  ``dim ker(B) == d``; ``T = d + r + 1`` is the theoretical minimum with one
  spare sample for rank conditions;
* ``heldout_rel_error`` on fresh-noise held-out observations. This is the
  metric that matters: under the current ``D == d`` concat construction the
  attacker's ``L_hat`` is **not** the matrix ``L`` (relative matrix difference
  can be ~0.33) yet is pointwise exact on the reachable state set, so only
  reporting ``decoder_rel_error_vs_true_L`` would wrongly conclude "attack
  failed";
* the attack cost vector of ``docs/work_factor_security.md`` section 2
  (analytic FLOPs estimate; an upper estimate, not an optimal-complexity lower
  bound, matching section 4.1 wording).

Algorithm (row-vector convention).  With ``c_t = h_t A + nu_t B``:

1. ``Q = qr(h_mat)``; ``resid = c_mat - Q (Q^T c_mat)``   (projects out col(H))
2. ``V = right_null_space(resid)``                        (== ker B when T-d>=r)
3. ``AV = lstsq(h_mat, c_mat @ V)``
4. ``L_hat = V @ pinv(AV)``                               (satisfies c L_hat = h)

Claims and non-claims: this is a known-plaintext algebraic attack. It needs no
kernel modification, no register access, no solution of a hard problem, and no
knowledge of the mixing basis ``M``; it needs the public base checkpoint (to
know ``h`` for chosen tokens) and observed mixed states ``c``. It does not
break the scheme on its own: it recovers only the semantic read map, which is
exactly the map the client's decode uses.

Usage:
    PYTHONPATH=src python scripts/verify_known_plaintext_decoder_recovery.py \
        --output results/raw/known_plaintext_decoder_recovery.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch

from fastprove.structured import generate_structured_basis

_FP64 = torch.float64
#: A recovery counts as exact when the held-out relative error is at most this.
#: The failing regime is O(0.1-1); FP64 round-off is ~1e-15 at unit noise scale
#: and grows to ~1e-9 when the in-cone noise is amplified by 1e6 (cancellation
#: in ``c @ L``, not a failure of the algebra). The threshold is therefore
#: scale-robust rather than pinned at 1e-10.
_RECOVERY_TOLERANCE = 1e-6


def right_null_space(matrix: torch.Tensor, tolerance: float = 1e-9) -> torch.Tensor:
    """Return a basis (columns) of ``{v : matrix @ v == 0}`` in FP64."""

    _, singular, vh = torch.linalg.svd(matrix, full_matrices=True)
    if singular.numel() == 0:
        return vh.T.contiguous()
    rank = int((singular > tolerance * max(1.0, float(singular.max()))).sum())
    return vh[rank:].T.contiguous()


def recover_decoder(
    h_mat: torch.Tensor, c_mat: torch.Tensor, *, tolerance: float = 1e-9
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Closed-form semantic-decoder recovery; see the module docstring."""

    if h_mat.ndim != 2 or c_mat.ndim != 2:
        raise ValueError("known-plaintext matrices must be rank two")
    if h_mat.shape[0] != c_mat.shape[0]:
        raise ValueError("h_mat and c_mat must share the sample axis")
    samples, signal_dim = h_mat.shape
    q_h, _ = torch.linalg.qr(h_mat.to(_FP64))
    residual = c_mat.to(_FP64) - q_h @ (q_h.T @ c_mat.to(_FP64))
    _, singular, _ = torch.linalg.svd(residual, full_matrices=True)
    rank = int((singular > tolerance * max(1.0, float(singular.max()))).sum())
    v = right_null_space(residual, tolerance)
    diagnostics: dict[str, Any] = {
        "samples": samples,
        "signal_dim": signal_dim,
        "residual_rank": rank,
        "null_space_dimension": int(v.shape[1]),
        "rank_condition_met": bool(v.shape[1] >= signal_dim),
    }
    if v.shape[1] < signal_dim:
        diagnostics["reason"] = "right null space smaller than the signal rank"
        return None, diagnostics
    av = torch.linalg.lstsq(h_mat.to(_FP64), c_mat.to(_FP64) @ v).solution
    return v @ torch.linalg.pinv(av), diagnostics


def _relative_error(predicted: torch.Tensor, target: torch.Tensor) -> float:
    denominator = float(target.norm())
    if denominator == 0.0:
        return float("inf")
    return float((predicted - target).norm()) / denominator


def _analytic_cost(
    samples: int, signal_dim: int, ambient: int, wall_clock: float
) -> dict[str, Any]:
    """Analytic cost vector; upper estimate, not a lower bound."""

    qr = 2.0 * samples * signal_dim * signal_dim
    residual = 2.0 * samples * signal_dim * ambient + samples * ambient
    svd = 4.0 * samples * ambient * ambient
    lstsq = 2.0 * samples * signal_dim * ambient
    left_inverse = 6.0 * signal_dim**3
    estimated_flops = qr + residual + svd + lstsq + left_inverse
    peak_memory = 8.0 * (
        samples * ambient + samples * samples + ambient * ambient
        + samples * signal_dim
    )
    per_target_flops = 2.0 * ambient * signal_dim
    return {
        "model": (
            "known-plaintext closed form: QR + SVD + lstsq, O(T N^2); analytic "
            "upper estimate, not an optimal-complexity lower bound"
        ),
        "estimated_flops": estimated_flops,
        "wall_clock_seconds": wall_clock,
        "peak_memory_bytes": int(peak_memory),
        "active_queries": samples,
        "passive_observations": 0,
        "training_data_records": 0,
        "setup_seconds": wall_clock,
        "per_target_seconds": wall_clock / max(samples, 1),
        "per_target_flops": per_target_flops,
        "right_censored": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--signal-dim", type=int, default=32)
    parser.add_argument("--noise-dim", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument(
        "--sample-counts",
        default=None,
        help="comma-separated T values; default sweeps around d + r",
    )
    parser.add_argument("--heldout-count", type=int, default=64)
    parser.add_argument(
        "--free-noise-scale",
        type=float,
        default=1.0,
        help="scale of the in-ker fresh noise per sample",
    )
    arguments = parser.parse_args()

    basis = generate_structured_basis(
        arguments.signal_dim,
        arguments.noise_dim,
        seed=arguments.seed,
        domain="known-plaintext-attack",
        block_size=arguments.block_size,
        dtype=_FP64,
    )
    signal_dim = arguments.signal_dim
    nuisance_rank = arguments.noise_dim
    ambient = basis.total_dim
    theoretical = signal_dim + nuisance_rank + 1
    if arguments.sample_counts:
        sample_counts = [
            int(value) for value in arguments.sample_counts.split(",")
        ]
    else:
        sample_counts = [
            2 * signal_dim,
            theoretical - 2,
            theoretical - 1,
            theoretical,
            theoretical + 1,
            theoretical + 8,
            4 * theoretical,
        ]
    generator = torch.Generator().manual_seed(arguments.seed)
    true_projection = basis.signal_projection()

    by_sample_count: list[dict[str, Any]] = []
    for samples in sample_counts:
        h_mat = torch.randn(samples, signal_dim, generator=generator, dtype=_FP64)
        noise = (
            torch.randn(samples, nuisance_rank, generator=generator, dtype=_FP64)
            * arguments.free_noise_scale
        )
        c_mat = basis.mix(torch.cat((h_mat, noise), dim=-1))
        started = time.perf_counter()
        decoder, diagnostics = recover_decoder(h_mat, c_mat)
        wall_clock = time.perf_counter() - started
        entry: dict[str, Any] = {"T": samples, **diagnostics}
        if decoder is None:
            entry["recovered"] = False
            entry["cost"] = _analytic_cost(samples, signal_dim, ambient, wall_clock)
            by_sample_count.append(entry)
            continue
        heldout_h = torch.randn(
            arguments.heldout_count, signal_dim, generator=generator, dtype=_FP64
        )
        heldout_noise = (
            torch.randn(
                arguments.heldout_count, nuisance_rank, generator=generator, dtype=_FP64
            )
            * arguments.free_noise_scale
        )
        heldout_c = basis.mix(torch.cat((heldout_h, heldout_noise), dim=-1))
        heldout_error = _relative_error(heldout_c @ decoder, heldout_h)
        entry.update(
            {
                "recovered": bool(heldout_error < _RECOVERY_TOLERANCE),
                "heldout_rel_error": heldout_error,
                "decoder_rel_error_vs_true_L": _relative_error(
                    decoder, true_projection
                ),
                "cost": _analytic_cost(samples, signal_dim, ambient, wall_clock),
            }
        )
        by_sample_count.append(entry)

    successful = [entry for entry in by_sample_count if entry.get("recovered")]
    record = {
        "claim": (
            "with known plaintext pairs the semantic decoder is recovered in "
            "closed form from T = d + nuisance_rank + 1 samples; the recovered "
            "map is exact on fresh-noise reachable states even when it differs "
            "from the true matrix L"
        ),
        "requires_known_plaintext_pairs": True,
        "requires_knowledge_of_the_mixing_basis": False,
        "requires_public_base_checkpoint": True,
        "requires_modified_kernel": False,
        "requires_register_access": False,
        "theoretical_min_samples": "d + nuisance_rank + 1",
        "geometry": {
            "d_semantic": signal_dim,
            "r_nuisance": nuisance_rank,
            "ambient_N": ambient,
            "condition_number": basis.condition_number,
            "block_size": arguments.block_size,
        },
        "free_noise_scale": arguments.free_noise_scale,
        "heldout_count": arguments.heldout_count,
        "seed": arguments.seed,
        "dtype": "float64",
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "by_sample_count": by_sample_count,
        "best_cost": min(
            (entry["cost"] for entry in successful),
            key=lambda cost: cost["estimated_flops"],
            default=None,
        ),
        "interpretation": (
            "heldout_rel_error and decoder_rel_error_vs_true_L are reported "
            "together because they diverge: under concat the attacker's map "
            "satisfies c L_hat = h on reachable states while L_hat != L as a "
            "matrix. Only the held-out metric decides whether the attack "
            "succeeded"
        ),
    }
    path = Path(arguments.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
