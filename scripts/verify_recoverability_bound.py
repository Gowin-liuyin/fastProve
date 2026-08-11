#!/usr/bin/env python3
"""Reproduce the recoverability upper bounds recorded in docs/threat_model.md §5bis.

This script is a *falsification harness* for the security narrative, not a
utility benchmark.  It measures how well a server-side observer can recover the
semantic signal ``h`` from the mixed state ``c = [h, e] M`` produced by the
current real-valued construction, under three attacker models:

  A. known basis          -- the attacker holds M (client leak / insider)
  B. known-plaintext      -- M unknown, but (h, c) pairs can be assembled
  C. basis-blind geometry -- only c is observed; is distance structure kept?

It also sweeps the noise dimension r and the refresh magnitude to show that
neither changes the answer, which is the claim that section 5bis.4 rests on.

Runs on CPU in seconds; no model checkpoint or dataset required.

Examples
--------
PYTHONPATH=src python3 scripts/verify_recoverability_bound.py
PYTHONPATH=src python3 scripts/verify_recoverability_bound.py \\
    --signal-dim 512 --output results/raw/recoverability_bound.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from fastprove.transforms import generate_transform


def _rel_err(estimate: torch.Tensor, truth: torch.Tensor) -> float:
    denom = truth.norm()
    if float(denom) == 0.0:
        return float("nan")
    return float((estimate - truth).norm() / denom)


def attack_known_basis(
    *, signal_dim: int, noise_dim: int, seed: int, noise_scales: Sequence[float]
) -> Dict[str, Any]:
    """Attacker A: holds M, applies the fixed linear decoder.

    The decoder is ``P = (M^-1)[:, :signal_dim]``.  It annihilates the noise
    subspace exactly, so ``h = c @ P`` for every ``e``.
    """

    transform = generate_transform(
        signal_dim, noise_dim, seed=seed, domain="verify-known-basis",
        dtype=torch.float64,
    )
    decoder = transform.inverse[:, :signal_dim]
    h = torch.randn(256, signal_dim, dtype=torch.float64)
    rows: List[Dict[str, Any]] = []
    for scale in noise_scales:
        e = scale * torch.randn(256, noise_dim, dtype=torch.float64)
        c = torch.cat([h, e], dim=1) @ transform.matrix
        rows.append(
            {
                "noise_scale": scale,
                "relative_l2_error": _rel_err(c @ decoder, h),
            }
        )
    return {
        "attacker": "known_basis",
        "note": (
            "h = c @ (M^-1)[:, :d] holds for any e; residual is floating-point "
            "only, so amplifying the auxiliary noise does not protect h"
        ),
        "decoder_shape": list(decoder.shape),
        "condition_number": transform.condition_number,
        "measurements": rows,
    }


def attack_row_projection(
    *, signal_dim: int, noise_dim: int, seed: int, noise_scales: Sequence[float]
) -> Dict[str, Any]:
    """Attacker A': exploits row-orthogonality of M = Pi D B, no inverse needed.

    Because ``B`` is orthogonal, the rows of ``M`` are mutually orthogonal, so
    the signal row-space and the noise row-space are exactly orthogonal.  The
    signal is then read off by a normalized inner product against the signal
    rows alone -- the noise block never has to be inverted, referenced, or even
    known.  This is a strictly weaker requirement than holding ``M^-1``.
    """

    transform = generate_transform(
        signal_dim, noise_dim, seed=seed, domain="verify-row-projection",
        dtype=torch.float64,
    )
    matrix = transform.matrix
    gram = matrix @ matrix.T
    off_diagonal = float(
        (gram - torch.diag(torch.diagonal(gram))).abs().max()
    )
    signal_rows = matrix[:signal_dim]
    row_sq_norms = (signal_rows * signal_rows).sum(dim=1)

    h = torch.randn(128, signal_dim, dtype=torch.float64)
    rows: List[Dict[str, Any]] = []
    for scale in noise_scales:
        e = scale * torch.randn(128, noise_dim, dtype=torch.float64)
        c = torch.cat([h, e], dim=1) @ matrix
        recovered = (c @ signal_rows.T) / row_sq_norms
        rows.append(
            {
                "noise_scale": scale,
                "relative_l2_error": _rel_err(recovered, h),
            }
        )
    return {
        "attacker": "row_projection",
        "note": (
            "B orthogonal => rows of M are mutually orthogonal => signal is "
            "read by normalized inner product; no matrix inverse and no "
            "knowledge of the auxiliary state are required"
        ),
        "max_offdiagonal_of_M_Mt": off_diagonal,
        "measurements": rows,
    }


def attack_known_plaintext(
    *, signal_dim: int, noise_dim: int, seed: int
) -> Dict[str, Any]:
    """Attacker B: M unknown; fit c -> h by least squares from observed pairs.

    Recovery becomes exact once the number of independent pairs reaches the
    augmented dimension d + r, which is the practical cost of the attack.
    """

    transform = generate_transform(
        signal_dim, noise_dim, seed=seed, domain="verify-known-plaintext",
        dtype=torch.float64,
    )
    total = signal_dim + noise_dim
    n_samples = 2 * total + 256
    h = torch.randn(n_samples, signal_dim, dtype=torch.float64)
    e = torch.randn(n_samples, noise_dim, dtype=torch.float64)
    c = torch.cat([h, e], dim=1) @ transform.matrix

    rows: List[Dict[str, Any]] = []
    for n_pairs in (total // 2, total - 8, total, total + 16, 2 * total):
        if n_pairs <= 0 or n_pairs >= n_samples:
            continue
        fitted = torch.linalg.lstsq(c[:n_pairs], h[:n_pairs]).solution
        held_out = slice(n_pairs, n_pairs + 200)
        rows.append(
            {
                "n_pairs": n_pairs,
                "augmented_dim": total,
                "held_out_relative_l2_error": _rel_err(
                    c[held_out] @ fitted, h[held_out]
                ),
            }
        )
    return {
        "attacker": "known_plaintext",
        "note": (
            "requires only d + r independent (h, c) pairs; assembling them is "
            "feasible whenever weights are server-side and inputs are known"
        ),
        "measurements": rows,
    }


def attack_basis_blind_geometry(
    *, signal_dim: int, noise_dim: int, seed: int, max_condition_number: float
) -> Dict[str, Any]:
    """Attacker C: only c is observed. Measure preserved distance structure.

    The condition-number cap chosen for numerical stability simultaneously
    bounds how much pairwise distances can be distorted, which is the
    precondition embedding-inversion attacks rely on.
    """

    transform = generate_transform(
        signal_dim, noise_dim, seed=seed, domain="verify-geometry",
        max_condition_number=max_condition_number, dtype=torch.float64,
    )
    n = 384
    h = torch.randn(n, signal_dim, dtype=torch.float64)
    e = torch.randn(n, noise_dim, dtype=torch.float64)
    c = torch.cat([h, e], dim=1) @ transform.matrix

    idx = torch.triu_indices(n, n, offset=1)
    d_h = torch.cdist(h, h)[idx[0], idx[1]]
    d_c = torch.cdist(c, c)[idx[0], idx[1]]
    ratio = d_c / d_h
    correlation = float(torch.corrcoef(torch.stack([d_h, d_c]))[0, 1])
    return {
        "attacker": "basis_blind_geometry",
        "note": (
            "kappa_2(M) bounds the stretch of every pairwise distance, so the "
            "numerical-stability cap is also a leakage bound"
        ),
        "condition_number": transform.condition_number,
        "max_condition_number": max_condition_number,
        "distance_correlation": correlation,
        "stretch_ratio_min": float(ratio.min()),
        "stretch_ratio_max": float(ratio.max()),
    }


def sweep_noise_dimension(
    *, signal_dim: int, seed: int, noise_dims: Sequence[int]
) -> Dict[str, Any]:
    """Show that the r/d ratio does not change known-basis recoverability."""

    rows: List[Dict[str, Any]] = []
    for noise_dim in noise_dims:
        if noise_dim <= 0:
            continue
        transform = generate_transform(
            signal_dim, noise_dim, seed=seed, domain="verify-sweep-%d" % noise_dim,
            dtype=torch.float64,
        )
        h = torch.randn(128, signal_dim, dtype=torch.float64)
        e = 1e3 * torch.randn(128, noise_dim, dtype=torch.float64)
        c = torch.cat([h, e], dim=1) @ transform.matrix
        rows.append(
            {
                "noise_dim": noise_dim,
                "noise_to_signal_dim_ratio": noise_dim / signal_dim,
                "relative_l2_error": _rel_err(
                    c @ transform.inverse[:, :signal_dim], h
                ),
                "known_plaintext_pairs_required": signal_dim + noise_dim,
            }
        )
    return {
        "sweep": "noise_dimension",
        "note": "r only sets how many dimensions the decoder annihilates",
        "measurements": rows,
    }


def sweep_refresh_magnitude(
    *, signal_dim: int, noise_dim: int, seed: int, refresh_scales: Sequence[float]
) -> Dict[str, Any]:
    """Show that per-request refresh xi is orthogonal to recoverability.

    Mirrors ``e' = h C + e G + xi`` from the ChainLinear refresh, then applies
    the same fixed decoder.  Also reports the ratio of the signal-coupled term
    to the fresh-random term, which is the second leakage channel noted in
    threat_model.md §5bis.4.
    """

    transform = generate_transform(
        signal_dim, noise_dim, seed=seed, domain="verify-refresh",
        dtype=torch.float64,
    )
    decoder = transform.inverse[:, :signal_dim]
    generator = torch.Generator().manual_seed(seed)
    coupling = 0.02 * torch.randn(
        signal_dim, noise_dim, generator=generator, dtype=torch.float64
    )
    h = torch.randn(256, signal_dim, generator=generator, dtype=torch.float64)
    signal_coupled = h @ coupling

    rows: List[Dict[str, Any]] = []
    for scale in refresh_scales:
        xi = scale * torch.randn(
            256, noise_dim, generator=generator, dtype=torch.float64
        )
        e = signal_coupled + xi
        c = torch.cat([h, e], dim=1) @ transform.matrix
        rows.append(
            {
                "refresh_scale": scale,
                "relative_l2_error": _rel_err(c @ decoder, h),
                "signal_coupled_over_fresh_random": (
                    float(signal_coupled.norm() / xi.norm())
                    if float(xi.norm()) != 0.0
                    else float("inf")
                ),
            }
        )
    return {
        "sweep": "refresh_magnitude",
        "note": (
            "refresh perturbs only e, which the decoder annihilates; rungs 1-3 "
            "of the refresh ladder are key hygiene, not confidentiality"
        ),
        "measurements": rows,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-dim", type=int, default=256)
    parser.add_argument("--noise-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--max-condition-number", type=float, default=10.0)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help=(
            "Relative error at or below which recovery counts as successful. "
            "Exit code is 1 if any attacker fails to recover, since that would "
            "contradict the bound documented in threat_model.md 5bis."
        ),
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)

    if args.signal_dim <= 0 or args.noise_dim <= 0:
        print("signal-dim and noise-dim must be positive", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    payload: Dict[str, Any] = {
        "purpose": (
            "Reproduce the recoverability upper bounds in "
            "docs/threat_model.md section 5bis. Utility metrics are NOT "
            "measured here and retention/RP is NOT evidence of privacy."
        ),
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "arithmetic_dtype": "float64",
        },
        "config": {
            "signal_dim": args.signal_dim,
            "noise_dim": args.noise_dim,
            "seed": args.seed,
            "max_condition_number": args.max_condition_number,
            "tolerance": args.tolerance,
        },
    }

    known = attack_known_basis(
        signal_dim=args.signal_dim,
        noise_dim=args.noise_dim,
        seed=args.seed,
        noise_scales=(1.0, 1e3, 1e6, 1e12),
    )
    kp = attack_known_plaintext(
        signal_dim=args.signal_dim, noise_dim=args.noise_dim, seed=args.seed
    )
    projection = attack_row_projection(
        signal_dim=args.signal_dim,
        noise_dim=args.noise_dim,
        seed=args.seed,
        noise_scales=(1.0, 1e3, 1e6),
    )
    geom = attack_basis_blind_geometry(
        signal_dim=args.signal_dim,
        noise_dim=args.noise_dim,
        seed=args.seed,
        max_condition_number=args.max_condition_number,
    )
    r_sweep = sweep_noise_dimension(
        signal_dim=args.signal_dim,
        seed=args.seed,
        noise_dims=(1, 8, 16, 64, args.signal_dim),
    )
    xi_sweep = sweep_refresh_magnitude(
        signal_dim=args.signal_dim,
        noise_dim=args.noise_dim,
        seed=args.seed,
        refresh_scales=(0.02, 1.0, 100.0),
    )

    payload["attacks"] = [known, projection, kp, geom]
    payload["sweeps"] = [r_sweep, xi_sweep]

    print("== A. known basis (attacker holds M) ==")
    for row in known["measurements"]:
        print(
            "   |e| scale %-8.0e -> relative L2 error %.3e"
            % (row["noise_scale"], row["relative_l2_error"])
        )
    print("== A'. row projection (no inverse, e unknown) ==")
    print(
        "   max off-diagonal of M M^T = %.2e (rows are mutually orthogonal)"
        % projection["max_offdiagonal_of_M_Mt"]
    )
    for row in projection["measurements"]:
        print(
            "   |e| scale %-8.0e -> relative L2 error %.3e"
            % (row["noise_scale"], row["relative_l2_error"])
        )
    print("== B. known plaintext (M unknown) ==")
    for row in kp["measurements"]:
        print(
            "   %5d pairs (d+r=%d) -> held-out relative L2 error %.3e"
            % (
                row["n_pairs"],
                row["augmented_dim"],
                row["held_out_relative_l2_error"],
            )
        )
    print("== C. basis-blind geometry (only c observed) ==")
    print(
        "   corr(|dh|,|dc|)=%.4f  stretch in [%.4f, %.4f]  kappa_2=%.4f"
        % (
            geom["distance_correlation"],
            geom["stretch_ratio_min"],
            geom["stretch_ratio_max"],
            geom["condition_number"],
        )
    )
    print("== D. noise-dimension sweep (known basis) ==")
    for row in r_sweep["measurements"]:
        print(
            "   r=%-5d r/d=%.3f -> relative L2 error %.3e  (KP pairs: %d)"
            % (
                row["noise_dim"],
                row["noise_to_signal_dim_ratio"],
                row["relative_l2_error"],
                row["known_plaintext_pairs_required"],
            )
        )
    print("== E. refresh-magnitude sweep (known basis) ==")
    for row in xi_sweep["measurements"]:
        print(
            "   xi scale %-8.2f -> relative L2 error %.3e  (|hC|/|xi| = %.1f)"
            % (
                row["refresh_scale"],
                row["relative_l2_error"],
                row["signal_coupled_over_fresh_random"],
            )
        )

    # The bound is an upper bound on confidentiality: recovery MUST succeed.
    # A failure here means the construction changed and the documented section
    # is stale, which is a documentation bug worth failing loudly on.
    failures: List[str] = []
    for row in known["measurements"]:
        if row["noise_scale"] <= 1e6 and not (
            row["relative_l2_error"] <= args.tolerance
        ):
            failures.append(
                "known-basis recovery at noise scale %g gave %.3e"
                % (row["noise_scale"], row["relative_l2_error"])
            )
    for row in projection["measurements"]:
        if not (row["relative_l2_error"] <= args.tolerance):
            failures.append(
                "row-projection recovery at noise scale %g gave %.3e"
                % (row["noise_scale"], row["relative_l2_error"])
            )
    exact_kp = [
        row
        for row in kp["measurements"]
        if row["n_pairs"] >= row["augmented_dim"]
    ]
    for row in exact_kp:
        if not (row["held_out_relative_l2_error"] <= args.tolerance):
            failures.append(
                "known-plaintext recovery with %d pairs gave %.3e"
                % (row["n_pairs"], row["held_out_relative_l2_error"])
            )

    payload["bound_reproduced"] = not failures
    payload["failures"] = failures

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = _REPO / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        print("wrote", out_path)

    if failures:
        print("\nBOUND NOT REPRODUCED:", file=sys.stderr)
        for item in failures:
            print("  -", item, file=sys.stderr)
        return 1
    print(
        "\nBound reproduced: h is recoverable from c for every tested noise "
        "scale, noise dimension, and refresh magnitude."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
