"""Record the free noise cone ``ker(L)`` as a first-class object (stage S0).

The dual-channel refactor makes ``R^N = ker(L) + row_space(L)`` an explicit
split. This script records the algebraic facts the refactor relies on, for one
deterministic structured basis:

* ``dim ker(L)`` equals ``noise_dim`` in the current ``D == signal_dim``
  construction;
* the free basis annihilates the semantic read matrix ``L`` to FP64 round-off;
* the free cone is the same subspace as the rows of ``M_bot``;
* noise in the free cone has zero semantic effect at any magnitude (nothing
  here is a privacy measurement -- channel 0 hiding is exactly zero by
  theorem 4 and must be recorded as zero).

Usage:
    PYTHONPATH=src python scripts/verify_free_noise_cone.py \
        --output results/raw/free_noise_cone_check.json
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from fastprove.structured import generate_structured_basis

_FP64 = torch.float64


def _principal_angle_max(left: torch.Tensor, right: torch.Tensor) -> float:
    # M_bot rows are not orthonormal, so orthogonalize both before comparing.
    left = torch.linalg.qr(left.T).Q.T
    right = torch.linalg.qr(right.T).Q.T
    cosines = torch.linalg.svdvals(left @ right.T).clamp(-1.0, 1.0)
    return float(torch.arccos(cosines.min()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--signal-dim", type=int, default=64)
    parser.add_argument("--noise-dim", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args()

    basis = generate_structured_basis(
        arguments.signal_dim,
        arguments.noise_dim,
        seed=arguments.seed,
        domain="free-noise-cone-check",
        block_size=arguments.block_size,
        dtype=_FP64,
    )
    basis.validate_integrity()
    free = basis.free_noise_basis()
    projection = basis.signal_projection()

    generator = torch.Generator().manual_seed(arguments.seed)
    pure = torch.zeros(8, basis.signal_dim, dtype=_FP64)
    concat_noise = torch.randn(
        8, basis.noise_dim, generator=generator, dtype=_FP64
    )
    injected = torch.cat((pure, concat_noise), dim=-1) @ basis.dense()
    pure_decomposition = basis.channel_decomposition(injected)

    generic = torch.randn(8, basis.total_dim, generator=generator, dtype=_FP64)
    generic_decomposition = basis.channel_decomposition(generic)

    # Theorem 4 demonstration: amplify in-cone noise by 1e12 and re-decode.
    signal = torch.randn(4, basis.signal_dim, generator=generator, dtype=_FP64)
    noise = torch.randn(4, basis.noise_dim, generator=generator, dtype=_FP64)
    mixed = basis.mix(torch.cat((signal, noise), dim=-1))
    direction = torch.randn(4, basis.noise_dim, generator=generator, dtype=_FP64)
    in_cone = torch.cat(
        (torch.zeros(4, basis.signal_dim, dtype=_FP64), direction), dim=-1
    ) @ basis.dense()
    huge = mixed + 1e12 * in_cone
    recovered = huge @ projection
    recovery_relative_error = float((recovered - signal).norm()) / float(
        signal.norm()
    )

    record = {
        "claim": (
            "ker(L) is an explicit subspace with dim == noise_dim in the "
            "current construction; it annihilates the semantic read matrix "
            "exactly, and its hiding contribution is exactly zero"
        ),
        "channel_0_information_theoretic_hiding_bits": 0.0,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "seed": arguments.seed,
        "arithmetic_dtype": "float64",
        "geometry": {
            "signal_dim": basis.signal_dim,
            "noise_dim": basis.noise_dim,
            "total_dim": basis.total_dim,
            "block_size": basis.block_size,
            "condition_number": basis.condition_number,
        },
        "dim_ker_L": int(free.shape[0]),
        "max_semantic_leak": float((free @ projection).abs().max()),
        "subspace_principal_angle_rad": _principal_angle_max(free, basis.noise_rows()),
        "pure_concat_noise": {
            "free_fraction": pure_decomposition["free_fraction"],
            "semantic_component_norm": pure_decomposition["semantic_component_norm"],
        },
        "generic_noise": {
            "free_fraction": generic_decomposition["free_fraction"],
            "semantic_component_norm": generic_decomposition["semantic_component_norm"],
            "induced_semantic_error_norm": generic_decomposition[
                "induced_semantic_error_norm"
            ],
        },
        "in_cone_noise_amplified_by": 1e12,
        "in_cone_h_recovery_relative_error": recovery_relative_error,
        "note_on_tolerance": (
            "arccos near 1 is ill-conditioned: 3e-8 rad corresponds to "
            "1 - cos = 5e-16, i.e. FP64 round-off"
        ),
        "interpretation": (
            "channel 0 is free of accuracy cost and free of hiding: it must be "
            "bookkept with information_theoretic_hiding_bits = 0, never "
            "presented as a privacy source"
        ),
    }
    path = Path(arguments.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
