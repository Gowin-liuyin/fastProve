"""Verify that the RMSNorm scale ``rho`` is recoverable from a fused kernel's output.

This closes the most common defence of the design: "``rho`` only lives in
registers inside the fused operation, it is never written back to memory, so an
observer cannot read it."

The defence fails. The fused RMS-QKV operation must emit ``Q`` (attention needs
it), and the server necessarily holds both the mixed state ``c`` and the deployed
weight ``W_q``. Since

    Q = (c @ W_q) / rho

a single elementwise division recovers ``rho`` exactly:

    rho = (c @ W_q) / Q

Therefore ``rho`` is not an internal temporary in any meaningful sense. It is
algebraically determined by quantities that must both exist on the server. No
kernel-fusion or register-residency assumption changes this, and the attacker
needs neither a modified kernel nor register access.

Usage:
    python scripts/verify_rho_recoverable_from_fused_output.py \\
        --output results/raw/rho_recoverable_from_fused_output.json
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from fastprove.structured import generate_structured_basis

_FP64 = torch.float64


def _run_case(
    *,
    signal_dim: int,
    noise_dim: int,
    block_size: int,
    head_dim: int,
    tokens: int,
    eps: float,
    seed: int,
    dtype: torch.dtype,
) -> dict:
    """Model one fused RMS-Q step and recover ``rho`` from its output only."""

    torch.manual_seed(seed)
    basis = generate_structured_basis(
        signal_dim,
        noise_dim,
        seed=seed,
        domain="rho-recovery",
        block_size=block_size,
        dtype=_FP64,
    )
    signal = torch.randn(tokens, signal_dim, dtype=_FP64)
    noise = torch.randn(tokens, noise_dim, dtype=_FP64)
    mixed = basis.mix(torch.cat((signal, noise), dim=-1))

    gamma = torch.rand(signal_dim, dtype=_FP64) + 0.5
    query_weight = torch.randn(signal_dim, head_dim, dtype=_FP64) * 0.1
    # Deployed weight the server holds, exactly as built by layers/deployed.py.
    deployed_query = (basis.signal_projection() * gamma[None, :]) @ query_weight

    # Inside the fused kernel: rho is computed and consumed, never written back.
    rho = torch.sqrt(signal.pow(2).mean(-1, keepdim=True) + eps)
    emitted_query = ((mixed @ deployed_query) / rho).to(dtype).to(_FP64)

    # Attacker view: c, deployed_query, emitted_query. No registers, no kernel
    # modification, no known-plaintext pair, no knowledge of the basis.
    prefix = mixed @ deployed_query
    ratio = prefix / emitted_query
    recovered_rho = ratio.median(dim=-1, keepdim=True).values

    true_norm = signal.norm(dim=-1, keepdim=True)
    recovered_norm = torch.sqrt(
        ((recovered_rho * recovered_rho - eps) * signal_dim).clamp_min(0)
    )
    return {
        "emitted_query_dtype": str(dtype).replace("torch.", ""),
        "rho_relative_error": float(
            ((recovered_rho - rho).abs() / rho).max()
        ),
        "signal_norm_relative_error": float(
            ((recovered_norm - true_norm).abs() / true_norm).max()
        ),
        "operations_needed": "one elementwise division and one median",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--signal-dim", type=int, default=256)
    parser.add_argument("--noise-dim", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    cases = [
        _run_case(
            signal_dim=arguments.signal_dim,
            noise_dim=arguments.noise_dim,
            block_size=arguments.block_size,
            head_dim=arguments.head_dim,
            tokens=arguments.tokens,
            eps=arguments.eps,
            seed=arguments.seed,
            dtype=dtype,
        )
        for dtype in (torch.float64, torch.float32, torch.bfloat16)
    ]

    record = {
        "claim": (
            "rho is recoverable from the fused kernel's emitted Q alone; the "
            "register-residency / no-write-back assumption does not protect it"
        ),
        "why": (
            "Q = (c @ W_q) / rho, and the server must hold c and W_q and must "
            "emit Q for attention, so rho = (c @ W_q) / Q"
        ),
        "requires_modified_kernel": False,
        "requires_register_access": False,
        "requires_known_plaintext_pairs": False,
        "requires_knowledge_of_the_mixing_basis": False,
        "signal_dim": arguments.signal_dim,
        "noise_dim": arguments.noise_dim,
        "block_size": arguments.block_size,
        "head_dim": arguments.head_dim,
        "tokens": arguments.tokens,
        "rms_epsilon": arguments.eps,
        "by_emitted_query_dtype": cases,
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
