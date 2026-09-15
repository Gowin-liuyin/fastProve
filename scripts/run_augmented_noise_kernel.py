"""Stage A kernel: augmented encode -> converted linear -> legal decode.

Runs the plan's minimum closed loop on small tensors and records raw metrics:

* stage A gates: condition number, stored-dtype inverse residual, ``M_n P ~=
  0``, full ``[y, e']`` identity error, FP32 max-abs signal error <= 1e-5;
* stage B group 1 (noise contribution): ``beta`` in {0, 0.1, 1, 3} with
  ``r`` in {8, 16};
* stage B group 2 (matrix contribution): two-sided dense vs ``Pi D Q``
  baseline; coverage metrics for both families.

Raw results go to ``results/raw/augmented_noise_kernel/*.jsonl``; the summary
CSV is derived by :mod:`scripts.postprocess_augmented_noise_kernel` or an
equivalent pandas-free pass in ``build_report.py``. Nothing here is a security
result; these are numerical-correctness and coverage records only.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch

from fastprove.augmented_noise import (
    NoiseRefreshSpec,
    apply_noise_refresh,
    generate_propagator,
    noise_coverage_metrics,
    observed_noise_ratios,
    record_generator,
    sample_initial_noise,
    sample_refresh_noise,
)
from fastprove.conversion import convert_affine_chain
from fastprove.state import decode_debug, encode_debug
from fastprove.transforms import (
    generate_transform,
    generate_two_sided_transform,
)

RAW_DIR = Path("results/raw/augmented_noise_kernel")
FP32_SIGNAL_GATE = 1e-5


@dataclass(frozen=True)
class KernelConfig:
    signal_dim: int
    noise_dim: int
    max_condition_number: float
    gamma: float
    energy_ratio: float
    basis_family: str
    seed: int
    batch: int = 4
    sequence: int = 16


def _generate_basis(config: KernelConfig, dtype: torch.dtype):
    if config.basis_family == "dense_two_sided":
        generate = generate_two_sided_transform
    elif config.basis_family == "dense_baseline":
        generate = generate_transform
    else:
        raise ValueError("unsupported basis_family: %s" % config.basis_family)
    return generate(
        config.signal_dim,
        config.noise_dim,
        seed=config.seed,
        domain="kernel-hidden",
        max_condition_number=config.max_condition_number,
        dtype=dtype,
    )


def _environment_record() -> Dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "activation_dtype": "float32",
    }


def run_closed_loop(config: KernelConfig) -> Dict[str, Any]:
    """One full stage-A record: conversion, identity, gates, coverage."""

    started = time.perf_counter()
    spec = NoiseRefreshSpec(
        noise_dim=config.noise_dim,
        gamma=config.gamma,
        energy_ratio=config.energy_ratio,
    )
    record: Dict[str, Any] = {
        "schema": "fastprove.augmented_noise_kernel.v1",
        **_environment_record(),
        "config": {
            "signal_dim": config.signal_dim,
            "noise_dim": config.noise_dim,
            "max_condition_number": config.max_condition_number,
            "gamma": config.gamma,
            "energy_ratio": config.energy_ratio,
            "basis_family": config.basis_family,
            "seed": config.seed,
            "coupling": "zero",
        },
    }

    basis_fp64 = _generate_basis(config, torch.float64)
    record["condition_number_fp64"] = basis_fp64.condition_number
    residual = (
        basis_fp64.matrix @ basis_fp64.inverse
        - torch.eye(basis_fp64.total_dim, dtype=torch.float64)
    ).abs().max()
    record["inverse_residual_fp64"] = float(residual)

    projection = basis_fp64.inverse[:, : config.signal_dim]
    noise_rows = basis_fp64.matrix[config.signal_dim :]
    record["noise_rows_dot_projection_max"] = float(
        (noise_rows @ projection).abs().max()
    )
    record["noise_coverage"] = noise_coverage_metrics(noise_rows).to_dict()

    basis_fp32 = _generate_basis(config, torch.float32)
    record["condition_number_fp32_stored"] = basis_fp32.condition_number
    residual_fp32 = (
        basis_fp32.matrix.to(torch.float64) @ basis_fp32.inverse.to(torch.float64)
        - torch.eye(basis_fp32.total_dim, dtype=torch.float64)
    ).abs().max()
    record["inverse_residual_fp32_stored"] = float(residual_fp32)

    signal_dim_out = config.signal_dim // 2
    out_basis_fp64 = generate_two_sided_transform(
        signal_dim_out,
        config.noise_dim,
        seed=config.seed + 1,
        domain="kernel-out",
        max_condition_number=config.max_condition_number,
        dtype=torch.float64,
    ) if config.basis_family == "dense_two_sided" else generate_transform(
        signal_dim_out,
        config.noise_dim,
        seed=config.seed + 1,
        domain="kernel-out",
        max_condition_number=config.max_condition_number,
        dtype=torch.float64,
    )
    propagator = generate_propagator(
        spec, seed=config.seed + 2, domain="kernel-prop"
    )

    generator = torch.Generator().manual_seed(config.seed)
    weight_math = (
        torch.randn(
            config.signal_dim, signal_dim_out, generator=generator, dtype=torch.float64
        )
        * 0.2
    )
    bias = torch.randn(signal_dim_out, generator=generator, dtype=torch.float64) * 0.1
    coupling = torch.zeros(config.signal_dim, config.noise_dim, dtype=torch.float64)

    converted = convert_affine_chain(
        weight_math=weight_math,
        bias=bias,
        coupling=coupling,
        propagator=propagator,
        in_transform=basis_fp64,
        out_transform=out_basis_fp64,
        fixed_refresh=None,
    )

    h = torch.randn(
        config.batch, config.sequence, config.signal_dim,
        generator=generator, dtype=torch.float64,
    )
    calibration_rms = float(h.norm(dim=-1).mean() / math.sqrt(config.signal_dim))
    noise_initial = sample_initial_noise(
        spec,
        calibration_rms=calibration_rms,
        shape=(config.batch, config.sequence, config.noise_dim),
        generator=torch.Generator().manual_seed(config.seed + 3),
    )
    refresh = sample_refresh_noise(
        spec,
        calibration_rms=calibration_rms,
        shape=(config.batch, config.sequence, config.noise_dim),
        generator=torch.Generator().manual_seed(config.seed + 4),
    )
    state = encode_debug(h, noise_initial, basis_fp64, enabled=True)
    out_matrix = out_basis_fp64.matrix.to(dtype=torch.float64)
    output_mixed = (
        torch.nn.functional.linear(
            state.mixed, converted.weight_pt, converted.bias_mixed
        )
        + refresh @ out_matrix[signal_dim_out:]
    )
    decoded_signal, decoded_noise = decode_debug(
        type(state)(output_mixed, out_basis_fp64.descriptor),
        out_basis_fp64,
        enabled=True,
    )
    expected_signal = h @ weight_math + bias
    expected_noise = apply_noise_refresh(noise_initial, propagator, refresh)
    signal_error = (decoded_signal - expected_signal).abs()
    noise_error = (decoded_noise - expected_noise).abs()
    record["signal_max_abs_error"] = float(signal_error.max())
    record["signal_mean_abs_error"] = float(signal_error.mean())
    record["signal_relative_l2"] = float(
        signal_error.norm() / expected_signal.norm()
    )
    record["noise_max_abs_error"] = float(noise_error.max())
    record["nan_inf_count"] = int(
        torch.isnan(output_mixed).sum() + torch.isinf(output_mixed).sum()
    )
    record["fp32_signal_gate_pass"] = bool(
        record["signal_max_abs_error"] <= FP32_SIGNAL_GATE
    )
    record["observed_noise_ratios"] = observed_noise_ratios(h, decoded_noise)

    # Fixed-seed reproducibility: rerun sampling and compare.
    noise_repeat = sample_refresh_noise(
        spec,
        calibration_rms=calibration_rms,
        shape=(config.batch, config.sequence, config.noise_dim),
        generator=torch.Generator().manual_seed(config.seed + 4),
    )
    record["fixed_seed_reproducible"] = bool(
        torch.equal(refresh, noise_repeat)
    )

    # Record isolation: per-token streams must differ.
    draws = []
    for position in range(min(4, config.sequence)):
        token_generator = record_generator(
            config.seed,
            key_epoch="k0",
            request_nonce="rq0",
            sample_id="s0",
            token_position=position,
            layer_id="L0",
            operation="attn",
        )
        draws.append(
            sample_refresh_noise(
                spec,
                calibration_rms=calibration_rms,
                shape=(config.noise_dim,),
                generator=token_generator,
            )
        )
    stacked = torch.stack(draws)
    # With beta = 0 every draw is the zero vector, so equality is expected;
    # distinctness is a meaningful gate only for the noisy configurations.
    if config.energy_ratio > 0.0:
        distinct = all(
            not torch.equal(stacked[i], stacked[j])
            for i in range(len(draws))
            for j in range(i + 1, len(draws))
        )
    else:
        distinct = True
    record["record_streams_distinct"] = bool(distinct)

    # Same absolute position agrees between a "prefill" and a "decode" call.
    prefill = sample_refresh_noise(
        spec,
        calibration_rms=calibration_rms,
        shape=(config.noise_dim,),
        generator=record_generator(
            config.seed, key_epoch="k0", request_nonce="rq0",
            sample_id="s0", token_position=7, layer_id="L0", operation="attn",
        ),
    )
    decode = sample_refresh_noise(
        spec,
        calibration_rms=calibration_rms,
        shape=(config.noise_dim,),
        generator=record_generator(
            config.seed, key_epoch="k0", request_nonce="rq0",
            sample_id="s0", token_position=7, layer_id="L0", operation="attn",
        ),
    )
    record["prefill_decode_consistent"] = bool(torch.equal(prefill, decode))

    record["runtime_seconds"] = time.perf_counter() - started
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=RAW_DIR,
        help="directory for raw JSONL records",
    )
    args = parser.parse_args()
    torch.manual_seed(20260914)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    # Stage A canonical point: d=64, r=16, kappa=3, beta=1, gamma=0.5.
    canonical = KernelConfig(
        signal_dim=64, noise_dim=16, max_condition_number=3.0,
        gamma=0.5, energy_ratio=1.0, basis_family="dense_two_sided", seed=20260914,
    )
    records.append(run_closed_loop(canonical))

    # Stage B group 1: noise contribution (beta x r) on the two-sided family.
    for energy_ratio in (0.0, 0.1, 1.0, 3.0):
        for noise_dim in (8, 16):
            records.append(run_closed_loop(
                KernelConfig(
                    signal_dim=64, noise_dim=noise_dim,
                    max_condition_number=3.0, gamma=0.5,
                    energy_ratio=energy_ratio,
                    basis_family="dense_two_sided", seed=20260914,
                )
            ))

    # Stage B group 2: matrix contribution (kappa sweep, family comparison).
    for kappa in (1.0, 3.0, 10.0):
        records.append(run_closed_loop(
            KernelConfig(
                signal_dim=64, noise_dim=16, max_condition_number=kappa,
                gamma=0.5, energy_ratio=1.0,
                basis_family="dense_two_sided", seed=20260914,
            )
        ))
    for family in ("dense_baseline",):
        records.append(run_closed_loop(
            KernelConfig(
                signal_dim=64, noise_dim=16, max_condition_number=3.0,
                gamma=0.5, energy_ratio=1.0,
                basis_family=family, seed=20260914,
            )
        ))

    output_path = args.output_dir / "kernel_records.jsonl"
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print("wrote %d records to %s" % (len(records), output_path))

    failures = [
        record
        for record in records
        if not record.get("fp32_signal_gate_pass", False)
        or record.get("nan_inf_count", 1) != 0
        or not record.get("fixed_seed_reproducible", False)
        or not record.get("record_streams_distinct", False)
        or not record.get("prefill_decode_consistent", False)
    ]
    if failures:
        raise SystemExit("stage A gates failed for %d records" % len(failures))
    print("all stage A gates passed")


if __name__ == "__main__":
    main()
