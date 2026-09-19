"""Build ``results/raw/attack_costs.jsonl`` from recorded attack artifacts.

This script does not run attacks and does not invent numbers. It restructures
the cost vectors and recovery values already present in recorded raw files
into the ``fastprove.attack.v1`` schema (``evaluation/attack_records.py``).
Attacks whose raw artifact predates cost instrumentation, or whose execution
needs the absent pretrained checkpoint, are emitted explicitly as
``status="not_executed"`` records with a reason: silent omission is forbidden
by AGENTS.md.

Usage:
    PYTHONPATH=src python scripts/build_attack_cost_records.py \
        --output results/raw/attack_costs.jsonl
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from fastprove.evaluation.attack_records import (  # noqa: E402
    build_attack_record,
    validate_attack_record,
)

RAW = _REPO / "results" / "raw"


def _load(name: str) -> dict[str, Any]:
    return json.loads((RAW / name).read_text(encoding="utf-8"))


def _environment(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "torch_version": record.get("torch_version", torch.__version__),
        "platform": record.get("platform", platform.platform()),
        "arithmetic_dtype": record.get("dtype", record.get("arithmetic_dtype")),
    }


def _known_plaintext_record() -> dict[str, Any]:
    raw = _load("known_plaintext_decoder_recovery.json")
    geometry = raw["geometry"]
    target = geometry["d_semantic"] + geometry["r_nuisance"] + 1
    primary = next(
        entry for entry in raw["by_sample_count"] if entry["T"] == target
    )
    return build_attack_record(
        attack_id="known_plaintext_decoder_recovery",
        tier=0,
        claim=raw["claim"],
        coa_view="cloud_coa",
        requires={
            "requires_known_plaintext_pairs": True,
            "requires_knowledge_of_the_mixing_basis": False,
            "requires_public_base_checkpoint": True,
            "requires_modified_kernel": False,
            "requires_register_access": False,
        },
        recovery={
            "metric": "heldout_rel_error_at_T_eq_d_plus_r_plus_1",
            "value": primary["heldout_rel_error"],
        },
        status="executed",
        config={
            "geometry": geometry,
            "primary_sample_count": target,
            "heldout_count": raw["heldout_count"],
            "seed": raw["seed"],
            "free_noise_scale": raw["free_noise_scale"],
        },
        environment=_environment(raw),
        cost=primary["cost"],
    )


def _gram_leakage_record() -> dict[str, Any]:
    raw = _load("gram_leakage.json")
    return build_attack_record(
        attack_id="gram_geometry_exposure",
        tier=0,
        claim=raw["claim"],
        coa_view="cloud_coa",
        requires={
            "requires_known_plaintext_pairs": False,
            "requires_knowledge_of_the_mixing_basis": False,
            "requires_public_base_checkpoint": False,
            "requires_modified_kernel": False,
            "requires_register_access": False,
            "requires_server_bundle": True,
        },
        recovery={
            "metric": "max_absolute_error_of_pairwise_distances",
            "value": raw["max_absolute_error_of_pairwise_distances"],
        },
        status="executed",
        config={
            "signal_dim": raw["signal_dim"],
            "noise_dim": raw["noise_dim"],
            "block_size": raw["block_size"],
            "samples": raw["samples"],
            "seed": raw["seed"],
        },
        environment=_environment(raw),
        cost=raw["cost"],
    )


def _rho_fused_record() -> dict[str, Any]:
    raw = _load("rho_recoverable_from_fused_output.json")
    worst = max(
        case["signal_norm_relative_error"]
        for case in raw["by_emitted_query_dtype"]
    )
    return build_attack_record(
        attack_id="rho_from_fused_output",
        tier=0,
        claim=raw["claim"],
        coa_view="cloud_coa",
        requires={
            "requires_known_plaintext_pairs": False,
            "requires_knowledge_of_the_mixing_basis": False,
            "requires_public_base_checkpoint": False,
            "requires_modified_kernel": False,
            "requires_register_access": False,
            "requires_server_bundle": True,
        },
        recovery={
            "metric": "worst_signal_norm_relative_error_over_emitted_dtypes",
            "value": worst,
        },
        status="executed",
        config={
            "signal_dim": raw["signal_dim"],
            "noise_dim": raw["noise_dim"],
            "head_dim": raw["head_dim"],
            "tokens": raw["tokens"],
            "seed": raw["seed"],
        },
        environment=_environment(raw),
        cost=raw["cost"],
    )


def _recoverability_bound_record() -> dict[str, Any]:
    raw = _load("recoverability_bound.json")
    known = next(
        attack for attack in raw["attacks"] if attack.get("sweep") is None
        and "measurements" in attack
    )
    worst = max(row["relative_l2_error"] for row in known["measurements"])
    return build_attack_record(
        attack_id="recoverability_bound_known_basis",
        tier=0,
        claim=raw["purpose"],
        coa_view="cloud_coa",
        requires={
            "requires_known_plaintext_pairs": True,
            "requires_knowledge_of_the_mixing_basis": True,
            "requires_public_base_checkpoint": False,
            "requires_modified_kernel": False,
            "requires_register_access": False,
        },
        recovery={
            "metric": "worst_relative_l2_error_known_basis_rows",
            "value": worst,
        },
        status="executed",
        config=dict(raw["config"], bound_reproduced=raw["bound_reproduced"]),
        environment=_environment(raw),
        cost=raw["cost"],
    )


def _not_executed(
    attack_id: str, raw_name: str, reason: str, tier: int = 0
) -> dict[str, Any]:
    raw = _load(raw_name)
    return build_attack_record(
        attack_id=attack_id,
        tier=tier,
        claim=raw["claim"],
        coa_view="cloud_coa",
        requires={
            "requires_known_plaintext_pairs": bool(
                raw.get("requires_known_plaintext_pairs", False)
            ),
            "requires_knowledge_of_the_mixing_basis": bool(
                raw.get("requires_knowledge_of_the_mixing_basis", False)
            ),
            "requires_public_base_checkpoint": bool(
                raw.get("requires_public_base_checkpoint", True)
            ),
            "requires_modified_kernel": bool(
                raw.get("requires_modified_kernel", False)
            ),
            "requires_register_access": bool(
                raw.get("requires_register_access", False)
            ),
        },
        recovery={"metric": "token_recovery", "value": None},
        status="not_executed",
        blocked_reason=reason,
        config={"model_path": raw.get("model_path")},
        environment=_environment(raw),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(RAW / "attack_costs.jsonl"))
    arguments = parser.parse_args()

    checkpoint_reason = (
        "pretrained checkpoint is not present on this machine "
        "(audit_pretrained_assets: checkpoint_bytes = 0); no silent fallback "
        "to a random tiny model is permitted"
    )
    records = [
        _known_plaintext_record(),
        _gram_leakage_record(),
        _rho_fused_record(),
        _recoverability_bound_record(),
        _not_executed(
            "token_recovery_from_rho",
            "token_recovery_from_rho.json",
            "recorded before S5 cost instrumentation; re-running requires the "
            "absent checkpoint. " + checkpoint_reason,
        ),
        _not_executed(
            "norm_leak_by_depth",
            "norm_leak_by_depth.json",
            "recorded before S5 cost instrumentation; re-running requires the "
            "absent checkpoint and prompt corpus. " + checkpoint_reason,
        ),
    ]
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        for record in records:
            validate_attack_record(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print("wrote %d records to %s" % (len(records), output))
    for record in records:
        status = record["status"]
        line = "  %-38s %-14s" % (record["attack_id"], status)
        if status == "executed":
            line += " flops=%.3g wall=%.4gs" % (
                record["cost"]["estimated_flops"],
                record["cost"]["wall_clock_seconds"],
            )
        print(line)


if __name__ == "__main__":
    main()
