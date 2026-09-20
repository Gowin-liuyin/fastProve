"""Replica sweep plumbing: run_id suffixing, pairing, marginal-cost rows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compare_rho_replica import (
    _pair_records,
    _strip_suffix,
    _summary,
    main as compare_main,
)
from scripts.run_accuracy_sweep import main as sweep_main

ROOT = Path(__file__).resolve().parents[2]
SWEEP_CONFIG = ROOT / "configs" / "eval_sweep.yaml"


def _record(run_id: str, rho: float, top1: float, ppl: float) -> dict:
    return {
        "run_id": run_id,
        "status": "success",
        "config": {
            "mode": "free_bounded",
            "tau_max": 0.03,
            "alpha": None,
            "preserve_top_k": None,
            "obfuscation": {"rho_noise_bound": rho},
        },
        "metrics": {
            "degradation": {
                "top1_absolute_drop": top1,
                "perplexity_relative_increase": ppl,
                "nll_absolute_increase": ppl / 2.0,
            },
            "softmax": {
                "js_divergence": 0.01,
                "actual_noise_infinity_norm": 0.02,
            },
        },
    }


def test_strip_suffix_requires_a_replica_id() -> None:
    assert _strip_suffix("free-tau0p1-rho1e-05", "-rho1e-05") == "free-tau0p1"
    with pytest.raises(ValueError, match="does not end with"):
        _strip_suffix("free-tau0p1", "-rho1e-05")


def test_pairing_and_marginal_rows() -> None:
    base = {
        "free-tau0p1": _record("free-tau0p1", 0.0, 0.0007, 0.001),
    }
    replica = {
        "free-tau0p1-rho1e-05": _record(
            "free-tau0p1-rho1e-05", 1.0e-5, 0.0010, 0.004
        ),
    }
    pairs = _pair_records(base, replica, "-rho1e-05")
    assert len(pairs) == 1
    from scripts.compare_rho_replica import _row

    row = _row(*pairs[0])
    assert row["top1_absolute_drop_marginal"] == pytest.approx(0.0003)
    assert row["perplexity_relative_increase_marginal"] == pytest.approx(0.003)
    summary = _summary([row])
    assert summary["point_count"] == 1
    assert summary["top1_absolute_drop_marginal_max"] == pytest.approx(0.0003)


def test_pairing_rejects_missing_base() -> None:
    base: dict = {}
    replica = {
        "free-tau0p1-rho1e-05": _record(
            "free-tau0p1-rho1e-05", 1.0e-5, 0.0, 0.0
        )
    }
    with pytest.raises(ValueError, match="without a base record"):
        _pair_records(base, replica, "-rho1e-05")


def test_compare_cli_writes_payload(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    replica_path = tmp_path / "replica.jsonl"
    output_json = tmp_path / "marginal.json"
    output_markdown = tmp_path / "marginal.md"
    base_path.write_text(
        json.dumps(_record("free-tau0p1", 0.0, 0.0007, 0.001)) + "\n",
        encoding="utf-8",
    )
    replica_path.write_text(
        json.dumps(_record("free-tau0p1-rho1e-05", 1.0e-5, 0.0010, 0.004))
        + "\n",
        encoding="utf-8",
    )
    status = compare_main(
        [
            "--base",
            str(base_path),
            "--replica",
            str(replica_path),
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
        ]
    )
    assert status == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "fastprove.rho_replica_marginal.v1"
    assert payload["summary"]["point_count"] == 1
    assert "free-tau0p1" in output_markdown.read_text(encoding="utf-8")


def test_sweep_plan_applies_replica_suffix(tmp_path: Path, capsys) -> None:
    manifest = tmp_path / "selection.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "ready_for_full_evaluation",
                "selected_run_ids": ["plaintext", "exact", "free-tau0p1"],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "replica.jsonl"
    status = sweep_main(
        [
            "--config",
            str(SWEEP_CONFIG),
            "--spec-ids-file",
            str(manifest),
            "--rho-replica",
            "1e-5",
            "--output",
            str(output),
        ]
    )
    assert status == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["rho_replica_bound"] == 1.0e-5
    assert plan["rho_replica_suffix"] == "-rho1e-05"
    run_ids = [spec["run_id"] for spec in plan["specs"]]
    assert run_ids == ["plaintext-rho1e-05", "exact-rho1e-05", "free-tau0p1-rho1e-05"]
    assert "--rho-replica 1e-05" in plan["next_command"]
    assert plan["spec_count"] == 3


def test_sweep_replica_requires_explicit_output(tmp_path: Path, capsys) -> None:
    manifest = tmp_path / "selection.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "ready_for_full_evaluation",
                "selected_run_ids": ["plaintext", "exact", "free-tau0p1"],
            }
        ),
        encoding="utf-8",
    )
    status = sweep_main(
        [
            "--config",
            str(SWEEP_CONFIG),
            "--spec-ids-file",
            str(manifest),
            "--rho-replica",
            "1e-5",
        ]
    )
    assert status == 2
    assert "explicit --output" in capsys.readouterr().err
