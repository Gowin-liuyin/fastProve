from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fastprove.evaluation.artifacts import build_run_record
from scripts.run_accuracy_sweep import (
    _exact_gate_skip_record,
    _failure_record,
    _order_specs_for_exact_gate,
    main as sweep_main,
)
from fastprove.evaluation.sweep import SweepSpec
from scripts.run_correctness import (
    _exact_succeeded,
    main as correctness_main,
)
from scripts.preflight_experiment import _accepted_evaluation_scope


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _valid_comparison_metrics() -> dict:
    return {
        "plaintext": {
            "negative_log_likelihood": 1.0,
            "perplexity": 2.0,
            "next_token_top1_accuracy": 0.5,
            "next_token_top5_accuracy": 0.9,
        },
        "obfuscated": {
            "negative_log_likelihood": 1.0,
            "perplexity": 2.0,
            "next_token_top1_accuracy": 0.5,
            "next_token_top5_accuracy": 0.9,
        },
        "agreement": {"next_token_top1_agreement": 1.0},
        "degradation": {
            "nll_absolute_increase": 0.0,
            "perplexity_absolute_increase": 0.0,
            "perplexity_relative_increase": 0.0,
            "top1_absolute_drop": 0.0,
            "top1_relative_drop": 0.0,
            "top5_absolute_drop": 0.0,
            "top5_relative_drop": 0.0,
        },
        "greedy": {
            "greedy_token_exact_match": 1.0,
            "greedy_sequence_exact_match": 1.0,
        },
        "softmax": {
            "kl_divergence": 0.0,
            "js_divergence": 0.0,
            "topk_overlap": 1.0,
            "rank_correlation": 1.0,
            "topk_changed_fraction": 0.0,
            "actual_noise_infinity_norm": 0.0,
            "zero_noise_query_fraction": 1.0,
            "attention_output_relative_l2_error": 0.0,
        },
        "performance": {
            "conversion_time_seconds": 0.0,
            "prefill_latency_seconds": 0.0,
            "decode_tpot_seconds": 0.0,
            "tokens_per_second": 0.0,
            "peak_memory_bytes": 0,
            "kv_cache_memory_bytes": 0,
        },
    }


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_correctness_cli_defaults_to_plan_only_and_writes_nothing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "must-not-exist.jsonl"
    completed = _run(
        "scripts/run_correctness.py",
        "--config",
        "configs/tiny_exact.yaml",
        "--output",
        str(output),
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["action"] == "plan"
    assert payload["execution_authorized"] is False
    assert payload["mode"] == "exact"
    assert payload["model_kind"] == "random_tiny_correctness_only"
    assert not output.exists()


def test_sweep_cli_defaults_to_plan_only_and_lists_all_72_specs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "must-not-exist.jsonl"
    completed = _run(
        "scripts/run_accuracy_sweep.py",
        "--config",
        "configs/eval_sweep.yaml",
        "--output",
        str(output),
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["action"] == "plan"
    assert payload["execution_authorized"] is False
    assert payload["spec_count"] == 72
    assert len(payload["specs"]) == 72
    assert {item["mode"] for item in payload["specs"]} == {
        "plaintext",
        "exact",
        "topk_preserving",
        "free_bounded",
    }
    assert len({item["run_id"] for item in payload["specs"]}) == 72
    assert not output.exists()


def test_sweep_plan_next_command_replays_stage_overrides(
    tmp_path: Path,
) -> None:
    output = tmp_path / "calibration.jsonl"
    completed = _run(
        "scripts/run_accuracy_sweep.py",
        "--config",
        "configs/eval_sweep.yaml",
        "--output",
        str(output),
        "--calibration-only",
        "--batch-size",
        "1",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    command = payload["next_command"]
    assert "--calibration-only" in command
    assert "--batch-size 1" in command
    assert "--execute-deferred" in command


def test_sweep_plan_records_explicit_runtime_overrides(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bf16-calibration.jsonl"
    completed = _run(
        "scripts/run_accuracy_sweep.py",
        "--config",
        "configs/eval_sweep.yaml",
        "--output",
        str(output),
        "--calibration-only",
        "--batch-size",
        "1",
        "--device",
        "mps",
        "--activation-dtype",
        "bf16",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["device"] == "mps"
    assert payload["activation_dtype"] == "bf16"
    assert "--device mps" in payload["next_command"]
    assert "--activation-dtype bf16" in payload["next_command"]


def test_pretrained_preflight_failure_writes_one_record_per_spec(
    tmp_path: Path,
) -> None:
    output = tmp_path / "preflight-failure.jsonl"
    completed = _run(
        "scripts/run_accuracy_sweep.py",
        "--config",
        "configs/eval_sweep.yaml",
        "--model-path",
        str(tmp_path / "missing-model"),
        "--dataset-cache",
        str(tmp_path / "missing-cache.pt"),
        "--output",
        str(output),
        "--execute-deferred",
    )

    assert completed.returncode == 2
    assert "failure_records" in completed.stderr
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 72
    assert {row["status"] for row in rows} == {"failure"}
    assert {row["stage"] for row in rows} == {"pretrained_preflight"}
    assert {row["config"]["mode"] for row in rows} == {
        "plaintext",
        "exact",
        "topk_preserving",
        "free_bounded",
    }


def test_sweep_plan_accepts_ready_candidate_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "candidate_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "ready_for_full_evaluation",
                "selected_run_ids": [
                    "plaintext",
                    "exact",
                    "topk-tau0-a0p5-k4",
                ],
            }
        ),
        encoding="utf-8",
    )
    completed = _run(
        "scripts/run_accuracy_sweep.py",
        "--config",
        "configs/eval_sweep.yaml",
        "--spec-ids-file",
        str(manifest),
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["all_spec_count"] == 72
    assert payload["spec_count"] == 3
    assert payload["spec_ids_file"] == str(manifest.resolve())
    assert "--spec-ids-file" in payload["next_command"]


def test_selected_specs_are_reordered_before_exact_gate() -> None:
    unordered = [
        SweepSpec("free-tau0p1", "free_bounded", 0.1, 0.1, None, None),
        SweepSpec("topk-tau0p1-a0p8-k4", "topk_preserving", 0.1, 0.1, 0.8, 4),
        SweepSpec("exact", "exact", 0.0, 0.0, None, None),
        SweepSpec("plaintext", "plaintext", 0.0, 0.0, None, None),
    ]

    ordered = _order_specs_for_exact_gate(unordered)

    assert [spec.mode for spec in ordered] == [
        "plaintext",
        "exact",
        "topk_preserving",
        "free_bounded",
    ]
    assert [spec.run_id for spec in ordered[2:]] == [
        "topk-tau0p1-a0p8-k4",
        "free-tau0p1",
    ]


def test_preflight_rejects_zero_generation_tokens_before_asset_access() -> None:
    completed = _run(
        "scripts/preflight_experiment.py",
        "--model-path",
        "/tmp/fastprove-missing-model",
        "--dataset-cache",
        "/tmp/fastprove-missing-cache.pt",
        "--device",
        "cpu",
        "--generation-tokens",
        "0",
    )

    assert completed.returncode != 0
    assert "generation and sample counts must be positive" in (
        completed.stdout + completed.stderr
    )


def test_preflight_requires_explicit_nonstandard_caption_opt_in() -> None:
    with pytest.raises(ValueError, match="non-standard caption-only"):
        _accepted_evaluation_scope(
            "caption_only_nonstandard_lm_candidate",
            accept_nonstandard_caption=False,
        )
    assert (
        _accepted_evaluation_scope(
            "caption_only_nonstandard_lm_candidate",
            accept_nonstandard_caption=True,
        )
        == "caption_only_nonstandard_lm_candidate"
    )


def test_report_cli_defaults_to_plan_only(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "derived"
    record = build_run_record(
        run_id="planned-exact",
        status="skipped",
        config={"mode": "exact", "tau_max": 0.0},
        seed=7,
        model={"id": "random-tiny"},
        dataset={"id": "synthetic"},
        environment={"device": "cpu"},
        sample_count=0,
        metrics={},
        elapsed_seconds=0.0,
        reason={"code": "unit_test", "message": "planned skip"},
    )
    (raw_dir / "one.jsonl").write_text(
        json.dumps(record, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    completed = _run(
        "scripts/build_report.py",
        "--raw",
        str(raw_dir),
        "--output",
        str(output),
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["action"] == "plan"
    assert payload["execution_authorized"] is False
    assert payload["record_count"] == 1
    assert payload["status_counts"] == {"skipped": 1}
    assert not (output / "REPORT.md").exists()


def test_report_cli_builds_only_after_explicit_execute(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "derived"
    record = build_run_record(
        run_id="planned-exact",
        status="skipped",
        config={"mode": "exact", "tau_max": 0.0},
        seed=7,
        model={"id": "random-tiny"},
        dataset={"id": "synthetic"},
        environment={"device": "cpu"},
        sample_count=0,
        metrics={},
        elapsed_seconds=0.0,
        reason={"code": "unit_test", "message": "planned skip"},
    )
    (raw_dir / "one.jsonl").write_text(
        json.dumps(record, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    completed = _run(
        "scripts/build_report.py",
        "--raw",
        str(raw_dir),
        "--output",
        str(output),
        "--execute",
        "--experiment-status",
        "not_run",
        "--pretrained-status",
        "not_run",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["action"] == "build_report"
    assert payload["execution_authorized"] is True
    assert (output / "REPORT.md").is_file()
    assert (output / "tables" / "summary.csv").is_file()
    assert (output / "figures" / "accuracy_vs_tau.png").is_file()
    assert (output / "figures" / "softmax_vs_tau.png").is_file()


def test_report_execute_refuses_empty_raw_without_overwriting_pending(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "derived"
    output.mkdir()
    pending = output / "REPORT.md"
    pending.write_text("pending evidence\n", encoding="utf-8")

    completed = _run(
        "scripts/build_report.py",
        "--raw",
        str(raw_dir),
        "--output",
        str(output),
        "--execute",
    )

    assert completed.returncode != 0
    assert "no raw run records" in completed.stderr
    assert pending.read_text(encoding="utf-8") == "pending evidence\n"
    assert not (output / "tables" / "summary.csv").exists()
    assert not (output / "figures" / "accuracy_vs_tau.png").exists()


def test_report_plan_derives_pretrained_status_from_raw_records(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "derived"
    record = build_run_record(
        run_id="pretrained-success",
        status="success",
        config={
            "mode": "plaintext",
            "experiment_name": "pretrained-smoke",
            "stage": "teacher_forced_eval",
            "expected_run_ids": ["pretrained-success"],
        },
        seed=7,
        model={"id": "pretrained-model", "pretrained": True},
        dataset={
            "id": "public-lm-data",
            "meaningful_lm_evidence": True,
        },
        environment={"device": "cpu"},
        sample_count=1,
        metrics=_valid_comparison_metrics(),
        elapsed_seconds=0.1,
    )
    (raw_dir / "one.jsonl").write_text(
        json.dumps(record, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    completed = _run(
        "scripts/build_report.py",
        "--raw",
        str(raw_dir),
        "--output",
        str(output),
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["pretrained_status"] == "completed"


def test_execution_flags_are_not_interchangeable(tmp_path: Path) -> None:
    correctness = _run(
        "scripts/run_correctness.py",
        "--execute-deferred",
    )
    sweep = _run(
        "scripts/run_accuracy_sweep.py",
        "--execute",
    )

    assert correctness.returncode != 0
    assert sweep.returncode != 0
    assert "unrecognized arguments" in correctness.stderr
    assert "unrecognized arguments" in sweep.stderr


def test_correctness_execute_refuses_nonempty_raw_destination_before_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "existing.jsonl"
    original = '{"run_id":"tiny-exact-correctness"}\n'
    output.write_text(original, encoding="utf-8")

    import fastprove.evaluation.correctness as correctness_module

    def forbidden_inference(**_: object) -> dict:
        raise AssertionError("correctness inference must not start")

    monkeypatch.setattr(
        correctness_module,
        "run_tiny_correctness",
        forbidden_inference,
    )

    return_code = correctness_main(
        [
            "--config",
            "configs/tiny_exact.yaml",
            "--output",
            str(output),
            "--execute",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert return_code == 2
    assert payload["action"] == "raw_output_preflight"
    assert payload["status"] == "refused"
    assert "new --output" in payload["error"]
    assert output.read_text(encoding="utf-8") == original


def test_sweep_execute_refuses_nonempty_raw_destination_before_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "existing.jsonl"
    original = '{"run_id":"plaintext"}\n'
    output.write_text(original, encoding="utf-8")

    import fastprove.evaluation.runner as runner_module

    def forbidden_inference(**_: object) -> dict:
        raise AssertionError("sweep inference must not start")

    monkeypatch.setattr(
        runner_module,
        "run_tiny_sweep_spec",
        forbidden_inference,
    )

    return_code = sweep_main(
        [
            "--config",
            "configs/eval_sweep.yaml",
            "--output",
            str(output),
            "--execute-deferred",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert return_code == 2
    assert payload["action"] == "raw_output_preflight"
    assert payload["status"] == "refused"
    assert "new --output" in payload["error"]
    assert output.read_text(encoding="utf-8") == original


def test_correctness_success_requires_exact_gate() -> None:
    assert _exact_succeeded(
        {
            "status": "success",
            "metrics": {"exact_gate": {"passed": True}},
        }
    )
    assert not _exact_succeeded(
        {
            "status": "success",
            "metrics": {"exact_gate": {"passed": False}},
        }
    )
    assert not _exact_succeeded(
        {
            "status": "failure",
            "metrics": {"exact_gate": {"passed": True}},
        }
    )


def test_sweep_non_success_records_carry_expected_run_manifest() -> None:
    spec = SweepSpec(
        run_id="free-tau0p1",
        mode="free_bounded",
        tau_max=0.1,
        tau_error=0.1,
        alpha=None,
        preserve_top_k=None,
    )
    experiment = {
        "seed": 7,
        "model_id": "unit-model",
        "dataset_id": "unit-data",
        "expected_run_ids": ["plaintext", "exact", "free-tau0p1"],
    }
    environment = {"device": "cpu"}

    failure = _failure_record(
        spec=spec,
        experiment=experiment,
        environment=environment,
        error=RuntimeError("intentional"),
    )
    skipped = _exact_gate_skip_record(
        spec=spec,
        experiment=experiment,
        environment=environment,
        reasons=["exact failed"],
    )

    expected = ["plaintext", "exact", "free-tau0p1"]
    assert failure["config"]["expected_run_ids"] == expected
    assert skipped["config"]["expected_run_ids"] == expected
