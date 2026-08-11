from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from fastprove.config import load_config
from fastprove.evaluation.correctness import run_tiny_correctness
from fastprove.evaluation.runner import (
    _config_for_spec,
    _debug_batch_to_cpu,
    _merge_staged_performance_metrics,
    _mean_metric_dicts,
    _resolve_runtime,
    environment_metadata,
    evaluate_exact_gate,
    run_tiny_sweep_spec,
)
from fastprove.evaluation.sweep import SweepSpec, enumerate_sweep_specs
from fastprove.models.obfuscated import ObfuscatedBlockDebug


ROOT = Path(__file__).resolve().parents[1]


def test_metric_aggregation_preserves_global_extrema_and_counts() -> None:
    result = _mean_metric_dicts(
        [
            {
                "max_absolute_error": 0.1,
                "actual_noise_infinity_norm": 0.03,
                "clean_boundary_margin_min": 0.2,
                "clean_boundary_margin_max": 0.8,
                "mean_absolute_error": 0.02,
                "nan_count": 1.0,
            },
            {
                "max_absolute_error": 0.3,
                "actual_noise_infinity_norm": 0.05,
                "clean_boundary_margin_min": 0.1,
                "clean_boundary_margin_max": 0.6,
                "mean_absolute_error": 0.04,
                "nan_count": 2.0,
            },
        ]
    )
    assert result["max_absolute_error"] == 0.3
    assert result["actual_noise_infinity_norm"] == 0.05
    assert result["clean_boundary_margin_min"] == 0.1
    assert result["clean_boundary_margin_max"] == 0.8
    assert result["mean_absolute_error"] == 0.03
    assert result["nan_count"] == 3.0


def test_staged_performance_merge_keeps_phase_peak_and_schema() -> None:
    plaintext = {
        "prefill": {
            "plaintext": {
                "mean_seconds": 1.0,
                "min_seconds": 0.9,
                "max_seconds": 1.1,
            }
        },
        "decode": {
            "plaintext": {
                "mean_seconds": 2.0,
                "min_seconds": 1.9,
                "max_seconds": 2.1,
                "tpot_seconds": 0.5,
                "tokens_per_second": 2.0,
            }
        },
        "peak_memory": {
            "status": "measured_process_rss",
            "bytes": 100,
            "process_peak_rss_bytes": 100,
            "mps_current_allocated_bytes": None,
        },
        "kv_cache": {"plaintext_bytes": 10},
    }
    obfuscated = {
        "prefill": {
            "mean_seconds": 3.0,
            "min_seconds": 2.9,
            "max_seconds": 3.1,
            "tokens_per_second": 4.0,
        },
        "decode": {
            "mean_seconds": 4.0,
            "min_seconds": 3.9,
            "max_seconds": 4.1,
            "tpot_seconds": 1.0,
            "tokens_per_second": 1.0,
        },
        "peak_memory": {
            "status": "measured_process_rss",
            "bytes": 250,
            "process_peak_rss_bytes": 250,
            "mps_current_allocated_bytes": None,
        },
        "obfuscated_kv_bytes": 20,
    }

    merged = _merge_staged_performance_metrics(
        plaintext=plaintext,
        obfuscated=obfuscated,
        conversion_seconds=0.25,
    )

    assert merged["conversion_time_seconds"] == 0.25
    assert merged["prefill"]["plaintext"]["mean_seconds"] == 1.0
    assert merged["prefill"]["obfuscated"]["mean_seconds"] == 3.0
    assert merged["decode"]["obfuscated"]["tpot_seconds"] == 1.0
    assert merged["peak_memory"]["bytes"] == 250
    assert merged["peak_memory"]["measurement_phases"]["plaintext"]["bytes"] == 100
    assert merged["kv_cache"]["obfuscated_bytes"] == 20


def test_debug_batch_is_detached_to_cpu_before_retention() -> None:
    tensor = torch.ones((1, 1), dtype=torch.float32)
    record = ObfuscatedBlockDebug(
        qk_score_error={"max_absolute_error": 0.0},
        softmax_error={"max_absolute_error": 0.0},
        clean_logits=tensor,
        noisy_logits=tensor,
        clean_probabilities=tensor,
        noisy_probabilities=tensor,
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
        logit_noise=tensor,
        tau=tensor,
        margin=tensor,
        clean_attention_output=tensor,
        attention_output=tensor,
        post_attention=tensor,
        attention_noise_state=tensor,
        z_prime=tensor,
        final_noise_state=tensor,
    )

    copied = _debug_batch_to_cpu((record,))

    assert copied[0].clean_logits.device.type == "cpu"
    assert copied[0].valid_mask.device.type == "cpu"
    assert copied[0].clean_logits is not tensor


def test_environment_metadata_records_accelerator_without_fallback() -> None:
    metadata = environment_metadata()
    assert metadata["python_version"]
    assert metadata["torch_version"]
    assert metadata["platform"]
    assert isinstance(metadata["cuda_available"], bool)
    assert isinstance(metadata["mps_available"], bool)
    assert metadata["cpu_count"] >= 1
    assert metadata["physical_memory_bytes"] is None or (
        metadata["physical_memory_bytes"] > 0
    )
    assert metadata["disk_free_bytes"] > 0


def test_deep_fp32_exact_gate_records_calibrated_softmax_envelope() -> None:
    record = {
        "status": "success",
        "config": {
            "mode": "exact",
            "runtime": {"device": "cpu", "activation_dtype": "float32"},
        },
        "environment": {
            "actual_device": "cpu",
            "checkpoint_compute_dtype": "float64",
        },
        "model": {"pretrained": True},
        "metrics": {
            "nan_inf_count": 0,
            "agreement": {"next_token_top1_agreement": 1.0},
            "greedy": {
                "greedy_token_exact_match": 1.0,
                "greedy_sequence_exact_match": 1.0,
            },
            "layer": {
                "logits": {
                    "max_absolute_error": 4e-4,
                    "relative_l2_error": 2e-5,
                },
                "qk_scores": {
                    "max_absolute_error": 1.5e-2,
                    "relative_l2_error": 2e-6,
                },
                "exact_softmax_probabilities": {
                    "max_absolute_error": 1.5e-3,
                    "relative_l2_error": 9e-6,
                },
            },
            "exact_gate_diagnostics": {
                "first_teacher_forced_argmax_mismatch": None,
                "first_greedy_mismatch": None,
            },
        },
    }
    passed, reasons = evaluate_exact_gate(record)
    assert passed is True
    assert reasons == []
    assert record["metrics"]["exact_gate_tolerances"][
        "max_softmax_absolute_error"
    ] == 2e-3
    assert record["metrics"]["exact_gate_tolerances"]["actual_device"] == "cpu"


def test_runtime_override_is_applied_to_each_sweep_config() -> None:
    config = _config_for_spec(
        load_config(ROOT / "configs" / "tiny_approx.yaml"),
        SweepSpec(
            run_id="runtime-override",
            mode="free_bounded",
            tau_max=0.01,
            tau_error=0.10,
            alpha=None,
            preserve_top_k=None,
        ),
        evaluation_override=None,
        runtime_override={"device": "cpu", "activation_dtype": "bfloat16"},
    )
    assert config.runtime.device == "cpu"
    assert config.runtime.activation_dtype == "bfloat16"


@pytest.mark.skipif(
    not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ),
    reason="MPS is unavailable",
)
def test_mps_fallback_env_is_rejected_before_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "tiny_exact.yaml")
    config = replace(
        config,
        runtime=replace(config.runtime, device="mps"),
    )
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(RuntimeError, match="MPS_FALLBACK"):
        _resolve_runtime(config)


def test_real_sweep_evaluation_override_constructs_without_metadata_fields() -> None:
    raw = yaml.safe_load((ROOT / "configs" / "eval_sweep.yaml").read_text())
    base = load_config(ROOT / "configs" / "tiny_approx.yaml")
    spec = enumerate_sweep_specs(ROOT / "configs" / "eval_sweep.yaml")[0]
    config = _config_for_spec(base, spec, raw["evaluation_override"])
    assert config.evaluation.sequence_length == 24
    assert config.evaluation.generation_tokens == 4
    assert config.evaluation.warmup_runs == 3
    assert config.evaluation.timed_runs == 10
    assert config.evaluation.bootstrap_replicates == 1000


def test_tiny_exact_correctness_writes_one_traceable_record(
    tmp_path: Path,
) -> None:
    output = tmp_path / "correctness.jsonl"
    record = run_tiny_correctness(
        config_path=ROOT / "configs" / "tiny_exact.yaml",
        output_path=output,
    )
    stored = [json.loads(line) for line in output.read_text().splitlines()]
    assert stored == [record]
    assert record["status"] == "success"
    assert record["config"]["expected_run_ids"] == [
        "tiny-exact-correctness"
    ]
    assert record["config"]["base_config_sha256"]
    assert record["environment"]["torch_version"]
    assert "numpy_version" in record["environment"]
    assert record["model"]["evidence_scope"] == "correctness_only_random_tiny"
    basis_manifest = record["model"]["obfuscation_manifest"]
    assert basis_manifest["hidden_basis"]["condition_number"] <= 10.0
    assert len(basis_manifest["value_bases"]) == 4
    assert all(
        item["condition_number"] <= 10.0
        for item in basis_manifest["value_bases"]
    )
    assert record["dataset"]["sample_ids_sha256"]
    assert record["metrics"]["nan_inf_count"] == 0
    assert (
        record["metrics"]["agreement"]["next_token_top1_agreement"] == 1.0
    )
    assert record["metrics"]["greedy"]["greedy_token_exact_match"] == 1.0
    assert record["metrics"]["layer"]["logits"]["max_absolute_error"] <= 2e-4
    assert record["metrics"]["exact_gate_diagnostics"] == {
        "first_teacher_forced_argmax_mismatch": None,
        "first_greedy_mismatch": None,
    }
    passed, reasons = evaluate_exact_gate(record)
    assert passed is True
    assert reasons == []
    assert record["metrics"]["exact_gate_tolerances"]["profile"] == "tiny_fp32_strict"
    assert record["metrics"]["exact_gate"] == {
        "passed": True,
        "reasons": [],
    }
    cache_metrics = record["metrics"]["performance"]["kv_cache"]
    assert cache_metrics["status"] == "measured_tensor_storage"
    assert cache_metrics["plaintext_bytes"] > 0
    assert cache_metrics["obfuscated_bytes"] >= cache_metrics["plaintext_bytes"]
    assert cache_metrics["decode_benchmark_uses_cache"] is True


def test_correctness_entrypoint_rejects_non_exact_config(
    tmp_path: Path,
) -> None:
    output = tmp_path / "must-not-exist.jsonl"
    with pytest.raises(ValueError, match="exact"):
        run_tiny_correctness(
            config_path=ROOT / "configs" / "tiny_approx.yaml",
            output_path=output,
        )
    assert not output.exists()


def test_tiny_free_bounded_runner_records_actual_noise_bound(
    tmp_path: Path,
) -> None:
    spec = SweepSpec(
        run_id="free-unit",
        mode="free_bounded",
        tau_max=0.03,
        tau_error=0.10,
        alpha=None,
        preserve_top_k=None,
    )
    output = tmp_path / "free.jsonl"
    record = run_tiny_sweep_spec(
        base_config_path=ROOT / "configs" / "tiny_approx.yaml",
        spec=spec,
        experiment={
            "seed": 20260731,
            "model_id": "fastprove-random-tiny-correctness-only",
            "dataset_id": "deterministic-synthetic-token-sequences",
        },
        evaluation_override={
            "sample_count": 2,
            "batch_size": 2,
            "sequence_length": 6,
            "generation_tokens": 1,
            "warmup_runs": 0,
            "timed_runs": 1,
        },
        output_path=output,
    )
    assert record["status"] == "success"
    assert record["config"]["mode"] == "free_bounded"
    assert (
        record["metrics"]["softmax"]["actual_noise_infinity_norm"]
        <= 0.03 + 1e-7
    )
    assert record["metrics"]["nan_inf_count"] == 0
    assert len(output.read_text().splitlines()) == 1
