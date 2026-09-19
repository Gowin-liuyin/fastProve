"""Decision ⑧ acceptance gate: fixed seeds reproduce identical metrics.

AGENTS.md's acceptance criteria require that fixed seeds reproduce identical
metrics, but before this module only component-level determinism was covered:
``tests/evaluation/test_evaluation_runner.py`` writes one traceable record
without re-running it.  This module closes the end-to-end gap.

Two levels are asserted:

1. the deterministic metrics pipeline (``accuracy`` + ``metrics``) feeding
   ``build_run_record`` twice produces a bit-identical record payload once the
   wall-clock timestamp is pinned;
2. ``run_tiny_sweep_spec`` executed twice with identical arguments produces an
   identical metrics payload apart from wall-clock/performance fields
   (``timestamp_utc``, ``elapsed_seconds``, ``metrics.performance``).
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import pytest
import torch

from fastprove.evaluation.accuracy import (
    compare_teacher_forced_metrics,
    make_synthetic_token_batch,
)
from fastprove.evaluation.artifacts import build_run_record
from fastprove.evaluation.metrics import tensor_error_metrics
from fastprove.evaluation.runner import run_tiny_sweep_spec
from fastprove.evaluation.sweep import SweepSpec

from tests._fixtures import run_record_metrics


def _reproducible_pipeline_record() -> Dict[str, Any]:
    """One deterministic pass over the metrics -> run-record pipeline."""

    tokens, _ = make_synthetic_token_batch(
        sample_count=2, sequence_length=6, vocab_size=32, seed=77
    )
    generator = torch.Generator().manual_seed(99)
    plaintext_logits = torch.randn(2, 6, 32, generator=generator)
    obfuscated_logits = plaintext_logits + 0.01 * torch.randn(
        2, 6, 32, generator=generator
    )
    mask = torch.ones_like(tokens, dtype=torch.bool)
    teacher_forced = compare_teacher_forced_metrics(
        plaintext_logits=plaintext_logits,
        obfuscated_logits=obfuscated_logits,
        input_ids=tokens,
        token_mask=mask,
        bootstrap_replicates=25,
        bootstrap_seed=123,
    )
    value = torch.tensor([[1.0, -2.0, 3.0]])
    error_metrics = tensor_error_metrics(value, value.clone() * 1.01)

    metrics = run_record_metrics(0.01)
    metrics["agreement"]["next_token_top1_agreement"] = teacher_forced[
        "agreement"
    ]["next_token_top1_agreement"]
    record = build_run_record(
        run_id="seed-reproducibility",
        status="success",
        config={"mode": "exact", "tau_max": 0.0, "stage": "full_evaluation"},
        seed=52,
        model={"id": "unit-model", "pretrained": False},
        dataset={"id": "unit-data", "meaningful_lm_evidence": False},
        environment={"actual_device": "cpu", "activation_dtype": "float32"},
        sample_count=2,
        metrics=metrics,
        elapsed_seconds=0.1,
    )
    record["teacher_forced"] = teacher_forced
    record["tensor_error"] = error_metrics
    return record


def test_metrics_and_run_record_pipeline_is_bitwise_reproducible() -> None:
    first = _reproducible_pipeline_record()
    second = _reproducible_pipeline_record()

    for record in (first, second):
        record["timestamp_utc"] = "1970-01-01T00:00:00+00:00"

    assert first == second


def _non_wall_clock_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    metrics = copy.deepcopy(record["metrics"])
    metrics.pop("performance", None)
    return metrics


@pytest.mark.slow
def test_tiny_sweep_metrics_are_reproducible_for_a_fixed_seed(
    project_root,
) -> None:
    spec = SweepSpec(
        run_id="free-repro",
        mode="free_bounded",
        tau_max=0.03,
        tau_error=0.10,
        alpha=None,
        preserve_top_k=None,
    )
    experiment = {
        "seed": 20260731,
        "model_id": "fastprove-random-tiny-correctness-only",
        "dataset_id": "deterministic-synthetic-token-sequences",
    }
    evaluation_override = {
        "sample_count": 2,
        "batch_size": 2,
        "sequence_length": 6,
        "generation_tokens": 1,
        "warmup_runs": 0,
        "timed_runs": 1,
    }
    base_config = project_root / "configs" / "tiny_approx.yaml"

    first = run_tiny_sweep_spec(
        base_config_path=base_config,
        spec=spec,
        experiment=dict(experiment),
        evaluation_override=dict(evaluation_override),
    )
    second = run_tiny_sweep_spec(
        base_config_path=base_config,
        spec=spec,
        experiment=dict(experiment),
        evaluation_override=dict(evaluation_override),
    )

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert first["run_id"] == second["run_id"] == spec.run_id
    assert first["seed"] == second["seed"] == experiment["seed"]
    # The performance block is the only wall-clock carrier inside metrics.
    assert first["metrics"]["performance"]
    assert second["metrics"]["performance"]
    assert _non_wall_clock_metrics(first) == _non_wall_clock_metrics(second)
