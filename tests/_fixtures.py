"""Deterministic object builders shared by ``tests/`` and ``evals/`` fixtures.

The pytest fixture declarations live in ``tests/conftest.py`` and
``evals/conftest.py``; both import the builders below so the fixture bodies are
defined exactly once.  This module must stay importable without pytest so that
``evals/conftest.py`` can reuse it during collection.

Everything in this module is a plain callable: test modules call the builders
directly when they need to vary parameters, and ``tests/conftest.py`` wraps a
few of them in pytest fixtures.  The builders intentionally return the exact
objects the per-file copies used to construct, so assertions stay unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from fastprove.config import ModelConfig, ObfuscationConfig, PrototypeConfig, load_config
from fastprove.evaluation.accuracy import make_synthetic_token_batch
from fastprove.evaluation.artifacts import build_run_record
from fastprove.layers.attention import ApproximationConfig, AttentionMode
from fastprove.layers.moe import ObfuscatedMoE
from fastprove.layers.router import RouterConfig, RouterMode
from fastprove.models.obfuscated import (
    ObfuscatedDecoderBlock,
    ObfuscatedTinyCausalLM,
)
from fastprove.models.plain import PlainDecoderBlock, PlainTinyCausalLM
from fastprove.structured import generate_structured_basis

from evals.conditions import (
    ConditionId,
    ConditionSpec,
    default_base_prototype_config,
    get_condition,
)
from evals.keys import MasterKey, generate_master_keys
from evals.model_factory import EvalModels, build_models


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# evals condition fixtures (used by tests/conftest.py and evals/conftest.py)


def build_base_config() -> PrototypeConfig:
    path = project_root() / "configs" / "tiny_exact.yaml"
    if path.is_file():
        return load_config(path)
    return default_base_prototype_config()


def build_master_keys(
    base_config: PrototypeConfig, *, count: int = 3
) -> List[MasterKey]:
    return generate_master_keys(
        count=count, base_seed=base_config.runtime.seed
    )


def build_synthetic_batch(
    base_config: PrototypeConfig,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    tokens, identifiers = make_synthetic_token_batch(
        sample_count=base_config.evaluation.sample_count,
        sequence_length=base_config.evaluation.sequence_length,
        vocab_size=base_config.model.vocab_size,
        seed=base_config.runtime.seed,
    )
    mask = torch.ones_like(tokens, dtype=torch.bool)
    return tokens, mask, identifiers


def build_condition(condition_id: ConditionId) -> ConditionSpec:
    return get_condition(condition_id)


def build_condition_models(
    base_config: PrototypeConfig,
    condition: ConditionSpec,
    master_keys: List[MasterKey],
) -> EvalModels:
    return build_models(
        base_config, condition, key=master_keys[0], debug_enabled=True
    )


# ---------------------------------------------------------------------------
# canonical small prototypes
#
# These replace the per-file ``_model_config`` / ``_obfuscation`` factories.
# The defaults are the magic numbers repeated across the suite
# (hidden_noise_dim=8, value_noise_dim_per_head=2, max_condition_number=10.0,
# noise_propagation_gamma=0.5, basis_block_size=8); every call site keeps its
# effective configuration by overriding only the fields it used to spell out.


def model_config(**overrides: Any) -> ModelConfig:
    """Tiny Llama-like ``ModelConfig`` with the suite's canonical dimensions."""

    values: Dict[str, Any] = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_sequence_length=32,
        rms_epsilon=1e-5,
        rope_theta=10000.0,
    )
    values.update(overrides)
    return ModelConfig(**values)


def obfuscation_config(
    *,
    refresh_mode: str = "fixed_debug",
    **overrides: Any,
) -> ObfuscationConfig:
    """``ObfuscationConfig`` with the suite's canonical obfuscation settings."""

    values: Dict[str, Any] = dict(
        hidden_noise_dim=8,
        value_noise_dim_per_head=2,
        max_condition_number=10.0,
        noise_propagation_gamma=0.5,
        refresh_mode=refresh_mode,
        basis_block_size=8,
    )
    values.update(overrides)
    return ObfuscationConfig(**values)


def unit_basis(
    signal_dim: int = 64,
    noise_dim: int = 8,
    *,
    block_size: int = 8,
    seed: int = 17,
    domain: str = "test",
    max_condition_number: float = 10.0,
    dtype: torch.dtype = torch.float64,
):
    """``generate_structured_basis`` with the suite's canonical arguments."""

    return generate_structured_basis(
        signal_dim,
        noise_dim,
        seed=seed,
        domain=domain,
        block_size=block_size,
        max_condition_number=max_condition_number,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# plain -> obfuscated conversion helpers


def tiny_lm_pair(
    config: Optional[ModelConfig] = None,
    *,
    plain_seed: int,
    conversion_seed: int,
    mode: AttentionMode = AttentionMode.EXACT,
    approximation: Optional[ApproximationConfig] = None,
    debug_enabled: bool = True,
    obfuscation: Optional[ObfuscationConfig] = None,
) -> Tuple[PlainTinyCausalLM, ObfuscatedTinyCausalLM]:
    """Build a matching plaintext / obfuscated tiny causal-LM pair."""

    if config is None:
        config = model_config()
    if obfuscation is None:
        obfuscation = obfuscation_config()
    plain = PlainTinyCausalLM(config, seed=plain_seed, debug_enabled=debug_enabled)
    converted = ObfuscatedTinyCausalLM.from_plain(
        plain,
        obfuscation=obfuscation,
        mode=mode,
        approximation=approximation,
        seed=conversion_seed,
        debug_enabled=debug_enabled,
    )
    return plain, converted


def decoder_block_pair(
    config: Optional[ModelConfig] = None,
    *,
    plain_seed: int = 101,
    layer_id: int = 0,
    conversion_seed: int = 202,
    mode: AttentionMode = AttentionMode.EXACT,
    approximation: Optional[ApproximationConfig] = None,
    debug_enabled: bool = True,
    obfuscation: Optional[ObfuscationConfig] = None,
) -> Tuple[PlainDecoderBlock, ObfuscatedDecoderBlock]:
    """Build a matching plaintext / obfuscated decoder-block pair."""

    if config is None:
        config = model_config()
    if obfuscation is None:
        obfuscation = obfuscation_config()
    plain = PlainDecoderBlock(
        config, seed=plain_seed, layer_id=layer_id, debug_enabled=debug_enabled
    )
    converted = ObfuscatedDecoderBlock.from_plain(
        plain,
        obfuscation=obfuscation,
        mode=mode,
        approximation=approximation,
        seed=conversion_seed,
        debug_enabled=debug_enabled,
    )
    return plain, converted


# ---------------------------------------------------------------------------
# run-record metric payloads
#
# Four near-identical copies existed in test_report_integrity.py,
# test_report_cli_coverage.py, test_cli_plan.py and test_run_record_schema.py.
# Two distinct payload families survive: the tau-parameterized core metrics
# (report_integrity + run_record_schema) and the all-zero comparison metrics
# (report_cli_coverage + cli_plan).


def run_record_metrics(tau: float) -> Dict[str, Any]:
    """Core-metrics payload parameterized by the perturbation strength tau."""

    return {
        "plaintext": {
            "negative_log_likelihood": 1.0,
            "perplexity": math.e,
            "next_token_top1_accuracy": 0.5,
            "next_token_top5_accuracy": 0.9,
        },
        "obfuscated": {
            "negative_log_likelihood": 1.0 + tau,
            "perplexity": math.e + tau,
            "next_token_top1_accuracy": 0.5 - tau,
            "next_token_top5_accuracy": 0.9 - tau,
        },
        "agreement": {"next_token_top1_agreement": 1.0 - tau},
        "degradation": {
            "nll_absolute_increase": tau,
            "nll_relative_increase": tau,
            "perplexity_absolute_increase": tau,
            "perplexity_relative_increase": tau / math.e,
            "top1_absolute_drop": tau,
            "top1_relative_drop": tau / 0.5,
            "top5_absolute_drop": tau,
            "top5_relative_drop": tau / 0.9,
        },
        "greedy": {
            "greedy_token_exact_match": 1.0 - tau,
            "greedy_sequence_exact_match": 1.0 - tau,
        },
        "softmax": {
            "kl_divergence": tau,
            "js_divergence": tau / 2,
            "topk_overlap": 1.0 - tau,
            "rank_correlation": 1.0 - tau,
            "topk_changed_fraction": tau,
            "actual_noise_infinity_norm": tau,
            "zero_noise_query_fraction": 1.0 if tau == 0 else 0.0,
            "attention_output_relative_l2_error": tau,
        },
        "performance": {
            "conversion_time_seconds": 0.01,
            "prefill_latency_seconds": 0.02,
            "decode_tpot_seconds": 0.003,
            "tokens_per_second": 333.0,
            "peak_memory_bytes": 1024,
            "kv_cache_memory_bytes": 512,
        },
    }


def comparison_metrics() -> Dict[str, Any]:
    """All-zero comparison payload used by the report CLI tests."""

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


# ---------------------------------------------------------------------------
# run-record constructors
#
# Five variants existed in test_report_integrity.py (success/failure),
# test_report_cli_coverage.py (``_success``), test_run_record_schema.py
# (``_valid_record``) and test_cli_plan.py (inline planned-skip records).


def success_record(
    run_id: str,
    *,
    mode: str = "free_bounded",
    tau: float = 0.01,
    raw_source: Optional[str] = "results/raw/attempt/eval.jsonl",
    raw_line: Optional[int] = 1,
) -> Dict[str, Any]:
    record = build_run_record(
        run_id=run_id,
        status="success",
        config={
            "mode": mode,
            "tau_max": tau,
            "alpha": 0.8,
            "preserve_top_k": 8,
            "stage": "full_evaluation",
            "evaluation": {
                "sequence_length": 128,
                "batch_size": 2,
            },
        },
        seed=52,
        model={
            "id": "unit-model",
            "revision": "revision-a",
            "weight_sha256": "weight-sha-a",
            "pretrained": False,
        },
        dataset={
            "id": "unit-data",
            "split": "test",
            "sample_ids_sha256": "sample-sha-a",
            "tokenized_inputs_sha256": "tokenized-sha-a",
            "meaningful_lm_evidence": False,
        },
        environment={
            "actual_device": "cpu",
            "activation_dtype": "float32",
            "batch_size": 2,
        },
        sample_count=4,
        metrics=run_record_metrics(tau),
        elapsed_seconds=0.1,
    )
    if raw_source is not None:
        record["_raw_source"] = raw_source
    if raw_line is not None:
        record["_raw_line"] = raw_line
    return record


def failure_record(
    run_id: str,
    *,
    status: str = "failure",
    tau: float = 0.05,
    raw_line: int = 9,
) -> Dict[str, Any]:
    common = {
        "run_id": run_id,
        "status": status,
        "config": {
            "mode": "topk_preserving",
            "tau_max": tau,
            "alpha": 0.95,
            "preserve_top_k": 16,
            "stage": "full_evaluation",
            "evaluation": {
                "sequence_length": 128,
                "batch_size": 2,
            },
        },
        "seed": 52,
        "model": {
            "id": "unit-model",
            "revision": "revision-a",
            "weight_sha256": "weight-sha-a",
            "pretrained": False,
        },
        "dataset": {
            "id": "unit-data",
            "split": "test",
            "sample_ids_sha256": "sample-sha-a",
            "tokenized_inputs_sha256": "tokenized-sha-a",
            "meaningful_lm_evidence": False,
        },
        "environment": {
            "actual_device": "cpu",
            "activation_dtype": "float32",
            "batch_size": 2,
        },
        "sample_count": 0,
        "metrics": {},
        "elapsed_seconds": 0.1,
    }
    if status == "failure":
        record = build_run_record(
            **common,
            error={"type": "RuntimeError", "message": "intentional failure"},
            stage="full_evaluation",
            last_completed_sample_id=None,
            partial_metrics_available=False,
        )
    else:
        record = build_run_record(
            **common,
            reason={
                "code": "not_selected",
                "message": "not selected after calibration",
            },
        )
    record["_raw_source"] = "results/raw/attempt/eval.jsonl"
    record["_raw_line"] = raw_line
    return record


def report_cli_success(
    run_id: str,
    *,
    mode: str,
    experiment_name: Optional[str] = None,
    stage: Optional[str] = None,
    expected_run_ids: Optional[List[str]] = None,
    pretrained: bool = False,
) -> Dict[str, Any]:
    """Success record used by the report-CLI coverage tests."""

    config: Dict[str, object] = {"mode": mode}
    if experiment_name is not None:
        config["experiment_name"] = experiment_name
    if stage is not None:
        config["stage"] = stage
    if expected_run_ids is not None:
        config["expected_run_ids"] = expected_run_ids
    metrics = comparison_metrics()
    if mode == "exact":
        metrics["exact_gate"] = {"passed": True, "reasons": []}
    return build_run_record(
        run_id=run_id,
        status="success",
        config=config,
        seed=7,
        model={"id": "model", "pretrained": pretrained},
        dataset={
            "id": "dataset",
            "meaningful_lm_evidence": pretrained,
        },
        environment={"device": "cpu"},
        sample_count=1,
        metrics=metrics,
        elapsed_seconds=0.01,
    )


def valid_run_record(**overrides: Any) -> Dict[str, Any]:
    """Strict-schema success record used by test_run_record_schema.py."""

    record = build_run_record(
        run_id="schema-valid",
        status="success",
        config={"mode": "exact"},
        seed=52,
        model={"id": "unit-model", "pretrained": False},
        dataset={
            "id": "unit-data",
            "meaningful_lm_evidence": False,
        },
        environment={"device": "cpu", "dtype": "float32"},
        sample_count=2,
        metrics=run_record_metrics(0.01),
        elapsed_seconds=0.1,
    )
    record.update(overrides)
    return record


def planned_skip_record() -> Dict[str, Any]:
    """Skipped record used by the report-CLI plan-only tests."""

    return build_run_record(
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


# ---------------------------------------------------------------------------
# MoE materials and builders


def moe_materials(
    *,
    signal: int = 32,
    noise: int = 4,
    experts: int = 8,
    intermediate: int = 48,
    seed: int = 7,
):
    """Deterministic basis / router / expert weights for the MoE tests."""

    basis = unit_basis(
        signal, noise, seed=seed, domain="moe", block_size=4
    )
    generator = torch.Generator().manual_seed(seed)
    gamma = torch.rand(signal, generator=generator) + 0.5
    router_w = torch.randn(signal, experts, generator=generator) * 0.2
    expert_weights = [
        (
            torch.randn(signal, intermediate, generator=generator) * 0.1,
            torch.randn(signal, intermediate, generator=generator) * 0.1,
            torch.randn(intermediate, signal, generator=generator) * 0.1,
        )
        for _ in range(experts)
    ]
    permutation = torch.randperm(experts, generator=generator)
    return basis, gamma, router_w, expert_weights, permutation


def build_moe(
    basis,
    gamma,
    router_w,
    expert_weights,
    permutation,
    *,
    top_k: int,
    mode: RouterMode,
    config: Optional[RouterConfig],
    seed: int = 7,
    debug_enabled: bool = True,
    signal_noise_coupling_scale: float = 0.02,
    free_noise_energy_ratio: float = 0.02,
) -> ObfuscatedMoE:
    """``ObfuscatedMoE`` with the standalone layer's pre-S1 equivalent scales."""

    return ObfuscatedMoE(
        basis=basis,
        gamma_ffn=gamma,
        router_weight_math=router_w,
        plain_experts=expert_weights,
        expert_permutation=permutation,
        top_k=top_k,
        router_mode=mode,
        router_config=config,
        seed=seed,
        layer_id="block-0",
        debug_enabled=debug_enabled,
        noise_propagation_gamma=0.5,
        refresh_mode="fixed_debug",
        # Pre-S1 equivalent scales: preserve the historical 0.02 behavior of
        # this standalone layer; production configs carry the same values.
        signal_noise_coupling_scale=signal_noise_coupling_scale,
        free_noise_energy_ratio=free_noise_energy_ratio,
    )


def production_moe(*, seed: int = 7):
    """A production-mode (no debug) MoE plus a deterministic mixed input.

    Returns ``(module, mixed)`` so exposure tests can exercise the production
    entry points without a debug-capable module.
    """

    basis, gamma, router_w, expert_weights, permutation = moe_materials(seed=seed)
    module = build_moe(
        basis,
        gamma,
        router_w,
        expert_weights,
        permutation,
        top_k=2,
        mode=RouterMode.EXACT,
        config=None,
        seed=seed,
        debug_enabled=False,
    )
    generator = torch.Generator().manual_seed(seed + 1)
    hidden = torch.randn(2, 5, basis.signal_dim, generator=generator)
    noise = torch.randn(2, 5, basis.noise_dim, generator=generator) * 0.05
    mixed = basis.mix(torch.cat((hidden, noise), dim=-1))
    return module, mixed


# Script loading is unified on plain ``from scripts.<name> import ...``; the
# previous ``importlib.util.spec_from_file_location`` idiom was removed when
# test_real_scenario_prompts.py switched to the package import.
