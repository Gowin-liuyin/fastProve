"""Executable tiny-model evaluation primitives.

These functions perform work only when called by an explicitly authorized CLI
path. Importing this module never starts an evaluation.
"""

from __future__ import annotations
import hashlib
import json
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ..config import PrototypeConfig, load_config
from ..layers.attention import ApproximationConfig, AttentionMode
from ..models.obfuscated import ObfuscatedBlockDebug, ObfuscatedTinyCausalLM
from ..models.plain import PlainTinyCausalLM
from ..pretrained.qwen2 import Qwen2Artifact, load_qwen2_artifact, load_qwen2_plain
from ..seed import RequestContext
from ..structured import reduction_compute_dtype_name
from .accuracy import (
    compare_teacher_forced_metrics,
    greedy_generation_metrics,
    make_synthetic_token_batch,
)
from .artifacts import (
    append_jsonl_record,
    build_run_record,
    validate_run_record,
)
from .diagnostics import _first_argmax_mismatch, _first_token_mismatch
from .environment import environment_metadata
from .exact_gate import _exact_gate_tolerances, evaluate_exact_gate
from .layer_metrics import (
    _debug_batch_to_cpu,
    _identity_softmax_metrics,
    _layer_metrics,
    _mean_metric_dicts,
)
from .metrics import tensor_error_metrics
from .performance import (
    _merge_staged_performance_metrics,
    _obfuscated_performance_metrics,
    _performance_metrics,
    _time_call,
)
from .pretrained import token_cache_provenance_issues
from .runtime import (
    _mps_allocated_bytes,
    _process_peak_rss_bytes,
    _release_runtime_memory,
    _resolve_runtime,
    _sync,
)
from .spec_config import _config_for_spec
from .sweep import SweepSpec
from .token_cache import load_token_cache


def _hash_identifiers(identifiers: Sequence[str]) -> str:
    payload = json.dumps(
        list(identifiers), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a local configuration file for run-level reproducibility."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _obfuscation_manifest(
    model: Optional[ObfuscatedTinyCausalLM],
) -> Optional[Dict[str, Any]]:
    """Return non-secret basis metadata for a converted evaluation model.

    The manifest records condition numbers and public fingerprints only.  It
    deliberately does not serialize mixing matrices, inverses, or seed
    material, while making the offline conditioning checks auditable.
    """

    if model is None:
        return None
    hidden = model.hidden_basis
    return {
        "max_condition_number": float(model.obfuscation.max_condition_number),
        "hidden_basis": {
            "signal_dim": int(hidden.signal_dim),
            "noise_dim": int(hidden.noise_dim),
            "condition_number": float(hidden.condition_number),
            "fingerprint": hidden.fingerprint,
        },
        "value_bases": [
            {
                "condition_number": float(condition_number),
                "fingerprint": fingerprint,
            }
            for layer_conditions, layer_fingerprints in zip(
                model.value_basis_condition_numbers,
                model.value_basis_fingerprints,
            )
            for condition_number, fingerprint in zip(
                layer_conditions, layer_fingerprints
            )
        ],
    }


def _execute_tiny(
    *,
    config: PrototypeConfig,
    spec: SweepSpec,
    experiment: Dict[str, Any],
    pretrained_path: Optional[str | Path] = None,
    pretrained_artifact: Optional[Qwen2Artifact] = None,
    dataset_cache_path: Optional[str | Path] = None,
) -> Tuple[Dict[str, Any], float]:
    started = time.perf_counter()
    device, dtype = _resolve_runtime(config)
    seed = int(experiment.get("seed", config.runtime.seed))
    artifact = pretrained_artifact
    token_cache = None
    if pretrained_path is not None and dataset_cache_path is None:
        raise ValueError("pretrained_path requires dataset_cache_path")
    if dataset_cache_path is not None and pretrained_path is None:
        raise ValueError("dataset_cache_path requires pretrained_path")
    if pretrained_path is not None:
        if artifact is None:
            artifact = load_qwen2_artifact(
                pretrained_path,
                max_sequence_length=(
                    config.evaluation.sequence_length
                    + config.evaluation.generation_tokens
                ),
                compute_hashes=True,
            )
        token_cache = load_token_cache(
            dataset_cache_path,
            expected_vocab_size=artifact.model_config.vocab_size,
            expected_sequence_length=config.evaluation.sequence_length,
        )
        provenance_issues = token_cache_provenance_issues(token_cache, artifact)
        if provenance_issues:
            raise ValueError(
                "token cache provenance is incomplete: "
                + "; ".join(provenance_issues)
            )
        if token_cache.sample_count < config.evaluation.sample_count:
            raise ValueError(
                "token cache has fewer samples than evaluation.sample_count"
            )
        config = replace(
            config,
            model=artifact.model_config,
        )
        tokens_cpu = token_cache.input_ids[: config.evaluation.sample_count]
        identifiers = list(token_cache.sample_ids[: config.evaluation.sample_count])
        token_mask_cpu = token_cache.token_mask[: config.evaluation.sample_count]
    else:
        tokens_cpu, identifiers = make_synthetic_token_batch(
            sample_count=config.evaluation.sample_count,
            sequence_length=config.evaluation.sequence_length,
            vocab_size=config.model.vocab_size,
            seed=seed,
        )
        token_mask_cpu = torch.ones_like(tokens_cpu, dtype=torch.bool)
    tokens = tokens_cpu.to(device=device)
    mask = token_mask_cpu.to(device=device)
    if artifact is not None:
        plain = load_qwen2_plain(
            artifact,
            device=device,
            dtype=dtype,
            seed=seed,
            debug_enabled=True,
        )
    else:
        plain = PlainTinyCausalLM(
            config.model, seed=seed, debug_enabled=True
        ).to(device=device, dtype=dtype)
    plain.eval()
    mode = AttentionMode(spec.mode)
    approximation = None
    if mode in (
        AttentionMode.TOPK_PRESERVING,
        AttentionMode.FREE_BOUNDED,
    ):
        approximation = ApproximationConfig(
            tau_max=spec.tau_max,
            tau_error=spec.tau_error,
            alpha=config.attention.alpha,
            preserve_top_k=config.attention.preserve_top_k,
        )

    plain_parts = []
    with torch.no_grad():
        for start in range(0, tokens.shape[0], config.evaluation.batch_size):
            stop = min(start + config.evaluation.batch_size, tokens.shape[0])
            batch = tokens[start:stop]
            batch_mask = mask[start:stop]
            plain_logits = plain(batch, token_mask=batch_mask)
            plain_parts.append(plain_logits.float().cpu())
    plaintext_logits = torch.cat(plain_parts, dim=0)
    prompt = tokens[: config.evaluation.batch_size]
    prompt_mask = mask[: config.evaluation.batch_size]
    plain_generated = plain.generate_greedy(
        prompt,
        max_new_tokens=config.evaluation.generation_tokens,
        token_mask=prompt_mask,
    )
    plain_generated_cpu = plain_generated.cpu()
    plain_performance = _performance_metrics(
        plain=plain,
        obfuscated=None,
        tokens=tokens,
        token_mask=mask,
        context=RequestContext(seed, config.runtime.request_id),
        evaluation=config.evaluation,
        device=device,
        conversion_seconds=0.0,
    )

    converted = None
    obfuscated = None
    conversion_seconds = 0.0
    obfuscated_parts = []
    debug_batches: List[Tuple[ObfuscatedBlockDebug, ...]] = []
    if mode != AttentionMode.PLAINTEXT:
        converted = ObfuscatedTinyCausalLM.from_plain(
            plain,
            obfuscation=config.obfuscation,
            mode=mode,
            approximation=approximation,
            seed=seed,
            debug_enabled=True,
        )
        obfuscated = converted.module.to(device=device, dtype=dtype)
        obfuscated.eval()
        conversion_seconds = converted.conversion_time_seconds
        codec = converted.token_codec
        # The converted module owns cloned deployed weights and no longer
        # needs the plaintext module.  Drop the latter before obfuscated
        # forwards/timing so the two full 1.5B copies are not resident during
        # the expensive part of the run.
        del plain
        _release_runtime_memory()
        with torch.no_grad():
            for start in range(0, tokens.shape[0], config.evaluation.batch_size):
                stop = min(start + config.evaluation.batch_size, tokens.shape[0])
                batch = tokens[start:stop]
                batch_mask = mask[start:stop]
                encoded = codec.encode(batch).to(device=device)
                context = RequestContext(
                    seed,
                    "%s:batch-%d"
                    % (config.runtime.request_id, start // config.evaluation.batch_size),
                )
                obfuscated_logits_permuted, debug = obfuscated.forward_debug(
                    encoded,
                    token_mask=batch_mask,
                    request_context=context,
                )
                obfuscated_parts.append(obfuscated_logits_permuted.float().cpu())
                debug_batches.append(_debug_batch_to_cpu(debug))
        # Column-permuted logits: L_plain[:, i] = L_tilde[:, tau(i)], so
        # decoding the columns is an index gather by ``tau``.
        obfuscated_logits = torch.cat(obfuscated_parts, dim=0)[
            ..., codec.permutation
        ]
        performance_context = RequestContext(
            seed, "%s:generation" % config.runtime.request_id
        )
        obfuscated_generated = codec.decode(
            obfuscated.generate_greedy(
                codec.encode(prompt).to(device=device),
                max_new_tokens=config.evaluation.generation_tokens,
                request_context=performance_context,
                token_mask=prompt_mask,
            )
        )
        obfuscated_performance = _obfuscated_performance_metrics(
            obfuscated=obfuscated,
            tokens=tokens,
            token_mask=mask,
            context=performance_context,
            evaluation=config.evaluation,
            device=device,
            token_codec=codec,
        )
        performance = _merge_staged_performance_metrics(
            plaintext=plain_performance,
            obfuscated=obfuscated_performance,
            conversion_seconds=conversion_seconds,
        )
    else:
        obfuscated_logits = plaintext_logits
        obfuscated_generated = plain_generated
        performance = plain_performance

    metrics = compare_teacher_forced_metrics(
        plaintext_logits=plaintext_logits,
        obfuscated_logits=obfuscated_logits,
        input_ids=tokens_cpu,
        token_mask=token_mask_cpu,
        bootstrap_replicates=config.evaluation.bootstrap_replicates,
        bootstrap_seed=seed,
    )
    layer, softmax, debug_nan_inf = _layer_metrics(
        debug_batches,
        top_k=config.attention.preserve_top_k,
    )
    layer["logits"] = tensor_error_metrics(
        plaintext_logits, obfuscated_logits
    )
    metrics["greedy"] = greedy_generation_metrics(
        plaintext_tokens=plain_generated_cpu,
        obfuscated_tokens=obfuscated_generated.cpu(),
        prompt_length=prompt.shape[1],
    )
    metrics["exact_gate_diagnostics"] = {
        "first_teacher_forced_argmax_mismatch": _first_argmax_mismatch(
            plaintext_logits, obfuscated_logits
        ),
        "first_greedy_mismatch": _first_token_mismatch(
            plain_generated_cpu,
            obfuscated_generated.cpu(),
            prompt_length=prompt.shape[1],
        ),
    }
    metrics["layer"] = layer
    metrics["softmax"] = softmax
    output_nan_inf = int(torch.isnan(obfuscated_logits).sum().item()) + int(
        torch.isinf(obfuscated_logits).sum().item()
    )
    metrics["nan_inf_count"] = output_nan_inf + debug_nan_inf
    metrics["performance"] = performance
    payload = {
        "metrics": metrics,
        "sample_count": config.evaluation.sample_count,
        "model": {
            "identifier": experiment.get(
                "model_id", "fastprove-random-tiny-correctness-only"
            ),
            "architecture": asdict(config.model),
            "base_weight_seed": seed,
            "evidence_scope": "correctness_only_random_tiny",
            "pretrained": False,
        },
        "dataset": {
            "identifier": experiment.get(
                "dataset_id", "deterministic-synthetic-token-sequences"
            ),
            "sample_ids_sha256": _hash_identifiers(identifiers),
            "sample_ids": identifiers,
            "provenance": "locally generated deterministic random tokens",
            "meaningful_lm_evidence": False,
        },
    }
    if artifact is not None:
        payload["model"] = {
            "identifier": experiment.get("model_id", "local-qwen2"),
            "revision": artifact.upstream_revision,
            "weight_sha256": artifact.weights_sha256,
            "config_sha256": artifact.config_sha256,
            "tokenizer_sha256": artifact.tokenizer_sha256,
            "tokenizer_config_sha256": artifact.tokenizer_config_sha256,
            "architecture": asdict(config.model),
            "checkpoint_manifest": artifact.to_dict(),
            "evidence_scope": experiment.get(
                "evaluation_scope", "language_model_accuracy"
            ),
            "pretrained": True,
        }
        payload["dataset"] = {
            "identifier": experiment.get("dataset_id", "local-token-cache"),
            "sample_ids_sha256": _hash_identifiers(identifiers),
            "tokenized_inputs_sha256": token_cache.content_sha256,
            "sample_ids": identifiers,
            "cache_path": str(Path(dataset_cache_path).expanduser().resolve()),
            "cache_content_sha256": token_cache.content_sha256,
            "cache_metadata": token_cache.metadata,
            "evaluation_scope": experiment.get(
                "evaluation_scope", "language_model_accuracy"
            ),
            "provenance": "validated local token cache",
            "meaningful_lm_evidence": True,
        }
        payload["dataset"]["meaningful_lm_evidence"] = bool(
            experiment.get("meaningful_lm_evidence", True)
        )
    obfuscation_manifest = _obfuscation_manifest(obfuscated)
    if obfuscation_manifest is not None:
        payload["model"]["obfuscation_manifest"] = obfuscation_manifest
    # Do not carry a full checkpoint or converted closures into the next sweep
    # point.  ``_execute_tiny`` already releases plaintext before the
    # obfuscated phase; this final release covers the end of each independent
    # record so a 72-point loop does not accumulate allocator-backed models.
    if mode == AttentionMode.PLAINTEXT:
        del plain
    else:
        del obfuscated
        del converted
    _release_runtime_memory()
    return payload, time.perf_counter() - started


def run_tiny_sweep_spec(
    *,
    base_config_path: Path,
    spec: SweepSpec,
    experiment: Dict[str, Any],
    evaluation_override: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None,
    pretrained_path: Optional[str | Path] = None,
    pretrained_artifact: Optional[Qwen2Artifact] = None,
    dataset_cache_path: Optional[str | Path] = None,
    runtime_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute and optionally append one independently traceable tiny run."""

    started = time.perf_counter()
    seed = int(experiment.get("seed", 0))
    base_config_resolved = Path(base_config_path).expanduser().resolve()
    config_provenance: Dict[str, Any] = {
        "base_config_path": str(base_config_resolved),
    }
    if base_config_resolved.is_file():
        config_provenance["base_config_sha256"] = _sha256_file(
            base_config_resolved
        )
    for key in ("sweep_config_path", "sweep_config_sha256"):
        if experiment.get(key) is not None:
            config_provenance[key] = experiment[key]
    config: Optional[PrototypeConfig] = None
    try:
        config = _config_for_spec(
            load_config(base_config_path),
            spec,
            evaluation_override,
            runtime_override,
        )
        payload, elapsed = _execute_tiny(
            config=config,
            spec=spec,
            experiment=experiment,
            pretrained_path=pretrained_path,
            pretrained_artifact=pretrained_artifact,
            dataset_cache_path=dataset_cache_path,
        )
        run_environment = environment_metadata()
        run_environment.update(
            {
                "actual_device": config.runtime.device,
                "activation_dtype": config.runtime.activation_dtype,
                "checkpoint_compute_dtype": reduction_compute_dtype_name(),
                "batch_size": config.evaluation.batch_size,
            }
        )
        record = build_run_record(
            run_id=spec.run_id,
            status="success",
            config={
                **spec.as_config(),
                **config_provenance,
                "experiment_name": experiment.get("name", "unnamed"),
                "stage": experiment.get("stage", "tiny_reference"),
                "evaluation_scope": experiment.get(
                    "evaluation_scope", "language_model_accuracy"
                ),
                "evaluation_stage": experiment.get(
                    "evaluation_stage", "unspecified"
                ),
                "effective_sample_count": int(
                    experiment.get("effective_sample_count", payload["sample_count"])
                ),
                **(
                    {"evidence": experiment["evidence"]}
                    if experiment.get("evidence") is not None
                    else {}
                ),
                "expected_run_ids": list(
                    experiment.get("expected_run_ids", [spec.run_id])
                ),
                "selected_spec_ids_file": experiment.get(
                    "selected_spec_ids_file"
                ),
                "runtime": asdict(config.runtime),
                "evaluation": asdict(config.evaluation),
                # Layer/Softmax diagnostics are intentionally captured through
                # the explicitly gated debug API for research metrics.  The
                # timed performance path below still calls production forward.
                "evaluation_debug_capture": spec.mode != "plaintext",
            },
            seed=seed,
            model=payload["model"],
            dataset=payload["dataset"],
            environment=run_environment,
            sample_count=payload["sample_count"],
            metrics=payload["metrics"],
            elapsed_seconds=elapsed,
        )
        if spec.mode == "exact":
            passed, reasons = evaluate_exact_gate(record)
            record["metrics"]["exact_gate"] = {
                "passed": passed,
                "reasons": reasons,
            }
            validate_run_record(record)
    except Exception as error:
        failure_environment = environment_metadata()
        planned_device = None
        planned_dtype = None
        if config is not None:
            planned_device = config.runtime.device
            planned_dtype = config.runtime.activation_dtype
        else:
            try:
                fallback_runtime = load_config(base_config_path).runtime
                planned_device = str(
                    (runtime_override or {}).get(
                        "device", fallback_runtime.device
                    )
                )
                planned_dtype = str(
                    (runtime_override or {}).get(
                        "activation_dtype", fallback_runtime.activation_dtype
                    )
                )
            except Exception:
                pass
        if planned_device is not None and planned_dtype is not None:
            failure_environment.update(
                {
                    "requested_device": planned_device,
                    "requested_activation_dtype": planned_dtype,
                    "device": planned_device,
                    "activation_dtype": planned_dtype,
                    "checkpoint_compute_dtype": reduction_compute_dtype_name(),
                }
            )
        requested_pretrained = bool(
            pretrained_artifact is not None
            or pretrained_path is not None
            or experiment.get("pretrained", False)
        )
        failure_model: Dict[str, Any] = {
            "identifier": experiment.get("model_id", "unknown"),
            "evidence_scope": (
                experiment.get(
                    "evaluation_scope", "language_model_accuracy"
                )
                if requested_pretrained
                else "correctness_only_random_tiny"
            ),
            "pretrained": requested_pretrained,
        }
        if pretrained_path is not None:
            failure_model["path"] = str(
                Path(pretrained_path).expanduser().resolve()
            )
        if pretrained_artifact is not None:
            failure_model["checkpoint_manifest"] = pretrained_artifact.to_dict()
            failure_model.update(
                {
                    "weight_sha256": pretrained_artifact.weights_sha256,
                    "config_sha256": pretrained_artifact.config_sha256,
                    "tokenizer_sha256": pretrained_artifact.tokenizer_sha256,
                    "tokenizer_config_sha256": pretrained_artifact.tokenizer_config_sha256,
                }
            )
        failure_dataset: Dict[str, Any] = {
            "identifier": experiment.get("dataset_id", "unknown"),
            "evaluation_scope": experiment.get(
                "evaluation_scope", "language_model_accuracy"
            ),
            "meaningful_lm_evidence": bool(
                experiment.get("meaningful_lm_evidence", False)
            ),
        }
        if dataset_cache_path is not None:
            failure_dataset["cache_path"] = str(
                Path(dataset_cache_path).expanduser().resolve()
            )
        if experiment.get("dataset_cache_sha256") is not None:
            failure_dataset["cache_content_sha256"] = str(
                experiment["dataset_cache_sha256"]
            )
        record = build_run_record(
            run_id=spec.run_id,
            status="failure",
            config={
                **spec.as_config(),
                **config_provenance,
                "experiment_name": experiment.get("name", "unnamed"),
                "stage": experiment.get("stage", "tiny_reference"),
                "evaluation_scope": experiment.get(
                    "evaluation_scope", "language_model_accuracy"
                ),
                "evaluation_stage": experiment.get(
                    "evaluation_stage", "unspecified"
                ),
                "effective_sample_count": int(
                    experiment.get("effective_sample_count", 0)
                ),
                **(
                    {"evidence": experiment["evidence"]}
                    if experiment.get("evidence") is not None
                    else {}
                ),
                "expected_run_ids": list(
                    experiment.get("expected_run_ids", [spec.run_id])
                ),
                "selected_spec_ids_file": experiment.get(
                    "selected_spec_ids_file"
                ),
                "evaluation_override": evaluation_override or {},
                "runtime_override": runtime_override or {},
            },
            seed=seed,
            model=failure_model,
            dataset=failure_dataset,
            environment=failure_environment,
            sample_count=0,
            metrics={},
            elapsed_seconds=time.perf_counter() - started,
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(
                    traceback.format_exception_only(type(error), error)
                ).strip(),
            },
            stage="tiny_model_execution",
            last_completed_sample_id=None,
            partial_metrics_available=False,
        )
    if output_path is not None:
        append_jsonl_record(output_path, record)
    _release_runtime_memory()
    return record
