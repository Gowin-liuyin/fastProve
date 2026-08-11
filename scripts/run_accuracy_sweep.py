#!/usr/bin/env python3
"""Plan or explicitly execute the complete 72-point aligned sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastprove.structured import reduction_compute_dtype_name
from fastprove.config import load_config  # noqa: E402
from fastprove.evaluation.artifacts import (  # noqa: E402
    append_jsonl_record,
    build_run_record,
    require_fresh_jsonl_destination,
    validate_run_record,
)
from fastprove.evaluation.sweep import (  # noqa: E402
    SweepSpec,
    enumerate_sweep_specs,
)
from fastprove.evaluation.pretrained import (  # noqa: E402
    token_cache_evidence_scope,
    token_cache_provenance_issues,
)
from fastprove.pretrained.qwen2 import Qwen2Artifact  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "List all sweep points. No model inference runs unless "
            "--execute-deferred is supplied."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval_sweep.yaml"),
        help="Sweep YAML containing the required tau/alpha/Top-k grid.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Raw JSONL destination; defaults to experiment.output_jsonl.",
    )
    parser.add_argument(
        "--execute-deferred",
        action="store_true",
        help=(
            "Explicitly authorize every listed tiny sweep run. This overrides "
            "execution.run_in_current_prototype_pass=false."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local Qwen2 artifact for meaningful LM evaluation",
    )
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        default=None,
        help="Validated token cache shared by every sweep point",
    )
    parser.add_argument(
        "--preflight-output",
        type=Path,
        default=None,
        help="Optional preflight manifest written before execution",
    )
    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="Run every point on calibration_sample_count instead of full_sample_count",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Explicit sample count override (recorded in every run)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Explicit inference batch-size override (use 1 for tight memory)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=None,
        help="Explicit performance warmup count override",
    )
    parser.add_argument(
        "--timed-runs",
        type=int,
        default=None,
        help="Explicit performance timing repetition override",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=None,
        help="Explicit paired sample-bootstrap replicate count override",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default=None,
        help="Explicit device override; unavailable accelerators fail without fallback",
    )
    parser.add_argument(
        "--activation-dtype",
        choices=("fp32", "bf16"),
        default=None,
        help="Explicit activation dtype override (fp32 or bf16)",
    )
    parser.add_argument(
        "--accept-nonstandard-caption",
        action="store_true",
        help=(
            "Explicitly accept a local Flickr/caption cache as a limited, "
            "non-standard LM evaluation scope."
        ),
    )
    parser.add_argument(
        "--spec-ids-file",
        type=Path,
        default=None,
        help=(
            "Optional candidate manifest JSON (or newline-delimited IDs) that "
            "selects a validated subset for full evaluation"
        ),
    )
    return parser


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sweep(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("sweep config must contain a mapping")
    for required in ("base_config", "experiment", "sweep", "execution"):
        if required not in raw:
            raise ValueError(f"sweep config is missing {required}")
    if not isinstance(raw["experiment"], dict):
        raise ValueError("experiment must be a mapping")
    if not isinstance(raw.get("evaluation_override", {}), dict):
        raise ValueError("evaluation_override must be a mapping")
    return raw


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _spec_payload(spec: SweepSpec) -> dict:
    return {"run_id": spec.run_id, **spec.as_config()}


def _order_specs_for_exact_gate(specs: Sequence[SweepSpec]) -> list[SweepSpec]:
    """Put the exact-gate baselines before every approximate point.

    The default sweep enumerator already emits this order, but a selected
    candidate manifest may have been produced from an out-of-order or
    parallel JSONL file.  Approximate records must never be skipped merely
    because their entry appeared before the exact baseline; they are ordered
    deterministically here and retain their relative order within each mode.
    """

    priority = {
        "plaintext": 0,
        "exact": 1,
        "topk_preserving": 2,
        "free_bounded": 3,
    }
    return [
        spec
        for _, spec in sorted(
            enumerate(specs),
            key=lambda item: (priority.get(item[1].mode, 99), item[0]),
        )
    ]


def _config_provenance(experiment: Dict[str, Any]) -> Dict[str, Any]:
    """Return configuration paths/hashes without inserting null placeholders."""

    return {
        key: experiment[key]
        for key in (
            "base_config_path",
            "base_config_sha256",
            "sweep_config_path",
            "sweep_config_sha256",
        )
        if experiment.get(key) is not None
    }


def _planned_runtime_metadata(
    environment: Dict[str, Any],
    *,
    base_config: Any,
    runtime_override: Dict[str, Any],
) -> Dict[str, Any]:
    """Annotate preflight/skip records with the requested runtime contract."""

    requested_device = str(
        runtime_override.get("device", base_config.runtime.device)
    )
    requested_dtype = str(
        runtime_override.get(
            "activation_dtype", base_config.runtime.activation_dtype
        )
    )
    enriched = dict(environment)
    enriched.update(
        {
            "requested_device": requested_device,
            "requested_activation_dtype": requested_dtype,
            # These are requested/planned fields until a successful runner
            # record replaces them with actual_device/activation_dtype.
            "device": requested_device,
            "activation_dtype": requested_dtype,
            # Derived, never hard-coded: task A4 made this FP32 on every
            # device and Stage B removed the FP64 checkpoint arithmetic.
            "checkpoint_compute_dtype": reduction_compute_dtype_name(),
        }
    )
    return enriched


def _load_selected_run_ids(path: Path) -> list[str]:
    """Load candidate IDs and require a ready selection manifest when present."""

    source = _resolve(path)
    if not source.is_file():
        raise FileNotFoundError("spec IDs file does not exist: %s" % source)
    text = source.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        ids = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        if isinstance(value, dict):
            if value.get("status") != "ready_for_full_evaluation":
                raise ValueError(
                    "candidate manifest status must be ready_for_full_evaluation"
                )
            ids = value.get("selected_run_ids")
        else:
            ids = value
    if not isinstance(ids, list) or not ids or any(
        not isinstance(item, str) or not item.strip() for item in ids
    ):
        raise ValueError("spec IDs file must contain a non-empty unique string list")
    normalized = [item.strip() for item in ids]
    if len(set(normalized)) != len(normalized):
        raise ValueError("spec IDs file must contain unique run IDs")
    return normalized


def _failure_record(
    *,
    spec: SweepSpec,
    experiment: Dict[str, Any],
    environment: Dict[str, Any],
    error: Exception,
    runtime_override: Optional[Dict[str, Any]] = None,
    record_stage: str = "tiny_sweep_point",
    pretrained_artifact: Optional[Qwen2Artifact] = None,
) -> dict:
    evaluation_scope = str(
        experiment.get("evaluation_scope", "language_model_accuracy")
    )
    return build_run_record(
        run_id=spec.run_id,
        status="failure",
        config={
            **spec.as_config(),
            **_config_provenance(experiment),
            "experiment_name": experiment.get("name", "unnamed"),
            "stage": experiment.get("stage", "tiny_reference"),
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
            "runtime_override": runtime_override or {},
            "selected_spec_ids_file": experiment.get("selected_spec_ids_file"),
            "evaluation_debug_capture": spec.mode != "plaintext",
        },
        seed=int(experiment["seed"]),
        model={
            "id": str(experiment["model_id"]),
            "revision": str(experiment.get("source_revision", "unavailable")),
            "pretrained": bool(experiment.get("pretrained", False)),
            "evidence_scope": (
                evaluation_scope
                if experiment.get("pretrained", False)
                else "correctness_only_random_tiny"
            ),
            **(
                {"path": str(experiment["model_path"])}
                if experiment.get("model_path") is not None
                else {}
            ),
            **(
                {
                    "checkpoint_manifest": pretrained_artifact.to_dict(),
                    "weight_sha256": pretrained_artifact.weights_sha256,
                    "config_sha256": pretrained_artifact.config_sha256,
                    "tokenizer_sha256": pretrained_artifact.tokenizer_sha256,
                    "tokenizer_config_sha256": pretrained_artifact.tokenizer_config_sha256,
                }
                if pretrained_artifact is not None
                else {}
            ),
        },
        dataset={
            "id": str(experiment["dataset_id"]),
            "evaluation_scope": evaluation_scope,
            "meaningful_lm_evidence": bool(
                experiment.get("meaningful_lm_evidence", False)
            ),
            **(
                {"path": str(experiment["dataset_cache_path"])}
                if experiment.get("dataset_cache_path") is not None
                else {}
            ),
            **(
                {"cache_content_sha256": str(experiment["dataset_cache_sha256"])}
                if experiment.get("dataset_cache_sha256") is not None
                else {}
            ),
        },
        environment=environment,
        sample_count=0,
        metrics={},
        elapsed_seconds=0.0,
        error={
            "type": type(error).__name__,
            "message": str(error),
        },
        stage=record_stage,
        last_completed_sample_id=None,
        partial_metrics_available=False,
    )


def _exact_gate_skip_record(
    *,
    spec: SweepSpec,
    experiment: Dict[str, Any],
    environment: Dict[str, Any],
    reasons: Sequence[str],
    runtime_override: Optional[Dict[str, Any]] = None,
    pretrained_artifact: Optional[Qwen2Artifact] = None,
) -> dict:
    evaluation_scope = str(
        experiment.get("evaluation_scope", "language_model_accuracy")
    )
    return build_run_record(
        run_id=spec.run_id,
        status="skipped",
        config={
            **spec.as_config(),
            **_config_provenance(experiment),
            "blocked_by": "mandatory_exact_gate",
            "experiment_name": experiment.get("name", "unnamed"),
            "stage": experiment.get("stage", "tiny_reference"),
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
            "runtime_override": runtime_override or {},
            "selected_spec_ids_file": experiment.get("selected_spec_ids_file"),
            "evaluation_debug_capture": False,
        },
        seed=int(experiment["seed"]),
        model={
            "id": str(experiment["model_id"]),
            "revision": str(experiment.get("source_revision", "unavailable")),
            "pretrained": bool(experiment.get("pretrained", False)),
            "evidence_scope": (
                evaluation_scope
                if experiment.get("pretrained", False)
                else "correctness_only_random_tiny"
            ),
            **(
                {"path": str(experiment["model_path"])}
                if experiment.get("model_path") is not None
                else {}
            ),
            **(
                {
                    "checkpoint_manifest": pretrained_artifact.to_dict(),
                    "weight_sha256": pretrained_artifact.weights_sha256,
                    "config_sha256": pretrained_artifact.config_sha256,
                    "tokenizer_sha256": pretrained_artifact.tokenizer_sha256,
                    "tokenizer_config_sha256": pretrained_artifact.tokenizer_config_sha256,
                }
                if pretrained_artifact is not None
                else {}
            ),
        },
        dataset={
            "id": str(experiment["dataset_id"]),
            "evaluation_scope": evaluation_scope,
            "meaningful_lm_evidence": bool(
                experiment.get("meaningful_lm_evidence", False)
            ),
            **(
                {"path": str(experiment["dataset_cache_path"])}
                if experiment.get("dataset_cache_path") is not None
                else {}
            ),
            **(
                {"cache_content_sha256": str(experiment["dataset_cache_sha256"])}
                if experiment.get("dataset_cache_sha256") is not None
                else {}
            ),
        },
        environment=environment,
        sample_count=0,
        metrics={},
        elapsed_seconds=0.0,
        reason={
            "code": "exact_gate_blocked",
            "message": "; ".join(reasons),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (args.model_path is None) != (args.dataset_cache is None):
        print("--model-path and --dataset-cache must be supplied together", file=sys.stderr)
        return 2
    config_path = _resolve(args.config)
    raw = _load_sweep(config_path)
    all_specs = enumerate_sweep_specs(config_path)
    specs = list(all_specs)
    selected_spec_ids_file: Optional[Path] = None
    if args.spec_ids_file is not None:
        try:
            selected_ids = _load_selected_run_ids(args.spec_ids_file)
            by_run_id = {spec.run_id: spec for spec in all_specs}
            unknown = sorted(set(selected_ids).difference(by_run_id))
            if unknown:
                raise ValueError(
                    "spec IDs file contains unknown run IDs: "
                    + ", ".join(unknown)
                )
            specs = [by_run_id[run_id] for run_id in selected_ids]
            selected_modes = {spec.mode for spec in specs}
            if not {"plaintext", "exact"}.issubset(selected_modes):
                raise ValueError(
                    "selected specs must include plaintext and exact baselines"
                )
            if not selected_modes.intersection(
                {"topk_preserving", "free_bounded"}
            ):
                raise ValueError(
                    "selected specs must include at least one approximate mode"
                )
            selected_spec_ids_file = _resolve(args.spec_ids_file)
        except (FileNotFoundError, ValueError) as error:
            print(
                json.dumps(
                    {
                        "action": "spec_selection_preflight",
                        "status": "failure",
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 2
    # The mandatory exact gate is stateful across the sweep loop.  Normalize
    # any user/manifest ordering before constructing expected_run_ids so that
    # approximate points cannot be skipped before the exact baseline runs.
    specs = _order_specs_for_exact_gate(specs)
    base_config_path = _project_path(raw["base_config"])
    base_config = load_config(base_config_path)
    experiment_section = raw["experiment"]
    configured_calibration = int(experiment_section.get("calibration_sample_count", 8))
    configured_full = int(experiment_section.get("full_sample_count", base_config.evaluation.sample_count))
    if args.sample_count is not None and args.sample_count < 1:
        print("--sample-count must be positive", file=sys.stderr)
        return 2
    if args.batch_size is not None and args.batch_size < 1:
        print("--batch-size must be positive", file=sys.stderr)
        return 2
    if args.warmup_runs is not None and args.warmup_runs < 0:
        print("--warmup-runs must be non-negative", file=sys.stderr)
        return 2
    if args.timed_runs is not None and args.timed_runs < 1:
        print("--timed-runs must be positive", file=sys.stderr)
        return 2
    if args.bootstrap_replicates is not None and args.bootstrap_replicates < 0:
        print("--bootstrap-replicates must be non-negative", file=sys.stderr)
        return 2
    effective_sample_count = (
        int(args.sample_count)
        if args.sample_count is not None
        else (configured_calibration if args.calibration_only else configured_full)
    )
    if effective_sample_count < 1:
        print("effective sample count must be positive", file=sys.stderr)
        return 2
    effective_evaluation_override = dict(raw.get("evaluation_override") or {})
    effective_evaluation_override["sample_count"] = effective_sample_count
    if args.batch_size is not None:
        effective_evaluation_override["batch_size"] = int(args.batch_size)
    if args.warmup_runs is not None:
        effective_evaluation_override["warmup_runs"] = int(args.warmup_runs)
    if args.timed_runs is not None:
        effective_evaluation_override["timed_runs"] = int(args.timed_runs)
    if args.bootstrap_replicates is not None:
        effective_evaluation_override["bootstrap_replicates"] = int(
            args.bootstrap_replicates
        )
    runtime_override: Dict[str, Any] = {}
    if args.device is not None:
        runtime_override["device"] = args.device
    if args.activation_dtype is not None:
        runtime_override["activation_dtype"] = {
            "fp32": "float32",
            "bf16": "bfloat16",
        }[args.activation_dtype]
    experiment = {
        **raw["experiment"],
        "expected_run_ids": [spec.run_id for spec in specs],
        "evaluation_stage": "calibration" if args.calibration_only else "full",
        "effective_sample_count": effective_sample_count,
        "sweep_config_path": str(config_path),
        "sweep_config_sha256": _sha256_file(config_path),
        "base_config_path": str(base_config_path),
        "base_config_sha256": _sha256_file(base_config_path),
    }
    if selected_spec_ids_file is not None:
        experiment["selected_spec_ids_file"] = str(selected_spec_ids_file)
    if args.model_path is not None:
        experiment["model_path"] = str(_resolve(args.model_path))
        experiment["dataset_cache_path"] = str(_resolve(args.dataset_cache))
    output_path = (
        _resolve(args.output)
        if args.output is not None
        else _project_path(experiment["output_jsonl"])
    )
    spec_payloads = [_spec_payload(spec) for spec in specs]
    planned_command = "python3 scripts/run_accuracy_sweep.py"
    planned_command += f" --config {shlex.quote(str(config_path))}"
    planned_command += f" --output {shlex.quote(str(output_path))}"
    if args.model_path is not None:
        planned_command += (
            f" --model-path {shlex.quote(str(_resolve(args.model_path)))}"
            f" --dataset-cache {shlex.quote(str(_resolve(args.dataset_cache)))}"
        )
    if args.preflight_output is not None:
        planned_command += f" --preflight-output {shlex.quote(str(_resolve(args.preflight_output)))}"
    if args.calibration_only:
        planned_command += " --calibration-only"
    if args.sample_count is not None:
        planned_command += f" --sample-count {int(args.sample_count)}"
    if args.batch_size is not None:
        planned_command += f" --batch-size {int(args.batch_size)}"
    if args.warmup_runs is not None:
        planned_command += f" --warmup-runs {int(args.warmup_runs)}"
    if args.timed_runs is not None:
        planned_command += f" --timed-runs {int(args.timed_runs)}"
    if args.bootstrap_replicates is not None:
        planned_command += (
            f" --bootstrap-replicates {int(args.bootstrap_replicates)}"
        )
    if args.device is not None:
        planned_command += f" --device {shlex.quote(args.device)}"
    if args.activation_dtype is not None:
        planned_command += f" --activation-dtype {shlex.quote(args.activation_dtype)}"
    if selected_spec_ids_file is not None:
        planned_command += f" --spec-ids-file {shlex.quote(str(selected_spec_ids_file))}"
    if args.accept_nonstandard_caption:
        planned_command += " --accept-nonstandard-caption"
    planned_command += " --execute-deferred"

    if not args.execute_deferred:
        print(
            json.dumps(
                {
                    "action": "plan",
                    "execution_authorized": False,
                    "deferred_by_config": not bool(
                        raw["execution"].get(
                            "run_in_current_prototype_pass", False
                        )
                    ),
                    "config": str(config_path),
                    "config_sha256": _sha256_file(config_path),
                    "base_config": str(base_config_path),
                    "base_config_sha256": _sha256_file(base_config_path),
                    "output": str(output_path),
                    "model_path": str(_resolve(args.model_path)) if args.model_path else None,
                    "dataset_cache": str(_resolve(args.dataset_cache)) if args.dataset_cache else None,
                    "evaluation_stage": experiment["evaluation_stage"],
                    "sample_count": effective_sample_count,
                    "batch_size": args.batch_size or base_config.evaluation.batch_size,
                    "warmup_runs": effective_evaluation_override.get(
                        "warmup_runs", base_config.evaluation.warmup_runs
                    ),
                    "timed_runs": effective_evaluation_override.get(
                        "timed_runs", base_config.evaluation.timed_runs
                    ),
                    "bootstrap_replicates": effective_evaluation_override.get(
                        "bootstrap_replicates",
                        base_config.evaluation.bootstrap_replicates,
                    ),
                    "device": args.device or base_config.runtime.device,
                    "activation_dtype": args.activation_dtype
                    or ("bf16" if base_config.runtime.activation_dtype == "bfloat16" else "fp32"),
                    "all_spec_count": len(all_specs),
                    "spec_count": len(specs),
                    "spec_ids_file": (
                        str(selected_spec_ids_file)
                        if selected_spec_ids_file is not None
                        else None
                    ),
                    "specs": spec_payloads,
                    "evidence_scope": (
                        (
                            "caption_only_nonstandard_lm_candidate"
                            if args.accept_nonstandard_caption
                            else "standard_causal_lm_evidence_required"
                        )
                        if args.model_path is not None
                        else "random_tiny_correctness_only_not_meaningful_lm_accuracy"
                    ),
                    "next_command": planned_command,
                },
                indent=2,
            )
        )
        return 0

    try:
        require_fresh_jsonl_destination(output_path)
    except FileExistsError as error:
        print(
            json.dumps(
                {
                    "action": "raw_output_preflight",
                    "execution_authorized": False,
                    "status": "refused",
                    "output": str(output_path),
                    "error": str(error),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    try:
        from fastprove.evaluation.runner import (
            environment_metadata,
            evaluate_exact_gate,
            run_tiny_sweep_spec,
        )

        environment = _planned_runtime_metadata(
            environment_metadata(),
            base_config=base_config,
            runtime_override=runtime_override,
        )
        runner_import_error: Exception | None = None
    except Exception as error:
        environment = {
            "status": "runner_unavailable",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        runner_import_error = error
        evaluate_exact_gate = None  # type: ignore[assignment]
        run_tiny_sweep_spec = None  # type: ignore[assignment]

    pretrained_artifact = None
    if args.model_path is not None:
        try:
            from fastprove.pretrained.qwen2 import load_qwen2_artifact
            from fastprove.evaluation.token_cache import load_token_cache

            evaluation_override = effective_evaluation_override
            eval_seq = int(evaluation_override.get("sequence_length", base_config.evaluation.sequence_length))
            eval_gen = int(evaluation_override.get("generation_tokens", base_config.evaluation.generation_tokens))
            pretrained_artifact = load_qwen2_artifact(
                _resolve(args.model_path),
                max_sequence_length=eval_seq + eval_gen,
                compute_hashes=True,
            )
            cache = load_token_cache(
                _resolve(args.dataset_cache),
                expected_vocab_size=pretrained_artifact.model_config.vocab_size,
                expected_sequence_length=eval_seq,
            )
            provenance_issues = token_cache_provenance_issues(
                cache, pretrained_artifact
            )
            if provenance_issues:
                raise ValueError(
                    "token cache provenance is incomplete: "
                    + "; ".join(provenance_issues)
                )
            required_samples = effective_sample_count
            if cache.sample_count < required_samples:
                raise ValueError("token cache has fewer samples than full_sample_count")
            evaluation_scope = token_cache_evidence_scope(cache)
            if (
                evaluation_scope == "caption_only_nonstandard_lm_candidate"
                and not args.accept_nonstandard_caption
            ):
                raise ValueError(
                    "token cache is a non-standard caption-only candidate; "
                    "rerun with --accept-nonstandard-caption after explicitly "
                    "choosing the limited evaluation scope"
                )
            experiment.update(
                {
                    "model_id": "local-qwen2",
                    "dataset_id": "validated-local-token-cache",
                    "source_revision": pretrained_artifact.upstream_revision,
                    "pretrained": True,
                    # A caption-only cache is useful for a limited comparison,
                    # but it is not a standard causal-LM benchmark.  Keep its
                    # raw records valid while preventing report aggregation
                    # from presenting them as language-model evidence.
                    "meaningful_lm_evidence": (
                        evaluation_scope == "language_model_accuracy"
                    ),
                    "stage": "pretrained_accuracy",
                    "evidence": (
                        "language_model_accuracy"
                        if evaluation_scope == "language_model_accuracy"
                        else "comparative_evaluation"
                    ),
                    "evaluation_scope": evaluation_scope,
                    "dataset_cache_sha256": cache.content_sha256,
                }
            )
            if args.preflight_output is not None:
                manifest = {
                    "status": "ready_for_sweep",
                    "model": pretrained_artifact.to_dict(),
                    "runtime": {
                        "requested_device": args.device
                        or base_config.runtime.device,
                        "requested_activation_dtype": args.activation_dtype
                        or (
                            "bf16"
                            if base_config.runtime.activation_dtype == "bfloat16"
                            else "fp32"
                        ),
                    },
                    "dataset": {
                        "path": str(_resolve(args.dataset_cache)),
                        "content_sha256": cache.content_sha256,
                        "sample_count": cache.sample_count,
                        "sequence_length": cache.sequence_length,
                        "evaluation_scope": evaluation_scope,
                        "metadata": cache.metadata,
                    },
                    "required_samples": required_samples,
                }
                preflight_path = _resolve(args.preflight_output)
                if preflight_path.exists():
                    raise FileExistsError("refusing to overwrite preflight output: %s" % preflight_path)
                preflight_path.parent.mkdir(parents=True, exist_ok=True)
                preflight_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
        except Exception as error:
            # Preserve one explicit failure record per configured point.  A
            # rejected model/cache must not leave an apparently empty attempt
            # that cannot be reconciled against the 72-point manifest.
            experiment.update(
                {
                    "model_id": "local-qwen2",
                    "dataset_id": "validated-local-token-cache",
                    "source_revision": (
                        pretrained_artifact.upstream_revision
                        if pretrained_artifact is not None
                        else "preflight_unavailable"
                    ),
                    "pretrained": True,
                    "meaningful_lm_evidence": False,
                    "stage": "pretrained_accuracy",
                    "evaluation_scope": "standard_causal_lm_evidence_required",
                    "evidence": "comparative_evaluation",
                }
            )
            preflight_status_counts: Dict[str, int] = {}
            for spec in specs:
                record = _failure_record(
                    spec=spec,
                    experiment=experiment,
                    environment=environment,
                    error=error,
                    runtime_override=runtime_override,
                    record_stage="pretrained_preflight",
                    pretrained_artifact=pretrained_artifact,
                )
                append_jsonl_record(output_path, record)
                status = str(record["status"])
                preflight_status_counts[status] = (
                    preflight_status_counts.get(status, 0) + 1
                )
            print(
                json.dumps(
                    {
                        "action": "pretrained_preflight",
                        "status": "failure",
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                        "output": str(output_path),
                        "failure_records": len(specs),
                        "status_counts": preflight_status_counts,
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 2

    status_counts: Dict[str, int] = {}
    exact_gate_passed = False
    exact_gate_reasons = ["exact gate has not run"]
    for spec in specs:
        if spec.mode in ("topk_preserving", "free_bounded") and (
            not exact_gate_passed
        ):
            record = _exact_gate_skip_record(
                spec=spec,
                experiment=experiment,
                environment=environment,
                reasons=exact_gate_reasons,
                runtime_override=runtime_override,
                pretrained_artifact=pretrained_artifact,
            )
            append_jsonl_record(output_path, record)
            status = str(record["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
            continue
        try:
            if runner_import_error is not None:
                raise RuntimeError(
                    "tiny sweep runner is unavailable"
                ) from runner_import_error
            assert run_tiny_sweep_spec is not None
            record = run_tiny_sweep_spec(
                base_config_path=base_config_path,
                spec=spec,
                experiment=experiment,
                evaluation_override=effective_evaluation_override,
                runtime_override=runtime_override,
                output_path=None,
                pretrained_path=_resolve(args.model_path) if args.model_path else None,
                pretrained_artifact=pretrained_artifact,
                dataset_cache_path=_resolve(args.dataset_cache) if args.dataset_cache else None,
            )
            validate_run_record(record)
            if spec.mode == "exact":
                assert evaluate_exact_gate is not None
                exact_gate_passed, exact_gate_reasons = (
                    evaluate_exact_gate(record)
                )
        except Exception as error:
            record = _failure_record(
                spec=spec,
                experiment=experiment,
                environment=environment,
                error=error,
                runtime_override=runtime_override,
                pretrained_artifact=pretrained_artifact,
            )
            if spec.mode == "exact":
                exact_gate_passed = False
                exact_gate_reasons = [
                    "exact gate execution failed: %s" % str(error)
                ]
        if record.get("status") != "success":
            # Failures/skips produced inside the runner may have only generic
            # environment metadata. Preserve the requested device/dtype and
            # checkpoint arithmetic contract even when no successful forward
            # was available to report actual fields.
            record["environment"] = _planned_runtime_metadata(
                record.get("environment", {}),
                base_config=base_config,
                runtime_override=runtime_override,
            )
            validate_run_record(record)
        append_jsonl_record(output_path, record)
        status = str(record["status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    print(
        json.dumps(
            {
                "action": "run_sweep",
                "execution_authorized": True,
                "output": str(output_path),
                "all_spec_count": len(all_specs),
                "spec_count": len(specs),
                "spec_ids_file": (
                    str(selected_spec_ids_file)
                    if selected_spec_ids_file is not None
                    else None
                ),
                "status_counts": status_counts,
                "exact_gate_passed": exact_gate_passed,
                "exact_gate_reasons": exact_gate_reasons,
            },
            indent=2,
        )
    )
    return 0 if status_counts == {"success": len(specs)} else 1


if __name__ == "__main__":
    raise SystemExit(main())
