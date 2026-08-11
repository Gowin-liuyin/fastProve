"""Local pretrained-asset audit and external-run proposal.

This module deliberately performs *discovery and validation only*.  It never
downloads a checkpoint or dataset and never allocates model weights.  The
resulting status document makes the difference between a complete evaluation
input set and a merely present-but-unsuitable local file explicit.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..pretrained.qwen2 import Qwen2Artifact, load_qwen2_artifact
from .token_cache import TokenCache, load_token_cache


PRETRAINED_STATUS_VERSION = "fastprove.pretrained_status.v1"
DEFAULT_SEARCH_ROOTS = (
    Path.home() / ".cache" / "huggingface" / "hub",
    Path.home() / ".cache" / "huggingface" / "datasets",
    Path.home() / "dr-claw" / "FL_unlearning" / "Experiment" / "datasets",
    Path.home() / "code" / "fastProve" / "data",
    Path.home() / "code" / "fastProve" / "datasets",
)
DEFAULT_QWEN_RELATIVE_PATH = Path(
    "dr-claw"
) / "基于协变混淆的边云协同推理" / "model_artifacts" / "raw_remote_pull" / "models" / "deepseek-r1-distill-qwen-1.5b" / "base_model"
SUPPORTED_DATA_SUFFIXES = frozenset(
    {".jsonl", ".json", ".txt", ".parquet", ".arrow", ".pt", ".bin"}
)


def token_cache_evidence_scope(cache: TokenCache) -> str:
    """Classify cache provenance without treating captions as a benchmark."""

    source_hint = (
        "%s %s"
        % (
            str(cache.metadata.get("source_path", "")),
            str(cache.metadata.get("evidence_scope", "")),
        )
    ).lower()
    if any(marker in source_hint for marker in ("flickr", "caption")):
        return "caption_only_nonstandard_lm_candidate"
    return "language_model_accuracy"


def token_cache_provenance_issues(
    cache: TokenCache, artifact: Qwen2Artifact
) -> List[str]:
    """Return missing/mismatched identity fields for a real-model cache."""

    metadata = cache.metadata
    issues: List[str] = []
    expected_fields = {
        "model_config_sha256": artifact.config_sha256,
        "tokenizer_sha256": artifact.tokenizer_sha256,
        "tokenizer_config_sha256": artifact.tokenizer_config_sha256,
    }
    for field, expected in expected_fields.items():
        actual = metadata.get(field)
        if actual != expected:
            issues.append(
                "%s missing or mismatched (expected %s)" % (field, expected)
            )
    try:
        actual_vocab_size = int(metadata.get("vocab_size", -1))
    except (TypeError, ValueError):
        actual_vocab_size = -1
    if actual_vocab_size != artifact.model_config.vocab_size:
        issues.append("vocab_size missing or mismatched")
    for field in ("source_path", "source_sha256", "source_format"):
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            issues.append("%s missing" % field)
    return issues


def default_local_qwen_path() -> Path:
    """Return the known local Qwen candidate without asserting it exists."""

    return (Path.home() / DEFAULT_QWEN_RELATIVE_PATH).expanduser().resolve()


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _safe_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _walk_candidate_files(
    roots: Sequence[Path], *, max_depth: int = 4, max_candidates: int = 80
) -> List[Dict[str, Any]]:
    """List likely local data candidates without reading their contents."""

    candidates: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            continue
        for current, directories, files in os.walk(root):
            current_path = Path(current)
            try:
                depth = len(current_path.relative_to(root).parts)
            except ValueError:
                depth = max_depth
            if depth >= max_depth:
                directories[:] = []
            for name in sorted(files):
                path = (current_path / name).resolve()
                if path.suffix.lower() not in SUPPORTED_DATA_SUFFIXES:
                    continue
                # Hugging Face's ``.no_exist`` markers and model/tokenizer
                # metadata are cache bookkeeping, not evaluation corpora.
                if ".no_exist" in path.parts or name in {
                    "version.txt",
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "dataset_info.json",
                    "dataset_infos.json",
                    "model.safetensors.index.json",
                }:
                    continue
                key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "path": key,
                        "suffix": path.suffix.lower(),
                        "kind": "candidate_local_data",
                    }
                )
                if len(candidates) >= max_candidates:
                    return candidates
    return candidates


def _root_inventory(roots: Sequence[Path]) -> List[Dict[str, Any]]:
    return [
        {
            "path": _safe_path(root),
            "exists": Path(root).expanduser().exists(),
            "is_directory": Path(root).expanduser().is_dir(),
        }
        for root in roots
    ]


def _model_entry(
    model_path: Optional[str | Path],
) -> tuple[Dict[str, Any], Optional[Qwen2Artifact]]:
    if model_path is None:
        return {
            "present": False,
            "path": None,
            "error": "no model path supplied",
        }, None
    path = Path(model_path).expanduser().resolve()
    entry: Dict[str, Any] = {"present": False, "path": str(path)}
    if not path.is_dir():
        entry["error"] = "model path is not a directory"
        return entry, None
    try:
        # This validates config, tokenizer files and Safetensors keys without
        # materializing the checkpoint tensors.
        artifact = load_qwen2_artifact(
            path, max_sequence_length=2048, compute_hashes=True
        )
    except Exception as error:
        entry["error"] = {"type": type(error).__name__, "message": str(error)}
        return entry, None
    entry.update(
        {
            "present": True,
            "architecture": artifact.architecture,
            "upstream_revision": artifact.upstream_revision,
            "checkpoint_manifest": artifact.to_dict(),
            "tokenizer_files": {
                "tokenizer_json": (path / "tokenizer.json").is_file(),
                "tokenizer_config_json": (path / "tokenizer_config.json").is_file(),
            },
        }
    )
    return entry, artifact


def _cache_entry(
    path: Path,
    *,
    artifact: Optional[Qwen2Artifact],
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"path": str(path), "present": path.is_file()}
    if not path.is_file():
        entry["error"] = "file does not exist"
        return entry
    entry["file_sha256"] = _sha256_file(path)
    try:
        cache: TokenCache = load_token_cache(
            path,
            expected_vocab_size=(
                artifact.model_config.vocab_size if artifact is not None else None
            ),
        )
    except Exception as error:
        entry["valid_aligned_token_cache"] = False
        entry["error"] = {"type": type(error).__name__, "message": str(error)}
        return entry
    entry.update(
        {
            "valid_aligned_token_cache": True,
            "content_sha256": cache.content_sha256,
            "sample_count": cache.sample_count,
            "sequence_length": cache.sequence_length,
            "metadata": cache.metadata,
        }
    )
    entry["evidence_scope"] = token_cache_evidence_scope(cache)
    if artifact is not None:
        provenance_issues = token_cache_provenance_issues(cache, artifact)
        entry["provenance_issues"] = provenance_issues
        entry["provenance_valid"] = not provenance_issues
    entry["nonstandard_caption_candidate"] = (
        entry["evidence_scope"] == "caption_only_nonstandard_lm_candidate"
    )
    if artifact is not None:
        entry["model_identity_match"] = cache.metadata.get(
            "model_config_sha256"
        ) in (None, artifact.config_sha256)
        entry["tokenizer_identity_match"] = cache.metadata.get(
            "tokenizer_sha256"
        ) in (None, artifact.tokenizer_sha256)
    return entry


def _proposed_external_run(
    *,
    model_entry: Dict[str, Any],
    calibration_count: int,
    full_count: int,
    sequence_length: int,
    generation_tokens: int,
) -> Dict[str, Any]:
    manifest = model_entry.get("checkpoint_manifest", {})
    checkpoint_bytes = int(manifest.get("weights_bytes", 0) or 0)
    # Keep this estimate aligned with preflight_experiment.py.  It is a plan,
    # not a reservation and not a measured peak.
    working_set_fp32 = int(checkpoint_bytes * 4.8)
    return {
        "model": {
            "identifier": "local-qwen2-1.5b",
            "path": model_entry.get("path"),
            "architecture": model_entry.get("architecture"),
        },
        "dataset": {
            "recommendation": "approved public WikiText-2 test split or an explicitly authorized local causal-LM corpus",
            "required_provenance": [
                "dataset name/config/split/revision",
                "source and license/authorization",
                "source SHA-256",
                "token-cache SHA-256",
            ],
        },
        "evaluation": {
            "calibration_sample_count": calibration_count,
            "full_sample_count": full_count,
            "sequence_length": sequence_length,
            "generation_tokens": generation_tokens,
            "dtype_order": ["float32_exact_gate_first", "bfloat16_separate"],
            "device_policy": "prefer CUDA with sufficient memory; never silently fallback",
            "sweep_points": 72,
        },
        "resource_estimate": {
            "checkpoint_bytes": checkpoint_bytes,
            "estimated_fp32_working_set_bytes": working_set_fp32,
            "estimated_fp32_working_set_gib": round(
                working_set_fp32 / (1024**3), 2
            ),
            "estimate_kind": "conservative_preflight_lower_bound_plus_conversion_overhead",
        },
        "permission_gate": "only the approved dataset/token-cache (and suitable compute if needed) remains external",
    }


def build_pretrained_status(
    *,
    model_path: Optional[str | Path] = None,
    dataset_paths: Sequence[str | Path] = (),
    search_roots: Sequence[str | Path] = DEFAULT_SEARCH_ROOTS,
    calibration_count: int = 8,
    full_count: int = 64,
    sequence_length: int = 24,
    generation_tokens: int = 4,
    accept_nonstandard_caption: bool = False,
) -> Dict[str, Any]:
    """Build a JSON-safe local asset status without network access."""

    if calibration_count < 1 or full_count < 1 or calibration_count > full_count:
        raise ValueError("calibration/full sample counts are invalid")
    if sequence_length < 2 or generation_tokens < 0:
        raise ValueError("sequence/generation lengths are invalid")
    roots = [Path(value).expanduser().resolve() for value in search_roots]
    resolved_model = (
        Path(model_path).expanduser().resolve()
        if model_path is not None
        else default_local_qwen_path()
    )
    model_entry, artifact = _model_entry(resolved_model)
    requested_data = [Path(value).expanduser().resolve() for value in dataset_paths]
    cache_entries = [
        _cache_entry(path, artifact=artifact)
        for path in requested_data
        if path.suffix.lower() == ".pt"
    ]
    discovered = _walk_candidate_files(roots)
    missing: List[str] = []
    if not model_entry.get("present"):
        missing.append("complete_local_llama_like_checkpoint_and_tokenizer")
    valid_cache = [
        item
        for item in cache_entries
        if item.get("valid_aligned_token_cache") is True
        and item.get("model_identity_match", True) is True
        and item.get("tokenizer_identity_match", True) is True
        and item.get("provenance_valid", True) is True
        and (
            accept_nonstandard_caption
            or item.get("nonstandard_caption_candidate") is not True
        )
        and int(item.get("sample_count", 0)) >= full_count
        and int(item.get("sequence_length", 0)) == sequence_length
    ]
    if not valid_cache:
        missing.append("approved_public_causal_lm_evaluation_data_and_aligned_token_cache")
    status = "ready_for_experiment" if not missing else "skipped_external_dependency"
    candidate_notes = []
    for item in discovered:
        lower = item["path"].lower()
        if any(marker in lower for marker in ("flickr", "caption", "mscoco", "coco")):
            item = dict(item)
            item["suitability"] = "present_but_nonstandard_caption_corpus"
            candidate_notes.append(item)
    return {
        "schema_version": PRETRAINED_STATUS_VERSION,
        "status": status,
        "timestamp_utc": _utc_now(),
        "network_access_attempted": False,
        "weights_allocated": False,
        "requested": {
            "model_path": str(resolved_model),
            "dataset_paths": [str(path) for path in requested_data],
            "calibration_sample_count": calibration_count,
            "full_sample_count": full_count,
            "sequence_length": sequence_length,
            "generation_tokens": generation_tokens,
            "accept_nonstandard_caption": accept_nonstandard_caption,
        },
        "model": model_entry,
        "evaluation_data": {
            "requested_cache_entries": cache_entries,
            "discovered_candidates": discovered,
            "not_accepted_candidates": candidate_notes,
        },
        "searched_cache_roots": _root_inventory(roots),
        "missing": missing,
        "proposed_external_run": _proposed_external_run(
            model_entry=model_entry,
            calibration_count=calibration_count,
            full_count=full_count,
            sequence_length=sequence_length,
            generation_tokens=generation_tokens,
        ),
        "authorization_required": (
            []
            if not missing
            else [
                "provide or authorize an evaluation corpus and aligned token cache",
                "authorize suitable compute if the 16-GiB host cannot complete FP32 conversion",
            ]
        ),
        "evaluation_scope": (
            "caption_only_nonstandard_lm_candidate"
            if accept_nonstandard_caption
            else "standard_causal_lm_evidence_required"
        ),
    }


def write_pretrained_status(
    path: str | Path,
    status: Dict[str, Any],
    *,
    allow_overwrite: bool = False,
) -> Path:
    """Write one status manifest, refusing accidental replacement by default."""

    import json

    target = Path(path).expanduser().resolve()
    if target.exists() and not allow_overwrite:
        raise FileExistsError(
            "refusing to overwrite pretrained status: %s" % target
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target
