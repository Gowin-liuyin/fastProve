#!/usr/bin/env python3
"""Plan or explicitly run the local random-Tiny-LM correctness check."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastprove.config import load_config  # noqa: E402
from fastprove.evaluation.artifacts import (  # noqa: E402
    append_jsonl_record,
    build_run_record,
    require_fresh_jsonl_destination,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Plan the tiny correctness run. No model code runs unless "
            "--execute is supplied."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tiny_exact.yaml"),
        help="Validated tiny prototype YAML configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/raw/correctness.jsonl"),
        help="Raw JSONL destination used only with --execute.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly authorize the local random-tiny correctness run.",
    )
    return parser


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _plan(config_path: Path, output_path: Path) -> dict:
    config = load_config(config_path)
    return {
        "action": "plan",
        "execution_authorized": False,
        "config": str(config_path),
        "output": str(output_path),
        "mode": config.attention.mode,
        "seed": config.runtime.seed,
        "model_kind": "random_tiny_correctness_only",
        "claims_allowed": [
            "mathematical_and_end_to_end_correctness",
            "floating_point_deviation",
        ],
        "claims_forbidden": [
            "meaningful_language_model_accuracy",
            "cryptographic_security",
        ],
        "next_command": (
            "python3 scripts/run_correctness.py"
            f" --config {config_path} --output {output_path} --execute"
        ),
    }


def _failure_record(
    *,
    config_path: Path,
    output_path: Path,
    error: Exception,
) -> dict:
    config = load_config(config_path)
    try:
        from fastprove.evaluation.runner import environment_metadata

        environment = environment_metadata()
    except Exception as metadata_error:  # pragma: no cover - defensive
        environment = {
            "status": "metadata_unavailable",
            "error_type": type(metadata_error).__name__,
            "error_message": str(metadata_error),
        }
    record = build_run_record(
        run_id="tiny-correctness",
        status="failure",
        config={
            "mode": config.attention.mode.value,
            "stage": "tiny_correctness",
            "prototype": asdict(config),
        },
        seed=config.runtime.seed,
        model={
            "id": "fastprove-random-tiny-correctness-only",
            "pretrained": False,
        },
        dataset={
            "id": "deterministic-synthetic-token-sequences",
            "meaningful_lm_evidence": False,
        },
        environment=environment,
        sample_count=0,
        metrics={},
        elapsed_seconds=0.0,
        error={
            "type": type(error).__name__,
            "message": str(error),
        },
        stage="correctness_cli_dispatch",
        last_completed_sample_id=None,
        partial_metrics_available=False,
    )
    append_jsonl_record(output_path, record)
    return record


def _exact_succeeded(record: dict) -> bool:
    return bool(
        record.get("status") == "success"
        and record.get("metrics", {})
        .get("exact_gate", {})
        .get("passed")
        is True
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = _resolve(args.config)
    output_path = _resolve(args.output)

    if not args.execute:
        print(json.dumps(_plan(config_path, output_path), indent=2))
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
        from fastprove.evaluation.correctness import run_tiny_correctness

        record = run_tiny_correctness(
            config_path=config_path,
            output_path=output_path,
        )
    except Exception as error:
        record = _failure_record(
            config_path=config_path,
            output_path=output_path,
            error=error,
        )
        print(
            json.dumps(
                {
                    "action": "run_correctness",
                    "execution_authorized": True,
                    "status": "failure",
                    "run_id": record["run_id"],
                    "output": str(output_path),
                    "error": record["error"],
                },
                indent=2,
            )
        )
        return 1

    print(
        json.dumps(
            {
                "action": "run_correctness",
                "execution_authorized": True,
                "status": record["status"],
                "exact_gate": record.get("metrics", {}).get(
                    "exact_gate",
                    {"passed": False, "reasons": ["missing exact gate"]},
                ),
                "run_id": record["run_id"],
                "output": str(output_path),
            },
            indent=2,
        )
    )
    return 0 if _exact_succeeded(record) else 1


if __name__ == "__main__":
    raise SystemExit(main())
