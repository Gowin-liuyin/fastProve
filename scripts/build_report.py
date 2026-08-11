#!/usr/bin/env python3
"""Plan or explicitly derive tables, figures, and REPORT.md from raw JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastprove.evaluation.artifacts import validate_run_record  # noqa: E402


REQUIRED_SWEEP_MODES = frozenset(
    {"plaintext", "exact", "topk_preserving", "free_bounded"}
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Inspect raw JSONL by default. Derived artifacts are written only "
            "when --execute is supplied."
        )
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("results/raw"),
        help="One JSONL file or a directory of JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Destination root for tables, figures, and REPORT.md.",
    )
    parser.add_argument(
        "--experiment-status",
        help="Explicit report status; otherwise derived from raw records.",
    )
    parser.add_argument(
        "--pretrained-status",
        help=(
            "Explicit pretrained-evaluation status; otherwise derived from "
            "meaningful pretrained raw records."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write derived report artifacts. This never runs model inference.",
    )
    return parser


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _raw_files(path: Path) -> List[Path]:
    if path.is_file():
        if path.suffix != ".jsonl":
            raise ValueError("--raw file must have a .jsonl suffix")
        return [path]
    if path.is_dir():
        return sorted(item for item in path.glob("*.jsonl") if item.is_file())
    raise FileNotFoundError(f"raw artifact path does not exist: {path}")


def _load_records(files: Iterable[Path]) -> List[dict]:
    records: List[dict] = []
    seen_run_ids = set()
    for path in files:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number}: record must be a JSON object"
                )
            validate_run_record(record)
            run_id = str(record["run_id"])
            if run_id in seen_run_ids:
                raise ValueError(f"duplicate run_id across raw inputs: {run_id}")
            seen_run_ids.add(run_id)
            annotated = dict(record)
            annotated["_raw_source"] = str(path.expanduser().resolve())
            annotated["_raw_line"] = line_number
            records.append(annotated)
    return records


def _record_scope(record: dict) -> tuple[str, str] | None:
    config = record.get("config")
    if not isinstance(config, dict):
        return None
    experiment_name = config.get("experiment_name")
    stage = config.get("stage")
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        return None
    if not isinstance(stage, str) or not stage.strip():
        return None
    return experiment_name.strip(), stage.strip()


def _expected_run_ids(record: dict) -> tuple[str, ...] | None:
    config = record.get("config")
    if not isinstance(config, dict) or "expected_run_ids" not in config:
        return None
    value = config["expected_run_ids"]
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        return ()
    return tuple(value)


def _is_sweep_scope(
    scope: tuple[str, str],
    records: Sequence[dict],
) -> bool:
    name, stage = scope
    if "sweep" in f"{name} {stage}".lower():
        return True
    modes = {
        record.get("config", {}).get("mode")
        for record in records
        if isinstance(record.get("config"), dict)
    }
    return REQUIRED_SWEEP_MODES.issubset(modes)


def _coverage_status(records: Sequence[dict]) -> str:
    """Derive status from an explicit run manifest or one named stage.

    ``config.expected_run_ids`` is the deliberately small manifest schema.
    Every declaration in one ``(experiment_name, stage)`` scope must agree.
    A sweep scope additionally needs all four required inference modes.
    Without a manifest, status remains partial because the intended coverage is
    unknowable even for a named non-sweep stage.
    """

    if not records or all(record["status"] == "skipped" for record in records):
        return "not_run"
    if any(record["status"] != "success" for record in records):
        return "partial_or_failed"
    for record in records:
        config = record.get("config")
        if not isinstance(config, dict) or config.get("mode") != "exact":
            continue
        metrics = record.get("metrics")
        exact_gate = (
            metrics.get("exact_gate")
            if isinstance(metrics, dict)
            else None
        )
        if (
            not isinstance(exact_gate, dict)
            or exact_gate.get("passed") is not True
        ):
            return "partial_or_failed"

    grouped: dict[tuple[str, str] | None, list[dict]] = {}
    for record in records:
        grouped.setdefault(_record_scope(record), []).append(record)
    if None in grouped:
        return "partial_or_failed"

    manifests_present = any(
        _expected_run_ids(record) is not None for record in records
    )
    if not manifests_present:
        return "partial_or_failed"

    for scope, scoped_records in grouped.items():
        assert scope is not None
        if any(
            _expected_run_ids(record) is None
            for record in scoped_records
        ):
            return "partial_or_failed"
        declarations = {
            expected
            for record in scoped_records
            if (expected := _expected_run_ids(record)) is not None
        }
        if len(declarations) != 1:
            return "partial_or_failed"
        expected = next(iter(declarations))
        if not expected:
            return "partial_or_failed"

        observed = {
            str(record["run_id"]): record for record in scoped_records
        }
        if set(expected) != set(observed):
            return "partial_or_failed"
        if any(observed[run_id]["status"] != "success" for run_id in expected):
            return "partial_or_failed"
        if _is_sweep_scope(scope, scoped_records):
            expected_modes = {
                observed[run_id].get("config", {}).get("mode")
                for run_id in expected
            }
            if not REQUIRED_SWEEP_MODES.issubset(expected_modes):
                return "partial_or_failed"
    # A candidate full run is intentionally not mislabeled as a complete
    # 72-point sweep.  The selected manifest is carried in every config so the
    # report can distinguish a completed subset from full coverage.
    if any(
        isinstance(record.get("config"), dict)
        and record["config"].get("selected_spec_ids_file")
        for record in records
    ):
        return "completed_selected_subset"
    return "completed"


def _derived_status(records: Sequence[dict]) -> str:
    return _coverage_status(records)


def _derived_pretrained_status(records: Sequence[dict]) -> str:
    """Derive pretrained status only from explicit model and dataset evidence."""

    relevant = []
    for record in records:
        model = record.get("model")
        dataset = record.get("dataset")
        if (
            isinstance(model, dict)
            and isinstance(dataset, dict)
            and model.get("pretrained") is True
            and dataset.get("meaningful_lm_evidence") is True
        ):
            relevant.append(record)
    if not relevant:
        return "not_run"

    return _coverage_status(relevant)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    raw_path = _resolve(args.raw)
    output_path = _resolve(args.output)
    try:
        files = _raw_files(raw_path)
        records = _load_records(files)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    derived_experiment_status = _derived_status(records)
    derived_pretrained_status = _derived_pretrained_status(records)
    if (
        args.experiment_status == "completed"
        and derived_experiment_status != "completed"
    ):
        parser.error(
            "--experiment-status completed cannot elevate incomplete raw "
            "evidence"
        )
    if (
        args.pretrained_status == "completed"
        and derived_pretrained_status != "completed"
    ):
        parser.error(
            "--pretrained-status completed cannot elevate incomplete raw "
            "evidence"
        )
    experiment_status = (
        args.experiment_status
        if args.experiment_status is not None
        else derived_experiment_status
    )
    pretrained_status = (
        args.pretrained_status
        if args.pretrained_status is not None
        else derived_pretrained_status
    )
    status_counts = dict(Counter(record["status"] for record in records))
    common = {
        "raw": str(raw_path),
        "output": str(output_path),
        "input_files": [str(path) for path in files],
        "record_count": len(records),
        "status_counts": status_counts,
        "experiment_status": experiment_status,
        "pretrained_status": pretrained_status,
    }

    if not args.execute:
        print(
            json.dumps(
                {
                    "action": "plan",
                    "execution_authorized": False,
                    **common,
                    "next_command": (
                        "python3 scripts/build_report.py"
                        f" --raw {raw_path} --output {output_path} --execute"
                    ),
                },
                indent=2,
            )
        )
        return 0

    if not records:
        parser.error(
            "no raw run records; refusing to overwrite pending report artifacts"
        )

    from fastprove.evaluation.report import build_report

    outputs = build_report(
        records=records,
        output_root=output_path,
        experiment_status=experiment_status,
        pretrained_status=pretrained_status,
        raw_sources=files,
    )
    print(
        json.dumps(
            {
                "action": "build_report",
                "execution_authorized": True,
                **common,
                "artifacts": {
                    "summary_csv": str(outputs.summary_csv),
                    "accuracy_figure": str(outputs.accuracy_figure),
                    "softmax_figure": str(outputs.softmax_figure),
                    "report_markdown": str(outputs.report_markdown),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
