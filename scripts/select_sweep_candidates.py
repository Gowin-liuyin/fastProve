#!/usr/bin/env python3
"""Select a traceable full-evaluation subset from calibration JSONL.

This command never runs inference and never edits the calibration file.  It
writes a JSON manifest suitable for ``run_accuracy_sweep.py
--spec-ids-file``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastprove.evaluation.selection import (  # noqa: E402
    SELECTION_SCHEMA_VERSION,
    load_jsonl_records,
    select_sweep_candidates,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Select full-evaluation candidates from calibration JSONL; never runs inference.",
    )
    parser.add_argument("--raw", required=True, type=Path, help="Calibration JSONL")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Candidate manifest JSON destination",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    raw = args.raw.expanduser().resolve()
    output = args.output.expanduser().resolve()
    try:
        records = load_jsonl_records(str(raw))
        manifest = select_sweep_candidates(records, source=str(raw))
        if output.exists() and not args.allow_overwrite:
            raise FileExistsError(
                "refusing to overwrite candidate manifest: %s" % output
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    except (FileNotFoundError, ValueError, FileExistsError) as error:
        print(
            json.dumps(
                {
                    "action": "select_sweep_candidates",
                    "status": "failure",
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "action": "select_sweep_candidates",
                "status": manifest["status"],
                "schema_version": SELECTION_SCHEMA_VERSION,
                "raw": str(raw),
                "output": str(output),
                "record_count": manifest["record_count"],
                "selected_count": len(manifest["selected_run_ids"]),
                "selected_run_ids": manifest["selected_run_ids"],
                "unselected_count": len(manifest["unselected_run_ids"]),
                "failure_count": len(manifest["failure_run_ids"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if manifest["status"] == "ready_for_full_evaluation" else 1


if __name__ == "__main__":
    raise SystemExit(main())
