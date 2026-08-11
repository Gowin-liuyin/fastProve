#!/usr/bin/env python3
"""Audit local pretrained assets and write the external-dependency manifest.

The command performs no downloads and no model inference.  It is intentionally
safe to run before an experiment authorization decision.
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

from fastprove.evaluation.pretrained import (  # noqa: E402
    DEFAULT_SEARCH_ROOTS,
    build_pretrained_status,
    default_local_qwen_path,
    write_pretrained_status,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit local Qwen/checkpoint and evaluation-data assets without network access."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=default_local_qwen_path(),
        help="Local Qwen2 artifact; defaults to the audited local candidate.",
    )
    parser.add_argument(
        "--dataset-path",
        action="append",
        default=[],
        type=Path,
        help="Optional local aligned .pt cache; may be repeated.",
    )
    parser.add_argument(
        "--search-root",
        action="append",
        type=Path,
        help="Root to inventory for likely local data candidates; may be repeated.",
    )
    parser.add_argument("--calibration-count", type=int, default=8)
    parser.add_argument("--full-count", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--generation-tokens", type=int, default=4)
    parser.add_argument(
        "--accept-nonstandard-caption",
        action="store_true",
        help="Treat a local Flickr/caption cache as an explicitly accepted limited evaluation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/raw/pretrained_evaluation_status.json"),
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    roots = args.search_root or list(DEFAULT_SEARCH_ROOTS)
    status = build_pretrained_status(
        model_path=args.model_path,
        dataset_paths=args.dataset_path,
        search_roots=roots,
        calibration_count=args.calibration_count,
        full_count=args.full_count,
        sequence_length=args.sequence_length,
        generation_tokens=args.generation_tokens,
        accept_nonstandard_caption=args.accept_nonstandard_caption,
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    try:
        target = write_pretrained_status(
            args.output, status, allow_overwrite=args.allow_overwrite
        )
    except FileExistsError as error:
        print(str(error), file=sys.stderr)
        return 2
    print("wrote %s" % target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
