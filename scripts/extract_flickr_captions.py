#!/usr/bin/env python3
"""Extract deterministic Flickr30k caption JSONL from a Karpathy JSON file.

This is an opt-in local-data preparation helper.  It performs no network
access, stores no image bytes, and labels the resulting corpus as a
non-standard caption-only language-model evaluation candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_caption_records(
    source: str | Path,
    *,
    split: str = "test",
    max_samples: int | None = None,
) -> List[Dict[str, str]]:
    """Return stable ``{id,text}`` records for one Karpathy split."""

    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError("caption source does not exist: %s" % source_path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        raise ValueError("caption source must contain an images list")
    records: List[Dict[str, str]] = []
    for image in payload["images"]:
        if not isinstance(image, dict) or image.get("split") != split:
            continue
        image_id = str(image.get("imgid", image.get("filename", "")))
        sentences = image.get("sentences", [])
        if not image_id or not isinstance(sentences, list):
            continue
        for index, sentence in enumerate(sentences):
            if not isinstance(sentence, dict):
                continue
            text = sentence.get("raw")
            if not isinstance(text, str) or not text.strip():
                continue
            sentence_id = str(sentence.get("sentid", index))
            records.append(
                {
                    "id": "flickr30k-%s-%s" % (image_id, sentence_id),
                    "text": text.strip(),
                }
            )
    records.sort(key=lambda item: item["id"])
    if len({item["id"] for item in records}) != len(records):
        raise ValueError("caption source contains duplicate derived IDs")
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        records = records[:max_samples]
    if not records:
        raise ValueError("caption source contains no records for split %r" % split)
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract local Flickr30k captions; no network or image loading."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.allow_overwrite:
        print("refusing to overwrite %s; use --allow-overwrite" % output, file=sys.stderr)
        return 2
    try:
        records = extract_caption_records(
            source, split=args.split, max_samples=args.max_samples
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and args.allow_overwrite:
        output.unlink()
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "status": "success",
                "evaluation_scope": "caption_only_nonstandard_lm_candidate",
                "source_path": str(source),
                "source_sha256": _sha256_file(source),
                "source_split": args.split,
                "output_path": str(output),
                "output_sha256": _sha256_file(output),
                "sample_count": len(records),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

