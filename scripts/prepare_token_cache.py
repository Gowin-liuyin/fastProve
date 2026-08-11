#!/usr/bin/env python3
"""Tokenize a local text/JSONL corpus into an aligned fastProve cache.

No network access is attempted.  The resulting file contains token IDs,
valid-token masks, stable sample IDs and hashes, but not raw text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastprove.evaluation.token_cache import build_token_cache, save_token_cache  # noqa: E402
from fastprove.pretrained.qwen2 import load_qwen2_artifact, load_qwen2_tokenizer  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic local token IDs for all aligned modes."
    )
    parser.add_argument("--model-path", required=True, type=Path, help="Local Qwen2 artifact directory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-jsonl", type=Path, help="JSONL with {id?, text} records")
    group.add_argument("--input-text", type=Path, help="UTF-8 text file, one example per non-empty line")
    parser.add_argument("--output", required=True, type=Path, help="New .pt token-cache path")
    parser.add_argument("--sequence-length", required=True, type=int)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--add-special-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-overwrite", action="store_true", help="Explicitly replace an existing cache")
    return parser


def _source_records(path: Path, *, jsonl: bool) -> Tuple[List[str], List[str]]:
    if not path.is_file():
        raise FileNotFoundError("input corpus does not exist: %s" % path)
    ids: List[str] = []
    texts: List[str] = []
    if jsonl:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("invalid JSONL at line %d: %s" % (line_number, error)) from error
            raw_text = value.get("text", value.get("prompt")) if isinstance(value, dict) else None
            if not isinstance(value, dict) or not isinstance(raw_text, str):
                raise ValueError("JSONL line %d must contain a string 'text' or 'prompt'" % line_number)
            sample_id = str(value.get("id", "line-%06d" % line_number))
            if "id" not in value and "request_id" in value:
                sample_id = str(value["request_id"])
            text = raw_text.strip()
            if not sample_id or not text:
                continue
            ids.append(sample_id)
            texts.append(text)
    else:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            text = line.strip()
            if text:
                ids.append("line-%06d" % line_number)
                texts.append(text)
    if not texts:
        raise ValueError("input corpus contains no non-empty examples")
    if len(set(ids)) != len(ids):
        raise ValueError("input corpus contains duplicate sample IDs")
    ordered = sorted(zip(ids, texts), key=lambda pair: pair[0])
    return [pair[0] for pair in ordered], [pair[1] for pair in ordered]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.sequence_length < 2:
        raise SystemExit("--sequence-length must be at least 2")
    if args.max_samples is not None and args.max_samples < 1:
        raise SystemExit("--max-samples must be positive")
    output = args.output.expanduser().resolve()
    if output.exists() and not args.allow_overwrite:
        raise SystemExit("refusing to overwrite %s; use --allow-overwrite" % output)
    corpus = (args.input_jsonl or args.input_text).expanduser().resolve()
    artifact = load_qwen2_artifact(
        args.model_path, max_sequence_length=args.sequence_length, compute_hashes=True
    )
    tokenizer = load_qwen2_tokenizer(artifact.root)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise SystemExit("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    ids, texts = _source_records(corpus, jsonl=args.input_jsonl is not None)
    if args.max_samples is not None:
        ids, texts = ids[: args.max_samples], texts[: args.max_samples]
    encoded = tokenizer(
        texts,
        add_special_tokens=bool(args.add_special_tokens),
        truncation=True,
        max_length=args.sequence_length,
        padding=False,
        return_attention_mask=False,
    )
    # Fixed full blocks keep greedy generation semantically correct: a padded
    # final position is not a valid prompt position for a causal LM.  Short
    # records are skipped and counted rather than silently evaluated with PAD.
    full = [
        (sample_id, tokens[: args.sequence_length])
        for sample_id, tokens in zip(ids, encoded["input_ids"])
        if len(tokens) >= args.sequence_length
    ]
    if not full:
        raise SystemExit(
            "no examples contain sequence-length full blocks; provide longer text or reduce --sequence-length"
        )
    ids = [sample_id for sample_id, _ in full]
    input_ids = torch.tensor([tokens for _, tokens in full], dtype=torch.int64)
    token_mask = torch.ones_like(input_ids, dtype=torch.bool)
    cache = build_token_cache(
        sample_ids=ids,
        input_ids=input_ids,
        token_mask=token_mask,
        metadata={
            "model_root": artifact.root,
            "model_config_sha256": artifact.config_sha256,
            "tokenizer_sha256": artifact.tokenizer_sha256,
            "tokenizer_config_sha256": artifact.tokenizer_config_sha256,
            "vocab_size": artifact.model_config.vocab_size,
            "sequence_length": args.sequence_length,
            "source_path": str(corpus),
            "source_sha256": _sha256(corpus),
            "source_format": "jsonl" if args.input_jsonl is not None else "text_lines",
            "short_examples_dropped": len(texts) - len(full),
            "add_special_tokens": bool(args.add_special_tokens),
            "padding_token_id": int(tokenizer.pad_token_id),
            "evidence_scope": "tokenized_local_corpus; not yet evaluated",
        },
    )
    if output.exists() and args.allow_overwrite:
        output.unlink()
    save_token_cache(output, cache)
    print(json.dumps({
        "status": "success",
        "output": str(output),
        "sample_count": cache.sample_count,
        "sequence_length": cache.sequence_length,
        "content_sha256": cache.content_sha256,
        "model": artifact.to_dict(),
        "metadata": cache.metadata,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
