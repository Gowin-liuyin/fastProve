#!/usr/bin/env python3
"""Compare a rho replica sweep against its rho=0 base records.

Replica run IDs are the base run ID plus the suffix emitted by
``scripts/run_accuracy_sweep.py --rho-replica`` (for example ``-rho1e-05``).
The tool never runs inference; it reads two JSONL files and reports the
marginal cost of the deployment-fold rho knob per selected candidate.

Exit status: 0 on success, 2 on malformed/unmatched inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("record %d is not a JSON object" % number)
        records.append(value)
    if not records:
        raise ValueError("no records in %s" % path)
    return records


def _index(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for record in records:
        run_id = record.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("record without a run_id")
        if run_id in index:
            raise ValueError("duplicate run_id: %s" % run_id)
        index[run_id] = record
    return index


def _strip_suffix(run_id: str, suffix: str) -> str:
    if not run_id.endswith(suffix):
        raise ValueError("run_id %r does not end with %r" % (run_id, suffix))
    base = run_id[: -len(suffix)]
    if not base:
        raise ValueError("run_id %r is empty after stripping %r" % (run_id, suffix))
    return base


def _pair_records(
    base: Dict[str, Dict[str, Any]],
    replica: Dict[str, Dict[str, Any]],
    suffix: str,
) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    pairs: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    missing_base: List[str] = []
    for run_id, replica_record in sorted(replica.items()):
        base_id = _strip_suffix(run_id, suffix)
        base_record = base.get(base_id)
        if base_record is None:
            missing_base.append(base_id)
            continue
        pairs.append((base_id, base_record, replica_record))
    if missing_base:
        raise ValueError(
            "replica points without a base record: " + ", ".join(missing_base)
        )
    if not pairs:
        raise ValueError("no replica/base pairs found")
    return pairs


def _metric(record: Dict[str, Any], *path: str, default: float = 0.0) -> float:
    value: Any = record.get("metrics", {})
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _row(
    base_id: str,
    base_record: Dict[str, Any],
    replica_record: Dict[str, Any],
) -> Dict[str, Any]:
    top1_base = _metric(base_record, "degradation", "top1_absolute_drop")
    top1_rep = _metric(replica_record, "degradation", "top1_absolute_drop")
    ppl_base = _metric(
        base_record, "degradation", "perplexity_relative_increase"
    )
    ppl_rep = _metric(
        replica_record, "degradation", "perplexity_relative_increase"
    )
    nll_base = _metric(base_record, "degradation", "nll_absolute_increase")
    nll_rep = _metric(replica_record, "degradation", "nll_absolute_increase")
    js_base = _metric(base_record, "softmax", "js_divergence")
    js_rep = _metric(replica_record, "softmax", "js_divergence")
    noise_base = _metric(
        base_record, "softmax", "actual_noise_infinity_norm"
    )
    noise_rep = _metric(
        replica_record, "softmax", "actual_noise_infinity_norm"
    )
    return {
        "run_id": base_id,
        "mode": base_record.get("config", {}).get("mode"),
        "tau_max": base_record.get("config", {}).get("tau_max"),
        "alpha": base_record.get("config", {}).get("alpha"),
        "preserve_top_k": base_record.get("config", {}).get("preserve_top_k"),
        "rho_base": _base_rho(base_record),
        "rho_replica": _base_rho(replica_record),
        "top1_absolute_drop_base": top1_base,
        "top1_absolute_drop_replica": top1_rep,
        "top1_absolute_drop_marginal": top1_rep - top1_base,
        "perplexity_relative_increase_base": ppl_base,
        "perplexity_relative_increase_replica": ppl_rep,
        "perplexity_relative_increase_marginal": ppl_rep - ppl_base,
        "nll_absolute_increase_base": nll_base,
        "nll_absolute_increase_replica": nll_rep,
        "nll_absolute_increase_marginal": nll_rep - nll_base,
        "js_divergence_base": js_base,
        "js_divergence_replica": js_rep,
        "js_divergence_marginal": js_rep - js_base,
        "actual_noise_infinity_norm_base": noise_base,
        "actual_noise_infinity_norm_replica": noise_rep,
    }


def _base_rho(record: Dict[str, Any]) -> Optional[float]:
    value = (
        record.get("config", {})
        .get("obfuscation", {})
        .get("rho_noise_bound")
    )
    return float(value) if isinstance(value, (int, float)) else None


def _summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def values(key: str) -> List[float]:
        return [float(row[key]) for row in rows]

    def maximum(key: str) -> float:
        return max(values(key))

    def mean(key: str) -> float:
        return sum(values(key)) / len(rows)

    return {
        "point_count": len(rows),
        "top1_absolute_drop_marginal_max": maximum("top1_absolute_drop_marginal"),
        "top1_absolute_drop_marginal_mean": mean("top1_absolute_drop_marginal"),
        "perplexity_relative_increase_marginal_max": maximum(
            "perplexity_relative_increase_marginal"
        ),
        "perplexity_relative_increase_marginal_mean": mean(
            "perplexity_relative_increase_marginal"
        ),
        "nll_absolute_increase_marginal_max": maximum(
            "nll_absolute_increase_marginal"
        ),
        "js_divergence_marginal_max": maximum("js_divergence_marginal"),
    }


def _markdown(rows: Sequence[Dict[str, Any]], suffix: str) -> str:
    lines = [
        "| run_id | ρ base → replica | top1 drop base → replica (Δ) | ppl rel inc base → replica (Δ) | JS Δ |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| %s | %s → %s | %.4f → %.4f (%+.4f) | %.3f%% → %.3f%% (%+.3f pp) | %+.2e |"
            % (
                row["run_id"],
                _fmt_rho(row["rho_base"]),
                _fmt_rho(row["rho_replica"]),
                row["top1_absolute_drop_base"],
                row["top1_absolute_drop_replica"],
                row["top1_absolute_drop_marginal"],
                100.0 * row["perplexity_relative_increase_base"],
                100.0 * row["perplexity_relative_increase_replica"],
                100.0 * row["perplexity_relative_increase_marginal"],
                row["js_divergence_marginal"],
            )
        )
    return "\n".join(lines) + "\n"


def _fmt_rho(value: Optional[float]) -> str:
    return "n/a" if value is None else ("%g" % value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--replica", type=Path, required=True)
    parser.add_argument(
        "--suffix",
        default="-rho1e-05",
        help="replica run_id suffix (default: -rho1e-05)",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        base_index = _index(_load_records(args.base))
        replica_index = _index(_load_records(args.replica))
        pairs = _pair_records(base_index, replica_index, args.suffix)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print("compare_rho_replica: %s" % error, file=sys.stderr)
        return 2

    rows = [_row(base_id, base, replica) for base_id, base, replica in pairs]
    payload = {
        "schema": "fastprove.rho_replica_marginal.v1",
        "base": str(args.base),
        "base_sha256": _file_sha256(args.base),
        "replica": str(args.replica),
        "replica_sha256": _file_sha256(args.replica),
        "suffix": args.suffix,
        "summary": _summary(rows),
        "points": rows,
    }
    markdown = _markdown(rows, args.suffix)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    summary = payload["summary"]
    print(
        "points=%d  Δtop1 max=%.4f mean=%.4f  Δppl_rel max=%.3f%%  ΔJS max=%.2e"
        % (
            summary["point_count"],
            summary["top1_absolute_drop_marginal_max"],
            summary["top1_absolute_drop_marginal_mean"],
            100.0 * summary["perplexity_relative_increase_marginal_max"],
            summary["js_divergence_marginal_max"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
