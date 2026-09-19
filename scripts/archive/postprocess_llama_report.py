#!/usr/bin/env python3
"""Recompute gates + REPORT.md from results/raw/llama_compare_P.json."""

from __future__ import annotations

import json
from pathlib import Path

from evals.layer1_operator_equiv import _chain_linear_identity_test
from evals.thresholds import evaluate_gates


def main() -> None:
    path = Path("results/raw/llama_compare_P.json")
    data = json.loads(path.read_text())
    primary = None
    for pair in data["pairs"]:
        if pair["mode"] == "full":
            primary = pair
            break
    if primary is None:
        primary = data["pairs"][0]
    chain = _chain_linear_identity_test(seed=20260802)
    gate_metrics = {
        "layer1": {
            "chain_linear": {
                "max_absolute_error": float(chain["max_absolute_error"])
            }
        },
        "layer2": {
            "rank_flip_rate": None,
            "top1_match": None,
            "topk_overlap": None,
            "causal_mask_match": None,
        },
        "layer3": {"expert_set_match": None},
        "layer4": {
            "lm_head_argmax_match": primary["lm_head_argmax_match"],
            "greedy_sequence_exact_match": primary[
                "greedy_sequence_exact_match"
            ],
            "ppl_relative_increase": primary["ppl_relative_increase"],
            "top1_absolute_drop_pp": primary["top1_absolute_drop_pp"],
        },
        "cache": {"cache_vs_nocache_identical": None},
        "utility": {
            "accuracy_drop_pp": primary["top1_absolute_drop_pp"],
            "ppl_relative_increase": primary["ppl_relative_increase"],
        },
    }
    summary = evaluate_gates(gate_metrics)
    data["chain_linear_unit_gate"] = chain
    data["aggregated_metrics"] = gate_metrics
    data["gates"] = summary.to_dict()
    data["primary_pair_mode"] = primary["mode"]
    path.write_text(json.dumps(data, indent=2, default=str))
    gate_table = summary.format_table()
    print(gate_table)

    lines = [
        "# Llama-3.2-3B-Instruct: plaintext vs current fastProve",
        "",
        "## Scope",
        "- Only plaintext base vs current fastProve conversion (structural / full).",
        "- Legacy ModelSplit obfuscation was NOT used as baseline or result.",
        "- Terminology: obfuscated-state relative to plaintext "
        "(real-valued augmented covariant obfuscation). NOT LWE-keyed and NOT "
        "LWE-inspired: the construction is exactly invertible over the reals, "
        "so h is recoverable from c (see docs/threat_model.md section 5bis).",
        "",
        "## Environment",
        "```json",
        json.dumps(data["environment"], indent=2),
        "```",
        "",
        "conda env: fastprove (offline clone of modelsplit when network blocked).",
        "",
        "## Plaintext base",
        "- path: `%s`" % data["plaintext_base"]["root"],
        "- architecture: `%s`" % data["plaintext_base"]["architecture"],
        "- weights_sha256: `%s`" % data["plaintext_base"]["weights_sha256"],
        "- config_sha256: `%s`" % data["plaintext_base"]["config_sha256"],
        "- is_plaintext_verified: %s"
        % data["plaintext_base"]["is_plaintext_verified"],
        "- verification notes: %s"
        % data["plaintext_base"]["verification_notes"],
        "",
        "## Evaluation design",
        "- sample_ids (shared): %s" % data["evaluation"]["sample_ids"],
        "- sequence_length=%s, gen_tokens=%s"
        % (
            data["evaluation"]["sequence_length"],
            data["evaluation"]["generation_tokens"],
        ),
        "- dtype=%s, device=%s"
        % (data["evaluation"]["dtype"], data["evaluation"]["device"]),
        "- master keys: independent conversion seeds (see pairs[].key)",
        "- modes: structural / full (current scheme only)",
        "",
        "## Results (current scheme only)",
        "",
        "| mode | key | top1 token agreement | greedy seq match | "
        "Delta PPL_rel | top1 drop (pp) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for pair in data["pairs"]:
        lines.append(
            "| %s | %s | %.6g | %.6g | %s | %.4g |"
            % (
                pair["mode"],
                pair["key"]["label"],
                pair["top1_token_agreement"],
                pair["greedy_sequence_exact_match"],
                pair["ppl_relative_increase"],
                pair["top1_absolute_drop_pp"],
            )
        )
    lines.extend(
        [
            "",
            "Units: agreement is a fraction; top1 drop is percentage points "
            "(pp); PPL uses relative increase.",
            "",
            "## Gates (primary = full mode first key)",
            "```",
            gate_table,
            "```",
            "",
            "FP32 ChainLinear unit gate max abs error = %.6g (threshold 1e-4)."
            % float(chain["max_absolute_error"]),
            "",
            "## Artifacts",
            "- results/raw/llama_compare_P.json",
            "- results/raw/llama_compare_run.log",
            "",
        ]
    )
    Path("results/REPORT.md").write_text("\n".join(lines))
    print("wrote results/REPORT.md")


if __name__ == "__main__":
    main()
