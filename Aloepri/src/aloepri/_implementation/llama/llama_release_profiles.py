from __future__ import annotations

from pathlib import Path

from src.aloepri._implementation.export.release_bundle_impl import StageKProfile, export_stage_k_release


DEFAULT_LLAMA_PAPER_CONSISTENT_DIR = "artifacts/stage_j_llama_instruct_paper_consistent"


def default_stage_k_llama_profiles() -> list[StageKProfile]:
    return [
        StageKProfile(
            name="default",
            source_dir=DEFAULT_LLAMA_PAPER_CONSISTENT_DIR,
            description="Default paper-consistent Llama-3.2-3B-Instruct Stage-J release profile.",
            recommended_use="Default delivery entry for the paper-consistent Llama Instruct deployment line.",
            correctness_evidence_file="outputs/stage_k_llama_release/correctness/default.json",
        ),
        StageKProfile(
            name="reference",
            source_dir=DEFAULT_LLAMA_PAPER_CONSISTENT_DIR,
            description="Reference paper-consistent Llama-3.2-3B-Instruct Stage-J release profile.",
            recommended_use="Audit and evidence entry for the same paper-consistent Llama Instruct deployment line.",
            correctness_evidence_file="outputs/stage_k_llama_release/correctness/reference.json",
        ),
    ]


def export_stage_k_llama_release(
    export_dir: str | Path,
    *,
    materialize: bool = False,
) -> dict:
    return export_stage_k_release(
        export_dir,
        profiles=default_stage_k_llama_profiles(),
        materialize=materialize,
        recommended_profile="default",
        reference_profile="reference",
        title="Stage-K Paper-Consistent Llama-3.2-3B-Instruct Release",
    )
