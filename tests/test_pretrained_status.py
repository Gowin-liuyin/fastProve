from __future__ import annotations

import json
from pathlib import Path

from fastprove.evaluation.pretrained import (
    PRETRAINED_STATUS_VERSION,
    build_pretrained_status,
    write_pretrained_status,
)


def test_pretrained_status_is_explicit_when_external_data_is_missing(
    tmp_path: Path,
) -> None:
    status = build_pretrained_status(
        model_path=tmp_path / "missing-model",
        search_roots=[tmp_path / "missing-cache-root"],
        calibration_count=2,
        full_count=4,
        sequence_length=8,
        generation_tokens=2,
    )

    assert status["schema_version"] == PRETRAINED_STATUS_VERSION
    assert status["status"] == "skipped_external_dependency"
    assert status["network_access_attempted"] is False
    assert status["weights_allocated"] is False
    assert "complete_local_llama_like_checkpoint_and_tokenizer" in status[
        "missing"
    ]
    assert "approved_public_causal_lm_evaluation_data_and_aligned_token_cache" in status[
        "missing"
    ]
    assert status["proposed_external_run"]["evaluation"]["sweep_points"] == 72
    assert status["evaluation_scope"] == "standard_causal_lm_evidence_required"


def test_pretrained_status_write_is_append_safe(tmp_path: Path) -> None:
    status = build_pretrained_status(
        model_path=tmp_path / "missing-model",
        search_roots=[],
    )
    target = tmp_path / "status.json"
    written = write_pretrained_status(target, status)
    assert written == target.resolve()
    assert json.loads(target.read_text(encoding="utf-8"))["status"] == (
        "skipped_external_dependency"
    )
