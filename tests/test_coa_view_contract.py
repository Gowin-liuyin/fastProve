"""COA view contract tests: secret isolation and view scoping (plan 4.3)."""

from __future__ import annotations

import json

import pytest

from fastprove.coa_views import (
    CLOUD_COA_EXTRA_INPUTS,
    FORBIDDEN_ATTACKER_INPUTS,
    TRANSCRIPT_COA_INPUTS,
    build_coa_view_contract,
    write_view_manifest,
)


def test_transcript_view_excludes_cloud_held_material() -> None:
    contract = build_coa_view_contract(
        "transcript_coa", key_epoch_count=3, observation_budget_per_epoch=100
    )
    assert contract.allowed_inputs == frozenset(TRANSCRIPT_COA_INPUTS)
    assert not (contract.allowed_inputs & set(CLOUD_COA_EXTRA_INPUTS))


def test_cloud_view_includes_converted_weights_and_injection_mapping() -> None:
    contract = build_coa_view_contract(
        "cloud_coa", key_epoch_count=3, observation_budget_per_epoch=100
    )
    assert "converted_weights" in contract.allowed_inputs
    assert "noise_injection_mapping" in contract.allowed_inputs
    assert contract.allowed_inputs >= frozenset(TRANSCRIPT_COA_INPUTS)


def test_no_view_includes_forbidden_material() -> None:
    for view in ("transcript_coa", "cloud_coa"):
        contract = build_coa_view_contract(
            view, key_epoch_count=1, observation_budget_per_epoch=10
        )
        assert not (contract.allowed_inputs & set(FORBIDDEN_ATTACKER_INPUTS))
        contract.assert_contract()


def test_unknown_view_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown COA view"):
        build_coa_view_contract(
            "server_coa", key_epoch_count=1, observation_budget_per_epoch=10
        )


def test_contract_rejects_leaked_forbidden_input() -> None:
    from fastprove.coa_views import COAViewContract

    contract = COAViewContract(
        view="transcript_coa",
        key_epoch_count=1,
        observation_budget_per_epoch=10,
        allowed_inputs=frozenset({"secret_basis"}),
        forbidden_inputs=frozenset(FORBIDDEN_ATTACKER_INPUTS),
    )
    with pytest.raises(ValueError, match="secret_basis"):
        contract.assert_contract()


def test_input_manifest_is_serializable_and_records_budget() -> None:
    contract = build_coa_view_contract(
        "cloud_coa",
        key_epoch_count=3,
        observation_budget_per_epoch=10000,
        extra={"tensor_shapes": "[64, 80]"},
    )
    manifest = contract.input_manifest()
    assert manifest["key_epoch_count"] == 3
    assert manifest["observation_budget_per_epoch"] == 10000
    assert "plaintext_ciphertext_pairs" in manifest["forbidden_inputs"]
    json.dumps(manifest)


def test_manifest_persists_to_file(tmp_path) -> None:
    contract = build_coa_view_contract(
        "transcript_coa", key_epoch_count=1, observation_budget_per_epoch=100
    )
    path = tmp_path / "coa_view_manifest.json"
    write_view_manifest(contract, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == contract.input_manifest()
