"""COA view contract: attacker inputs are an explicit, isolated list.

Plan sections 4.3 and 10 (stage B/C): the two observation scopes
``transcript_coa`` and ``cloud_coa`` must be separate configuration inputs;
an attack process must never receive target-key plaintext-ciphertext pairs,
the secret basis, or decoding interfaces. This module provides the view
contract builder used by the augmented-noise COA evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Tuple

#: Inputs an observer of transmission records may see (plan 4.3).
TRANSCRIPT_COA_INPUTS: Tuple[str, ...] = (
    "mixed_input_records",
    "mixed_output_records",
    "public_metadata",
    "public_base_weights",
)

#: Additional inputs held by an executing cloud observer (plan 4.3).
CLOUD_COA_EXTRA_INPUTS: Tuple[str, ...] = (
    "converted_weights",
    "runtime_config",
    "plain_outputs",
    "caches",
    "visible_random_states",
    "noise_injection_mapping",
)

#: Material that must never appear in any COA view (plan 4.2).
FORBIDDEN_ATTACKER_INPUTS: Tuple[str, ...] = (
    "target_key_material",
    "secret_basis",
    "basis_inverse",
    "plaintext_ciphertext_pairs",
    "debug_decode_api",
    "private_reproduction_seeds",
    "internal_clean_tensors",
)

VALID_VIEWS: Tuple[str, ...] = ("transcript_coa", "cloud_coa")


@dataclass(frozen=True)
class COAViewContract:
    """Explicit input whitelist for one attack run."""

    view: str
    key_epoch_count: int
    observation_budget_per_epoch: int
    allowed_inputs: FrozenSet[str]
    forbidden_inputs: FrozenSet[str]
    extra: Dict[str, Any] = field(default_factory=dict)

    def input_manifest(self) -> Dict[str, Any]:
        """Machine-readable record of what the attack process may read."""

        return {
            "view": self.view,
            "allowed_inputs": sorted(self.allowed_inputs),
            "forbidden_inputs": sorted(self.forbidden_inputs),
            "key_epoch_count": self.key_epoch_count,
            "observation_budget_per_epoch": self.observation_budget_per_epoch,
            "extra": dict(self.extra),
        }

    def assert_contract(self) -> None:
        """Reject a contract that leaks forbidden material."""

        overlap = self.allowed_inputs & self.forbidden_inputs
        if overlap:
            raise ValueError(
                "COA view %s leaks forbidden inputs: %s"
                % (self.view, sorted(overlap))
            )
        if "plaintext_ciphertext_pairs" in self.allowed_inputs:
            raise ValueError("COA views must not include plaintext-ciphertext pairs")
        if self.view == "transcript_coa" and (
            self.allowed_inputs & set(CLOUD_COA_EXTRA_INPUTS)
        ):
            raise ValueError(
                "transcript_coa view must not include cloud-held material"
            )


def build_coa_view_contract(
    view: str,
    *,
    key_epoch_count: int,
    observation_budget_per_epoch: int,
    extra: Dict[str, Any] | None = None,
) -> COAViewContract:
    """Build one of the two plan-defined observation scopes."""

    if view not in VALID_VIEWS:
        raise ValueError("unknown COA view: %s" % view)
    if key_epoch_count < 1:
        raise ValueError("key_epoch_count must be at least one")
    if observation_budget_per_epoch < 1:
        raise ValueError("observation_budget_per_epoch must be at least one")
    if view == "transcript_coa":
        allowed = set(TRANSCRIPT_COA_INPUTS)
    else:
        allowed = set(TRANSCRIPT_COA_INPUTS) | set(CLOUD_COA_EXTRA_INPUTS)
    contract = COAViewContract(
        view=view,
        key_epoch_count=key_epoch_count,
        observation_budget_per_epoch=observation_budget_per_epoch,
        allowed_inputs=frozenset(allowed),
        forbidden_inputs=frozenset(FORBIDDEN_ATTACKER_INPUTS),
        extra=dict(extra or {}),
    )
    contract.assert_contract()
    return contract


def write_view_manifest(contract: COAViewContract, path) -> None:
    """Persist the input manifest next to raw attack results."""

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(contract.input_manifest(), handle, indent=2, sort_keys=True)
        handle.write("\n")
