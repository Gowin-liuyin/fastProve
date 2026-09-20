"""Base-sweep rho policy (docs/PROBLEMS.md §3.5, decision 丙 base).

Every base sweep point must run with ``rho_noise_bound = 0`` so that accuracy
deltas are attributable to the logit-noise knobs alone. The deployment-fold
rho is only used by explicit replica runs. These tests pin the contract at
the config-resolution boundary and keep the YAML declaration in sync.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fastprove.config import load_config
from fastprove.evaluation.spec_config import (
    BASE_SWEEP_RHO_NOISE_BOUND,
    _config_for_spec,
)
from fastprove.evaluation.sweep import enumerate_sweep_specs

ROOT = Path(__file__).resolve().parents[2]
SWEEP_CONFIG = ROOT / "configs" / "eval_sweep.yaml"
BASE_CONFIG = ROOT / "configs" / "tiny_approx.yaml"


def _sweep_config() -> dict:
    return yaml.safe_load(SWEEP_CONFIG.read_text(encoding="utf-8"))


def test_all_72_base_specs_resolve_to_zero_rho() -> None:
    raw = _sweep_config()
    base = load_config(BASE_CONFIG)
    specs = enumerate_sweep_specs(SWEEP_CONFIG)
    assert len(specs) == 72
    mode_counts: dict[str, int] = {}
    for spec in specs:
        config = _config_for_spec(
            base,
            spec,
            raw.get("evaluation_override"),
            obfuscation_override=raw.get("obfuscation_override"),
        )
        assert config.obfuscation.rho_noise_bound == 0.0, spec.run_id
        assert config.obfuscation.refresh_mode == "per_request", spec.run_id
        mode_counts[spec.mode] = mode_counts.get(spec.mode, 0) + 1
    assert mode_counts == {
        "plaintext": 1,
        "exact": 1,
        "topk_preserving": 63,
        "free_bounded": 7,
    }


def test_code_default_is_zero_without_any_override() -> None:
    base = load_config(BASE_CONFIG)
    spec = enumerate_sweep_specs(SWEEP_CONFIG)[1]
    config = _config_for_spec(base, spec, None)
    assert config.obfuscation.rho_noise_bound == BASE_SWEEP_RHO_NOISE_BOUND
    assert config.obfuscation.rho_noise_bound == 0.0


def test_explicit_replica_override_applies() -> None:
    base = load_config(BASE_CONFIG)
    spec = enumerate_sweep_specs(SWEEP_CONFIG)[1]
    config = _config_for_spec(
        base,
        spec,
        None,
        obfuscation_override={"rho_noise_bound": 1.0e-5},
    )
    assert config.obfuscation.rho_noise_bound == 1.0e-5
    assert config.obfuscation.refresh_mode == "per_request"


def test_unknown_obfuscation_override_key_is_rejected() -> None:
    base = load_config(BASE_CONFIG)
    spec = enumerate_sweep_specs(SWEEP_CONFIG)[1]
    with pytest.raises(ValueError, match="unknown fields"):
        _config_for_spec(
            base, spec, None, obfuscation_override={"not_a_field": 1}
        )


def test_yaml_declares_the_code_default() -> None:
    raw = _sweep_config()
    declared = raw["obfuscation_override"]["rho_noise_bound"]
    assert declared == BASE_SWEEP_RHO_NOISE_BOUND


def test_gate_flag_is_synced_with_completed_base_sweep() -> None:
    raw = _sweep_config()
    execution = raw["execution"]
    assert execution["run_in_current_prototype_pass"] is True
    assert "completed" in execution["reason"]
