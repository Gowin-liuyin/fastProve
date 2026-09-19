"""S2 dual-channel budget ledger: guards, bounds and manifest round trip.

The channel-0 guard is the important one: theorem 4 says in-cone noise hides
nothing, so any ledger claiming nonzero channel-0 hiding bits must fail
construction. These tests were checked to turn red when the corresponding
validations are removed (see the stage S2 report).
"""

from __future__ import annotations

import json
import math

import pytest

from fastprove.config import (
    AttentionConfig,
    EvaluationConfig,
    PrototypeConfig,
    RuntimeConfig,
)
from fastprove.converter import ModelConverter
from fastprove.layers.attention import AttentionMode
from fastprove.models.plain import PlainTinyCausalLM
from fastprove.noise_budget import (
    BOUND_SOURCES,
    ChannelBudget,
    DualChannelBudget,
    semantic_hiding_bits,
)

from tests._fixtures import model_config, obfuscation_config, unit_basis


def _config(
    *,
    mode: AttentionMode = AttentionMode.EXACT,
    rho: float = 0.0,
    tau_max: float = 0.0,
    tau_error: float = 0.0,
    hidden_noise_dim: int = 8,
) -> PrototypeConfig:
    return PrototypeConfig(
        model=model_config(),
        obfuscation=obfuscation_config(
            refresh_mode="per_request",
            hidden_noise_dim=hidden_noise_dim,
            rho_noise_bound=rho,
        ),
        attention=AttentionConfig(
            mode=mode,
            tau_max=tau_max,
            tau_error=tau_error,
            alpha=0.8,
            preserve_top_k=4,
        ),
        runtime=RuntimeConfig(
            seed=7,
            request_id="noise-budget",
            device="cpu",
            activation_dtype="float32",
            debug_enabled=False,
        ),
        evaluation=EvaluationConfig(
            batch_size=2,
            sequence_length=16,
            sample_count=2,
            generation_tokens=2,
            warmup_runs=0,
            timed_runs=1,
        ),
    )


def _basis(noise_dim: int = 8, seed: int = 11):
    return unit_basis(
        32, noise_dim, seed=seed, domain="noise-budget-test", block_size=8
    )


def test_channel_zero_rejects_nonzero_hiding_claim() -> None:
    with pytest.raises(ValueError, match="zero information-theoretic hiding"):
        ChannelBudget(
            channel=0,
            dimension=8,
            energy_ratio=0.1,
            accuracy_cost_bound=0.0,
            information_theoretic_hiding_bits=3.0,
            bound_source=None,
        )


def test_channel_zero_rejects_nonzero_accuracy_cost() -> None:
    with pytest.raises(ValueError, match="zero accuracy cost"):
        ChannelBudget(
            channel=0,
            dimension=8,
            energy_ratio=0.1,
            accuracy_cost_bound=0.5,
            information_theoretic_hiding_bits=0.0,
            bound_source=None,
        )


def test_channel_one_requires_bound_source() -> None:
    with pytest.raises(ValueError, match="whitelisted bound source"):
        ChannelBudget(
            channel=1,
            dimension=32,
            energy_ratio=0.1,
            accuracy_cost_bound=0.1,
            information_theoretic_hiding_bits=None,
            bound_source=None,
        )


def test_channel_one_rejects_unknown_bound_source() -> None:
    with pytest.raises(ValueError, match="whitelisted bound source"):
        ChannelBudget(
            channel=1,
            dimension=32,
            energy_ratio=0.1,
            accuracy_cost_bound=0.1,
            information_theoretic_hiding_bits=None,
            bound_source="vibes",
        )
    assert set(BOUND_SOURCES) == {"margin_delta_k", "value_diameter_D_V", "rho"}


def test_channel_one_rejects_infinite_accuracy_cost() -> None:
    with pytest.raises(ValueError, match="finite accuracy_cost_bound"):
        ChannelBudget(
            channel=1,
            dimension=32,
            energy_ratio=0.1,
            accuracy_cost_bound=math.inf,
            information_theoretic_hiding_bits=None,
            bound_source="rho",
        )


def test_exact_mode_forces_zero_semantic_energy() -> None:
    """An `exact` ledger label is only legal with zero semantic noise."""

    exact = _config()
    budget = DualChannelBudget.from_config(exact)
    assert budget.mode == "exact"
    assert budget.manifest()["exact_signal_path"] is True
    with pytest.raises(ValueError, match="zero semantic noise"):
        DualChannelBudget(
            free=budget.free,
            semantic=ChannelBudget(
                channel=1,
                dimension=32,
                energy_ratio=1e-3,
                accuracy_cost_bound=1e-3,
                information_theoretic_hiding_bits=None,
                bound_source="rho",
            ),
            sources={"rho": 1e-3, "margin_delta_k": 0.0},
            mode="exact",
        )


def test_exact_attention_with_rho_noise_is_labeled_as_compromise() -> None:
    """The accepted deployment compromise must not be mislabeled exact."""

    budget = DualChannelBudget.from_config(_config(rho=1e-5))
    assert budget.mode == "exact_attention_with_semantic_noise"
    assert budget.manifest()["exact_signal_path"] is False
    # The same rho value with approximate attention keeps the plain label.
    approximate = DualChannelBudget.from_config(
        _config(mode=AttentionMode.TOPK_PRESERVING, rho=1e-3, tau_max=0.03)
    )
    assert approximate.mode == "topk_preserving"
    assert approximate.semantic.energy_ratio == 1e-3


def test_approximate_config_uses_the_dominant_bound_source() -> None:
    budget = DualChannelBudget.from_config(
        _config(
            mode=AttentionMode.TOPK_PRESERVING,
            rho=1e-5,
            tau_max=0.03,
            tau_error=0.10,
        )
    )
    assert budget.semantic.accuracy_cost_bound == 0.10
    assert budget.semantic.bound_source == "margin_delta_k"
    assert budget.sources == {"rho": 1e-5, "margin_delta_k": 0.10}


def test_ledger_rejects_understated_bound() -> None:
    config = _config(
        mode=AttentionMode.TOPK_PRESERVING, rho=1e-5, tau_max=0.03, tau_error=0.10
    )
    budget = DualChannelBudget.from_config(config)
    understated = DualChannelBudget(
        free=budget.free,
        semantic=ChannelBudget(
            channel=1,
            dimension=32,
            energy_ratio=1e-5,
            accuracy_cost_bound=0.001,
            information_theoretic_hiding_bits=None,
            bound_source="rho",
        ),
        sources={"rho": 1e-5, "margin_delta_k": 0.10},
        mode="topk_preserving",
    )
    with pytest.raises(ValueError, match="does not cover"):
        understated.assert_consistent_with(config)


def test_budget_dimension_matches_free_noise_basis() -> None:
    config = _config()
    basis = _basis(noise_dim=8)
    budget = DualChannelBudget.from_config(config, basis=basis)
    assert budget.free.dimension == basis.free_noise_basis().shape[0] == 8
    budget.assert_consistent_with(config, basis=basis)

    mismatched = _basis(noise_dim=16, seed=12)
    with pytest.raises(ValueError, match="dim ker"):
        budget.assert_consistent_with(config, basis=mismatched)


def test_semantic_hiding_bits_matches_reference_table() -> None:
    # Table from docs/noise_placement_explained.pdf (T9): sigma_delta/sigma_h
    # -> bits retained per coordinate.
    assert semantic_hiding_bits(1e-5) == pytest.approx(16.61, abs=0.01)
    assert semantic_hiding_bits(0.577) == pytest.approx(1.00, abs=0.01)
    assert semantic_hiding_bits(1.0) == pytest.approx(0.5, abs=1e-12)
    assert semantic_hiding_bits(0.0) == math.inf
    with pytest.raises(ValueError):
        semantic_hiding_bits(-1.0)


def test_manifest_round_trips() -> None:
    budget = DualChannelBudget.from_config(_config())
    manifest = budget.manifest()
    assert manifest["schema"] == "fastprove.noise_budget.v1"
    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["free"]["information_theoretic_hiding_bits"] == 0.0
    assert "not a privacy source" in manifest["free"]["statement"]
    assert manifest["semantic"]["bound_source"] == "rho"


def test_conversion_writes_noise_budget_next_to_runtime_config(tmp_path) -> None:
    config = _config()
    plain = PlainTinyCausalLM(config.model, seed=1, debug_enabled=False).eval()
    server = tmp_path / "server"
    client = tmp_path / "client"
    ModelConverter().convert(
        plain, config, client_secret_dir=client, server_model_dir=server
    )
    budget = json.loads((server / "noise_budget.json").read_text(encoding="utf-8"))
    manifest = json.loads((server / "manifest.json").read_text(encoding="utf-8"))
    assert budget["free"]["dimension"] == config.obfuscation.hidden_noise_dim
    assert budget["semantic"]["energy_ratio"] == config.obfuscation.rho_noise_bound
    assert manifest["noise_budget_sha256"]
