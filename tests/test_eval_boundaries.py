from __future__ import annotations

from pathlib import Path

import torch

from evals.conditions import ConditionId, get_condition
from evals.keys import generate_master_keys
from evals.model_factory import build_models
from fastprove.config import load_config


def test_public_key_record_does_not_contain_seed_material() -> None:
    record = generate_master_keys(count=1, base_seed=123)[0].to_dict()
    assert record["seed_material_recorded"] is False
    assert len(record["key_fingerprint"]) == 64
    assert "master_seed" not in record
    assert "conversion_seed" not in record


def test_bf16_condition_casts_models_and_conversion_buffers() -> None:
    config = load_config(Path("configs/tiny_exact.yaml"))
    models = build_models(
        config,
        get_condition(ConditionId.P2),
        key=generate_master_keys(count=1, base_seed=config.runtime.seed)[0],
        debug_enabled=True,
    )
    assert models.dtype == torch.bfloat16
    assert models.plain.embedding.weight.dtype == torch.bfloat16
    assert models.obfuscated is not None
    assert models.obfuscated.embedding_weight.dtype == torch.bfloat16
    assert models.obfuscated.blocks[0].common_qk.dtype == torch.bfloat16
    assert models.obfuscation_manifest is not None
    assert models.obfuscation_manifest["hidden_basis"]["condition_number"] <= 10.0
    assert all(
        item["condition_number"] <= 10.0
        for item in models.obfuscation_manifest["value_bases"]
    )
