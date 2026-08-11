"""Shared pytest fixtures for the evals suite.

These fixtures load the tiny model, generate independent master keys, and
build condition cells without starting full evaluation on import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
import torch

from fastprove.config import PrototypeConfig, load_config
from fastprove.evaluation.accuracy import make_synthetic_token_batch

from evals.conditions import (
    ConditionId,
    ConditionSpec,
    default_base_prototype_config,
    get_condition,
)
from evals.keys import MasterKey, generate_master_keys
from evals.model_factory import EvalModels, build_models


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def base_config() -> PrototypeConfig:
    path = _project_root() / "configs" / "tiny_exact.yaml"
    if path.is_file():
        return load_config(path)
    return default_base_prototype_config()


@pytest.fixture(scope="session")
def master_keys(base_config: PrototypeConfig) -> list[MasterKey]:
    return generate_master_keys(count=3, base_seed=base_config.runtime.seed)


@pytest.fixture
def synthetic_batch(base_config: PrototypeConfig):
    tokens, ids = make_synthetic_token_batch(
        sample_count=base_config.evaluation.sample_count,
        sequence_length=base_config.evaluation.sequence_length,
        vocab_size=base_config.model.vocab_size,
        seed=base_config.runtime.seed,
    )
    mask = torch.ones_like(tokens, dtype=torch.bool)
    return tokens, mask, ids


@pytest.fixture
def condition_p2() -> ConditionSpec:
    return get_condition(ConditionId.P2)


@pytest.fixture
def models_p2(
    base_config: PrototypeConfig,
    master_keys: list[MasterKey],
    condition_p2: ConditionSpec,
) -> EvalModels:
    return build_models(
        base_config, condition_p2, key=master_keys[0], debug_enabled=True
    )


@pytest.fixture
def models_f1(
    base_config: PrototypeConfig, master_keys: list[MasterKey]
) -> EvalModels:
    return build_models(
        base_config,
        get_condition(ConditionId.F1),
        key=master_keys[0],
        debug_enabled=True,
    )
