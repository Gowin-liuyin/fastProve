"""Deterministic object builders shared by ``tests/`` and ``evals/`` fixtures.

The pytest fixture declarations live in ``tests/conftest.py`` and
``evals/conftest.py``; both import the builders below so the fixture bodies are
defined exactly once.  This module must stay importable without pytest so that
``evals/conftest.py`` can reuse it during collection.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

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


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_base_config() -> PrototypeConfig:
    path = project_root() / "configs" / "tiny_exact.yaml"
    if path.is_file():
        return load_config(path)
    return default_base_prototype_config()


def build_master_keys(
    base_config: PrototypeConfig, *, count: int = 3
) -> List[MasterKey]:
    return generate_master_keys(
        count=count, base_seed=base_config.runtime.seed
    )


def build_synthetic_batch(
    base_config: PrototypeConfig,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    tokens, identifiers = make_synthetic_token_batch(
        sample_count=base_config.evaluation.sample_count,
        sequence_length=base_config.evaluation.sequence_length,
        vocab_size=base_config.model.vocab_size,
        seed=base_config.runtime.seed,
    )
    mask = torch.ones_like(tokens, dtype=torch.bool)
    return tokens, mask, identifiers


def build_condition(condition_id: ConditionId) -> ConditionSpec:
    return get_condition(condition_id)


def build_condition_models(
    base_config: PrototypeConfig,
    condition: ConditionSpec,
    master_keys: List[MasterKey],
) -> EvalModels:
    return build_models(
        base_config, condition, key=master_keys[0], debug_enabled=True
    )
