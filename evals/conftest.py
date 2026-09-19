"""Shared pytest fixtures for the evals suite.

Fixture bodies live in ``tests/_fixtures.py`` so ``tests/conftest.py`` and this
module expose the same builders without duplication.  Importing that module is
safe under pytest because ``pythonpath = [".", "src"]`` puts the repository
root on ``sys.path`` before collection.
"""

from __future__ import annotations

from typing import List, Tuple

import pytest
import torch

from fastprove.config import PrototypeConfig

from evals.conditions import ConditionId, ConditionSpec
from evals.keys import MasterKey
from evals.model_factory import EvalModels
from tests import _fixtures


@pytest.fixture(scope="session")
def base_config() -> PrototypeConfig:
    return _fixtures.build_base_config()


@pytest.fixture(scope="session")
def master_keys(base_config: PrototypeConfig) -> List[MasterKey]:
    return _fixtures.build_master_keys(base_config)


@pytest.fixture
def synthetic_batch(base_config: PrototypeConfig) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    return _fixtures.build_synthetic_batch(base_config)


@pytest.fixture
def condition_p2() -> ConditionSpec:
    return _fixtures.build_condition(ConditionId.P2)


@pytest.fixture
def models_p2(
    base_config: PrototypeConfig,
    master_keys: List[MasterKey],
    condition_p2: ConditionSpec,
) -> EvalModels:
    return _fixtures.build_condition_models(
        base_config, condition_p2, master_keys
    )


@pytest.fixture
def models_f1(
    base_config: PrototypeConfig, master_keys: List[MasterKey]
) -> EvalModels:
    return _fixtures.build_condition_models(
        base_config, _fixtures.build_condition(ConditionId.F1), master_keys
    )
