"""Shared pytest fixtures for the fastprove test suite.

Object builders live in ``tests/_fixtures.py`` so ``evals/conftest.py`` can
expose the same fixtures without duplicating bodies.  This file stays at the
top level of ``tests/`` so its fixtures apply to every subdirectory
(``math/``, ``layers/``, ``models/``, ``deployment/``, ``security/``,
``evaluation/``, ``cli/``, ``pretrained/``).

Fixture policy: parametrized builders stay plain callables in
``tests/_fixtures.py`` (tests import and call them), while pytest fixtures are
declared here only for the shared objects that are convenient to inject.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pytest
import torch

from fastprove.config import PrototypeConfig
from fastprove.layers.moe import ObfuscatedMoE

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


@pytest.fixture(scope="session")
def project_root() -> Path:
    return _fixtures.project_root()


@pytest.fixture
def production_moe() -> Tuple[ObfuscatedMoE, torch.Tensor]:
    """Production-mode (``debug_enabled=False``) MoE plus its mixed input."""

    return _fixtures.production_moe()
