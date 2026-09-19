"""Decision ⑧ acceptance gate: production MoE must not expose router logits.

``tests/layers/test_moe.py`` checks routing *correctness* through the
explicitly gated debug API; this module checks the *exposure boundary* of the
production API.  AGENTS.md requires that production-mode outputs never expose
decoded hidden states, noise states, attention probabilities, or router
logits, so an honest-but-curious observer of the production path must receive
only the mixed increment.

Grounding: ``ObfuscatedMoE.forward`` returns a bare ``MixedState`` and routes
through ``_route`` with ``debug_enabled=False``; ``forward_debug`` is the only
API that returns ``MoEDebug`` and it raises ``PermissionError`` when debug is
disabled (``src/fastprove/layers/moe.py``).
"""

from __future__ import annotations

import pytest

from fastprove.layers.moe import ObfuscatedMoE
from fastprove.seed import RequestContext
from fastprove.state import MixedState


def test_production_moe_forward_returns_only_the_mixed_increment(
    production_moe,
) -> None:
    module, mixed = production_moe
    assert isinstance(module, ObfuscatedMoE)
    assert module.debug_enabled is False

    output = module(mixed, RequestContext(3, "moe-exposure"))

    assert isinstance(output, MixedState)
    assert output.mixed.shape == mixed.shape
    for attribute in (
        "router_debug",
        "router_logits",
        "gate_weights",
        "physical_expert_indices",
        "probabilities",
    ):
        assert not hasattr(output, attribute), attribute


def test_production_moe_debug_api_raises_permission_error(production_moe) -> None:
    module, mixed = production_moe

    with pytest.raises(PermissionError, match="debug"):
        module.forward_debug(mixed, RequestContext(3, "moe-exposure"))
