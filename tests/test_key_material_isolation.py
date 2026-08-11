"""Server-side state_dict must not contain key material."""

from __future__ import annotations

import torch

from fastprove.config import ModelConfig, ObfuscationConfig
from fastprove.layers.attention import AttentionMode
from fastprove.models.obfuscated import ObfuscatedTinyCausalLM
from fastprove.models.plain import PlainTinyCausalLM

#: Substrings that indicate conversion/client-side key material. A server
#: checkpoint containing any of these would let an operator decode the mixed
#: state directly, which is exactly what the conversion is supposed to withhold.
_FORBIDDEN_SUBSTRINGS = (
    "rotation",
    "common_qk",
    "coupling",
    "propagator",
    "refresh",
    "gram",
    "perm_in",
    "perm_out",
    "scales",
    "blocks.0.blocks",
    "inverse",
)


def _converted():
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_sequence_length=16,
    )
    obfuscation = ObfuscationConfig(
        hidden_noise_dim=8,
        value_noise_dim_per_head=2,
        max_condition_number=10.0,
        noise_propagation_gamma=0.5,
        refresh_mode="per_request",
        basis_block_size=8,
    )
    plain = PlainTinyCausalLM(config, seed=1, debug_enabled=False)
    return ObfuscatedTinyCausalLM.from_plain(
        plain,
        obfuscation=obfuscation,
        mode=AttentionMode.EXACT,
        approximation=None,
        seed=1,
        debug_enabled=False,
    ).module


def _is_deployed_artifact(key: str) -> bool:
    """Deployed weights are the shipped server artifact, not raw key material.

    ``deployed_*`` weights are produced by the offline fusion
    (``layers/deployed.py``); they deliberately contain the ``(G-I) M_bot``
    and ``N`` factors whose recoverability consequence is recorded in
    ``docs/threat_model.md`` (B1.3). They must ship, so the raw-key-material
    substring check below skips them.
    """

    return key.rsplit(".", 1)[-1].startswith("deployed_")


def test_state_dict_excludes_key_material() -> None:
    keys = [
        key
        for key in _converted().state_dict().keys()
        if not _is_deployed_artifact(key)
    ]
    offending = [
        key
        for key in keys
        for token in _FORBIDDEN_SUBSTRINGS
        if token in key.lower()
    ]
    assert offending == [], "key material leaked into state_dict: %s" % offending


def test_state_dict_still_contains_deployed_weights() -> None:
    keys = " ".join(_converted().state_dict().keys())
    for required in ("embedding.table", "deployed_head", "kv_index"):
        assert required in keys


def test_key_material_is_still_reachable_as_a_runtime_buffer() -> None:
    """Non-persistent buffers must still exist for the forward pass."""

    module = _converted()
    names = dict(module.named_buffers())
    for required in (
        "blocks.0.common_qk",
        "blocks.0.attention_noise_coupling",
    ):
        assert required in names


def test_saved_and_reloaded_state_dict_round_trips(tmp_path) -> None:
    module = _converted()
    path = tmp_path / "server.pt"
    torch.save(module.state_dict(), path)
    reloaded = torch.load(path, weights_only=True)
    assert set(reloaded.keys()) == set(module.state_dict().keys())


def test_reloaded_server_checkpoint_cannot_reconstruct_the_basis() -> None:
    """A server checkpoint alone must not carry any basis factor."""

    state = _converted().state_dict()
    for key, value in state.items():
        assert "gram" not in key
        assert "perm" not in key or "kv_index" in key
        assert isinstance(value, torch.Tensor)
