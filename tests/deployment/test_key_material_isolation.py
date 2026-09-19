"""Server-side state_dict must not contain key material."""

from __future__ import annotations

import torch

from tests._fixtures import model_config, obfuscation_config, tiny_lm_pair

#: Substrings that indicate conversion/client-side key material. A server
#: checkpoint containing any of these would let an operator decode the mixed
#: state directly, which is exactly what the conversion is supposed to withhold.
#:
#: ``common_qk`` is deliberately absent from this list: it is applied after
#: RoPE and cannot be absorbed into a deployed weight, so the server must hold
#: it (task C4 reclassifies it as server-visible key material and the threat
#: model records that the QK geometry is not protected).
_FORBIDDEN_SUBSTRINGS = (
    "rotation",
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
    config = model_config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        max_sequence_length=16,
    )
    _, converted = tiny_lm_pair(
        config,
        plain_seed=1,
        conversion_seed=1,
        debug_enabled=False,
        obfuscation=obfuscation_config(refresh_mode="per_request"),
    )
    return converted.module


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
    """A server checkpoint alone must not carry any basis factor.

    ``deployed_gram_blocks`` / ``deployed_gram_perm`` are the block-diagonal
    ``A_gram = P P.T`` artifacts the server needs for the ``rho`` statistic;
    their noise-subspace content is already implied by the shipped
    ``noise_read`` (B1.3), so they are allowed. Anything that would decode
    the state (M, M^-1, P, N, scales, permutations) stays forbidden.
    """

    state = _converted().state_dict()
    for key, value in state.items():
        if key.rsplit(".", 1)[-1].startswith("deployed_gram"):
            continue
        assert "gram" not in key
        assert "perm" not in key or "kv_index" in key
        assert isinstance(value, torch.Tensor)
