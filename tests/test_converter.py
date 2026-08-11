"""ModelConverter: client/server bundle separation (task C4)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from fastprove.config import ModelConfig, ObfuscationConfig, PrototypeConfig
from fastprove.converter import ModelConverter, load_client, load_server_model
from fastprove.layers.attention import AttentionMode
from fastprove.models.plain import PlainTinyCausalLM
from fastprove.seed import RequestContext


def _config() -> PrototypeConfig:
    from fastprove.config import (
        AttentionConfig,
        EvaluationConfig,
        RuntimeConfig,
    )

    return PrototypeConfig(
        model=ModelConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_sequence_length=32,
        ),
        obfuscation=ObfuscationConfig(
            hidden_noise_dim=8,
            value_noise_dim_per_head=2,
            max_condition_number=10.0,
            noise_propagation_gamma=0.5,
            refresh_mode="per_request",
            basis_block_size=8,
        ),
        attention=AttentionConfig(
            mode=AttentionMode.EXACT,
            tau_max=0.0,
            tau_error=0.0,
            alpha=0.8,
            preserve_top_k=4,
        ),
        runtime=RuntimeConfig(
            seed=7,
            request_id="converter",
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


def _convert(tmp_path: Path):
    config = _config()
    plain = PlainTinyCausalLM(config.model, seed=1, debug_enabled=False).eval()
    server = tmp_path / "server"
    client = tmp_path / "client"
    manifest = ModelConverter().convert(
        plain,
        config,
        client_secret_dir=client,
        server_model_dir=server,
    )
    return plain, config, manifest, server, client


def test_server_bundle_contains_no_key_material(tmp_path) -> None:
    _, _, _, server, _ = _convert(tmp_path)
    manifest = json.loads(
        (server / "manifest.json").read_text(encoding="utf-8")
    )
    for key in manifest["server_state_keys"]:
        lowered = key.lower()
        for token in (
            "perm_in",
            "perm_out",
            "scales",
            "inverse",
            "tau",
            "token_codec",
            "rotation",
            "propagator",
        ):
            assert token not in lowered, "%s leaked into the server bundle" % key


def test_server_model_runs_without_the_client_bundle(tmp_path) -> None:
    plain, config, _, server, client = _convert(tmp_path)
    import shutil

    shutil.rmtree(client)  # client bundle deleted: server must still run
    module = load_server_model(server)
    module.eval()
    codec = plain  # placeholder to satisfy type checkers
    del codec
    runtime = json.loads(
        (server / "runtime_config.json").read_text(encoding="utf-8")
    )
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        logits = module(
            ids,
            request_context=RequestContext(3, "server-alone"),
        )
    assert logits.shape == (1, 4, config.model.vocab_size)
    assert torch.isfinite(logits).all()
    assert runtime["server_visible_key_material"]["common_qk"]


def test_client_bundle_can_decode_server_output(tmp_path) -> None:
    plain, _, _, server, client = _convert(tmp_path)
    module = load_server_model(server)
    module.eval()
    bundle = load_client(client)
    ids = torch.tensor([[2, 4, 6, 8]])
    with torch.no_grad():
        plain_logits = plain(ids)
        obf_logits = module(
            bundle.encode(ids),
            request_context=RequestContext(5, "client-decode"),
        )
    decoded = bundle.decode_logits(obf_logits)
    torch.testing.assert_close(decoded, plain_logits, atol=2e-4, rtol=2e-4)


def test_conversion_rejects_a_mismatched_architecture(tmp_path) -> None:
    # ModelConfig already rejects odd head_dim, so bypass its validation to
    # exercise the converter's own defense-in-depth check.
    config = _config()
    object.__setattr__(config.model, "hidden_size", 30)  # 30 % 2 == 0
    object.__setattr__(config.model, "num_attention_heads", 2)  # head_dim 15
    object.__setattr__(config.model, "num_key_value_heads", 1)
    plain = PlainTinyCausalLM(config.model, seed=1, debug_enabled=False).eval()
    import pytest

    with pytest.raises(ValueError, match="even for RoPE"):
        ModelConverter().convert(
            plain,
            config,
            client_secret_dir=tmp_path / "c",
            server_model_dir=tmp_path / "s",
        )


def test_manifest_fingerprints_cross_validate(tmp_path) -> None:
    import hashlib

    plain, config, manifest, server, client = _convert(tmp_path)
    server_manifest = json.loads(
        (server / "manifest.json").read_text(encoding="utf-8")
    )
    client_manifest = json.loads(
        (client / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.runtime_config_sha256 == server_manifest[
        "runtime_config_sha256"
    ]
    assert client_manifest["server_runtime_config_sha256"] == manifest.runtime_config_sha256
    digest = hashlib.sha256()
    digest.update((client / "manifest.json").read_bytes())
    assert manifest.client_manifest_sha256 == digest.hexdigest()
    assert client_manifest["token_codec_fingerprint"]
    assert client_manifest["hidden_basis_fingerprint"]


def test_converted_model_matches_plaintext_end_to_end(tmp_path) -> None:
    plain, config, _, server, client = _convert(tmp_path)
    module = load_server_model(server)
    module.eval()
    bundle = load_client(client)
    ids = torch.tensor([[1, 3, 5, 7]])
    with torch.no_grad():
        plain_gen = plain.generate_greedy(ids, max_new_tokens=2)
        obf_gen = bundle.decode(
            module.generate_greedy(
                bundle.encode(ids),
                max_new_tokens=2,
                request_context=RequestContext(9, "e2e"),
            )
        )
    assert torch.equal(obf_gen, plain_gen)
