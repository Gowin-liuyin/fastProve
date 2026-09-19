"""Conversion writer: server model directory plus client secret bundle."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import torch
import torch.nn as nn

from .basis_io import (
    _basis_factors,
    _fastprove_version,
    _rebuild_hidden_basis,
    _rebuild_value_bases,
)
from ..codec import generate_token_codec
from ..config import PrototypeConfig
from .hygiene import (
    ConversionManifest,
    _assert_server_bundle_is_clean,
    _require_supported_architecture,
)
from ..layers.attention import ApproximationConfig, AttentionMode
from .load import _sha256_file
from ..models.obfuscated import ObfuscatedTinyCausalLM
from ..noise_budget import DualChannelBudget


class ModelConverter:
    """Write a server model and a separate client secret bundle."""

    def convert(
        self,
        plain_model: nn.Module,
        config: PrototypeConfig,
        client_secret_dir: str | Path,
        server_model_dir: str | Path,
    ) -> ConversionManifest:
        """Write a server model and a separate client secret bundle."""

        started = time.perf_counter()
        plain_model.eval()
        _require_supported_architecture(plain_model)
        model = config.model
        if model.hidden_size % model.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )
        if model.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        total = model.hidden_size + config.obfuscation.hidden_noise_dim
        if total % config.obfuscation.basis_block_size != 0:
            raise ValueError(
                "basis_block_size must divide hidden_size + hidden_noise_dim"
            )
        tied_embedding = plain_model.lm_head.weight is plain_model.embedding.weight
        if not tied_embedding:
            tied_embedding = torch.equal(
                plain_model.lm_head.weight.detach().cpu(),
                plain_model.embedding.weight.detach().cpu(),
            )

        approximation = None
        if config.attention.mode in (
            AttentionMode.TOPK_PRESERVING,
            AttentionMode.FREE_BOUNDED,
        ):
            approximation = ApproximationConfig(
                tau_max=config.attention.tau_max,
                tau_error=config.attention.tau_error,
                alpha=config.attention.alpha,
                preserve_top_k=config.attention.preserve_top_k,
            )
        token_codec = generate_token_codec(
            model.vocab_size,
            seed=config.runtime.seed,
            domain="tiny-lm-vocab",
        )
        converted = ObfuscatedTinyCausalLM.from_plain(
            plain_model,
            obfuscation=config.obfuscation,
            mode=config.attention.mode,
            approximation=approximation,
            seed=config.runtime.seed,
            debug_enabled=False,
            token_codec=token_codec,
        )
        module = converted.module.eval()

        server_dir = Path(server_model_dir)
        secret_dir = Path(client_secret_dir)
        server_dir.mkdir(parents=True, exist_ok=True)
        secret_dir.mkdir(parents=True, exist_ok=True)

        state = module.state_dict()
        _assert_server_bundle_is_clean(state)
        hidden_basis = _rebuild_hidden_basis(config, module)
        # S2: the dual-channel ledger must agree with the configuration before
        # any artifact is written; a mismatch fails conversion here.
        noise_budget = DualChannelBudget.from_config(config, basis=hidden_basis)
        noise_budget.assert_consistent_with(config, basis=hidden_basis)
        weights_format = "torch"
        try:
            import safetensors.torch  # type: ignore

            safetensors.torch.save_file(state, server_dir / "model.safetensors")
            weights_format = "safetensors"
        except ImportError:
            torch.save(state, server_dir / "model.pt")

        runtime = {
            "schema": "fastprove.server.runtime.v1",
            "mode": str(module.mode.value)
            if hasattr(module.mode, "value")
            else str(module.mode),
            "approximation": (
                None
                if approximation is None
                else {
                    "tau_max": approximation.tau_max,
                    "tau_error": approximation.tau_error,
                    "alpha": approximation.alpha,
                    "preserve_top_k": approximation.preserve_top_k,
                }
            ),
            "model": {
                "vocab_size": model.vocab_size,
                "hidden_size": model.hidden_size,
                "intermediate_size": model.intermediate_size,
                "num_layers": model.num_layers,
                "num_attention_heads": model.num_attention_heads,
                "num_key_value_heads": model.num_key_value_heads,
                "max_sequence_length": model.max_sequence_length,
                "rms_epsilon": model.rms_epsilon,
                "rope_theta": model.rope_theta,
                "qkv_bias": model.qkv_bias,
            },
            "obfuscation": {
                "hidden_noise_dim": config.obfuscation.hidden_noise_dim,
                "value_noise_dim_per_head": (
                    config.obfuscation.value_noise_dim_per_head
                ),
                "max_condition_number": config.obfuscation.max_condition_number,
                "noise_propagation_gamma": (
                    config.obfuscation.noise_propagation_gamma
                ),
                "refresh_mode": config.obfuscation.refresh_mode,
                "basis_block_size": config.obfuscation.basis_block_size,
                "lm_head_mode": config.obfuscation.lm_head_mode,
                "rho_noise_bound": config.obfuscation.rho_noise_bound,
                "free_noise_energy_ratio": (
                    config.obfuscation.free_noise_energy_ratio
                ),
                "signal_noise_coupling_scale": (
                    config.obfuscation.signal_noise_coupling_scale
                ),
            },
            "hidden_basis": {
                "signal_dim": module.hidden_basis.signal_dim,
                "noise_dim": module.hidden_basis.noise_dim,
                "condition_number": module.hidden_basis.condition_number,
                "fingerprint": module.hidden_basis.fingerprint,
            },
            "value_basis_condition_numbers": [
                list(row)
                for row in module.value_basis_condition_numbers
            ],
            "value_basis_fingerprints": [
                list(row) for row in module.value_basis_fingerprints
            ],
            "cache_identity": [block.cache_identity for block in module.blocks],
            "server_visible_key_material": {
                "common_qk": (
                    "applied after RoPE; cannot be absorbed into a deployed "
                    "weight; the server must hold it and the QK geometry is "
                    "not protected (docs/threat_model.md)"
                )
            },
        }
        runtime_path = server_dir / "runtime_config.json"
        runtime_path.write_text(
            json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
        )
        noise_budget_path = server_dir / "noise_budget.json"
        noise_budget_path.write_text(
            json.dumps(noise_budget.manifest(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        server_manifest = {
            "schema": "fastprove.server.manifest.v1",
            "fastprove_version": _fastprove_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "converted_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tied_embedding": tied_embedding,
            "weights_format": weights_format,
            "runtime_config_sha256": _sha256_file(runtime_path),
            "noise_budget_sha256": _sha256_file(noise_budget_path),
            "server_state_keys": list(state.keys()),
        }
        manifest_path = server_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(server_manifest, indent=2) + "\n", encoding="utf-8"
        )

        # -- client bundle -------------------------------------------------
        token_codec.save_client_secret(secret_dir / "token_codec.pt")
        value_bases = _rebuild_value_bases(model, config)
        torch.save(
            {
                "hidden": _basis_factors(hidden_basis),
                "value": [
                    [_basis_factors(basis) for basis in layer]
                    for layer in value_bases
                ],
            },
            secret_dir / "bases.pt",
        )
        client_manifest = {
            "schema": "fastprove.client.manifest.v1",
            "token_codec_fingerprint": token_codec.fingerprint,
            "hidden_basis_fingerprint": hidden_basis.fingerprint,
            "value_basis_fingerprints": [
                [basis.fingerprint for basis in layer]
                for layer in value_bases
            ],
            "server_runtime_config_sha256": _sha256_file(runtime_path),
            "warning": (
                "client material must never be copied into a server directory"
            ),
        }
        (secret_dir / "manifest.json").write_text(
            json.dumps(client_manifest, indent=2) + "\n", encoding="utf-8"
        )

        return ConversionManifest(
            fastprove_version=_fastprove_version(),
            torch_version=torch.__version__,
            platform=platform.platform(),
            converted_at_utc=server_manifest["converted_at_utc"],
            conversion_time_seconds=time.perf_counter() - started,
            server_model_dir=str(server_dir),
            client_secret_dir=str(secret_dir),
            weights_format=weights_format,
            model_sha256={"runtime_config": server_manifest["runtime_config_sha256"]},
            runtime_config_sha256=server_manifest["runtime_config_sha256"],
            client_manifest_sha256=_sha256_file(secret_dir / "manifest.json"),
            tied_embedding=tied_embedding,
            server_state_keys=tuple(state.keys()),
        )
