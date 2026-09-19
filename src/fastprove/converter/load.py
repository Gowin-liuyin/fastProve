"""Server-bundle loading: rebuild a runnable module from deployed artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..augmented_noise import deployed_free_noise_sigma
from ..config import ModelConfig, ObfuscationConfig
from ..layers.attention import (
    ApproximationConfig,
    AttentionMode,
    ObfuscatedAttention,
)
from ..layers.embedding import SecureEmbedding
from ..models.obfuscated import (
    ObfuscatedDecoderBlock,
    ObfuscatedTinyCausalLM,
    _rms_scale_from_factors,
)
from ..transforms import BasisDescriptor


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _descend(module: nn.Module, path: str) -> nn.Module:
    current = module
    for part in path.split("."):
        current = getattr(current, part)
    return current


def _restore_block(
    runtime: Dict[str, Any], layer_index: int
) -> ObfuscatedDecoderBlock:
    block = ObfuscatedDecoderBlock.__new__(ObfuscatedDecoderBlock)
    nn.Module.__init__(block)
    block.config = _runtime_model_config(runtime)
    block.obfuscation = _runtime_obfuscation(runtime)
    block.mode = AttentionMode(runtime["mode"])
    block.layer_id = layer_index
    block.debug_enabled = False
    block.noise_injection_enabled = True
    block.refresh_noise_scale = 1.0
    # Same derivation as ``ObfuscatedDecoderBlock.__init__``; the server bundle
    # records the scale in runtime_config.json, so this is FP-identical.
    block._free_noise_sigma = float(
        deployed_free_noise_sigma(
            block.obfuscation.free_noise_energy_ratio,
            block.obfuscation.hidden_noise_dim,
        )
    )
    block.hidden_basis = BasisDescriptor(**runtime["hidden_basis"])
    block.attention = ObfuscatedAttention(
        mode=block.mode,
        approximation=_runtime_approximation(runtime),
        layer_id="block-%d-attention" % layer_index,
        debug_enabled=False,
    )
    return block


def _runtime_model_config(runtime: Dict[str, Any]):
    return ModelConfig(**runtime["model"])


def _runtime_obfuscation(runtime: Dict[str, Any]):
    return ObfuscationConfig(**runtime["obfuscation"])


def _runtime_approximation(
    runtime: Dict[str, Any],
) -> Optional[ApproximationConfig]:
    if runtime.get("approximation") is None:
        return None
    return ApproximationConfig(**runtime["approximation"])


def load_server_model(server_model_dir: str | Path) -> ObfuscatedTinyCausalLM:
    """Rebuild the runnable server module from the server bundle alone."""

    server_dir = Path(server_model_dir)
    runtime = json.loads(
        (server_dir / "runtime_config.json").read_text(encoding="utf-8")
    )
    weights_path = server_dir / ("model.pt" if (server_dir / "model.pt").exists()
                                  else "model.safetensors")
    if weights_path.suffix == ".safetensors":
        import safetensors.torch  # type: ignore

        state = dict(safetensors.torch.load_file(str(weights_path)))
    else:
        state = torch.load(weights_path, weights_only=True)

    config = _runtime_model_config(runtime)
    module = ObfuscatedTinyCausalLM.__new__(ObfuscatedTinyCausalLM)
    nn.Module.__init__(module)
    module.config = config
    module.obfuscation = _runtime_obfuscation(runtime)
    module.mode = AttentionMode(runtime["mode"])
    module.debug_enabled = False
    module.noise_injection_enabled = True
    module.initial_refresh_noise_scale = 1.0
    # Derived exactly as in ``__init__`` from the recorded config; the server
    # bundle alone is sufficient, no client material is read.
    module._free_noise_sigma = float(
        deployed_free_noise_sigma(
            module.obfuscation.free_noise_energy_ratio,
            module.obfuscation.hidden_noise_dim,
        )
    )
    module.hidden_basis = BasisDescriptor(**runtime["hidden_basis"])
    module.value_basis_condition_numbers = tuple(
        tuple(row) for row in runtime["value_basis_condition_numbers"]
    )
    module.value_basis_fingerprints = tuple(
        tuple(row) for row in runtime["value_basis_fingerprints"]
    )
    module.embedding = SecureEmbedding.__new__(SecureEmbedding)
    nn.Module.__init__(module.embedding)
    module.embedding.vocab_size = config.vocab_size
    module.embedding.basis_descriptor = module.hidden_basis
    module.blocks = nn.ModuleList(
        _restore_block(runtime, index)
        for index in range(config.num_layers)
    )
    for name, value in state.items():
        parent_path, _, leaf = name.rpartition(".")
        parent = module if not parent_path else _descend(module, parent_path)
        parent.register_buffer(leaf, value.clone())

    module._rms_scale = _rms_scale_from_factors(
        config.hidden_size,
        module.deployed_gram_blocks,
        module.deployed_gram_perm,
    )
    module._debug_basis_unmix = None
    cache_identity = runtime.get("cache_identity") or []
    for block in module.blocks:
        block._rms_scale = _rms_scale_from_factors(
            config.hidden_size,
            block.deployed_gram_blocks,
            block.deployed_gram_perm,
        )
        block._debug_basis_unmix = None
        if block.layer_id < len(cache_identity):
            block.cache_identity = cache_identity[block.layer_id]
    return module
