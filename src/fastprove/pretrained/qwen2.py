"""Offline Qwen2 checkpoint adapter for the fastProve reference path.

The adapter deliberately uses ``safetensors.safe_open`` instead of loading a
second Hugging Face model.  It streams one tensor at a time into the existing
Llama-like reference module, which keeps the 1.5B checkpoint usable on the
16-GiB MPS/CPU host.  It is a weight-layout adapter, not a new model
implementation: all inference still goes through ``PlainTinyCausalLM`` and
``ObfuscatedTinyCausalLM``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch

from ..config import ModelConfig
from ..models.plain import PlainTinyCausalLM


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(root: Path, name: str) -> Path:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError("Qwen2 artifact is missing %s" % path)
    return path


@dataclass(frozen=True)
class Qwen2Artifact:
    """Immutable local-checkpoint manifest and mapped model dimensions."""

    root: str
    model_config: ModelConfig
    config_sha256: str
    tokenizer_sha256: str
    tokenizer_config_sha256: str
    weights_sha256: str
    weights_bytes: int
    upstream_revision: str
    architecture: str
    max_position_embeddings: int

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "architecture": self.architecture,
            "model_config": asdict(self.model_config),
            "config_sha256": self.config_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "tokenizer_config_sha256": self.tokenizer_config_sha256,
            "weights_sha256": self.weights_sha256,
            "weights_bytes": self.weights_bytes,
            "upstream_revision": self.upstream_revision,
            "max_position_embeddings": self.max_position_embeddings,
        }


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object: %s" % path)
    return value


def _qwen_model_config(raw: Mapping[str, object], *, max_sequence_length: int) -> ModelConfig:
    required = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "rms_norm_eps",
        "rope_theta",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError("Qwen2 config missing fields: %s" % missing)
    max_position = int(raw["max_position_embeddings"])
    if max_sequence_length < 1 or max_sequence_length > max_position:
        raise ValueError(
            "requested sequence length %d is outside Qwen2 context %d"
            % (max_sequence_length, max_position)
        )
    return ModelConfig(
        vocab_size=int(raw["vocab_size"]),
        hidden_size=int(raw["hidden_size"]),
        intermediate_size=int(raw["intermediate_size"]),
        num_layers=int(raw["num_hidden_layers"]),
        num_attention_heads=int(raw["num_attention_heads"]),
        num_key_value_heads=int(raw["num_key_value_heads"]),
        max_sequence_length=int(max_sequence_length),
        rms_epsilon=float(raw["rms_norm_eps"]),
        rope_theta=float(raw["rope_theta"]),
        qkv_bias=True,
    )


def _safetensor_keys(weights: Path) -> Tuple[str, ...]:
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "Qwen2 loading requires safetensors; install the project optional "
            "pretrained dependencies"
        ) from error
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        return tuple(handle.keys())


def load_qwen2_artifact(
    root: str | Path,
    *,
    max_sequence_length: int,
    compute_hashes: bool = True,
) -> Qwen2Artifact:
    """Validate a local Qwen2 directory without allocating model weights."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError("Qwen2 artifact directory does not exist: %s" % root_path)
    config_path = _require_file(root_path, "config.json")
    tokenizer_path = _require_file(root_path, "tokenizer.json")
    tokenizer_config_path = _require_file(root_path, "tokenizer_config.json")
    weights_path = _require_file(root_path, "model.safetensors")
    raw = _load_json(config_path)
    architecture = str((raw.get("architectures") or [""])[0])
    if raw.get("model_type") != "qwen2" or architecture != "Qwen2ForCausalLM":
        raise ValueError(
            "expected Qwen2ForCausalLM artifact, got model_type=%r architecture=%r"
            % (raw.get("model_type"), architecture)
        )
    keys = set(_safetensor_keys(weights_path))
    for layer in range(int(raw["num_hidden_layers"])):
        for suffix in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ):
            expected = "model.layers.%d.%s" % (layer, suffix)
            if expected not in keys:
                raise ValueError("Qwen2 checkpoint missing %s" % expected)
    for expected in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        if expected not in keys:
            raise ValueError("Qwen2 checkpoint missing %s" % expected)
    model_config = _qwen_model_config(
        raw, max_sequence_length=int(max_sequence_length)
    )
    upstream_revision = str(raw.get("_commit_hash", "revision_unavailable"))
    if compute_hashes:
        config_sha256 = _sha256_file(config_path)
        tokenizer_sha256 = _sha256_file(tokenizer_path)
        tokenizer_config_sha256 = _sha256_file(tokenizer_config_path)
        weights_sha256 = _sha256_file(weights_path)
    else:
        config_sha256 = tokenizer_sha256 = tokenizer_config_sha256 = "not_computed"
        weights_sha256 = "not_computed"
    return Qwen2Artifact(
        root=str(root_path),
        model_config=model_config,
        config_sha256=config_sha256,
        tokenizer_sha256=tokenizer_sha256,
        tokenizer_config_sha256=tokenizer_config_sha256,
        weights_sha256=weights_sha256,
        weights_bytes=weights_path.stat().st_size,
        upstream_revision=upstream_revision,
        architecture=architecture,
        max_position_embeddings=int(raw["max_position_embeddings"]),
    )


def _target_tensors(model: PlainTinyCausalLM) -> Dict[str, torch.Tensor]:
    return {
        **dict(model.named_parameters()),
        **dict(model.named_buffers()),
    }


def _copy_tensor(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    source_name: str,
) -> None:
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError(
            "%s shape mismatch: checkpoint=%s target=%s"
            % (source_name, tuple(source.shape), tuple(target.shape))
        )
    if not source.is_floating_point() or not target.is_floating_point():
        raise ValueError("%s must be floating point" % source_name)
    with torch.no_grad():
        target.copy_(source.to(device=target.device, dtype=target.dtype))


def _qwen_mapping(config: ModelConfig) -> Dict[str, str]:
    mapping = {
        "model.embed_tokens.weight": "embedding.weight",
        "model.norm.weight": "final_norm_weight",
        "lm_head.weight": "lm_head.weight",
    }
    for layer in range(config.num_layers):
        prefix = "model.layers.%d." % layer
        target = "blocks.%d." % layer
        mapping.update(
            {
                prefix + "input_layernorm.weight": target + "attention_norm_weight",
                prefix + "post_attention_layernorm.weight": target + "ffn_norm_weight",
                prefix + "self_attn.q_proj.weight": target + "q_proj.weight",
                prefix + "self_attn.k_proj.weight": target + "k_proj.weight",
                prefix + "self_attn.v_proj.weight": target + "v_proj.weight",
                prefix + "self_attn.o_proj.weight": target + "o_proj.weight",
                prefix + "self_attn.q_proj.bias": target + "q_proj.bias",
                prefix + "self_attn.k_proj.bias": target + "k_proj.bias",
                prefix + "self_attn.v_proj.bias": target + "v_proj.bias",
                prefix + "mlp.gate_proj.weight": target + "gate_proj.weight",
                prefix + "mlp.up_proj.weight": target + "up_proj.weight",
                prefix + "mlp.down_proj.weight": target + "down_proj.weight",
            }
        )
    return mapping


def load_qwen2_plain(
    artifact: Qwen2Artifact,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
    debug_enabled: bool = False,
) -> PlainTinyCausalLM:
    """Stream a validated Qwen2 checkpoint into the plaintext reference model."""

    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Qwen2 adapter supports FP32 or BF16 only")
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise RuntimeError("Qwen2 loading requires safetensors") from error
    weights_path = Path(artifact.root) / "model.safetensors"
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            model = PlainTinyCausalLM(
                artifact.model_config, seed=seed, debug_enabled=debug_enabled
            )
    finally:
        torch.set_default_dtype(old_default_dtype)
    model = model.to_empty(device=device)
    # ``to_empty`` intentionally leaves non-parameter buffers uninitialised;
    # restore deterministic structural buffers that are not present in the
    # checkpoint state dict.
    for block in model.blocks:
        block.kv_index.copy_(
            torch.arange(
                artifact.model_config.num_attention_heads,
                device=block.kv_index.device,
            )
            // artifact.model_config.query_heads_per_kv_head
        )
    model.eval()
    targets = _target_tensors(model)
    mapping = _qwen_mapping(artifact.model_config)
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        for source_name, target_name in mapping.items():
            if source_name not in handle.keys():
                raise ValueError("Qwen2 checkpoint missing %s" % source_name)
            if target_name not in targets:
                raise ValueError("fastProve target missing %s" % target_name)
            _copy_tensor(
                handle.get_tensor(source_name),
                targets[target_name],
                source_name=source_name,
            )
    return model


def load_qwen2_tokenizer(root: str | Path):
    """Load the local tokenizer without any network fallback."""

    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise RuntimeError("Qwen2 tokenizer loading requires transformers") from error
    return AutoTokenizer.from_pretrained(
        str(Path(root).expanduser().resolve()),
        local_files_only=True,
        use_fast=True,
    )
