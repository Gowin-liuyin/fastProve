"""Offline Llama (LlamaForCausalLM) checkpoint adapter for fastProve.

Streams sharded ``safetensors`` into ``PlainTinyCausalLM`` without loading a
second Hugging Face module. Supports tied embeddings (no separate
``lm_head.weight``). This is a weight-layout adapter only: inference still uses
the reference plaintext / obfuscated modules.

**Plaintext verification:** rejects directories whose weight maps contain
legacy ModelSplit / non-Llama obfuscation tensor names. Does **not** load
legacy-obfuscated trees as a base.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from ..config import ModelConfig
from ..models.plain import PlainTinyCausalLM

# Weight-name substrings that indicate a non-plaintext / legacy-obfuscation tree.
_LEGACY_MARKERS: Tuple[str, ...] = (
    "noise",
    "mixed",
    "obfus",
    "augment",
    "propagator",
    "coupling",
    "xi_",
    "perm_voc",
    "vocab_perm",
    "chain_linear",
    "value_mixed",
)


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
        raise FileNotFoundError("Llama artifact is missing %s" % path)
    return path


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object: %s" % path)
    return value


@dataclass(frozen=True)
class LlamaArtifact:
    """Immutable local Llama checkpoint manifest."""

    root: str
    model_config: ModelConfig
    config_sha256: str
    tokenizer_sha256: str
    tokenizer_config_sha256: str
    weights_sha256: str  # digest over sorted shard digests
    shard_sha256: Dict[str, str]
    weights_bytes: int
    upstream_revision: str
    architecture: str
    max_position_embeddings: int
    tie_word_embeddings: bool
    is_plaintext_verified: bool
    verification_notes: Tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "architecture": self.architecture,
            "model_config": asdict(self.model_config),
            "config_sha256": self.config_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "tokenizer_config_sha256": self.tokenizer_config_sha256,
            "weights_sha256": self.weights_sha256,
            "shard_sha256": dict(self.shard_sha256),
            "weights_bytes": self.weights_bytes,
            "upstream_revision": self.upstream_revision,
            "max_position_embeddings": self.max_position_embeddings,
            "tie_word_embeddings": self.tie_word_embeddings,
            "is_plaintext_verified": self.is_plaintext_verified,
            "verification_notes": list(self.verification_notes),
            "scheme": "plaintext_llama_base",
            "legacy_modelsplit_obfuscation": False,
        }


def _llama_model_config(
    raw: Mapping[str, object], *, max_sequence_length: int
) -> ModelConfig:
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
        raise ValueError("Llama config missing fields: %s" % missing)
    max_position = int(raw["max_position_embeddings"])
    if max_sequence_length < 1 or max_sequence_length > max_position:
        raise ValueError(
            "requested sequence length %d is outside Llama context %d"
            % (max_sequence_length, max_position)
        )
    # Llama-3.x typically has no QKV bias.
    attention_bias = bool(raw.get("attention_bias", False))
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
        qkv_bias=attention_bias,
    )


def _weight_map(root: Path) -> Dict[str, str]:
    """Return tensor_name -> relative shard path."""

    single = root / "model.safetensors"
    index = root / "model.safetensors.index.json"
    if index.is_file():
        raw = _load_json(index)
        weight_map = raw.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Llama index weight_map is empty")
        return {str(k): str(v) for k, v in weight_map.items()}
    if single.is_file():
        # Discover keys without loading tensors.
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise RuntimeError(
                "Llama loading requires safetensors; install pretrained deps"
            ) from error
        with safe_open(str(single), framework="pt", device="cpu") as handle:
            return {key: single.name for key in handle.keys()}
    raise FileNotFoundError(
        "Llama artifact has neither model.safetensors nor "
        "model.safetensors.index.json under %s" % root
    )


def _reject_legacy_keys(keys: Iterable[str]) -> None:
    bad = sorted(
        {
            key
            for key in keys
            if any(marker in key.lower() for marker in _LEGACY_MARKERS)
        }
    )
    if bad:
        raise ValueError(
            "refusing non-plaintext / legacy-obfuscation weight keys: %s"
            % (bad[:12],)
        )


def verify_plaintext_llama_tree(root: str | Path) -> Tuple[bool, List[str]]:
    """Return (ok, notes) for a local Llama directory without loading weights."""

    notes: List[str] = []
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        return False, ["directory missing: %s" % root_path]
    try:
        raw = _load_json(root_path / "config.json")
    except (OSError, ValueError, FileNotFoundError) as exc:
        return False, ["config.json unreadable: %s" % exc]
    model_type = raw.get("model_type")
    architectures = raw.get("architectures") or []
    if model_type != "llama":
        return False, ["model_type is %r, expected 'llama'" % model_type]
    if "LlamaForCausalLM" not in architectures:
        notes.append("architectures=%r (expected LlamaForCausalLM)" % architectures)
        return False, notes
    for key in raw:
        if any(marker in str(key).lower() for marker in _LEGACY_MARKERS):
            return False, ["suspicious config key %r" % key]
    try:
        weight_map = _weight_map(root_path)
    except (OSError, ValueError, FileNotFoundError, RuntimeError) as exc:
        return False, ["weight map error: %s" % exc]
    try:
        _reject_legacy_keys(weight_map.keys())
    except ValueError as exc:
        return False, [str(exc)]
    required_prefixes = (
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
    )
    for name in required_prefixes:
        if name not in weight_map:
            return False, ["missing standard weight %s" % name]
    tie = bool(raw.get("tie_word_embeddings", False))
    if not tie and "lm_head.weight" not in weight_map:
        return False, ["untied model missing lm_head.weight"]
    if tie:
        notes.append("tie_word_embeddings=true; lm_head shares embed_tokens")
    notes.append("standard LlamaForCausalLM weight map; no legacy markers")
    notes.append(
        "NOT a legacy ModelSplit obfuscation artifact; safe as plaintext base"
    )
    return True, notes


def load_llama_artifact(
    root: str | Path,
    *,
    max_sequence_length: int,
    compute_hashes: bool = True,
) -> LlamaArtifact:
    """Validate a local Llama directory and optionally hash weight shards."""

    root_path = Path(root).expanduser().resolve()
    ok, notes = verify_plaintext_llama_tree(root_path)
    if not ok:
        raise ValueError(
            "Llama tree failed plaintext verification: %s" % "; ".join(notes)
        )
    config_path = _require_file(root_path, "config.json")
    tokenizer_path = _require_file(root_path, "tokenizer.json")
    tokenizer_config_path = _require_file(root_path, "tokenizer_config.json")
    raw = _load_json(config_path)
    weight_map = _weight_map(root_path)
    _reject_legacy_keys(weight_map.keys())
    model_config = _llama_model_config(
        raw, max_sequence_length=int(max_sequence_length)
    )
    # Validate every layer key exists in the map.
    for layer in range(model_config.num_layers):
        for suffix in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ):
            expected = "model.layers.%d.%s" % (layer, suffix)
            if expected not in weight_map:
                raise ValueError("Llama checkpoint missing %s" % expected)
    for expected in ("model.embed_tokens.weight", "model.norm.weight"):
        if expected not in weight_map:
            raise ValueError("Llama checkpoint missing %s" % expected)
    tie = bool(raw.get("tie_word_embeddings", False))
    if not tie and "lm_head.weight" not in weight_map:
        raise ValueError("untied Llama checkpoint missing lm_head.weight")
    shards = sorted(set(weight_map.values()))
    shard_paths = [root_path / name for name in shards]
    for path in shard_paths:
        if not path.is_file():
            raise FileNotFoundError("missing weight shard %s" % path)
    weights_bytes = sum(path.stat().st_size for path in shard_paths)
    if compute_hashes:
        config_sha256 = _sha256_file(config_path)
        tokenizer_sha256 = _sha256_file(tokenizer_path)
        tokenizer_config_sha256 = _sha256_file(tokenizer_config_path)
        shard_sha256 = {
            path.name: _sha256_file(path) for path in shard_paths
        }
        # Stable aggregate over sorted shard digests.
        aggregate = hashlib.sha256()
        for name in sorted(shard_sha256):
            aggregate.update(name.encode("utf-8"))
            aggregate.update(shard_sha256[name].encode("utf-8"))
        weights_sha256 = aggregate.hexdigest()
    else:
        config_sha256 = tokenizer_sha256 = tokenizer_config_sha256 = "not_computed"
        shard_sha256 = {path.name: "not_computed" for path in shard_paths}
        weights_sha256 = "not_computed"
    architecture = str((raw.get("architectures") or [""])[0])
    upstream_revision = str(raw.get("_commit_hash", "revision_unavailable"))
    return LlamaArtifact(
        root=str(root_path),
        model_config=model_config,
        config_sha256=config_sha256,
        tokenizer_sha256=tokenizer_sha256,
        tokenizer_config_sha256=tokenizer_config_sha256,
        weights_sha256=weights_sha256,
        shard_sha256=shard_sha256,
        weights_bytes=weights_bytes,
        upstream_revision=upstream_revision,
        architecture=architecture,
        max_position_embeddings=int(raw["max_position_embeddings"]),
        tie_word_embeddings=tie,
        is_plaintext_verified=True,
        verification_notes=tuple(notes),
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


def _llama_mapping(config: ModelConfig) -> Dict[str, str]:
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
                prefix
                + "post_attention_layernorm.weight": target
                + "ffn_norm_weight",
                prefix + "self_attn.q_proj.weight": target + "q_proj.weight",
                prefix + "self_attn.k_proj.weight": target + "k_proj.weight",
                prefix + "self_attn.v_proj.weight": target + "v_proj.weight",
                prefix + "self_attn.o_proj.weight": target + "o_proj.weight",
                prefix + "mlp.gate_proj.weight": target + "gate_proj.weight",
                prefix + "mlp.up_proj.weight": target + "up_proj.weight",
                prefix + "mlp.down_proj.weight": target + "down_proj.weight",
            }
        )
        if config.qkv_bias:
            mapping.update(
                {
                    prefix + "self_attn.q_proj.bias": target + "q_proj.bias",
                    prefix + "self_attn.k_proj.bias": target + "k_proj.bias",
                    prefix + "self_attn.v_proj.bias": target + "v_proj.bias",
                }
            )
    return mapping


def load_llama_plain(
    artifact: LlamaArtifact,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
    debug_enabled: bool = False,
) -> PlainTinyCausalLM:
    """Stream a verified plaintext Llama checkpoint into the reference model."""

    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Llama adapter supports FP32 or BF16 only")
    if not artifact.is_plaintext_verified:
        raise ValueError("refusing to load artifact that failed plaintext verification")
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("Llama loading requires safetensors") from error

    root = Path(artifact.root)
    weight_map = _weight_map(root)
    mapping = _llama_mapping(artifact.model_config)

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

    # Group target loads by shard for one open per file.
    by_shard: Dict[str, List[Tuple[str, str]]] = {}
    for source_name, target_name in mapping.items():
        if source_name == "lm_head.weight" and artifact.tie_word_embeddings:
            continue
        if source_name not in weight_map:
            if source_name == "lm_head.weight" and artifact.tie_word_embeddings:
                continue
            raise ValueError("Llama checkpoint missing %s" % source_name)
        if target_name not in targets:
            raise ValueError("fastProve target missing %s" % target_name)
        by_shard.setdefault(weight_map[source_name], []).append(
            (source_name, target_name)
        )

    for shard_name, pairs in by_shard.items():
        shard_path = root / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for source_name, target_name in pairs:
                if source_name not in available:
                    raise ValueError(
                        "shard %s missing %s" % (shard_name, source_name)
                    )
                _copy_tensor(
                    handle.get_tensor(source_name),
                    targets[target_name],
                    source_name=source_name,
                )

    if artifact.tie_word_embeddings:
        with torch.no_grad():
            targets["lm_head.weight"].copy_(targets["embedding.weight"])
    return model


def load_llama_tokenizer(root: str | Path):
    """Load the local tokenizer without network fallback."""

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Llama tokenizer loading requires transformers"
        ) from error
    return AutoTokenizer.from_pretrained(
        str(Path(root).expanduser().resolve()),
        local_files_only=True,
        use_fast=True,
    )
