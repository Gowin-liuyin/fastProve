"""Shared checkpoint I/O helpers for the Llama and Qwen2 adapters.

These helpers were duplicated verbatim in ``llama.py`` and ``qwen2.py``;
the adapter-specific weight maps and verification logic stay in the
respective modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch

from ..config import ModelConfig
from ..models.plain import PlainTinyCausalLM


def require_file(root: Path, name: str, *, artifact_label: str) -> Path:
    """Return ``root/name`` or raise with the adapter-specific label."""

    path = root / name
    if not path.is_file():
        raise FileNotFoundError(
            "%s artifact is missing %s" % (artifact_label, path)
        )
    return path


def load_json(path: Path) -> dict:
    """Load a JSON object, rejecting non-object roots."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object: %s" % path)
    return value


def target_tensors(model: PlainTinyCausalLM) -> Dict[str, torch.Tensor]:
    """Map parameter and buffer names to their tensors."""

    return {
        **dict(model.named_parameters()),
        **dict(model.named_buffers()),
    }


def copy_tensor(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    source_name: str,
) -> None:
    """Copy a checkpoint tensor into its target, preserving target dtype."""

    if tuple(source.shape) != tuple(target.shape):
        raise ValueError(
            "%s shape mismatch: checkpoint=%s target=%s"
            % (source_name, tuple(source.shape), tuple(target.shape))
        )
    if not source.is_floating_point() or not target.is_floating_point():
        raise ValueError("%s must be floating point" % source_name)
    with torch.no_grad():
        target.copy_(source.to(device=target.device, dtype=target.dtype))


def build_meta_model(
    model_config: ModelConfig,
    *,
    seed: int,
    debug_enabled: bool,
    dtype: torch.dtype,
    device: str | torch.device,
) -> PlainTinyCausalLM:
    """Materialize a reference model on ``device`` and restore structural buffers.

    ``to_empty`` intentionally leaves non-parameter buffers uninitialised;
    the KV-head index buffer is not present in either checkpoint and must be
    restored deterministically.
    """

    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            model = PlainTinyCausalLM(
                model_config, seed=seed, debug_enabled=debug_enabled
            )
    finally:
        torch.set_default_dtype(old_default_dtype)
    model = model.to_empty(device=device)
    for block in model.blocks:
        block.kv_index.copy_(
            torch.arange(
                model_config.num_attention_heads,
                device=block.kv_index.device,
            )
            // model_config.query_heads_per_kv_head
        )
    model.eval()
    return model
