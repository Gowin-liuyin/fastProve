from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from src.aloepri._implementation.export.tensor_io import safetensors_shapes


REQUIRED_STANDARD_KEYS = [
    "model.embed_tokens.weight",
    "lm_head.weight",
]


def build_stage_j_standard_weight_proof(server_dir: str | Path) -> dict[str, Any]:
    server_dir = Path(server_dir)
    result: dict[str, Any] = {
        "is_standard_weight_export": False,
        "validation_scope": "config_and_all_tensor_names_and_shapes_only",
        "actual_model_load_tested": False, "numerical_equivalence_tested": False, "privacy_tested": False,
        "layout": "invalid", "standard_key_count": 0, "buffered_key_count": 0,
        "missing_standard_keys": [], "unexpected_keys": [], "shape_mismatches": {},
        "sample_standard_keys": [], "sample_buffered_keys": [],
    }
    try:
        if not (server_dir / "config.json").is_file():
            raise FileNotFoundError("config.json is required.")
        shapes = safetensors_shapes(server_dir)
        buffered = [key for key in shapes if key.startswith(("buffer::", "param::"))]
        result.update(standard_key_count=len(shapes) - len(buffered), buffered_key_count=len(buffered),
                      sample_standard_keys=[key for key in shapes if key not in buffered][:20],
                      sample_buffered_keys=buffered[:20])
        config = AutoConfig.from_pretrained(server_dir, trust_remote_code=False, local_files_only=True)
        factory = AutoModel if config.model_type == "bert" else AutoModelForCausalLM
        # meta 只建立结构，不分配一个与原模型同样大的随机模型。
        with torch.device("meta"):
            model = factory.from_config(config, trust_remote_code=False)
        expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
        missing = set(expected) - set(shapes)
        if getattr(config, "tie_word_embeddings", False) and "model.embed_tokens.weight" in shapes:
            missing.discard("lm_head.weight")
        unexpected = set(shapes) - set(expected)
        mismatch = {key: {"expected": expected[key], "actual": shapes[key]}
                    for key in expected.keys() & shapes.keys() if expected[key] != shapes[key]}
        passed = not missing and not unexpected and not mismatch and not buffered
        result.update(is_standard_weight_export=passed,
                      layout="standard_weight_visible" if passed else "invalid_standard_structure",
                      missing_standard_keys=sorted(missing), unexpected_keys=sorted(unexpected),
                      shape_mismatches=mismatch)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        result["error"] = str(exc)
    return result
