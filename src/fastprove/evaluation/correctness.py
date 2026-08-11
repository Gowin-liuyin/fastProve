"""Exact tiny-model correctness entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..config import load_config
from ..layers.attention import AttentionMode
from .runner import run_tiny_sweep_spec
from .sweep import SweepSpec


def run_tiny_correctness(
    *, config_path: Path, output_path: Path
) -> Dict[str, object]:
    """Run the exact random-tiny gate and append one raw JSONL record."""

    config = load_config(config_path)
    if config.attention.mode != AttentionMode.EXACT:
        raise ValueError("correctness entrypoint requires an exact config")
    return run_tiny_sweep_spec(
        base_config_path=config_path,
        spec=SweepSpec(
            run_id="tiny-exact-correctness",
            mode="exact",
            tau_max=0.0,
            tau_error=0.0,
            alpha=None,
            preserve_top_k=None,
        ),
        experiment={
            "seed": config.runtime.seed,
            "model_id": "fastprove-random-tiny-correctness-only",
            "dataset_id": "deterministic-synthetic-token-sequences",
            "expected_run_ids": ["tiny-exact-correctness"],
        },
        output_path=output_path,
    )
