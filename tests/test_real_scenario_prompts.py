"""Tests for real-scenario prompt bank used by large-scale compare."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load(name: str, rel: str):
    path = Path(__file__).resolve().parents[1] / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_real_scenario_prompts_ge_1500() -> None:
    mod = _load("build_real", "scripts/build_real_scenario_prompts.py")
    prompts = mod.build_real_scenario_prompts(1500, seed=20260802)
    assert len(prompts) == 1500
    assert all(isinstance(p, str) and p.strip() for p in prompts)
    # Deterministic
    again = mod.build_real_scenario_prompts(1500, seed=20260802)
    assert prompts == again
    # Diversity: not all identical
    assert len(set(prompts)) > 100


def test_compare_cli_load_prompt_texts_real_scenario() -> None:
    cmp_mod = _load("cmp", "scripts/run_pretrained_compare.py")
    texts = cmp_mod.load_prompt_texts(
        prompt_file=None,
        sample_count=1500,
        seed=1,
        real_scenario=True,
    )
    assert len(texts) == 1500
