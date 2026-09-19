"""Tests for real-scenario prompt bank used by large-scale compare."""

from __future__ import annotations

from scripts.build_real_scenario_prompts import build_real_scenario_prompts
from scripts.run_pretrained_compare import load_prompt_texts


def test_build_real_scenario_prompts_ge_1500() -> None:
    prompts = build_real_scenario_prompts(1500, seed=20260802)
    assert len(prompts) == 1500
    assert all(isinstance(p, str) and p.strip() for p in prompts)
    # Deterministic
    again = build_real_scenario_prompts(1500, seed=20260802)
    assert prompts == again
    # Diversity: not all identical
    assert len(set(prompts)) > 100


def test_compare_cli_load_prompt_texts_real_scenario() -> None:
    texts = load_prompt_texts(
        prompt_file=None,
        sample_count=1500,
        seed=1,
        real_scenario=True,
    )
    assert len(texts) == 1500
