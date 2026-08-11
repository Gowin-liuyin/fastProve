"""Integration tests for the shipped pretrained compare path (no 3B download)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from fastprove.config import ModelConfig, ObfuscationConfig
from fastprove.layers.attention import AttentionMode
from fastprove.models.obfuscated import ObfuscatedTinyCausalLM
from fastprove.models.plain import PlainTinyCausalLM
from fastprove.seed import RequestContext
from fastprove.evaluation.accuracy import (
    compare_teacher_forced_metrics,
    make_synthetic_token_batch,
)

from evals.keys import generate_master_keys
from evals.model_factory import zero_noise_injection_


def _load_compare_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_pretrained_compare.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_pretrained_compare_shipped", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_scheme_conversion_pair_no_legacy_arm() -> None:
    """Drive real from_plain conversion + paired metrics (shipped path)."""

    config = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_sequence_length=32,
    )
    plain = PlainTinyCausalLM(config, seed=11, debug_enabled=True)
    plain.eval()
    keys = generate_master_keys(count=3, base_seed=99)
    tokens, ids = make_synthetic_token_batch(
        sample_count=2, sequence_length=8, vocab_size=64, seed=7
    )
    mask = torch.ones_like(tokens, dtype=torch.bool)
    obfuscation = ObfuscationConfig(
        hidden_noise_dim=4,
        value_noise_dim_per_head=1,
        max_condition_number=10.0,
        noise_propagation_gamma=0.5,
        refresh_mode="fixed_debug",
        basis_block_size=12,
    )
    compare = _load_compare_module()
    records = []
    for key in keys:
        converted = ObfuscatedTinyCausalLM.from_plain(
            plain,
            obfuscation=obfuscation,
            mode=AttentionMode.EXACT,
            approximation=None,
            seed=key.conversion_seed(),
            debug_enabled=True,
        )
        module = converted.module
        zero_noise_injection_(module)  # structural
        ctx = RequestContext(key.request_seed("t"), "test")
        with torch.no_grad():
            p_logits = plain(tokens, token_mask=mask)
            o_logits = module(tokens, token_mask=mask, request_context=ctx)
        tf = compare_teacher_forced_metrics(
            plaintext_logits=p_logits,
            obfuscated_logits=o_logits,
            input_ids=tokens,
            token_mask=mask,
        )
        greedy = compare.greedy_unpadded_pair(
            plain=plain,
            obfuscated=module,
            tokens=tokens,
            mask=mask,
            sample_ids=ids,
            generation_tokens=2,
            request_context=ctx,
        )
        records.append(
            {
                "key": key.label,
                "sample_ids": ids,
                "legacy_modelsplit_obfuscation": False,
                "scheme": "current_fastprove_covariant",
                "agreement": tf["agreement"]["next_token_top1_agreement"],
                "greedy_sequence_exact_match": greedy[
                    "greedy_sequence_exact_match"
                ],
                "greedy_n_samples": greedy["greedy_n_samples"],
                "greedy_sample_ids": greedy["greedy_sample_ids"],
            }
        )
    assert len(records) == 3
    assert all(r["legacy_modelsplit_obfuscation"] is False for r in records)
    assert all(r["scheme"] == "current_fastprove_covariant" for r in records)
    assert all(r["agreement"] >= 0.99 for r in records)
    assert all(r["greedy_sequence_exact_match"] >= 0.99 for r in records)
    assert all(r["sample_ids"] == ids for r in records)
    # All sample_ids must enter greedy (not a batch_size=1 slice).
    assert all(r["greedy_n_samples"] == len(ids) for r in records)
    assert all(r["greedy_sample_ids"] == ids for r in records)


def test_greedy_unpadded_evaluates_all_rows_and_strips_pad() -> None:
    """Right-padded rows must strip pad so last index is last real token."""

    compare = _load_compare_module()
    config = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_sequence_length=32,
    )
    plain = PlainTinyCausalLM(config, seed=3, debug_enabled=True)
    plain.eval()
    converted = ObfuscatedTinyCausalLM.from_plain(
        plain,
        obfuscation=ObfuscationConfig(
            hidden_noise_dim=4,
            value_noise_dim_per_head=1,
            max_condition_number=10.0,
            noise_propagation_gamma=0.5,
            refresh_mode="fixed_debug",
            basis_block_size=12,
        ),
        mode=AttentionMode.EXACT,
        approximation=None,
        seed=5,
        debug_enabled=True,
    )
    module = converted.module
    zero_noise_injection_(module)

    # Four right-padded prompts of different valid lengths.
    tokens = torch.zeros(4, 12, dtype=torch.long)
    mask = torch.zeros(4, 12, dtype=torch.bool)
    for i, length in enumerate((3, 5, 7, 9)):
        tokens[i, :length] = torch.arange(1, length + 1)
        mask[i, :length] = True
    sample_ids = ["prompt-%04d" % i for i in range(4)]
    ctx = RequestContext(1, "pad-strip")
    greedy = compare.greedy_unpadded_pair(
        plain=plain,
        obfuscated=module,
        tokens=tokens,
        mask=mask,
        sample_ids=sample_ids,
        generation_tokens=2,
        request_context=ctx,
    )
    assert greedy["greedy_n_samples"] == 4
    assert greedy["greedy_sample_ids"] == sample_ids
    assert greedy["greedy_prompt_lengths"] == [3, 5, 7, 9]
    assert greedy["greedy_padding_stripped"] is True


def test_run_pair_reports_e2e_logit_separate_from_chainlinear_unit() -> None:
    """Shipped _run_pair must expose e2e logit error without claiming ChainLinear."""

    compare = _load_compare_module()
    config = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_sequence_length=32,
    )
    plain = PlainTinyCausalLM(config, seed=9, debug_enabled=True)
    plain.eval()
    converted = ObfuscatedTinyCausalLM.from_plain(
        plain,
        obfuscation=ObfuscationConfig(
            hidden_noise_dim=4,
            value_noise_dim_per_head=1,
            max_condition_number=10.0,
            noise_propagation_gamma=0.5,
            refresh_mode="fixed_debug",
            basis_block_size=12,
        ),
        mode=AttentionMode.EXACT,
        approximation=None,
        seed=11,
        debug_enabled=True,
    )
    module = converted.module
    zero_noise_injection_(module)
    tokens, ids = make_synthetic_token_batch(
        sample_count=3, sequence_length=6, vocab_size=64, seed=13
    )
    mask = torch.ones_like(tokens, dtype=torch.bool)
    pair = compare._run_pair(
        plain=plain,
        obfuscated=module,
        tokens=tokens,
        mask=mask,
        sample_ids=ids,
        mode_name="structural",
        key_meta={"label": "key-00"},
        conversion_seconds=0.0,
        structural_noise_zeroed=True,
        request_seed=17,
        generation_tokens=2,
        batch_size=1,
        device=torch.device("cpu"),
    )
    assert pair["greedy_n_samples"] == 3
    assert pair["greedy"]["greedy_sample_ids"] == ids
    assert "e2e_logit_max_absolute_error" in pair
    assert "logits_e2e" in pair
    # Must not re-label e2e logits as ChainLinear in the pair record.
    assert "chain_linear" not in pair
    assert pair["legacy_modelsplit_obfuscation"] is False
