from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from fastprove.evaluation.accuracy import (
    compare_teacher_forced_metrics,
    greedy_generation_metrics,
    make_synthetic_token_batch,
)


def test_teacher_forced_metrics_match_direct_cross_entropy() -> None:
    input_ids = torch.tensor([[0, 1, 2, 3]])
    plain_logits = torch.tensor(
        [
            [
                [0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 3.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    obfuscated_logits = plain_logits.clone()
    obfuscated_logits[0, 1] = torch.tensor([3.0, 0.0, 0.0, 0.0])
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    result = compare_teacher_forced_metrics(
        plaintext_logits=plain_logits,
        obfuscated_logits=obfuscated_logits,
        input_ids=input_ids,
        token_mask=mask,
    )
    expected_nll = F.cross_entropy(
        plain_logits[:, :-1].reshape(-1, 4),
        input_ids[:, 1:].reshape(-1),
        reduction="mean",
    ).item()
    assert abs(result["plaintext"]["negative_log_likelihood"] - expected_nll) < 1e-7
    assert abs(
        result["plaintext"]["perplexity"] - math.exp(expected_nll)
    ) < 1e-6
    assert result["plaintext"]["next_token_top1_accuracy"] == 1.0
    assert result["plaintext"]["next_token_top5_accuracy"] == 1.0
    assert result["obfuscated"]["next_token_top1_accuracy"] == pytest.approx(
        2 / 3
    )
    assert result["agreement"]["next_token_top1_agreement"] == pytest.approx(
        2 / 3
    )
    assert result["degradation"]["top1_absolute_drop"] == pytest.approx(
        1 / 3
    )
    assert result["degradation"]["top1_relative_drop"] == pytest.approx(
        1 / 3
    )
    assert result["degradation"]["perplexity_absolute_increase"] > 0
    assert result["degradation"]["nll_relative_increase"] == pytest.approx(
        result["degradation"]["nll_absolute_increase"]
        / result["plaintext"]["negative_log_likelihood"]
    )
    assert result["token_count"] == 3


def test_padding_tokens_are_excluded_from_accuracy() -> None:
    input_ids = torch.tensor([[0, 1, 2, 3]])
    logits = torch.zeros(1, 4, 4)
    logits[0, 0, 1] = 5
    logits[0, 1, 2] = 5
    mask = torch.tensor([[True, True, False, False]])
    result = compare_teacher_forced_metrics(
        plaintext_logits=logits,
        obfuscated_logits=logits,
        input_ids=input_ids,
        token_mask=mask,
    )
    assert result["token_count"] == 1
    assert result["plaintext"]["next_token_top1_accuracy"] == 1.0


def test_noncontiguous_valid_mask_is_rejected_for_next_token_metrics() -> None:
    input_ids = torch.tensor([[0, 1, 2]])
    logits = torch.zeros(1, 3, 3)
    mask = torch.tensor([[True, False, True]])
    with pytest.raises(ValueError, match="adjacent valid"):
        compare_teacher_forced_metrics(
            plaintext_logits=logits,
            obfuscated_logits=logits,
            input_ids=input_ids,
            token_mask=mask,
        )


def test_greedy_generation_metrics_report_token_and_sequence_match() -> None:
    plain = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    obfuscated = torch.tensor([[1, 2, 3, 9], [5, 6, 7, 8]])
    metrics = greedy_generation_metrics(
        plaintext_tokens=plain,
        obfuscated_tokens=obfuscated,
        prompt_length=2,
    )
    assert metrics["greedy_token_exact_match"] == 3 / 4
    assert metrics["greedy_sequence_exact_match"] == 0.5


def test_synthetic_token_batch_is_reproducible_with_stable_ids() -> None:
    first_tokens, first_ids = make_synthetic_token_batch(
        sample_count=5, sequence_length=8, vocab_size=32, seed=77
    )
    second_tokens, second_ids = make_synthetic_token_batch(
        sample_count=5, sequence_length=8, vocab_size=32, seed=77
    )
    assert torch.equal(first_tokens, second_tokens)
    assert first_ids == second_ids
    assert first_tokens.shape == (5, 8)
    assert all(identifier.startswith("synthetic-") for identifier in first_ids)


def test_relative_accuracy_drop_is_undefined_for_zero_plaintext_baseline() -> None:
    input_ids = torch.tensor([[0, 1, 2]])
    plain_logits = torch.zeros(1, 3, 3)
    plain_logits[:, :-1, 0] = 2.0
    obfuscated_logits = plain_logits.clone()
    obfuscated_logits[0, 0, 1] = 4.0
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    result = compare_teacher_forced_metrics(
        plaintext_logits=plain_logits,
        obfuscated_logits=obfuscated_logits,
        input_ids=input_ids,
        token_mask=mask,
    )
    assert result["plaintext"]["next_token_top1_accuracy"] == 0.0
    assert result["degradation"]["top1_relative_drop"] is None


def test_sample_bootstrap_is_deterministic_and_resamples_whole_examples() -> None:
    input_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
        ]
    )
    plain_logits = torch.zeros(3, 4, 4)
    plain_logits[:, :-1].scatter_(
        -1, input_ids[:, 1:].unsqueeze(-1), 4.0
    )
    obfuscated_logits = plain_logits.clone()
    obfuscated_logits[1, 1, input_ids[1, 2]] = -2.0
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    first = compare_teacher_forced_metrics(
        plaintext_logits=plain_logits,
        obfuscated_logits=obfuscated_logits,
        input_ids=input_ids,
        token_mask=mask,
        bootstrap_replicates=200,
        bootstrap_seed=123,
    )
    second = compare_teacher_forced_metrics(
        plaintext_logits=plain_logits,
        obfuscated_logits=obfuscated_logits,
        input_ids=input_ids,
        token_mask=mask,
        bootstrap_replicates=200,
        bootstrap_seed=123,
    )
    assert first["bootstrap"] == second["bootstrap"]
    bootstrap = first["bootstrap"]
    assert bootstrap["unit"] == "sample"
    assert bootstrap["replicates"] == 200
    assert bootstrap["confidence_level"] == 0.95
    assert (
        bootstrap["metrics"]["top1_absolute_drop"]["high"]
        >= bootstrap["metrics"]["top1_absolute_drop"]["low"]
    )
