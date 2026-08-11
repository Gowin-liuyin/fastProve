from __future__ import annotations

import math

import torch

from fastprove.evaluation.metrics import (
    attention_distribution_metrics,
    tensor_error_metrics,
)


def test_tensor_error_metrics_are_exact_for_identical_tensors() -> None:
    value = torch.tensor([[1.0, -2.0, 3.0]])
    metrics = tensor_error_metrics(value, value.clone())
    assert metrics["max_absolute_error"] == 0.0
    assert metrics["mean_absolute_error"] == 0.0
    assert metrics["relative_l2_error"] == 0.0
    assert metrics["cosine_similarity"] == 1.0
    assert metrics["nan_count"] == 0
    assert metrics["inf_count"] == 0


def test_tensor_error_metrics_count_non_finite_values() -> None:
    reference = torch.zeros(3)
    actual = torch.tensor([float("nan"), float("inf"), 1.0])
    metrics = tensor_error_metrics(reference, actual)
    assert metrics["nan_count"] == 1
    assert metrics["inf_count"] == 1
    assert math.isinf(metrics["max_absolute_error"])


def test_attention_metrics_cover_divergence_ranking_and_output_error() -> None:
    clean_probabilities = torch.tensor(
        [[[[0.6, 0.3, 0.1, 0.0], [0.5, 0.5, 0.0, 0.0]]]]
    )
    noisy_probabilities = torch.tensor(
        [[[[0.2, 0.3, 0.5, 0.0], [0.5, 0.5, 0.0, 0.0]]]]
    )
    clean_logits = torch.tensor(
        [[[[3.0, 2.0, 1.0, -torch.inf], [1.0, 1.0, -torch.inf, -torch.inf]]]]
    )
    noisy_logits = torch.tensor(
        [[[[1.0, 2.0, 3.0, -torch.inf], [1.0, 1.0, -torch.inf, -torch.inf]]]]
    )
    valid = torch.tensor(
        [[[[True, True, True, False], [True, True, False, False]]]]
    )
    noise = torch.tensor(
        [[[[0.04, -0.02, -0.02, 0.0], [0.0, 0.0, 0.0, 0.0]]]]
    )
    margin = torch.tensor([[[[1.0], [0.0]]]])
    clean_output = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    noisy_output = torch.tensor([[[[1.1, 1.9], [3.0, 4.0]]]])

    metrics = attention_distribution_metrics(
        clean_probabilities=clean_probabilities,
        noisy_probabilities=noisy_probabilities,
        clean_logits=clean_logits,
        noisy_logits=noisy_logits,
        valid_mask=valid,
        noise=noise,
        margin=margin,
        top_k=1,
        clean_output=clean_output,
        noisy_output=noisy_output,
    )
    assert metrics["kl_divergence"] > 0
    assert metrics["js_divergence"] > 0
    assert 0 <= metrics["js_divergence"] <= math.log(2)
    assert metrics["topk_overlap"] == 0.5
    assert metrics["topk_changed_fraction"] == 0.5
    assert metrics["rank_correlation"] == 0.0
    assert abs(metrics["actual_noise_infinity_norm"] - 0.04) < 1e-7
    assert metrics["zero_noise_query_fraction"] == 0.5
    assert metrics["clean_boundary_margin_mean"] == 0.5
    assert metrics["attention_output_relative_l2_error"] > 0


def test_masked_values_do_not_affect_attention_metrics() -> None:
    probabilities = torch.tensor([[[[0.75, 0.25, 0.0]]]])
    valid = torch.tensor([[[[True, True, False]]]])
    clean_logits = torch.tensor([[[[2.0, 1.0, -torch.inf]]]])
    noisy_logits = torch.tensor([[[[2.0, 1.0, -torch.inf]]]])
    noise = torch.tensor([[[[0.0, 0.0, 999.0]]]])
    metrics = attention_distribution_metrics(
        clean_probabilities=probabilities,
        noisy_probabilities=probabilities,
        clean_logits=clean_logits,
        noisy_logits=noisy_logits,
        valid_mask=valid,
        noise=noise,
        margin=torch.tensor([[[[1.0]]]]),
        top_k=1,
    )
    assert metrics["kl_divergence"] == 0.0
    assert metrics["js_divergence"] == 0.0
    assert metrics["topk_overlap"] == 1.0
    assert metrics["actual_noise_infinity_norm"] == 0.0

