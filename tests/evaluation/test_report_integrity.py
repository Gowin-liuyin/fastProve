from __future__ import annotations

import copy
from pathlib import Path

import pytest
from matplotlib.figure import Figure

from fastprove.evaluation.report import (
    _cohort_key,
    _rows,
    build_report,
)

from tests._fixtures import failure_record as _failure_record
from tests._fixtures import success_record as _success_record


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("model_revision", "revision-b"),
        ("model_weight_sha256", "weight-sha-b"),
        ("model_config_sha256", "config-sha-b"),
        ("model_tokenizer_sha256", "tokenizer-sha-b"),
        ("dataset_split", "validation"),
        ("device", "mps"),
        ("dtype", "bfloat16"),
        ("sequence_length", 256),
        ("batch_size", 4),
        ("generation_tokens", 8),
        ("sample_count", 8),
        ("tokenized_inputs_sha256", "tokenized-sha-b"),
        ("sample_ids_sha256", "sample-sha-b"),
    ],
)
def test_cohort_key_separates_every_fairness_dimension(
    field: str, changed: object
) -> None:
    baseline = {
        "model_id": "model",
        "model_revision": "revision-a",
        "model_weight_sha256": "weight-sha-a",
        "model_config_sha256": "config-sha-a",
        "model_tokenizer_sha256": "tokenizer-sha-a",
        "dataset_id": "data",
        "dataset_split": "test",
        "seed": 52,
        "stage": "full",
        "device": "cpu",
        "dtype": "float32",
        "sequence_length": 128,
        "batch_size": 2,
        "generation_tokens": 4,
        "sample_count": 4,
        "tokenized_inputs_sha256": "tokenized-sha-a",
        "sample_ids_sha256": "sample-sha-a",
    }
    candidate = copy.deepcopy(baseline)
    candidate[field] = changed

    assert _cohort_key(candidate) != _cohort_key(baseline)


def test_rows_preserve_fairness_and_direct_raw_provenance() -> None:
    row = _rows([_success_record("traceable")])[0]

    assert row["sequence_length"] == 128
    assert row["batch_size"] == 2
    assert row["tokenized_inputs_sha256"] == "tokenized-sha-a"
    assert row["raw_source"] == "results/raw/attempt/eval.jsonl"
    assert row["raw_line"] == 1


def test_rows_extract_qwen_hashes_from_checkpoint_manifest_and_cache() -> None:
    record = _success_record("nested-provenance")
    record["model"].pop("revision")
    record["model"].pop("weight_sha256")
    record["model"]["checkpoint_manifest"] = {
        "upstream_revision": "qwen-revision",
        "weights_sha256": "qwen-weight-sha",
        "config_sha256": "qwen-config-sha",
    }
    record["dataset"].pop("tokenized_inputs_sha256")
    record["dataset"]["cache_content_sha256"] = "cache-content-sha"

    row = _rows([record])[0]

    assert row["model_revision"] == "qwen-revision"
    assert row["model_weight_sha256"] == "qwen-weight-sha"
    assert row["model_config_sha256"] == "qwen-config-sha"
    assert row["tokenized_inputs_sha256"] == "cache-content-sha"


def test_report_figures_render_successes_as_unconnected_scatter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots: list[tuple[int, int]] = []

    def inspect_figure(self: Figure, *_args: object, **_kwargs: object) -> None:
        line_count = sum(len(axis.lines) for axis in self.axes)
        collection_count = sum(len(axis.collections) for axis in self.axes)
        snapshots.append((line_count, collection_count))

    monkeypatch.setattr(Figure, "savefig", inspect_figure)
    records = [
        _success_record("low", tau=0.01, raw_line=1),
        _failure_record("failed-middle", tau=0.05, raw_line=2),
        _success_record("high", tau=0.1, raw_line=3),
    ]

    output = build_report(
        records=records,
        output_root=tmp_path,
        experiment_status="unit_test",
        pretrained_status="not_run",
    )
    report = output.report_markdown.read_text(encoding="utf-8")

    assert snapshots == [(0, 2), (0, 2)]
    assert "不连线的散点" in report


def test_report_tables_include_raw_locations_and_failure_parameters(
    tmp_path: Path,
) -> None:
    exact = _success_record("exact", mode="exact", tau=0.0, raw_line=4)
    approximate = _success_record("approx", raw_line=5)
    failure = _failure_record("failed", raw_line=6)
    output = build_report(
        records=[exact, approximate, failure],
        output_root=tmp_path,
        experiment_status="unit_test",
        pretrained_status="not_run",
    )
    report = output.report_markdown.read_text(encoding="utf-8")

    assert "weight SHA-256" in report
    assert "tokenized SHA-256" in report
    assert "sample IDs SHA-256" in report
    assert "weight-sha-a" in report
    assert "tokenized-sha-a" in report
    assert report.count("raw source") >= 6
    assert report.count("raw line") >= 6
    assert "results/raw/attempt/eval.jsonl" in report
    assert "| failed | failure | topk_preserving | 0.05 | 0.95 | 16 |" in report


def test_missing_direct_raw_location_is_explicitly_not_available(
    tmp_path: Path,
) -> None:
    output = build_report(
        records=[
            _success_record(
                "unlocated",
                mode="exact",
                tau=0.0,
                raw_source=None,
                raw_line=None,
            )
        ],
        output_root=tmp_path,
        experiment_status="unit_test",
        pretrained_status="not_run",
    )
    report = output.report_markdown.read_text(encoding="utf-8")

    assert "| unlocated | not_available | not_available |" in report
