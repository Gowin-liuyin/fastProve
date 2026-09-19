"""Record the S2 dual-channel ledger produced by a real conversion.

Runs ``ModelConverter`` for one config into a temporary directory and copies
the resulting ``noise_budget.json`` (plus the obfuscation section of
``runtime_config.json``) into a raw record. The conversion itself is where the
ledger is validated; this script makes the artifact auditable.

Usage:
    PYTHONPATH=src python scripts/record_noise_budget.py \
        --config configs/tiny_approx.yaml \
        --output results/raw/noise_budget_tiny_approx.json
"""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from pathlib import Path

import torch

from fastprove.config import load_config
from fastprove.converter import ModelConverter
from fastprove.models.plain import PlainTinyCausalLM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    plain = PlainTinyCausalLM(
        config.model, seed=config.runtime.seed, debug_enabled=False
    ).eval()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ModelConverter().convert(
            plain,
            config,
            client_secret_dir=root / "client",
            server_model_dir=root / "server",
        )
        budget = json.loads(
            (root / "server" / "noise_budget.json").read_text(encoding="utf-8")
        )
        runtime = json.loads(
            (root / "server" / "runtime_config.json").read_text(encoding="utf-8")
        )
    record = {
        "schema": "fastprove.noise_budget_record.v1",
        "config": str(arguments.config),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "seed": config.runtime.seed,
        "runtime_config_obfuscation": runtime["obfuscation"],
        "noise_budget": budget,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
