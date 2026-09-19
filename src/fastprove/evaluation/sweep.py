"""Configuration-only enumeration of the required noise ablation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
import math
from pathlib import Path
from typing import List, Optional, Sequence

import yaml

from ..io import resolve_path


@dataclass(frozen=True)
class SweepSpec:
    """One independently recorded evaluation configuration."""

    run_id: str
    mode: str
    tau_max: float
    tau_error: float
    alpha: Optional[float]
    preserve_top_k: Optional[int]

    def as_config(self) -> dict:
        """Return JSON-serializable attention settings."""

        return {
            "mode": self.mode,
            "tau_max": self.tau_max,
            "tau_error": self.tau_error,
            "alpha": self.alpha,
            "preserve_top_k": self.preserve_top_k,
        }


def _label(value: float) -> str:
    return ("%g" % value).replace(".", "p").replace("-", "m")


def enumerate_sweep_specs(path: Path) -> List[SweepSpec]:
    """Enumerate baselines, all 63 Top-k points, and 7 free points."""

    root = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(root, dict) or not isinstance(root.get("sweep"), dict):
        raise ValueError("sweep config must contain a sweep mapping")
    sweep = root["sweep"]
    tau_values = [float(value) for value in sweep["tau_max"]]
    alpha_values = [float(value) for value in sweep["alpha"]]
    top_k_values = [int(value) for value in sweep["preserve_top_k"]]
    tau_error = float(sweep["tau_error"])
    if (
        any(not math.isfinite(value) or value < 0 for value in tau_values)
        or not math.isfinite(tau_error)
        or tau_error < 0
    ):
        raise ValueError("tau values must be non-negative")
    if any(not 0 < value < 1 for value in alpha_values):
        raise ValueError("alpha values must lie in (0, 1)")
    if any(value < 1 for value in top_k_values):
        raise ValueError("preserve_top_k values must be positive")

    specs = [
        SweepSpec("plaintext", "plaintext", 0.0, 0.0, None, None),
        SweepSpec("exact", "exact", 0.0, 0.0, None, None),
    ]
    for tau, alpha, top_k in product(
        tau_values, alpha_values, top_k_values
    ):
        specs.append(
            SweepSpec(
                run_id="topk-tau%s-a%s-k%d"
                % (_label(tau), _label(alpha), top_k),
                mode="topk_preserving",
                tau_max=tau,
                tau_error=tau_error,
                alpha=alpha,
                preserve_top_k=top_k,
            )
        )
    for tau in tau_values:
        specs.append(
            SweepSpec(
                run_id="free-tau%s" % _label(tau),
                mode="free_bounded",
                tau_max=tau,
                tau_error=tau_error,
                alpha=None,
                preserve_top_k=None,
            )
        )
    if len({spec.run_id for spec in specs}) != len(specs):
        raise ValueError("sweep generated duplicate run IDs")
    return specs


def _resolve(path: Path) -> Path:
    return resolve_path(path)


def _spec_payload(spec: SweepSpec) -> dict:
    return {"run_id": spec.run_id, **spec.as_config()}


def _order_specs_for_exact_gate(specs: Sequence[SweepSpec]) -> list[SweepSpec]:
    """Put the exact-gate baselines before every approximate point.

    The default sweep enumerator already emits this order, but a selected
    candidate manifest may have been produced from an out-of-order or
    parallel JSONL file.  Approximate records must never be skipped merely
    because their entry appeared before the exact baseline; they are ordered
    deterministically here and retain their relative order within each mode.
    """

    priority = {
        "plaintext": 0,
        "exact": 1,
        "topk_preserving": 2,
        "free_bounded": 3,
    }
    return [
        spec
        for _, spec in sorted(
            enumerate(specs),
            key=lambda item: (priority.get(item[1].mode, 99), item[0]),
        )
    ]


def _load_selected_run_ids(path: Path) -> list[str]:
    """Load candidate IDs and require a ready selection manifest when present."""

    source = _resolve(path)
    if not source.is_file():
        raise FileNotFoundError("spec IDs file does not exist: %s" % source)
    text = source.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        ids = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        if isinstance(value, dict):
            if value.get("status") != "ready_for_full_evaluation":
                raise ValueError(
                    "candidate manifest status must be ready_for_full_evaluation"
                )
            ids = value.get("selected_run_ids")
        else:
            ids = value
    if not isinstance(ids, list) or not ids or any(
        not isinstance(item, str) or not item.strip() for item in ids
    ):
        raise ValueError("spec IDs file must contain a non-empty unique string list")
    normalized = [item.strip() for item in ids]
    if len(set(normalized)) != len(normalized):
        raise ValueError("spec IDs file must contain unique run IDs")
    return normalized
