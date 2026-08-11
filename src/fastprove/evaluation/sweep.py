"""Configuration-only enumeration of the required noise ablation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from pathlib import Path
from typing import List, Optional

import yaml


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
