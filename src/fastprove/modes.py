"""Neutral execution-mode enum shared by config, layers, and evaluation.

Dependency-free so ``config.py`` and ``noise_budget.py`` do not import the
attention kernels. ``fastprove.layers.attention`` re-exports the name.
"""

from __future__ import annotations

from enum import Enum


class AttentionMode(str, Enum):
    """Supported attention execution modes."""

    PLAINTEXT = "plaintext"
    EXACT = "exact"
    TOPK_PRESERVING = "topk_preserving"
    FREE_BOUNDED = "free_bounded"
