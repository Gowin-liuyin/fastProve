"""Runtime device selection and memory-release helpers."""

from __future__ import annotations

import gc
import os
import platform
import resource
from typing import Optional, Tuple

import torch

from ..config import PrototypeConfig


def _resolve_runtime(config: PrototypeConfig) -> Tuple[torch.device, torch.dtype]:
    requested = config.runtime.device
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no fallback")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable; no fallback")
    if requested == "mps" and os.environ.get(
        "PYTORCH_ENABLE_MPS_FALLBACK", ""
    ).lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            "PYTORCH_ENABLE_MPS_FALLBACK is enabled; refusing a run that may "
            "silently execute operators on CPU"
        )
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[config.runtime.activation_dtype]
    return torch.device(requested), dtype


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _release_runtime_memory() -> None:
    """Release references/cached allocator blocks between independent runs."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        empty_cache = getattr(torch.mps, "empty_cache", None)
        if empty_cache is not None:
            empty_cache()


def _process_peak_rss_bytes() -> Optional[int]:
    """Return process peak RSS in bytes where the host exposes it."""

    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None
    # macOS reports bytes; Linux and most other Unix systems report KiB.
    return value if platform.system() == "Darwin" else value * 1024


def _mps_allocated_bytes() -> Optional[int]:
    """Return the MPS allocator counter when supported by this PyTorch build."""

    if not hasattr(torch, "mps") or not torch.backends.mps.is_available():
        return None
    current = getattr(torch.mps, "current_allocated_memory", None)
    if current is None:
        return None
    try:
        return int(current())
    except (RuntimeError, TypeError, ValueError):
        return None
