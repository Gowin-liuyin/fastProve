"""Execution-environment metadata capture."""

from __future__ import annotations

import os
import platform
import shutil
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def environment_meta(device: torch.device) -> Dict[str, Any]:
    cuda = torch.cuda.is_available()
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_memory_total_bytes": (
            int(torch.cuda.get_device_properties(0).total_memory) if cuda else None
        ),
    }


def environment_metadata() -> Dict[str, Any]:
    """Return JSON-safe software and accelerator metadata without fallback."""

    def package_version(name: str) -> Optional[str]:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            return None

    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    device_names: Dict[str, Optional[str]] = {
        "cuda": torch.cuda.get_device_name(0) if cuda_available else None,
        "mps": "Apple Metal Performance Shaders" if mps_available else None,
    }
    physical_memory = None
    try:
        physical_memory = int(
            os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        )
    except (AttributeError, OSError, ValueError):
        pass
    disk = shutil.disk_usage(Path.cwd())
    mps_fallback_env = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "")
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": package_version("numpy"),
        "transformers_version": package_version("transformers"),
        "safetensors_version": package_version("safetensors"),
        "fastprove_version": package_version("fastprove") or "source-tree",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": int(os.cpu_count() or 1),
        "physical_memory_bytes": physical_memory,
        "disk_total_bytes": int(disk.total),
        "disk_free_bytes": int(disk.free),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "mps_available": mps_available,
        "mps_fallback_env": mps_fallback_env or None,
        "device_names": device_names,
    }
