"""Generic JSON persistence helpers shared by experiment scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def resolve_path(path: Path) -> Path:
    """Expand ``~`` and resolve a path (shared by experiment scripts)."""

    return path.expanduser().resolve()


def write_json_file(
    results: Dict[str, Any],
    output: str | Path,
    *,
    base_dir: Optional[str | Path] = None,
) -> Path:
    """Persist a JSON document, resolving relative paths against base_dir."""

    out_path = Path(output)
    if not out_path.is_absolute():
        out_path = (
            Path(base_dir) if base_dir is not None else Path.cwd()
        ) / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    return out_path
