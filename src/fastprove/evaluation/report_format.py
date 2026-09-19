"""Formatting helpers for report derivation."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence


def _get(
    record: dict, *path: str, default: Any = None
) -> Optional[float]:
    value = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first(mapping: object, *keys: str, default: object = None) -> object:
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def _as_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_value(value: object, digits: int = 6) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "not_available" if value is None else str(value)
    return ("%%.%dg" % digits) % numeric


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    rendered = [list(str(value) for value in row) for row in rows]
    if not rendered:
        return "_没有可用记录。_"
    head = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"
        for row in rendered
    ]
    return "\n".join([head, separator, *body])
