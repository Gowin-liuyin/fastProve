"""Deterministic, domain-separated random seed utilities."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict

import torch


def _encode_domain_value(value: Any) -> Dict[str, Any]:
    if value is None:
        return {"type": "none", "value": None}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("seed domain floats must be finite")
        return {"type": "float", "value": value.hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    raise TypeError(
        "seed domain values must be None, bool, int, finite float, or str"
    )


def derive_seed(global_seed: int, *domain: Any) -> int:
    """Derive a stable non-negative 63-bit seed.

    Python's process-randomized ``hash`` is deliberately not used. Domain values
    are serialized with explicit type information so that ``1`` and ``"1"`` do
    not collide.
    """

    typed_domain = [_encode_domain_value(value) for value in domain]
    payload = json.dumps(
        {"global_seed": int(global_seed), "domain": typed_domain},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63)


def make_generator(global_seed: int, *domain: Any) -> torch.Generator:
    """Return an independent CPU generator for a derived domain."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(derive_seed(global_seed, *domain))
    return generator


@dataclass(frozen=True)
class RequestContext:
    """Stable request identity used for per-request noise derivation."""

    global_seed: int
    request_id: str

    def seed_for(self, *domain: Any) -> int:
        """Derive a request-scoped seed."""

        return derive_seed(self.global_seed, "request", self.request_id, *domain)

    def generator_for(self, *domain: Any) -> torch.Generator:
        """Return a request-scoped CPU generator."""

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed_for(*domain))
        return generator
