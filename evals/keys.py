"""Independent obfuscation master-key generation (protocol §6.1).

fastProve has at least three randomness classes:

1. **Obfuscation-key randomness** — permutations, bases, noise subspaces.
2. **Refresh-noise randomness** — per-request ξ under a fixed key.
3. **Generation randomness** — temperature / top-p sampling.

This module addresses class 1: each master key yields one full model
conversion. Confidence intervals must cover both sample and key variance
(stratified bootstrap).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, List, Sequence

from fastprove.seed import derive_seed


@dataclass(frozen=True)
class MasterKey:
    """One independent obfuscation master key."""

    key_id: int
    master_seed: int
    label: str

    def conversion_seed(self) -> int:
        """Seed passed to model conversion (transforms, permutations)."""

        return derive_seed(self.master_seed, "obfuscation-master-key", self.key_id)

    def request_seed(self, request_id: str) -> int:
        """Global seed for a request under this key."""

        return derive_seed(
            self.master_seed, "request-under-key", self.key_id, request_id
        )

    def public_fingerprint(self) -> str:
        """Return an audit identifier without exposing a derivation seed."""

        payload = ("fastprove-master-key-v1:" + str(self.master_seed)).encode(
            "ascii"
        )
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict:
        """Return a public audit record without secret seed material.

        The master and conversion seeds are key material in this prototype:
        publishing either one lets a reader regenerate the deterministic
        conversion basis.  Reproducibility is obtained from the private run
        manifest, not by placing secrets in ``results/raw``.
        """

        return {
            "key_id": self.key_id,
            "label": self.label,
            "key_fingerprint": self.public_fingerprint(),
            "seed_material_recorded": False,
        }


def generate_master_keys(
    *,
    count: int = 5,
    base_seed: int = 20260731,
    labels: Sequence[str] | None = None,
) -> List[MasterKey]:
    """Generate ``count`` independent master keys (protocol: ≥3, default 5).

    Parameters
    ----------
    count:
        Number of independent keys. Must be ≥ 1; protocol recommends ≥ 3.
    base_seed:
        Root seed; each key derives a distinct master_seed via domain separation.
    labels:
        Optional human labels; defaults to ``key-00``, ``key-01``, …
    """

    if count < 1:
        raise ValueError("key count must be at least 1")
    if count < 3:
        # Protocol recommends ≥3; allow fewer only for smoke tests.
        pass
    if labels is not None and len(labels) != count:
        raise ValueError("labels length must match key count")
    keys: List[MasterKey] = []
    for index in range(count):
        master_seed = derive_seed(base_seed, "master-key-pool", index)
        label = labels[index] if labels is not None else "key-%02d" % index
        keys.append(
            MasterKey(key_id=index, master_seed=master_seed, label=label)
        )
    return keys


def ensure_min_keys(keys: Iterable[MasterKey], *, minimum: int = 3) -> List[MasterKey]:
    """Return the key list after validating the protocol minimum."""

    materialised = list(keys)
    if len(materialised) < minimum:
        raise ValueError(
            "protocol §6.1 requires at least %d independent keys; got %d"
            % (minimum, len(materialised))
        )
    return materialised
