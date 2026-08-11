"""Protocol-aligned five-layer evaluation suite for fastProve.

This package implements the authoritative evaluation protocol in
``docs/literature-review/reports/evaluation_protocol_fastProve.md``.

Terminology (protocol §8.2): report results as *obfuscated-state relative to
plaintext*, not as cryptographic ciphertext. fastProve is an augmented
covariant obfuscation prototype with bounded auxiliary noise, not standard
ciphertext or fully encrypted inference.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
