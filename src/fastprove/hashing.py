"""Shared file hashing helper.

Single implementation of the seven copies previously scattered across
pretrained loaders, evaluation code, the converter, and scripts. The chunk
size never changes the digest; callers that historically read in 1 MiB
chunks (converter) and 8 MiB chunks (everything else) produce identical
values.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union


def sha256_file(
    path: Union[str, Path], *, chunk_size: int = 8 * 1024 * 1024
) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
