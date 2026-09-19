"""Server-bundle safety gates for client/server conversion bundles.

This module also defines :class:`ConversionManifest`, the recorded metadata of
one conversion: it is the return value of ``ModelConverter.convert`` and its
fields are computed by the gates and the writer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import torch

from ..models.plain import PlainTinyCausalLM


#: Substrings that must never appear in the server bundle. ``common_qk`` is
#: deliberately absent from this list: it is applied after RoPE and cannot be
#: absorbed into a deployed weight, so the server must hold it. It is instead
#: explicitly annotated in ``runtime_config.json`` as server-visible key
#: material and the threat model records that the QK geometry is not
#: protected. Do not rename it to dodge the check. The ``deployed_*`` family
#: is the shipped server artifact whose recorded consequences are in
#: docs/threat_model.md (B1.3).
_FORBIDDEN_SERVER_KEY_LEAF_TOKENS = (
    "perm_in",
    "perm_out",
    "scales",
    "blocks",
    "gram",
    "inverse",
    "rotation",
    "coupling",
    "propagator",
    "tau",
    "token_codec",
    "basis",
)


@dataclass(frozen=True)
class ConversionManifest:
    """Recorded metadata for one client/server conversion."""

    fastprove_version: str
    torch_version: str
    platform: str
    converted_at_utc: str
    conversion_time_seconds: float
    server_model_dir: str
    client_secret_dir: str
    weights_format: str
    model_sha256: Dict[str, str]
    runtime_config_sha256: str
    client_manifest_sha256: str
    tied_embedding: bool
    server_state_keys: Sequence[str] = field(default_factory=tuple)


def _assert_server_bundle_is_clean(state: Dict[str, torch.Tensor]) -> None:
    """Fail conversion if key material reached the server bundle.

    Only the leaf name is scanned: module paths such as ``blocks.0.*`` are
    structural and not key material.
    """

    offending = []
    for key in state:
        leaf = key.rsplit(".", 1)[-1]
        if leaf.startswith("deployed_"):
            continue
        for token in _FORBIDDEN_SERVER_KEY_LEAF_TOKENS:
            if token in leaf.lower():
                offending.append(key)
                break
    if offending:
        raise RuntimeError(
            "server checkpoint contains key material: %s" % offending
        )


def _require_supported_architecture(plain: PlainTinyCausalLM) -> None:
    """Reject architectures the converter does not understand."""

    for name, module in plain.named_modules():
        class_name = module.__class__.__name__
        if not any(
            marker in class_name
            for marker in (
                "DecoderBlock",
                "TinyCausalLM",
                "Embedding",
                "Linear",
                "ModuleList",
            )
        ):
            raise NotImplementedError(
                "unsupported module %s of type %s" % (name, class_name)
            )
