"""Basis factor serialization and deterministic basis reconstruction."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ..config import PrototypeConfig
from ..models.obfuscated import ObfuscatedTinyCausalLM
from ..structured import StructuredBasis, generate_structured_basis


def _fastprove_version() -> str:
    try:
        from importlib import metadata

        return metadata.version("fastprove")
    except Exception:
        return "0.1.0"


def _basis_factors(basis: StructuredBasis) -> Dict[str, Any]:
    return {
        "perm_in": basis.perm_in,
        "scales": basis.scales,
        "blocks": basis.blocks,
        "perm_out": basis.perm_out,
        "gram_blocks": basis.gram_blocks,
        "signal_dim": basis.signal_dim,
        "noise_dim": basis.noise_dim,
        "condition_number": basis.condition_number,
        "fingerprint": basis.fingerprint,
    }


def _rebuild_hidden_basis(
    config: PrototypeConfig, module: ObfuscatedTinyCausalLM
) -> StructuredBasis:
    """Reconstruct the hidden basis deterministically from the conversion seed."""

    return generate_structured_basis(
        config.model.hidden_size,
        config.obfuscation.hidden_noise_dim,
        seed=config.runtime.seed,
        domain="tiny-lm-shared-hidden-basis",
        block_size=config.obfuscation.basis_block_size,
        max_condition_number=config.obfuscation.max_condition_number,
    )


def _rebuild_value_bases(
    model, config: PrototypeConfig
) -> Tuple[Tuple[StructuredBasis, ...], ...]:
    """Reconstruct per-layer value bases deterministically."""

    head_dim = model.head_dim
    value_noise = config.obfuscation.value_noise_dim_per_head
    return tuple(
        tuple(
            generate_structured_basis(
                head_dim,
                value_noise,
                seed=config.runtime.seed,
                domain="block-%d-value-%d" % (layer, head),
                block_size=head_dim + value_noise,
                max_condition_number=config.obfuscation.max_condition_number,
            )
            for head in range(model.num_key_value_heads)
        )
        for layer in range(model.num_layers)
    )
