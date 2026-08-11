"""Client/server conversion bundles (manual section 55, task C4).

``ModelConverter.convert`` writes two strictly separated directories:

* ``server_model_dir``: the deployed weights the server needs to run forward
  (state_dict with deployed weights, pre-mixed vocabulary, permuted head, the
  server-visible ``common_qk`` and the Gram artifacts for ``rho``), plus a
  runtime configuration and a manifest. It contains **no** inverse vocabulary
  and no basis factors.
* ``client_secret_dir``: the vocabulary codec (``tau``/``tau^-1``) and the
  full basis factors needed to decode states in debug mode.

``load_server_model`` reconstructs a runnable module from the server bundle
alone; ``load_client`` restores the client-side codec and debug client.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .config import PrototypeConfig
from .layers.attention import ApproximationConfig, AttentionMode, ObfuscatedAttention
from .layers.embedding import SecureEmbedding
from .models.obfuscated import (
    ObfuscatedBlockClient,
    ObfuscatedDecoderBlock,
    ObfuscatedTinyCausalLM,
    _rms_scale_from_factors,
)
from .models.plain import PlainTinyCausalLM
from .state import MixedState
from .structured import StructuredBasis
from .transforms import BasisDescriptor

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


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _descend(module: nn.Module, path: str) -> nn.Module:
    current = module
    for part in path.split("."):
        current = getattr(current, part)
    return current


def _restore_block(
    runtime: Dict[str, Any], layer_index: int
) -> ObfuscatedDecoderBlock:
    block = ObfuscatedDecoderBlock.__new__(ObfuscatedDecoderBlock)
    nn.Module.__init__(block)
    block.config = _runtime_model_config(runtime)
    block.obfuscation = _runtime_obfuscation(runtime)
    block.mode = AttentionMode(runtime["mode"])
    block.layer_id = layer_index
    block.debug_enabled = False
    block.noise_injection_enabled = True
    block.refresh_noise_scale = 1.0
    block.hidden_basis = BasisDescriptor(**runtime["hidden_basis"])
    block.attention = ObfuscatedAttention(
        mode=block.mode,
        approximation=_runtime_approximation(runtime),
        layer_id="block-%d-attention" % layer_index,
        debug_enabled=False,
    )
    return block


def _runtime_model_config(runtime: Dict[str, Any]):
    from .config import ModelConfig

    return ModelConfig(**runtime["model"])


def _runtime_obfuscation(runtime: Dict[str, Any]):
    from .config import ObfuscationConfig

    return ObfuscationConfig(**runtime["obfuscation"])


def _runtime_approximation(
    runtime: Dict[str, Any],
) -> Optional[ApproximationConfig]:
    if runtime.get("approximation") is None:
        return None
    return ApproximationConfig(**runtime["approximation"])


def load_server_model(server_model_dir: str | Path) -> ObfuscatedTinyCausalLM:
    """Rebuild the runnable server module from the server bundle alone."""

    server_dir = Path(server_model_dir)
    runtime = json.loads(
        (server_dir / "runtime_config.json").read_text(encoding="utf-8")
    )
    weights_path = server_dir / ("model.pt" if (server_dir / "model.pt").exists()
                                  else "model.safetensors")
    if weights_path.suffix == ".safetensors":
        import safetensors.torch  # type: ignore

        state = dict(safetensors.torch.load_file(str(weights_path)))
    else:
        state = torch.load(weights_path, weights_only=True)

    config = _runtime_model_config(runtime)
    module = ObfuscatedTinyCausalLM.__new__(ObfuscatedTinyCausalLM)
    nn.Module.__init__(module)
    module.config = config
    module.obfuscation = _runtime_obfuscation(runtime)
    module.mode = AttentionMode(runtime["mode"])
    module.debug_enabled = False
    module.noise_injection_enabled = True
    module.initial_refresh_noise_scale = 1.0
    module.hidden_basis = BasisDescriptor(**runtime["hidden_basis"])
    module.value_basis_condition_numbers = tuple(
        tuple(row) for row in runtime["value_basis_condition_numbers"]
    )
    module.value_basis_fingerprints = tuple(
        tuple(row) for row in runtime["value_basis_fingerprints"]
    )
    module.embedding = SecureEmbedding.__new__(SecureEmbedding)
    nn.Module.__init__(module.embedding)
    module.embedding.vocab_size = config.vocab_size
    module.embedding.basis_descriptor = module.hidden_basis
    module.blocks = nn.ModuleList(
        _restore_block(runtime, index)
        for index in range(config.num_layers)
    )
    for name, value in state.items():
        parent_path, _, leaf = name.rpartition(".")
        parent = module if not parent_path else _descend(module, parent_path)
        parent.register_buffer(leaf, value.clone())

    module._rms_scale = _rms_scale_from_factors(
        config.hidden_size,
        module.deployed_gram_blocks,
        module.deployed_gram_perm,
    )
    module._debug_basis_unmix = None
    cache_identity = runtime.get("cache_identity") or []
    for block in module.blocks:
        block._rms_scale = _rms_scale_from_factors(
            config.hidden_size,
            block.deployed_gram_blocks,
            block.deployed_gram_perm,
        )
        block._debug_basis_unmix = None
        if block.layer_id < len(cache_identity):
            block.cache_identity = cache_identity[block.layer_id]
    return module


class ClientBundle:
    """Client-side material restored from the client secret directory."""

    def __init__(
        self,
        token_codec,
        hidden_basis: StructuredBasis,
        *,
        debug_enabled: bool = True,
    ) -> None:
        from .codec import TokenCodec

        self.token_codec: TokenCodec = token_codec
        self.hidden_basis = hidden_basis
        self.client = ObfuscatedBlockClient(
            hidden_basis, debug_enabled=debug_enabled
        )

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.token_codec.encode(token_ids)

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.token_codec.decode(token_ids)

    def decode_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Recover plaintext-domain logits from column-permuted output."""

        return logits[..., self.token_codec.permutation.to(logits.device)]


def load_client(client_secret_dir: str | Path) -> ClientBundle:
    """Restore the codec and debug client from the client secret directory."""

    secret_dir = Path(client_secret_dir)
    from .codec import TokenCodec

    codec_state = torch.load(secret_dir / "token_codec.pt", weights_only=True)
    token_codec = TokenCodec(
        permutation=codec_state["permutation"],
        inverse_permutation=codec_state["inverse_permutation"],
        vocab_size=codec_state["vocab_size"],
        fingerprint=codec_state["fingerprint"],
    )
    basis_state = torch.load(secret_dir / "bases.pt", weights_only=True)
    hidden = basis_state["hidden"]
    hidden_basis = StructuredBasis(**hidden)
    return ClientBundle(token_codec, hidden_basis)


class ModelConverter:
    """Write a server model and a separate client secret bundle."""

    def convert(
        self,
        plain_model: nn.Module,
        config: PrototypeConfig,
        client_secret_dir: str | Path,
        server_model_dir: str | Path,
    ) -> ConversionManifest:
        """Write a server model and a separate client secret bundle."""

        from .codec import generate_token_codec
        from .models.obfuscated import ObfuscatedTinyCausalLM

        started = time.perf_counter()
        plain_model.eval()
        _require_supported_architecture(plain_model)
        model = config.model
        if model.hidden_size % model.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )
        if model.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        total = model.hidden_size + config.obfuscation.hidden_noise_dim
        if total % config.obfuscation.basis_block_size != 0:
            raise ValueError(
                "basis_block_size must divide hidden_size + hidden_noise_dim"
            )
        tied_embedding = plain_model.lm_head.weight is plain_model.embedding.weight
        if not tied_embedding:
            tied_embedding = torch.equal(
                plain_model.lm_head.weight.detach().cpu(),
                plain_model.embedding.weight.detach().cpu(),
            )

        approximation = None
        if config.attention.mode in (
            AttentionMode.TOPK_PRESERVING,
            AttentionMode.FREE_BOUNDED,
        ):
            approximation = ApproximationConfig(
                tau_max=config.attention.tau_max,
                tau_error=config.attention.tau_error,
                alpha=config.attention.alpha,
                preserve_top_k=config.attention.preserve_top_k,
            )
        token_codec = generate_token_codec(
            model.vocab_size,
            seed=config.runtime.seed,
            domain="tiny-lm-vocab",
        )
        converted = ObfuscatedTinyCausalLM.from_plain(
            plain_model,
            obfuscation=config.obfuscation,
            mode=config.attention.mode,
            approximation=approximation,
            seed=config.runtime.seed,
            debug_enabled=False,
            token_codec=token_codec,
        )
        module = converted.module.eval()

        server_dir = Path(server_model_dir)
        secret_dir = Path(client_secret_dir)
        server_dir.mkdir(parents=True, exist_ok=True)
        secret_dir.mkdir(parents=True, exist_ok=True)

        state = module.state_dict()
        _assert_server_bundle_is_clean(state)
        weights_format = "torch"
        try:
            import safetensors.torch  # type: ignore

            safetensors.torch.save_file(state, server_dir / "model.safetensors")
            weights_format = "safetensors"
        except ImportError:
            torch.save(state, server_dir / "model.pt")

        runtime = {
            "schema": "fastprove.server.runtime.v1",
            "mode": str(module.mode.value)
            if hasattr(module.mode, "value")
            else str(module.mode),
            "approximation": (
                None
                if approximation is None
                else {
                    "tau_max": approximation.tau_max,
                    "tau_error": approximation.tau_error,
                    "alpha": approximation.alpha,
                    "preserve_top_k": approximation.preserve_top_k,
                }
            ),
            "model": {
                "vocab_size": model.vocab_size,
                "hidden_size": model.hidden_size,
                "intermediate_size": model.intermediate_size,
                "num_layers": model.num_layers,
                "num_attention_heads": model.num_attention_heads,
                "num_key_value_heads": model.num_key_value_heads,
                "max_sequence_length": model.max_sequence_length,
                "rms_epsilon": model.rms_epsilon,
                "rope_theta": model.rope_theta,
                "qkv_bias": model.qkv_bias,
            },
            "obfuscation": {
                "hidden_noise_dim": config.obfuscation.hidden_noise_dim,
                "value_noise_dim_per_head": (
                    config.obfuscation.value_noise_dim_per_head
                ),
                "max_condition_number": config.obfuscation.max_condition_number,
                "noise_propagation_gamma": (
                    config.obfuscation.noise_propagation_gamma
                ),
                "refresh_mode": config.obfuscation.refresh_mode,
                "basis_block_size": config.obfuscation.basis_block_size,
                "lm_head_mode": config.obfuscation.lm_head_mode,
            },
            "hidden_basis": {
                "signal_dim": module.hidden_basis.signal_dim,
                "noise_dim": module.hidden_basis.noise_dim,
                "condition_number": module.hidden_basis.condition_number,
                "fingerprint": module.hidden_basis.fingerprint,
            },
            "value_basis_condition_numbers": [
                list(row)
                for row in module.value_basis_condition_numbers
            ],
            "value_basis_fingerprints": [
                list(row) for row in module.value_basis_fingerprints
            ],
            "cache_identity": [block.cache_identity for block in module.blocks],
            "server_visible_key_material": {
                "common_qk": (
                    "applied after RoPE; cannot be absorbed into a deployed "
                    "weight; the server must hold it and the QK geometry is "
                    "not protected (docs/threat_model.md)"
                )
            },
        }
        runtime_path = server_dir / "runtime_config.json"
        runtime_path.write_text(
            json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
        )
        server_manifest = {
            "schema": "fastprove.server.manifest.v1",
            "fastprove_version": _fastprove_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "converted_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tied_embedding": tied_embedding,
            "weights_format": weights_format,
            "runtime_config_sha256": _sha256_file(runtime_path),
            "server_state_keys": list(state.keys()),
        }
        manifest_path = server_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(server_manifest, indent=2) + "\n", encoding="utf-8"
        )

        # -- client bundle -------------------------------------------------
        token_codec.save_client_secret(secret_dir / "token_codec.pt")
        hidden_basis = _rebuild_hidden_basis(config, module)
        value_bases = _rebuild_value_bases(model, config)
        torch.save(
            {
                "hidden": _basis_factors(hidden_basis),
                "value": [
                    [_basis_factors(basis) for basis in layer]
                    for layer in value_bases
                ],
            },
            secret_dir / "bases.pt",
        )
        client_manifest = {
            "schema": "fastprove.client.manifest.v1",
            "token_codec_fingerprint": token_codec.fingerprint,
            "hidden_basis_fingerprint": hidden_basis.fingerprint,
            "value_basis_fingerprints": [
                [basis.fingerprint for basis in layer]
                for layer in value_bases
            ],
            "server_runtime_config_sha256": _sha256_file(runtime_path),
            "warning": (
                "client material must never be copied into a server directory"
            ),
        }
        (secret_dir / "manifest.json").write_text(
            json.dumps(client_manifest, indent=2) + "\n", encoding="utf-8"
        )

        return ConversionManifest(
            fastprove_version=_fastprove_version(),
            torch_version=torch.__version__,
            platform=platform.platform(),
            converted_at_utc=server_manifest["converted_at_utc"],
            conversion_time_seconds=time.perf_counter() - started,
            server_model_dir=str(server_dir),
            client_secret_dir=str(secret_dir),
            weights_format=weights_format,
            model_sha256={"runtime_config": server_manifest["runtime_config_sha256"]},
            runtime_config_sha256=server_manifest["runtime_config_sha256"],
            client_manifest_sha256=_sha256_file(secret_dir / "manifest.json"),
            tied_embedding=tied_embedding,
            server_state_keys=tuple(state.keys()),
        )


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

    from .structured import generate_structured_basis

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

    from .structured import generate_structured_basis

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
