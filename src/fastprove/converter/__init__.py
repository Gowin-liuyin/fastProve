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

The implementation lives in the sibling modules ``hygiene`` (safety gates and
the conversion manifest), ``load`` (server bundle loading), ``client`` (client
bundle), ``save`` (the conversion writer) and ``basis_io`` (basis factor
serialization). Every top-level name the former single module exposed is
re-exported here so ``fastprove.converter.<name>`` keeps working.
"""

from __future__ import annotations

from ..augmented_noise import deployed_free_noise_sigma
from .basis_io import (
    _basis_factors,
    _fastprove_version,
    _rebuild_hidden_basis,
    _rebuild_value_bases,
)
from .client import ClientBundle, load_client
from ..config import PrototypeConfig
from .hygiene import (
    ConversionManifest,
    _assert_server_bundle_is_clean,
    _FORBIDDEN_SERVER_KEY_LEAF_TOKENS,
    _require_supported_architecture,
)
from ..layers.attention import (
    ApproximationConfig,
    AttentionMode,
    ObfuscatedAttention,
)
from ..layers.embedding import SecureEmbedding
from .load import (
    _descend,
    _restore_block,
    _runtime_approximation,
    _runtime_model_config,
    _runtime_obfuscation,
    _sha256_file,
    load_server_model,
)
from ..models.obfuscated import (
    ObfuscatedBlockClient,
    ObfuscatedDecoderBlock,
    ObfuscatedTinyCausalLM,
    _rms_scale_from_factors,
)
from ..models.plain import PlainTinyCausalLM
from ..noise_budget import DualChannelBudget
from .save import ModelConverter
from ..state import MixedState
from ..structured import StructuredBasis
from ..transforms import BasisDescriptor

__all__ = [
    "ApproximationConfig",
    "AttentionMode",
    "BasisDescriptor",
    "ClientBundle",
    "ConversionManifest",
    "DualChannelBudget",
    "ModelConverter",
    "MixedState",
    "ObfuscatedAttention",
    "ObfuscatedBlockClient",
    "ObfuscatedDecoderBlock",
    "ObfuscatedTinyCausalLM",
    "PlainTinyCausalLM",
    "PrototypeConfig",
    "SecureEmbedding",
    "StructuredBasis",
    "_FORBIDDEN_SERVER_KEY_LEAF_TOKENS",
    "_assert_server_bundle_is_clean",
    "_basis_factors",
    "_descend",
    "_fastprove_version",
    "_rebuild_hidden_basis",
    "_rebuild_value_bases",
    "_require_supported_architecture",
    "_restore_block",
    "_rms_scale_from_factors",
    "_runtime_approximation",
    "_runtime_model_config",
    "_runtime_obfuscation",
    "_sha256_file",
    "deployed_free_noise_sigma",
    "load_client",
    "load_server_model",
]
