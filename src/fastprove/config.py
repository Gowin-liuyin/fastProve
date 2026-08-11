"""Validated configuration dataclasses and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict

import yaml

from .layers.attention import AttentionMode


@dataclass(frozen=True)
class ModelConfig:
    """Tiny Llama-like model dimensions."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_sequence_length: int
    rms_epsilon: float = 1e-5
    rope_theta: float = 10000.0
    qkv_bias: bool = False

    def __post_init__(self) -> None:
        positive_values = (
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.max_sequence_length,
        )
        if any(value <= 0 for value in positive_values):
            raise ValueError("model dimensions must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "number of attention heads must be divisible by KV heads"
            )
        if self.head_dim % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE")
        if self.rms_epsilon <= 0 or self.rope_theta <= 0:
            raise ValueError("RMS epsilon and RoPE theta must be positive")

    @property
    def head_dim(self) -> int:
        """Per-head Query/Key/Value signal dimension."""

        return self.hidden_size // self.num_attention_heads

    @property
    def query_heads_per_kv_head(self) -> int:
        """Number of Query heads sharing one KV head."""

        return self.num_attention_heads // self.num_key_value_heads


@dataclass(frozen=True)
class ObfuscationConfig:
    """Augmented-state dimensions and transform constraints."""

    hidden_noise_dim: int
    value_noise_dim_per_head: int
    max_condition_number: float
    noise_propagation_gamma: float
    refresh_mode: str
    basis_block_size: int = 16

    def __post_init__(self) -> None:
        if self.hidden_noise_dim <= 0 or self.value_noise_dim_per_head <= 0:
            raise ValueError("noise dimensions must be positive")
        if self.basis_block_size <= 0:
            raise ValueError("basis_block_size must be positive")
        if not math.isfinite(self.max_condition_number) or self.max_condition_number < 1:
            raise ValueError("max_condition_number must be at least one")
        if not math.isfinite(self.noise_propagation_gamma) or not 0 < self.noise_propagation_gamma < 1:
            raise ValueError("noise propagation gamma must be in (0, 1)")
        if self.refresh_mode not in ("fixed_debug", "per_request"):
            raise ValueError("unsupported refresh_mode")


@dataclass(frozen=True)
class AttentionConfig:
    """Attention mode and optional bounded-noise settings."""

    mode: AttentionMode
    tau_max: float
    tau_error: float
    alpha: float
    preserve_top_k: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", AttentionMode(self.mode))
        if not math.isfinite(self.tau_max) or not math.isfinite(self.tau_error):
            raise ValueError("tau bounds must be finite")
        if self.tau_max < 0 or self.tau_error < 0:
            raise ValueError("tau bounds must be non-negative")
        if not math.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if self.preserve_top_k < 1:
            raise ValueError("preserve_top_k must be positive")
        if self.mode in (AttentionMode.PLAINTEXT, AttentionMode.EXACT) and (
            self.tau_max != 0 or self.tau_error != 0
        ):
            raise ValueError("plaintext/exact modes require zero noise bounds")


@dataclass(frozen=True)
class RuntimeConfig:
    """Deterministic runtime selection."""

    seed: int
    request_id: str
    device: str
    activation_dtype: str
    debug_enabled: bool

    def __post_init__(self) -> None:
        if self.device not in ("cpu", "mps", "cuda"):
            raise ValueError("unsupported device")
        if self.activation_dtype not in ("float32", "bfloat16"):
            raise ValueError("unsupported activation dtype")
        if not self.request_id:
            raise ValueError("request_id must be non-empty")


@dataclass(frozen=True)
class EvaluationConfig:
    """Small aligned evaluation settings."""

    batch_size: int
    sequence_length: int
    sample_count: int
    generation_tokens: int
    warmup_runs: int
    timed_runs: int
    bootstrap_replicates: int = 1000

    def __post_init__(self) -> None:
        values = (
            self.batch_size,
            self.sequence_length,
            self.sample_count,
            self.generation_tokens,
            self.timed_runs,
        )
        if any(value <= 0 for value in values) or self.warmup_runs < 0:
            raise ValueError("evaluation sizes must be positive")
        if self.bootstrap_replicates < 0:
            raise ValueError("bootstrap_replicates must be non-negative")


@dataclass(frozen=True)
class PrototypeConfig:
    """Complete prototype configuration."""

    model: ModelConfig
    obfuscation: ObfuscationConfig
    attention: AttentionConfig
    runtime: RuntimeConfig
    evaluation: EvaluationConfig

    def __post_init__(self) -> None:
        if (
            self.evaluation.sequence_length + self.evaluation.generation_tokens
            > self.model.max_sequence_length
        ):
            raise ValueError("evaluation and generation exceed model context")
        total = self.model.hidden_size + self.obfuscation.hidden_noise_dim
        block = self.obfuscation.basis_block_size
        if total % block != 0:
            raise ValueError(
                "basis_block_size %d must divide hidden_size + hidden_noise_dim "
                "= %d; adjust hidden_noise_dim or basis_block_size"
                % (block, total)
            )
        # Value bases always use a single dense orthogonal block: head_dim +
        # value_noise_dim_per_head is small (130 for a 3B model, ~0.12% of the
        # per-token MAC budget), so no block partition is needed there and no
        # divisibility constraint applies to it.


def _mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("%s must be a mapping" % name)
    return value


def load_config(path: Path) -> PrototypeConfig:
    """Load and validate a tiny YAML configuration."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = _mapping(raw, "config")
    required = ("model", "obfuscation", "attention", "runtime", "evaluation")
    missing = [name for name in required if name not in root]
    if missing:
        raise ValueError("missing config sections: %s" % ", ".join(missing))
    return PrototypeConfig(
        model=ModelConfig(**_mapping(root["model"], "model")),
        obfuscation=ObfuscationConfig(
            **_mapping(root["obfuscation"], "obfuscation")
        ),
        attention=AttentionConfig(
            **_mapping(root["attention"], "attention")
        ),
        runtime=RuntimeConfig(**_mapping(root["runtime"], "runtime")),
        evaluation=EvaluationConfig(
            **_mapping(root["evaluation"], "evaluation")
        ),
    )
