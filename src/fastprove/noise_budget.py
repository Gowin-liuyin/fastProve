"""Dual-channel noise budget ledger (stage S2).

The ambient augmented state space splits as ``R^N = ker(L) + row_space(L)``
(``L`` is the semantic read matrix, ``c @ L = h``). Every noise injection is
assigned to exactly one channel:

* **channel 0** (``ker(L)``): the only subspace that commutes exactly with
  nonlinear checkpoints. It costs no accuracy and provides **exactly zero**
  information-theoretic hiding: ``h = c @ L`` is deterministic in ``c``, so an
  observer knowing ``L`` recovers ``h`` regardless of the in-cone magnitude
  (theorem 4, T4 in ``results/raw/noise_ordering_analysis_20260919.json``).
  Channel 0 must never be described as a privacy source.
* **channel 1** (semantic direction): every unit of noise here perturbs ``h``
  by exactly ``n @ L`` (theorem 2) and therefore must carry a computable
  accuracy bound whose source is one of the whitelisted mechanisms:
  the Top-k margin ``Delta_k`` / Value diameter ``D_V`` / RMS scale ``rho``.

The ledger turns "which channel does each knob belong to, what does it buy,
what does it cost" into a machine-readable record written next to the server
runtime configuration, so a mismatch fails at conversion time instead of
being discovered by reading prose. See
``docs/IMPLEMENTATION_PROMPT_dual_channel.md`` S2 and
``docs/noise_placement_explained.pdf`` T1-T9.

Non-claims: nothing here asserts a bit-security level, IND-CPA, LWE, or
differential privacy. ``information_theoretic_hiding_bits`` is an
information-theoretic retention figure for a Gaussian channel, not a security
parameter (see :func:`semantic_hiding_bits`).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .config import PrototypeConfig
from .layers.attention import AttentionMode
from .structured import StructuredBasis

#: Whitelisted sources of a computable channel-1 accuracy bound. A channel-1
#: budget without one of these is an unbounded claim and is rejected.
BOUND_SOURCES = ("margin_delta_k", "value_diameter_D_V", "rho")

_EXACT_FAMILY = (AttentionMode.PLAINTEXT, AttentionMode.EXACT)


def semantic_hiding_bits(sigma_delta_over_sigma_h: float) -> float:
    """Retained bits per coordinate for a Gaussian semantic channel.

    For ``h -> h + delta`` with isotropic Gaussian signal and noise, the
    retained mutual information per coordinate is
    ``0.5 * log2(1 + SNR)`` with ``SNR = (sigma_h / sigma_delta)^2``
    (theorem 9 / T9). ``ratio == 0`` returns ``inf``: without semantic noise
    there is no hiding at all.

    This is a retention figure under stated Gaussian assumptions, not a
    security level: it does not model the known-plaintext algebraic attacks
    recorded in ``docs/PROBLEMS.md``.
    """

    if not math.isfinite(sigma_delta_over_sigma_h) or sigma_delta_over_sigma_h < 0:
        raise ValueError("ratio must be finite and non-negative")
    if sigma_delta_over_sigma_h == 0.0:
        return math.inf
    return 0.5 * math.log2(
        1.0 + 1.0 / (sigma_delta_over_sigma_h * sigma_delta_over_sigma_h)
    )


@dataclass(frozen=True)
class ChannelBudget:
    """One channel's noise budget and its audited consequences."""

    channel: int
    dimension: int
    energy_ratio: float
    accuracy_cost_bound: float | None
    information_theoretic_hiding_bits: float | None
    bound_source: str | None

    def __post_init__(self) -> None:
        if self.channel not in (0, 1):
            raise ValueError("channel must be 0 (ker L) or 1 (semantic)")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if not math.isfinite(self.energy_ratio) or self.energy_ratio < 0:
            raise ValueError("energy_ratio must be finite and non-negative")
        if self.channel == 0:
            # Theorem 4: in-cone noise costs nothing and hides nothing. Any
            # other accounting is a bookkeeping error, not a modeling choice.
            if self.accuracy_cost_bound != 0.0:
                raise ValueError(
                    "channel 0 has exactly zero accuracy cost; got %r"
                    % (self.accuracy_cost_bound,)
                )
            if self.information_theoretic_hiding_bits != 0.0:
                raise ValueError(
                    "channel 0 has exactly zero information-theoretic hiding "
                    "(theorem 4: h = c @ L is deterministic regardless of "
                    "in-cone magnitude); got %r"
                    % (self.information_theoretic_hiding_bits,)
                )
            if self.bound_source is not None:
                raise ValueError("channel 0 has no channel-1 bound source")
        else:
            if self.bound_source not in BOUND_SOURCES:
                raise ValueError(
                    "channel 1 requires a whitelisted bound source %s; got %r"
                    % (", ".join(BOUND_SOURCES), self.bound_source)
                )
            if self.accuracy_cost_bound is None or not math.isfinite(
                self.accuracy_cost_bound
            ):
                raise ValueError(
                    "channel 1 requires a finite accuracy_cost_bound"
                )
            if self.accuracy_cost_bound < 0:
                raise ValueError("accuracy_cost_bound must be non-negative")
            if self.information_theoretic_hiding_bits is not None and (
                math.isnan(self.information_theoretic_hiding_bits)
                or self.information_theoretic_hiding_bits < 0
            ):
                raise ValueError("hiding bits must be non-negative or None")


@dataclass(frozen=True)
class DualChannelBudget:
    """The two-channel split with machine-readable accounting.

    ``mode`` is the *effective* mode of the ledger, which is stricter than the
    attention mode alone:

    * ``plaintext`` / ``exact`` require zero semantic noise (the signal path is
      mathematically exact);
    * ``exact_attention_with_semantic_noise`` records the accepted deployment
      compromise (exact attention covariance plus a nonzero channel-1 knob such
      as ``rho_noise_bound``) without mislabeling it as exact. The recorded
      consequence of that compromise is in ``docs/PROBLEMS.md`` section 6;
    * otherwise the attention mode is carried through unchanged.

    Constructing an ``exact`` or ``plaintext`` budget with nonzero semantic
    noise raises, so a ledger cannot claim exactness while paying channel-1
    accuracy.
    """

    free: ChannelBudget
    semantic: ChannelBudget
    sources: dict[str, float]
    mode: str

    def __post_init__(self) -> None:
        if self.free.channel != 0 or self.semantic.channel != 1:
            raise ValueError("free must be channel 0 and semantic channel 1")
        if self.mode in ("plaintext", "exact"):
            if (
                self.semantic.energy_ratio != 0.0
                or self.semantic.accuracy_cost_bound != 0.0
            ):
                raise ValueError(
                    "%s mode requires zero semantic noise; got energy_ratio=%r, "
                    "accuracy_cost_bound=%r"
                    % (
                        self.mode,
                        self.semantic.energy_ratio,
                        self.semantic.accuracy_cost_bound,
                    )
                )

    @classmethod
    def from_config(
        cls,
        config: PrototypeConfig,
        *,
        basis: StructuredBasis | None = None,
        router_tau_max: float = 0.0,
    ) -> "DualChannelBudget":
        """Derive the ledger from a validated prototype configuration.

        ``router_tau_max`` folds the optional MoE router margin bound in;
        ``PrototypeConfig`` has no router section, so MoE callers pass it
        explicitly. The channel-1 accuracy bound is the maximum over the
        recorded sources, all of which remain visible in ``manifest()``.

        Exact attention with a nonzero channel-1 knob is labeled
        ``exact_attention_with_semantic_noise`` instead of ``exact``: the
        signal path is then approximate even though the attention covariance
        is exact (docs/PROBLEMS.md section 6 records that deployment
        compromise).
        """

        if not math.isfinite(router_tau_max) or router_tau_max < 0:
            raise ValueError("router_tau_max must be finite and non-negative")
        mode = AttentionMode(config.attention.mode)
        free_dimension = (
            int(basis.free_noise_basis().shape[0])
            if basis is not None
            else int(config.obfuscation.hidden_noise_dim)
        )
        sources = {
            "rho": float(config.obfuscation.rho_noise_bound),
            "margin_delta_k": float(
                max(
                    config.attention.tau_max,
                    config.attention.tau_error,
                    router_tau_max,
                )
            ),
        }
        semantic_cost = max(sources.values())
        if mode in _EXACT_FAMILY and semantic_cost != 0.0:
            effective_mode = "exact_attention_with_semantic_noise"
        else:
            effective_mode = mode.value
        bound_source = max(
            ("rho", "margin_delta_k"), key=lambda name: sources[name]
        )
        free = ChannelBudget(
            channel=0,
            dimension=free_dimension,
            energy_ratio=float(config.obfuscation.free_noise_energy_ratio),
            accuracy_cost_bound=0.0,
            information_theoretic_hiding_bits=0.0,
            bound_source=None,
        )
        semantic = ChannelBudget(
            channel=1,
            dimension=int(config.model.hidden_size),
            energy_ratio=float(config.obfuscation.rho_noise_bound),
            accuracy_cost_bound=semantic_cost,
            information_theoretic_hiding_bits=None,
            bound_source=bound_source,
        )
        return cls(
            free=free,
            semantic=semantic,
            sources=sources,
            mode=effective_mode,
        )

    def manifest(self) -> dict[str, object]:
        """Machine-readable record written next to raw results."""

        return {
            "schema": "fastprove.noise_budget.v1",
            "mode": self.mode,
            "exact_signal_path": (
                self.semantic.energy_ratio == 0.0
                and self.semantic.accuracy_cost_bound == 0.0
            ),
            "free": {
                **asdict(self.free),
                "statement": (
                    "channel 0 (ker L): zero accuracy cost, zero "
                    "information-theoretic hiding by theorem 4; not a privacy "
                    "source"
                ),
            },
            "semantic": {
                **asdict(self.semantic),
                "statement": (
                    "channel 1 (semantic direction): perturbs h by exactly "
                    "n @ L (theorem 2); the bound source is %s"
                    % self.semantic.bound_source
                ),
            },
            "semantic_sources": dict(self.sources),
            "channel_1_hiding_bits_note": (
                "left None in the config-derived ledger: it depends on the "
                "signal scale sigma_h and is computed per experiment via "
                "semantic_hiding_bits()"
            ),
        }

    def assert_consistent_with(
        self,
        config: PrototypeConfig,
        *,
        basis: StructuredBasis | None = None,
    ) -> None:
        """Fail conversion when the ledger disagrees with the configuration.

        When ``basis`` is provided, the channel-0 dimension is checked against
        ``ker(L)`` itself (``basis.free_noise_basis().shape[0]``), not merely
        against the configured noise width.
        """

        if self.free.energy_ratio != config.obfuscation.free_noise_energy_ratio:
            raise ValueError(
                "free channel energy_ratio %r does not match config %r"
                % (
                    self.free.energy_ratio,
                    config.obfuscation.free_noise_energy_ratio,
                )
            )
        if self.free.dimension != config.obfuscation.hidden_noise_dim:
            raise ValueError(
                "free channel dimension %d does not match hidden_noise_dim %d"
                % (self.free.dimension, config.obfuscation.hidden_noise_dim)
            )
        if basis is not None:
            observed = int(basis.free_noise_basis().shape[0])
            if self.free.dimension != observed:
                raise ValueError(
                    "free channel dimension %d does not match dim ker(L) = %d"
                    % (self.free.dimension, observed)
                )
        if self.semantic.energy_ratio != config.obfuscation.rho_noise_bound:
            raise ValueError(
                "semantic channel energy_ratio %r does not match "
                "rho_noise_bound %r"
                % (self.semantic.energy_ratio, config.obfuscation.rho_noise_bound)
            )
        required = max(
            config.obfuscation.rho_noise_bound,
            config.attention.tau_max,
            config.attention.tau_error,
        )
        if self.semantic.accuracy_cost_bound is None or (
            self.semantic.accuracy_cost_bound < required
        ):
            raise ValueError(
                "semantic accuracy_cost_bound %r does not cover the "
                "configured channel-1 noise %r"
                % (self.semantic.accuracy_cost_bound, required)
            )
        if AttentionMode(config.attention.mode) in _EXACT_FAMILY:
            # Exact attention may only be labeled exact when it carries no
            # semantic noise; the compromise label is the truthful alternative.
            if self.mode not in ("exact", "plaintext",
                                 "exact_attention_with_semantic_noise"):
                raise ValueError(
                    "exact-family attention requires an exact-family ledger "
                    "mode; got %r" % self.mode
                )
            if self.mode == "exact_attention_with_semantic_noise" and (
                self.semantic.energy_ratio == 0.0
                and self.semantic.accuracy_cost_bound == 0.0
            ):
                raise ValueError(
                    "ledger labeled exact_attention_with_semantic_noise but "
                    "carries zero semantic noise"
                )
        elif self.mode != AttentionMode(config.attention.mode).value:
            raise ValueError(
                "ledger mode %r does not match attention mode %r"
                % (self.mode, AttentionMode(config.attention.mode).value)
            )
