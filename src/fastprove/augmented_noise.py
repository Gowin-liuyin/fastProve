"""Independent per-record auxiliary noise (plan sections 7.4 and 7.6).

First-version defaults follow the augmented-noise plan:

* signal-to-noise coupling ``C = 0`` so the contribution of the independent
  random terms is separated from any signal-linear image;
* propagator ``G = gamma P_e`` with a fixed signed permutation and
  ``0 < gamma < 1``;
* bounded uniform component distributions with variance matched to a fixed
  energy scale derived from a *public calibration* RMS, not per-request
  magnitudes.

All sampling is domain separated by
``(key_epoch, request_nonce, sample_id, absolute_token_position, layer_id,
operation, head_id)`` and is reproducible experiment randomness, not a
deployment secrecy mechanism (plan 7.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .seed import derive_seed
from .transforms import generate_signed_permutation

_SQRT3 = math.sqrt(3.0)


@dataclass(frozen=True)
class NoiseRefreshSpec:
    """Static parameters of the independent noise chain.

    ``energy_ratio`` is ``beta`` in the plan; the component standard deviation
    is ``sigma_e = beta * H / sqrt(r)`` where ``H`` is the calibration RMS of
    the hidden vectors at the relevant checkpoint.
    """

    noise_dim: int
    gamma: float
    energy_ratio: float

    def __post_init__(self) -> None:
        if self.noise_dim <= 0:
            raise ValueError("noise_dim must be positive")
        if not math.isfinite(self.gamma) or not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must be in (0, 1)")
        if not math.isfinite(self.energy_ratio) or self.energy_ratio < 0.0:
            raise ValueError("energy_ratio must be finite and non-negative")

    def component_sigma(self, calibration_rms: float) -> float:
        """Return ``sigma_e`` for a public calibration RMS value."""

        if not math.isfinite(calibration_rms) or calibration_rms < 0.0:
            raise ValueError("calibration_rms must be finite and non-negative")
        return self.energy_ratio * calibration_rms / math.sqrt(self.noise_dim)


def generate_propagator(
    spec: NoiseRefreshSpec,
    *,
    seed: int,
    domain: str,
) -> torch.Tensor:
    """Return the fixed signed-permutation propagator ``G = gamma P_e``.

    Offline key material; generated once per key period and validated like any
    other transform factor. FP64 storage for conversion use.
    """

    permutation = generate_signed_permutation(
        spec.noise_dim, seed=seed, domain=domain, dtype=torch.float64
    )
    return spec.gamma * permutation


def sample_uniform(
    shape: Tuple[int, ...],
    *,
    half_width: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample ``U[-half_width, half_width]`` component-wise."""

    if not math.isfinite(half_width) or half_width < 0.0:
        raise ValueError("half_width must be finite and non-negative")
    unit = torch.rand(shape, generator=generator, dtype=torch.float64)
    return (2.0 * unit - 1.0) * half_width


def record_generator(
    global_seed: int,
    *,
    key_epoch: str,
    request_nonce: str,
    sample_id: str,
    token_position: int,
    layer_id: str,
    operation: str,
    head_id: int = 0,
) -> torch.Generator:
    """Return a generator for one record's noise draw.

    Domain separation uses absolute token positions so prefill and decode
    agree at the same position, and different samples never share a stream
    regardless of batch arrangement (plan 7.4).
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        derive_seed(
            global_seed,
            "augmented-noise",
            key_epoch,
            request_nonce,
            sample_id,
            int(token_position),
            layer_id,
            operation,
            int(head_id),
        )
    )
    return generator


def sample_initial_noise(
    spec: NoiseRefreshSpec,
    *,
    calibration_rms: float,
    shape: Tuple[int, ...],
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample ``e_0`` with components ``U[-sqrt(3) sigma_e, sqrt(3) sigma_e]``."""

    sigma = spec.component_sigma(calibration_rms)
    return sample_uniform(shape, half_width=_SQRT3 * sigma, generator=generator)


def sample_refresh_noise(
    spec: NoiseRefreshSpec,
    *,
    calibration_rms: float,
    shape: Tuple[int, ...],
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample ``xi`` with components ``U[-sqrt(3) sigma_e sqrt(1-gamma^2), ...]``.

    The variance ``sigma_e^2 (1 - gamma^2)`` makes the stationary covariance of
    the linear chain equal to ``sigma_e^2 I`` under the plan's conditional
    proposition. This is not a stationarity proof for a full Transformer.
    """

    sigma = spec.component_sigma(calibration_rms)
    half_width = _SQRT3 * sigma * math.sqrt(1.0 - spec.gamma * spec.gamma)
    return sample_uniform(shape, half_width=half_width, generator=generator)


def apply_noise_refresh(
    noise: torch.Tensor,
    propagator: torch.Tensor,
    refresh: torch.Tensor,
) -> torch.Tensor:
    """Return ``gamma e P_e + xi`` (plan 7.4) with ``C = 0``."""

    if noise.shape != refresh.shape:
        raise ValueError("noise and refresh shapes must match")
    if noise.shape[-1] != propagator.shape[0]:
        raise ValueError("noise width does not match propagator")
    if not propagator.is_floating_point() or not noise.is_floating_point():
        raise ValueError("noise chain tensors must use a floating dtype")
    compute = noise.detach().to(dtype=torch.float64) @ propagator.detach().to(
        dtype=torch.float64
    ) + refresh.detach().to(dtype=torch.float64)
    return compute.to(dtype=noise.dtype)


@dataclass(frozen=True)
class NoiseCoverage:
    """Coverage statistics of the noise rows ``M_n`` (plan 7.6)."""

    row_norm_min: float
    row_norm_max: float
    near_zero_coordinate_fraction: float
    effective_rank: float

    def to_dict(self) -> dict:
        return {
            "row_norm_min": self.row_norm_min,
            "row_norm_max": self.row_norm_max,
            "near_zero_coordinate_fraction": self.near_zero_coordinate_fraction,
            "effective_rank": self.effective_rank,
        }


def noise_coverage_metrics(
    noise_rows: torch.Tensor,
    *,
    near_zero_threshold: float = 1e-6,
) -> NoiseCoverage:
    """Measure how mixed-state coordinates are covered by auxiliary noise.

    For ``Cov(e) = sigma_e^2 I`` the conditional variance of mixed coordinate
    ``j`` is ``sigma_e^2 ||M_n[:, j]||^2``; a coordinate with a near-zero column
    norm receives essentially no noise. Effective rank follows Roy & Vetterli.
    """

    if noise_rows.ndim != 2:
        raise ValueError("noise_rows must be a rank-two matrix [r, n]")
    if not torch.isfinite(noise_rows).all():
        raise ValueError("noise_rows must be finite")
    rows64 = noise_rows.detach().to(dtype=torch.float64)
    column_norms = rows64.T.norm(dim=1)
    singular_values = torch.linalg.svdvals(rows64)
    energy = singular_values**2
    total = float(energy.sum())
    if total <= 0.0:
        raise ValueError("noise_rows has zero energy; no coverage possible")
    probabilities = energy / total
    entropy = float(-(probabilities * probabilities.clamp_min(1e-300).log()).sum())
    effective_rank = float(math.exp(entropy))
    return NoiseCoverage(
        row_norm_min=float(column_norms.min()),
        row_norm_max=float(column_norms.max()),
        near_zero_coordinate_fraction=float(
            (column_norms < near_zero_threshold).to(torch.float64).mean()
        ),
        effective_rank=effective_rank,
        )


def observed_noise_ratios(
    signal: torch.Tensor,
    noise: torch.Tensor,
) -> dict:
    """Return observed ``||e||/||h||`` percentiles for one checkpoint.

    The public scale fixes only the *expected* ratio; per-token actual ratios
    are recorded so numerical-budget violations are visible (plan 7.4).
    """

    signal_norms = signal.detach().to(dtype=torch.float64).norm(dim=-1)
    noise_norms = noise.detach().to(dtype=torch.float64).norm(dim=-1)
    if torch.any(signal_norms == 0.0):
        raise ValueError("zero-norm signal cannot produce a meaningful ratio")
    ratios = noise_norms / signal_norms
    quantiles = (0.5, 0.9, 0.99)
    values = torch.quantile(
        ratios.flatten().cpu(),
        torch.tensor(quantiles, dtype=torch.float64),
    )
    return {
        "median": float(values[0]),
        "p90": float(values[1]),
        "p99": float(values[2]),
        "max": float(ratios.max()),
    }
