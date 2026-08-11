"""Generation and validation of well-conditioned mixing transforms."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

from .seed import make_generator


def _validate_transform_dtype(dtype: torch.dtype) -> None:
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("transform deployment dtype must be FP32 or FP64")


def _fingerprint(
    matrix: torch.Tensor, signal_dim: int, noise_dim: int
) -> str:
    raw = matrix.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
    header = ("%d:%d:" % (signal_dim, noise_dim)).encode("ascii")
    return hashlib.sha256(header + raw.tobytes()).hexdigest()


@dataclass(frozen=True)
class BasisDescriptor:
    """Public state metadata that contains no decoding matrix."""

    signal_dim: int
    noise_dim: int
    condition_number: float
    fingerprint: str

    @property
    def total_dim(self) -> int:
        """Augmented dimension."""

        return self.signal_dim + self.noise_dim


@dataclass(frozen=True)
class BasisTransform:
    """Client/conversion-side mixing basis and precomputed inverse."""

    matrix: torch.Tensor
    inverse: torch.Tensor
    signal_dim: int
    noise_dim: int
    condition_number: float
    fingerprint: str

    def __post_init__(self) -> None:
        total_dim = self.signal_dim + self.noise_dim
        if self.signal_dim <= 0 or self.noise_dim <= 0:
            raise ValueError("signal_dim and noise_dim must be positive")
        if self.matrix.shape != (total_dim, total_dim):
            raise ValueError("mixing matrix shape does not match dimensions")
        if self.inverse.shape != self.matrix.shape:
            raise ValueError("inverse shape must match mixing matrix")
        _validate_transform_dtype(self.matrix.dtype)
        _validate_transform_dtype(self.inverse.dtype)
        if (
            self.matrix.dtype != self.inverse.dtype
            or self.matrix.device != self.inverse.device
        ):
            raise ValueError("matrix and inverse dtype/device must match")
        if not torch.isfinite(self.matrix).all() or not torch.isfinite(
            self.inverse
        ).all():
            raise ValueError("mixing transform must be finite")
        if not math.isfinite(self.condition_number) or self.condition_number < 1:
            raise ValueError("condition number must be finite and at least one")
        if self.fingerprint != _fingerprint(
            self.matrix, self.signal_dim, self.noise_dim
        ):
            raise ValueError("basis fingerprint does not match its matrix")
        matrix_fp64 = self.matrix.detach().to(device="cpu", dtype=torch.float64)
        inverse_fp64 = self.inverse.detach().to(
            device="cpu", dtype=torch.float64
        )
        identity = torch.eye(total_dim, dtype=torch.float64)
        if not torch.allclose(
            matrix_fp64 @ inverse_fp64,
            identity,
            atol=2e-5,
            rtol=2e-5,
        ):
            raise ValueError("stored inverse does not invert the mixing matrix")
        actual_condition = float(torch.linalg.cond(matrix_fp64).item())
        condition_tolerance = max(1e-5, actual_condition * 1e-5)
        if abs(actual_condition - self.condition_number) > condition_tolerance:
            raise ValueError(
                "declared condition number does not match mixing matrix"
            )

    @property
    def total_dim(self) -> int:
        """Augmented dimension."""

        return self.signal_dim + self.noise_dim

    @property
    def descriptor(self) -> BasisDescriptor:
        """Return production-safe basis metadata without an inverse."""

        return BasisDescriptor(
            signal_dim=self.signal_dim,
            noise_dim=self.noise_dim,
            condition_number=self.condition_number,
            fingerprint=self.fingerprint,
        )

    def validate_integrity(self) -> None:
        """Re-check integrity after construction.

        ``frozen=True`` prevents attribute rebinding but cannot prevent an
        in-place mutation of a tensor.  Conversion/debug boundaries therefore
        call this method before using the basis, so a stale fingerprint or
        inverse cannot silently decode a state with the wrong transform.
        """

        if self.fingerprint != _fingerprint(
            self.matrix, self.signal_dim, self.noise_dim
        ):
            raise ValueError("basis matrix was mutated after construction")
        matrix_fp64 = self.matrix.detach().to(device="cpu", dtype=torch.float64)
        inverse_fp64 = self.inverse.detach().to(
            device="cpu", dtype=torch.float64
        )
        identity = torch.eye(self.total_dim, dtype=torch.float64)
        if not torch.allclose(
            matrix_fp64 @ inverse_fp64,
            identity,
            atol=2e-5,
            rtol=2e-5,
        ):
            raise ValueError("basis inverse no longer matches its matrix")


def generate_orthogonal(
    dim: int,
    *,
    seed: int,
    domain: str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a deterministic dense orthogonal matrix via FP64 QR."""

    if dim <= 0:
        raise ValueError("dim must be positive")
    _validate_transform_dtype(dtype)
    generator = make_generator(seed, domain, "orthogonal", dim)
    raw = torch.randn(dim, dim, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(raw)
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(0)
    return q.to(dtype=dtype)


def generate_signed_permutation(
    dim: int,
    *,
    seed: int,
    domain: str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a deterministic signed permutation matrix."""

    if dim <= 0:
        raise ValueError("dim must be positive")
    _validate_transform_dtype(dtype)
    generator = make_generator(seed, domain, "signed-permutation", dim)
    permutation = torch.randperm(dim, generator=generator)
    signs = torch.randint(0, 2, (dim,), generator=generator, dtype=torch.int64)
    signs = signs.to(torch.float64).mul_(2).sub_(1)
    matrix = torch.zeros(dim, dim, dtype=torch.float64)
    matrix[torch.arange(dim), permutation] = signs
    return matrix.to(dtype=dtype)


def generate_transform(
    signal_dim: int,
    noise_dim: int,
    *,
    seed: int,
    domain: str,
    max_condition_number: float = 10.0,
    dtype: torch.dtype = torch.float32,
) -> BasisTransform:
    """Generate ``M = Pi D B`` and precompute its inverse offline.

    The orthogonal/permutation factors have unit condition number. Diagonal
    log-scales are bounded so their ratio cannot exceed the requested maximum.
    Validation and inversion are performed in FP64; deployed matrices use the
    requested dtype.
    """

    if signal_dim <= 0 or noise_dim <= 0:
        raise ValueError("signal_dim and noise_dim must be positive")
    _validate_transform_dtype(dtype)
    if max_condition_number < 1.0 or not math.isfinite(max_condition_number):
        raise ValueError("max_condition_number must be finite and at least one")
    total_dim = signal_dim + noise_dim
    generator = make_generator(
        seed, domain, "basis", signal_dim, noise_dim, total_dim
    )
    permutation_indices = torch.randperm(total_dim, generator=generator)
    permutation = torch.eye(total_dim, dtype=torch.float64)[permutation_indices]
    orthogonal = generate_orthogonal(
        total_dim,
        seed=seed,
        domain="%s-basis-%d-%d" % (domain, signal_dim, noise_dim),
        dtype=torch.float64,
    )

    log_half_range = 0.5 * math.log(max_condition_number)
    unit = torch.rand(total_dim, generator=generator, dtype=torch.float64)
    log_scales = (2.0 * unit - 1.0) * log_half_range
    diagonal = torch.diag(torch.exp(log_scales))
    matrix_fp64 = permutation @ diagonal @ orthogonal
    condition = float(torch.linalg.cond(matrix_fp64).item())
    tolerance = max(1e-10, max_condition_number * 1e-10)
    if not math.isfinite(condition) or condition > max_condition_number + tolerance:
        raise ValueError(
            "generated transform condition %.6g exceeds limit %.6g"
            % (condition, max_condition_number)
        )
    inverse_fp64 = torch.linalg.solve(
        matrix_fp64, torch.eye(total_dim, dtype=torch.float64)
    )
    matrix = matrix_fp64.to(dtype=dtype)
    inverse = inverse_fp64.to(dtype=dtype)
    return BasisTransform(
        matrix=matrix,
        inverse=inverse,
        signal_dim=signal_dim,
        noise_dim=noise_dim,
        condition_number=condition,
        fingerprint=_fingerprint(matrix, signal_dim, noise_dim),
    )
