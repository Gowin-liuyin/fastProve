"""Structured mixing basis ``M = Pi1 @ D @ B @ Pi2`` with O(n*b) application.

Mathematics (row-vector convention, ``c = [h, e] @ M``):

* ``Pi1``, ``Pi2`` are permutations, ``D`` is a bounded nonzero diagonal, and
  ``B`` is block diagonal orthogonal with uniform block width ``b``.
* ``kappa_2(M) == kappa_2(D)`` because permutations and block-orthogonal
  factors have unit condition number.
* Applying ``M`` or ``M^-1`` costs ``O(n*b)`` instead of ``O(n^2)``.
* ``A_gram = P @ P.T`` with ``P = M^-1[:, :d]`` is block diagonal in ``Pi2``
  order, so ``||h||^2 = c @ A_gram @ c.T`` is an ``O(n*b)`` blockwise
  contraction that materializes only one scalar per block. This is what lets
  the deployed forward pass obtain the RMSNorm scale without decoding ``h``.

Offline generation, inversion and validation are FP64. Deployed factors are
stored in the requested dtype (FP32 by default).

Security note: every attribute of :class:`StructuredBasis` is conversion- or
client-side key material. ``descriptor`` is the only production-safe view.
Block-local mixing lowers the cost of a known-plaintext recovery of ``M`` from
``O(n)`` to ``O(b)`` samples per block relative to a dense basis; see
``docs/threat_model.md`` for the recorded consequence.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

from .seed import make_generator
from .transforms import BasisDescriptor

_FP64 = torch.float64


def _validate_dtype(dtype: torch.dtype) -> None:
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("structured basis dtype must be FP32 or FP64")


def _fingerprint(
    perm_in: torch.Tensor,
    scales: torch.Tensor,
    blocks: torch.Tensor,
    perm_out: torch.Tensor,
    signal_dim: int,
    noise_dim: int,
) -> str:
    """Hash the basis factors.

    Floating factors are canonicalized to FP64 before hashing so that the
    fingerprint identifies the *numerical values* actually stored, independent
    of whether the caller hashes before or after a dtype cast. Callers must
    still hash the tensors they store, not pre-cast copies.
    """

    digest = hashlib.sha256()
    digest.update(("%d:%d:" % (signal_dim, noise_dim)).encode("ascii"))
    for tensor in (perm_in, scales, blocks, perm_out):
        value = tensor.detach().to(device="cpu").contiguous()
        if value.is_floating_point():
            value = value.to(dtype=_FP64).contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class StructuredBasis:
    """Structured invertible mixing basis with cheap application."""

    perm_in: torch.Tensor
    scales: torch.Tensor
    blocks: torch.Tensor
    perm_out: torch.Tensor
    gram_blocks: torch.Tensor
    signal_dim: int
    noise_dim: int
    condition_number: float
    fingerprint: str

    def __post_init__(self) -> None:
        total = self.signal_dim + self.noise_dim
        if self.signal_dim <= 0 or self.noise_dim <= 0:
            raise ValueError("signal_dim and noise_dim must be positive")
        if self.perm_in.shape != (total,) or self.perm_out.shape != (total,):
            raise ValueError("permutation shape must be [signal_dim+noise_dim]")
        if self.perm_in.dtype != torch.int64 or self.perm_out.dtype != torch.int64:
            raise ValueError("permutations must be int64")
        for permutation in (self.perm_in, self.perm_out):
            if not torch.equal(
                torch.sort(permutation.cpu()).values, torch.arange(total)
            ):
                raise ValueError("permutation must be a bijection on [0, n)")
        if self.scales.shape != (total,):
            raise ValueError("scales shape must be [signal_dim+noise_dim]")
        if not torch.isfinite(self.scales).all() or torch.any(self.scales == 0):
            raise ValueError("diagonal scales must be finite and nonzero")
        if self.blocks.ndim != 3 or self.blocks.shape[1] != self.blocks.shape[2]:
            raise ValueError("blocks must have shape [count, b, b]")
        count, block, _ = self.blocks.shape
        if count * block != total:
            raise ValueError("block partition does not cover the basis")
        if self.gram_blocks.shape != self.blocks.shape:
            raise ValueError("gram_blocks shape must match blocks")
        _validate_dtype(self.scales.dtype)
        _validate_dtype(self.blocks.dtype)
        _validate_dtype(self.gram_blocks.dtype)
        if not math.isfinite(self.condition_number) or self.condition_number < 1:
            raise ValueError("condition number must be finite and at least one")
        if self.fingerprint != _fingerprint(
            self.perm_in,
            self.scales,
            self.blocks,
            self.perm_out,
            self.signal_dim,
            self.noise_dim,
        ):
            raise ValueError("basis fingerprint does not match its factors")
        blocks64 = self.blocks.detach().to(device="cpu", dtype=_FP64)
        identity = torch.eye(block, dtype=_FP64).expand(count, block, block)
        # FP32-deployed blocks are orthogonal only to FP32 precision once cast
        # back to FP64, so the tolerance must follow the stored dtype.
        atol = 1e-10 if self.blocks.dtype == _FP64 else 5e-6
        if not torch.allclose(
            blocks64 @ blocks64.transpose(-1, -2), identity, atol=atol, rtol=0.0
        ):
            raise ValueError("basis blocks are not orthogonal")

    @property
    def total_dim(self) -> int:
        """Augmented dimension ``n = signal_dim + noise_dim``."""

        return self.signal_dim + self.noise_dim

    @property
    def block_size(self) -> int:
        """Uniform orthogonal block width ``b``."""

        return int(self.blocks.shape[1])

    @property
    def block_count(self) -> int:
        """Number of orthogonal blocks ``m = n / b``."""

        return int(self.blocks.shape[0])

    @property
    def descriptor(self) -> BasisDescriptor:
        """Production-safe metadata that carries no matrix material."""

        return BasisDescriptor(
            signal_dim=self.signal_dim,
            noise_dim=self.noise_dim,
            condition_number=self.condition_number,
            fingerprint=self.fingerprint,
        )

    @staticmethod
    def _blockwise(x: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
        count, block, _ = blocks.shape
        segments = x.reshape(*x.shape[:-1], count, block)
        mixed = torch.einsum("...mi,mij->...mj", segments, blocks)
        return mixed.reshape(*x.shape)

    def mix(self, augmented: torch.Tensor) -> torch.Tensor:
        """Return ``augmented @ M`` in ``O(n*b)``."""

        if augmented.shape[-1] != self.total_dim:
            raise ValueError("augmented width does not match basis")
        device, dtype = augmented.device, augmented.dtype
        gathered = augmented[..., torch.argsort(self.perm_in).to(device)]
        scaled = gathered * self.scales.to(device=device, dtype=dtype)
        mixed = self._blockwise(scaled, self.blocks.to(device=device, dtype=dtype))
        return mixed[..., torch.argsort(self.perm_out).to(device)]

    def unmix(self, mixed: torch.Tensor) -> torch.Tensor:
        """Return ``mixed @ M^-1`` in ``O(n*b)``.

        Conversion/debug only. The deployed forward pass must not call this;
        deployed weights absorb ``M^-1`` offline instead.
        """

        if mixed.shape[-1] != self.total_dim:
            raise ValueError("mixed width does not match basis")
        device, dtype = mixed.device, mixed.dtype
        gathered = mixed[..., self.perm_out.to(device)]
        transposed = (
            self.blocks.transpose(-1, -2).contiguous().to(device=device, dtype=dtype)
        )
        unblocked = self._blockwise(gathered, transposed)
        scaled = unblocked / self.scales.to(device=device, dtype=dtype)
        return scaled[..., self.perm_in.to(device)]

    def signal_norm_squared(self, mixed: torch.Tensor) -> torch.Tensor:
        """Return ``||h||^2`` from ``mixed`` without materializing ``h``.

        Blockwise contraction of ``c A c^T`` with ``A = P P^T``, which is block
        diagonal in ``perm_out`` order. Only one scalar per block is
        materialized, so the signal is never reconstructed. FP32 reduction;
        clamped at zero because ``A`` is positive semidefinite but finite
        precision can yield a tiny negative value.

        **Numerical constraint (important).** ``A`` annihilates the auxiliary
        subspace by cancellation, so the FP32 relative error of the result grows
        like ``(||e|| / ||h||)^2``:

        =============== =================
        ``||e||/||h||``  FP32 rel. error
        =============== =================
        1                4e-7
        10               1e-6
        100              1e-4
        1000             2e-2
        =============== =================

        The deployed noise scales must therefore keep ``||e|| / ||h||`` bounded;
        :func:`auxiliary_magnitude_bound` and the conversion-time check in
        ``layers/deployed.py`` enforce it. Raising the auxiliary magnitude for
        stronger obfuscation degrades the accuracy of ``rho`` and hence the
        signal path. Do not silently widen this bound.
        """

        if mixed.shape[-1] != self.total_dim:
            raise ValueError("mixed width does not match basis")
        device = mixed.device
        gathered = mixed[..., self.perm_out.to(device)].to(dtype=torch.float32)
        segments = gathered.reshape(
            *mixed.shape[:-1], self.block_count, self.block_size
        )
        gram = self.gram_blocks.to(device=device, dtype=torch.float32)
        contracted = torch.einsum("...mi,mij,...mj->...", segments, gram, segments)
        return contracted.clamp_min(0.0)

    def rms_scale(self, mixed: torch.Tensor, eps: float) -> torch.Tensor:
        """Return ``sqrt(||h||^2 / d + eps)`` with shape ``[..., 1]`` in FP32."""

        if eps <= 0:
            raise ValueError("eps must be positive")
        squared = self.signal_norm_squared(mixed)
        return torch.sqrt(squared / float(self.signal_dim) + float(eps)).unsqueeze(-1)

    def dense(self) -> torch.Tensor:
        """Return dense FP64 ``M``. Offline conversion/validation only."""

        return self.mix(torch.eye(self.total_dim, dtype=_FP64))

    def dense_inverse(self) -> torch.Tensor:
        """Return dense FP64 ``M^-1``. Offline conversion only."""

        return self.unmix(torch.eye(self.total_dim, dtype=_FP64))

    def signal_projection(self) -> torch.Tensor:
        """Return ``P = M^-1[:, :d]`` in FP64. Offline conversion only."""

        return self.dense_inverse()[:, : self.signal_dim]

    def noise_projection(self) -> torch.Tensor:
        """Return ``N = M^-1[:, d:]`` in FP64. Offline conversion only."""

        return self.dense_inverse()[:, self.signal_dim :]

    def signal_rows(self) -> torch.Tensor:
        """Return ``M_top = M[:d]`` in FP64. Offline conversion only."""

        return self.dense()[: self.signal_dim]

    def noise_rows(self) -> torch.Tensor:
        """Return ``M_bot = M[d:]`` in FP64. Offline conversion only."""

        return self.dense()[self.signal_dim :]

    def validate_integrity(self) -> None:
        """Re-check factors and the dense round trip after construction.

        ``frozen=True`` blocks attribute rebinding but not in-place tensor
        mutation, so conversion/debug boundaries call this before trusting a
        basis. The round-trip tolerance follows the stored dtype: FP32-deployed
        factors invert only to FP32 precision.
        """

        if self.fingerprint != _fingerprint(
            self.perm_in,
            self.scales,
            self.blocks,
            self.perm_out,
            self.signal_dim,
            self.noise_dim,
        ):
            raise ValueError("basis factors were mutated after construction")
        product = self.dense() @ self.dense_inverse()
        atol = 1e-9 if self.blocks.dtype == _FP64 else 1e-4
        if not torch.allclose(
            product, torch.eye(self.total_dim, dtype=_FP64), atol=atol, rtol=0.0
        ):
            raise ValueError("structured inverse does not invert the basis")


def _gram_blocks_from(basis: StructuredBasis) -> torch.Tensor:
    """Extract block-diagonal blocks of ``P @ P.T`` in ``perm_out`` order."""

    projection = basis.signal_projection()
    gram = projection @ projection.T
    order = basis.perm_out
    permuted = gram[order][:, order]
    count, block = basis.block_count, basis.block_size
    residual = permuted.clone()
    for index in range(count):
        low, high = index * block, (index + 1) * block
        residual[low:high, low:high] = 0
    deviation = float(residual.abs().max())
    if deviation > 1e-9:
        raise ValueError(
            "signal Gram is not block diagonal (max off-block %.3e)" % deviation
        )
    return torch.stack(
        [
            permuted[
                index * block : (index + 1) * block,
                index * block : (index + 1) * block,
            ]
            for index in range(count)
        ]
    )


def generate_structured_basis(
    signal_dim: int,
    noise_dim: int,
    *,
    seed: int,
    domain: str,
    block_size: int,
    max_condition_number: float = 10.0,
    dtype: torch.dtype = torch.float32,
) -> StructuredBasis:
    """Generate ``M = Pi1 D B Pi2`` deterministically.

    ``block_size`` must divide ``signal_dim + noise_dim``. If the total is at
    most ``block_size`` a single dense orthogonal block is used, which keeps
    small Value bases (``head_dim + value_noise_dim_per_head``) valid.
    """

    if signal_dim <= 0 or noise_dim <= 0:
        raise ValueError("signal_dim and noise_dim must be positive")
    _validate_dtype(dtype)
    if not math.isfinite(max_condition_number) or max_condition_number < 1.0:
        raise ValueError("max_condition_number must be finite and at least one")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    total = signal_dim + noise_dim
    effective_block = block_size if total > block_size else total
    if total % effective_block != 0:
        raise ValueError(
            "block_size %d must divide signal_dim + noise_dim = %d"
            % (block_size, total)
        )
    count = total // effective_block

    generator = make_generator(
        seed, domain, "structured-basis", signal_dim, noise_dim
    )
    perm_in = torch.randperm(total, generator=generator)
    perm_out = torch.randperm(total, generator=generator)
    half_range = 0.5 * math.log(max_condition_number)
    unit = torch.rand(total, generator=generator, dtype=_FP64)
    scales = torch.exp((2.0 * unit - 1.0) * half_range)
    raw_blocks = []
    for _ in range(count):
        raw = torch.randn(
            effective_block, effective_block, generator=generator, dtype=_FP64
        )
        orthogonal, upper = torch.linalg.qr(raw)
        signs = torch.sign(torch.diagonal(upper))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        raw_blocks.append(orthogonal * signs.unsqueeze(0))
    blocks = torch.stack(raw_blocks)

    condition = float(scales.max() / scales.min())
    tolerance = max(1e-9, max_condition_number * 1e-9)
    if condition > max_condition_number + tolerance:
        raise ValueError(
            "generated basis condition %.6g exceeds limit %.6g"
            % (condition, max_condition_number)
        )

    # Cast to the deployed dtype first, then hash and derive the Gram blocks, so
    # that every stored quantity is consistent with the recorded fingerprint.
    stored_scales = scales.to(dtype=dtype)
    stored_blocks = blocks.to(dtype=dtype)
    fingerprint = _fingerprint(
        perm_in, stored_scales, stored_blocks, perm_out, signal_dim, noise_dim
    )
    partial = StructuredBasis(
        perm_in=perm_in,
        scales=stored_scales,
        blocks=stored_blocks,
        perm_out=perm_out,
        gram_blocks=torch.zeros_like(stored_blocks),
        signal_dim=signal_dim,
        noise_dim=noise_dim,
        condition_number=condition,
        fingerprint=fingerprint,
    )
    gram_blocks = _gram_blocks_from(partial).to(dtype=dtype)
    return StructuredBasis(
        perm_in=perm_in,
        scales=stored_scales,
        blocks=stored_blocks,
        perm_out=perm_out,
        gram_blocks=gram_blocks,
        signal_dim=signal_dim,
        noise_dim=noise_dim,
        condition_number=condition,
        fingerprint=fingerprint,
    )


def structured_condition_number(basis: StructuredBasis) -> float:
    """Return the FP64 dense 2-norm condition number for validation records."""

    return float(torch.linalg.cond(basis.dense()).item())


#: Largest ``||e|| / ||h||`` for which the FP32 blockwise Gram reduction keeps
#: the relative error of ``||h||^2`` below roughly 1e-5. Measured, not assumed;
#: see :meth:`StructuredBasis.signal_norm_squared` and
#: ``tests/test_structured_basis.py::test_gram_accuracy_degrades_with_noise``.
AUXILIARY_MAGNITUDE_BOUND = 30.0


def auxiliary_magnitude_bound() -> float:
    """Return the validated ``||e||/||h||`` bound for FP32 Gram reduction."""

    return AUXILIARY_MAGNITUDE_BOUND


def check_auxiliary_magnitude(
    signal: torch.Tensor, noise: torch.Tensor, *, context: str
) -> float:
    """Validate that ``||e||/||h||`` stays inside the FP32 Gram bound.

    Returns the observed ratio so callers can record it. Raises when the bound
    is exceeded, because past that point ``rho`` silently loses accuracy and
    corrupts the signal path rather than only the auxiliary path.
    """

    signal_norm = float(signal.detach().float().norm())
    noise_norm = float(noise.detach().float().norm())
    if signal_norm == 0.0:
        return 0.0
    ratio = noise_norm / signal_norm
    if ratio > AUXILIARY_MAGNITUDE_BOUND:
        raise ValueError(
            "%s: auxiliary/signal magnitude ratio %.3g exceeds the validated "
            "FP32 Gram bound %.3g; reduce the noise coupling or propagator "
            "scale, or switch the rho reduction to FP64"
            % (context, ratio, AUXILIARY_MAGNITUDE_BOUND)
        )
    return ratio
