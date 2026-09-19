"""Deployment policy constants shared by config, layers, and evaluation.

This module is intentionally dependency-free (torch only) so that
``config.py`` and ``noise_budget.py`` do not have to import the heavier
``structured``/``layers`` modules just to validate a bound. The names are
re-exported from their historical modules for compatibility.

Reference policy: ``||e|| / ||h||`` must stay inside the FP32 Gram bound, and
RMS statistics run in FP32 on the deployed path (AGENTS.md).
"""

from __future__ import annotations

import torch


#: Largest ``||e|| / ||h||`` for which the FP32 blockwise Gram reduction keeps
#: the relative error of ``||h||^2`` below roughly 1e-5. Measured, not assumed;
#: see :meth:`StructuredBasis.signal_norm_squared` and
#: ``tests/test_structured_basis.py::test_gram_accuracy_degrades_with_noise``.
AUXILIARY_MAGNITUDE_BOUND = 30.0


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


#: Dtype of the runtime reduction that produces the RMSNorm scale ``rho``.
#: This is the only elevated-precision arithmetic left on the deployed path
#: after Stage B removed the checkpoint mix/unmix, and AGENTS.md requires FP32
#: or better for RMS statistics. Experiment records must derive the reported
#: dtype from here instead of hard-coding a literal, so that a record always
#: states the dtype that actually executed.
REDUCTION_COMPUTE_DTYPE = torch.float32


def reduction_compute_dtype_name() -> str:
    """Return the executed reduction dtype for experiment records.

    Deployed inference performs no FP64 arithmetic on any device. Prior to task
    A4 the checkpoint mix/unmix ran in FP64 on CPU, and several record writers
    hard-coded ``"float64" if device == "cpu" else "float32"``. That literal
    became false when A4 switched to FP32 and Stage B deleted the checkpoint
    arithmetic entirely; this function is the single source of truth.
    """

    return str(REDUCTION_COMPUTE_DTYPE).replace("torch.", "")
