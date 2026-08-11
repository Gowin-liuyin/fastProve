"""Offline construction of deployed weights for the non-decoding forward pass.

Every matrix produced here absorbs ``M``, ``M^-1`` and the RMSNorm gamma offline
so that the deployed block runs with the *same GEMM count as the plaintext
block* and never materializes the plaintext hidden state ``h``, the plaintext
attention context ``O``, or the plaintext FFN output.

Row-vector convention throughout: ``y = x @ W``. Every returned tensor is in
mathematical layout ``[in, out]``.

Algebra (all identities verified in FP64 by ``docs/design_identity_check.py``)::

    c              = [h, e] @ M                        server state, width n
    P              = M^-1[:, :d]                       signal read-out
    N              = M^-1[:, d:]                       auxiliary read-out
    M_top, M_bot   = M[:d], M[d:]
    rho            = sqrt(||h||^2 / d + eps)           from basis.rms_scale(c)

    Q              = (c @ Wq) / rho          Wq = P @ diag(gamma_a) @ W_Q
    K              = (c @ Wk) / rho          Wk = P @ diag(gamma_a) @ W_K
    c_V[head]      = (c @ Wv[head]) / rho    Wv = [P diag(gamma_a) W_V | C_V] @ M_V

    c_post         = c + einsum(mixed_ctx, Wattn) + (c @ N) @ Wnz_a + xi_a
    z'             = silu(c_post @ Wg / rho2) * (c_post @ Wu / rho2)
    c_next         = c_post + z' @ Wffn + (c_post @ N) @ Wnz_f + xi_f

where ``Wattn[q_head]`` fuses the per-KV-head Value unmix, ``W_O``, the residual
re-mix ``M_top``, the signal noise coupling ``C_O @ M_bot`` and the auxiliary
term, and ``Wffn`` fuses ``Pf^T Df^-1 W_d``, ``M_top``, ``C_d @ M_bot`` and
``C_z @ M_bot``.

Security consequences that must not be omitted from the threat model:

1. ``Wnz_a`` / ``Wnz_f`` are applied as ``(c @ N) @ Wnz``. ``N`` therefore sits
   in the deployed weights, so an observer of the deployed weights recovers the
   auxiliary state ``e`` up to an invertible ``r x r`` map. This is unavoidable
   for any rank-``r`` factorization of ``N (G - I) M_bot`` (verified: the rank is
   exactly ``r`` and ``M @ U`` has a zero signal block for every factorization).
   It does not additionally expose ``h`` beyond what ``M`` already does, but it
   does make the ``e ~ h C`` channel of ``docs/threat_model.md`` 5bis.6 directly
   exploitable.
2. The deployed matrices are products containing ``P`` and ``M_top``. They are
   server-side weights by construction. Recovering ``M`` from them is a
   factorization problem, not a hard cryptographic one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ..structured import StructuredBasis, check_auxiliary_magnitude

_FP64 = torch.float64


def _require_math_layout(
    weight: torch.Tensor, rows: int, name: str
) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError("%s must be a matrix in math layout [in, out]" % name)
    if weight.shape[0] != rows:
        raise ValueError(
            "%s input dimension %d does not match expected %d; a PyTorch-layout "
            "weight was probably passed without transposing"
            % (name, weight.shape[0], rows)
        )
    return weight.detach().cpu().to(dtype=_FP64)


@dataclass(frozen=True)
class DeployedAttentionWeights:
    """Deployed attention-segment weights in math layout."""

    query: torch.Tensor            # [n, H * dh]
    key: torch.Tensor              # [n, Hkv * dh]
    value: torch.Tensor            # [Hkv, n, dh + rh]
    query_bias: torch.Tensor       # [H * dh]
    key_bias: torch.Tensor         # [Hkv * dh]
    value_bias: torch.Tensor       # [Hkv, dh + rh]
    output: torch.Tensor           # [H, dh + rh, n]
    noise_out: torch.Tensor        # [r, n]
    refresh_out: torch.Tensor      # [r, n]  (fixed_debug only; zeros otherwise)


@dataclass(frozen=True)
class DeployedFeedForwardWeights:
    """Deployed FFN-segment weights in math layout."""

    gate: torch.Tensor             # [n, dff]
    up: torch.Tensor               # [n, dff]
    output: torch.Tensor           # [dff, n]
    noise_out: torch.Tensor        # [r, n]
    refresh_out: torch.Tensor      # [r, n]
    neuron_permutation: torch.Tensor  # [dff] int64, debug/validation only
    neuron_scale: torch.Tensor        # [dff], debug/validation only


def build_deployed_attention(
    *,
    basis: StructuredBasis,
    value_bases: Tuple[StructuredBasis, ...],
    gamma_attention: torch.Tensor,
    q_weight_math: torch.Tensor,
    k_weight_math: torch.Tensor,
    v_weight_math: torch.Tensor,
    o_weight_math: torch.Tensor,
    q_bias: Optional[torch.Tensor],
    k_bias: Optional[torch.Tensor],
    v_bias: Optional[torch.Tensor],
    value_signal_coupling: torch.Tensor,
    signal_noise_coupling: torch.Tensor,
    noise_propagator: torch.Tensor,
    auxiliary_to_hidden: torch.Tensor,
    fixed_refresh: Optional[torch.Tensor],
    kv_index: torch.Tensor,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
) -> DeployedAttentionWeights:
    """Fuse the attention segment into deployed weights.

    Args:
        basis: hidden/residual mixing basis, signal width ``d``.
        value_bases: one basis per KV head, signal width ``head_dim``.
        gamma_attention: RMSNorm scale before Q/K/V, shape ``[d]``.
        q_weight_math: ``[d, H * dh]``.
        k_weight_math: ``[d, Hkv * dh]``.
        v_weight_math: ``[d, Hkv * dh]``.
        o_weight_math: ``[H * dh, d]``.
        q_bias / k_bias / v_bias: optional projection biases (Qwen2 has them).
        value_signal_coupling: ``[n, Hkv, rh]``, maps ``c`` to Value noise.
        signal_noise_coupling: ``[d, r]``, the ``C_O`` of ``e' = dH C_O + e G``.
        noise_propagator: ``[r, r]``, the ``G`` of the auxiliary chain.
        auxiliary_to_hidden: ``[rh, r]``, folds Value noise into hidden noise.
        fixed_refresh: ``[r]`` for ``fixed_debug``, else ``None``.
        kv_index: ``[H]`` mapping each query head to its KV head.
        head_dim: ``dh``.

    Returns:
        Deployed weights whose forward use is described in the module docstring.
    """

    signal_dim = basis.signal_dim
    noise_dim = basis.noise_dim
    total = basis.total_dim
    kv_heads = len(value_bases)
    if kv_index.ndim != 1:
        raise ValueError("kv_index must be a vector of length H")
    heads = int(kv_index.numel())
    value_noise = value_bases[0].noise_dim
    for candidate in value_bases:
        if candidate.signal_dim != head_dim:
            raise ValueError("Value basis signal width must equal head_dim")
        if candidate.noise_dim != value_noise:
            raise ValueError("all Value bases must share the noise width")
    if torch.any(kv_index < 0) or torch.any(kv_index >= kv_heads):
        raise ValueError("kv_index contains an invalid KV head")

    q_math = _require_math_layout(q_weight_math, signal_dim, "q_weight_math")
    k_math = _require_math_layout(k_weight_math, signal_dim, "k_weight_math")
    v_math = _require_math_layout(v_weight_math, signal_dim, "v_weight_math")
    o_math = _require_math_layout(
        o_weight_math, heads * head_dim, "o_weight_math"
    )
    if gamma_attention.shape != (signal_dim,):
        raise ValueError("gamma_attention shape must be [d]")
    if value_signal_coupling.shape != (total, kv_heads, value_noise):
        raise ValueError("value_signal_coupling shape must be [n, Hkv, rh]")
    if signal_noise_coupling.shape != (signal_dim, noise_dim):
        raise ValueError("signal_noise_coupling shape must be [d, r]")
    if noise_propagator.shape != (noise_dim, noise_dim):
        raise ValueError("noise_propagator shape must be [r, r]")
    if auxiliary_to_hidden.shape != (value_noise, noise_dim):
        raise ValueError("auxiliary_to_hidden shape must be [rh, r]")

    projection = basis.signal_projection()
    top = basis.signal_rows()
    bottom = basis.noise_rows()
    gamma = gamma_attention.detach().cpu().to(dtype=_FP64)
    coupling_v = value_signal_coupling.detach().cpu().to(dtype=_FP64)
    coupling_o = signal_noise_coupling.detach().cpu().to(dtype=_FP64)
    propagator = noise_propagator.detach().cpu().to(dtype=_FP64)
    auxiliary = auxiliary_to_hidden.detach().cpu().to(dtype=_FP64)

    # ``P @ diag(gamma)`` == ``P * gamma[None, :]`` because P is [n, d].
    absorbed = projection * gamma[None, :]
    deployed_query = absorbed @ q_math
    deployed_key = absorbed @ k_math

    value_mix = torch.stack([item.dense() for item in value_bases])
    value_unmix = torch.stack([item.dense_inverse() for item in value_bases])
    deployed_value = torch.stack(
        [
            torch.cat(
                (
                    absorbed
                    @ v_math[:, head * head_dim : (head + 1) * head_dim],
                    coupling_v[:, head],
                ),
                dim=-1,
            )
            @ value_mix[head]
            for head in range(kv_heads)
        ]
    )

    residual_signal = top + coupling_o @ bottom
    auxiliary_out = auxiliary @ bottom
    deployed_output = torch.stack(
        [
            value_unmix[int(kv_index[head])][:, :head_dim]
            @ o_math[head * head_dim : (head + 1) * head_dim]
            @ residual_signal
            + (1.0 / heads)
            * value_unmix[int(kv_index[head])][:, head_dim:]
            @ auxiliary_out
            for head in range(heads)
        ]
    )

    identity = torch.eye(noise_dim, dtype=_FP64)
    deployed_noise_out = (propagator - identity) @ bottom
    if fixed_refresh is None:
        refresh_out = torch.zeros(noise_dim, total, dtype=_FP64)
    else:
        if fixed_refresh.shape != (noise_dim,):
            raise ValueError("fixed_refresh shape must be [r]")
        refresh_out = torch.diag(
            fixed_refresh.detach().cpu().to(dtype=_FP64)
        ) @ bottom

    zero_q = torch.zeros(q_math.shape[1], dtype=_FP64)
    zero_k = torch.zeros(k_math.shape[1], dtype=_FP64)
    bias_q = zero_q if q_bias is None else q_bias.detach().cpu().to(dtype=_FP64)
    bias_k = zero_k if k_bias is None else k_bias.detach().cpu().to(dtype=_FP64)
    if v_bias is None:
        bias_v = torch.zeros(kv_heads, head_dim + value_noise, dtype=_FP64)
    else:
        raw_v = v_bias.detach().cpu().to(dtype=_FP64)
        bias_v = torch.stack(
            [
                torch.cat(
                    (
                        raw_v[head * head_dim : (head + 1) * head_dim],
                        torch.zeros(value_noise, dtype=_FP64),
                    )
                )
                @ value_mix[head]
                for head in range(kv_heads)
            ]
        )

    return DeployedAttentionWeights(
        query=deployed_query.to(dtype=dtype),
        key=deployed_key.to(dtype=dtype),
        value=deployed_value.to(dtype=dtype),
        query_bias=bias_q.to(dtype=dtype),
        key_bias=bias_k.to(dtype=dtype),
        value_bias=bias_v.to(dtype=dtype),
        output=deployed_output.to(dtype=dtype),
        noise_out=deployed_noise_out.to(dtype=dtype),
        refresh_out=refresh_out.to(dtype=dtype),
    )


def build_deployed_feed_forward(
    *,
    basis: StructuredBasis,
    gamma_ffn: torch.Tensor,
    gate_weight_math: torch.Tensor,
    up_weight_math: torch.Tensor,
    down_weight_math: torch.Tensor,
    neuron_permutation: torch.Tensor,
    neuron_scale: torch.Tensor,
    swiglu_noise_coupling: torch.Tensor,
    down_noise_coupling: torch.Tensor,
    noise_propagator: torch.Tensor,
    fixed_refresh: Optional[torch.Tensor],
    dtype: torch.dtype = torch.float32,
) -> DeployedFeedForwardWeights:
    """Fuse the FFN segment into deployed weights.

    Uses the SwiGLU covariance ``silu(g Pf) * (u Df Pf) = (silu(g) * u) Df Pf``
    and the compensating Down conversion ``Pf^T Df^-1 W_d``.
    """

    signal_dim = basis.signal_dim
    noise_dim = basis.noise_dim
    total = basis.total_dim
    gate_math = _require_math_layout(
        gate_weight_math, signal_dim, "gate_weight_math"
    )
    up_math = _require_math_layout(up_weight_math, signal_dim, "up_weight_math")
    intermediate = gate_math.shape[1]
    down_math = _require_math_layout(
        down_weight_math, intermediate, "down_weight_math"
    )
    if up_math.shape[1] != intermediate:
        raise ValueError("gate and up must share the intermediate dimension")
    if down_math.shape[1] != signal_dim:
        raise ValueError("down output dimension must equal d")
    if gamma_ffn.shape != (signal_dim,):
        raise ValueError("gamma_ffn shape must be [d]")
    if neuron_permutation.shape != (intermediate,):
        raise ValueError("neuron_permutation shape must be [dff]")
    if neuron_scale.shape != (intermediate,):
        raise ValueError("neuron_scale shape must be [dff]")
    if not torch.equal(
        torch.sort(neuron_permutation.cpu()).values, torch.arange(intermediate)
    ):
        raise ValueError("neuron_permutation must be a bijection")
    if torch.any(neuron_scale == 0) or not torch.isfinite(neuron_scale).all():
        raise ValueError("neuron_scale must be finite and nonzero")
    if swiglu_noise_coupling.shape != (intermediate, noise_dim):
        raise ValueError("swiglu_noise_coupling shape must be [dff, r]")
    if down_noise_coupling.shape != (signal_dim, noise_dim):
        raise ValueError("down_noise_coupling shape must be [d, r]")
    if noise_propagator.shape != (noise_dim, noise_dim):
        raise ValueError("noise_propagator shape must be [r, r]")

    projection = basis.signal_projection()
    top = basis.signal_rows()
    bottom = basis.noise_rows()
    gamma = gamma_ffn.detach().cpu().to(dtype=_FP64)
    permutation = neuron_permutation.detach().cpu()
    scale = neuron_scale.detach().cpu().to(dtype=_FP64)
    coupling_z = swiglu_noise_coupling.detach().cpu().to(dtype=_FP64)
    coupling_d = down_noise_coupling.detach().cpu().to(dtype=_FP64)
    propagator = noise_propagator.detach().cpu().to(dtype=_FP64)

    absorbed = projection * gamma[None, :]
    deployed_gate = absorbed @ gate_math[:, permutation]
    deployed_up = absorbed @ (
        up_math[:, permutation] * scale[permutation][None, :]
    )
    # Down conversion ``Pf^T Df^-1 W_d``: permute rows, then scale each row.
    down_converted = (1.0 / scale[permutation])[:, None] * down_math[permutation]
    deployed_output = (
        down_converted @ (top + coupling_d @ bottom) + coupling_z @ bottom
    )
    identity = torch.eye(noise_dim, dtype=_FP64)
    deployed_noise_out = (propagator - identity) @ bottom
    if fixed_refresh is None:
        refresh_out = torch.zeros(noise_dim, total, dtype=_FP64)
    else:
        if fixed_refresh.shape != (noise_dim,):
            raise ValueError("fixed_refresh shape must be [r]")
        refresh_out = torch.diag(
            fixed_refresh.detach().cpu().to(dtype=_FP64)
        ) @ bottom

    return DeployedFeedForwardWeights(
        gate=deployed_gate.to(dtype=dtype),
        up=deployed_up.to(dtype=dtype),
        output=deployed_output.to(dtype=dtype),
        noise_out=deployed_noise_out.to(dtype=dtype),
        refresh_out=refresh_out.to(dtype=dtype),
        neuron_permutation=permutation.to(dtype=torch.int64),
        neuron_scale=scale.to(dtype=dtype),
    )


def validate_auxiliary_budget(
    *,
    basis: StructuredBasis,
    sample_signal: torch.Tensor,
    signal_noise_coupling: torch.Tensor,
    noise_propagator: torch.Tensor,
    context: str,
) -> float:
    """Check the ``||e||/||h||`` bound required by the FP32 Gram reduction.

    Estimates the steady-state auxiliary magnitude of the geometric chain
    ``e_{k+1} = h C + e_k G`` as ``||h C|| / (1 - ||G||_2)`` and validates it
    against :data:`structured.AUXILIARY_MAGNITUDE_BOUND`. Call this once per
    conversion, not per forward.
    """

    coupling = signal_noise_coupling.detach().to(dtype=torch.float32)
    propagator_norm = float(
        torch.linalg.matrix_norm(
            noise_propagator.detach().to(dtype=torch.float32), ord=2
        )
    )
    if propagator_norm >= 1.0:
        raise ValueError(
            "%s: auxiliary propagator spectral norm %.4f is not contractive; "
            "the chain diverges and rho loses accuracy"
            % (context, propagator_norm)
        )
    signal = sample_signal.detach().to(dtype=torch.float32)
    single_step = signal @ coupling
    steady_state = single_step / (1.0 - propagator_norm)
    return check_auxiliary_magnitude(signal, steady_state, context=context)
