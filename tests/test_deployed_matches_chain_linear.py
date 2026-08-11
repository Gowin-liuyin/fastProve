"""The fused deployed weights must equal the general ChainLinear composition.

``ChainLinear`` implements the manual's general augmented affine map
``K = [[W, C], [0, G]]`` with ``W_tilde = M_in^-1 K M_out``. The deployed
weights of Stage B are a *fusion* of several such maps. This test pins the
equivalence so that ``ChainLinear`` remains the documented reference form
rather than dead code.
"""

from __future__ import annotations

import torch

from fastprove.layers.linear import ChainLinear
from fastprove.structured import generate_structured_basis

_FP64 = torch.float64


def test_deployed_residual_map_equals_chain_linear() -> None:
    """A single deployed residual step equals a ChainLinear with W = I + dW."""

    torch.manual_seed(0)
    signal_dim, noise_dim = 32, 8
    basis = generate_structured_basis(
        signal_dim, noise_dim, seed=1, domain="eq", block_size=8, dtype=_FP64
    )
    top = basis.signal_rows()
    bottom = basis.noise_rows()
    noise_read = basis.noise_projection()

    weight = torch.randn(signal_dim, signal_dim, dtype=_FP64) * 0.1
    coupling = torch.randn(signal_dim, noise_dim, dtype=_FP64) * 0.02
    propagator = 0.5 * torch.eye(noise_dim, dtype=_FP64)

    signal = torch.randn(4, signal_dim, dtype=_FP64)
    noise = torch.randn(4, noise_dim, dtype=_FP64) * 0.1
    mixed = basis.mix(torch.cat((signal, noise), dim=-1))

    # Deployed (fused) form, as used by Stage B.
    delta = signal @ weight
    identity = torch.eye(noise_dim, dtype=_FP64)
    fused = (
        mixed
        + delta @ (top + coupling @ bottom)
        + (mixed @ noise_read) @ ((propagator - identity) @ bottom)
    )

    # General ChainLinear form: signal map I + W, noise map C and G.
    expected_signal = signal + delta
    expected_noise = signal @ weight @ coupling + noise @ propagator
    reference = basis.mix(
        torch.cat((expected_signal, expected_noise), dim=-1)
    )

    assert torch.allclose(fused, reference, atol=1e-10)


def test_chain_linear_still_satisfies_its_documented_identity() -> None:
    """Guard the general primitive itself (manual section 11)."""

    torch.manual_seed(1)
    signal_dim, noise_dim = 16, 4
    basis_in = generate_structured_basis(
        signal_dim, noise_dim, seed=2, domain="in", block_size=4, dtype=_FP64
    )
    basis_out = generate_structured_basis(
        signal_dim, noise_dim, seed=3, domain="out", block_size=4, dtype=_FP64
    )
    weight = torch.randn(signal_dim, signal_dim, dtype=_FP64) * 0.1
    bias = torch.randn(signal_dim, dtype=_FP64) * 0.05
    coupling = torch.randn(signal_dim, noise_dim, dtype=_FP64) * 0.02
    propagator = 0.5 * torch.eye(noise_dim, dtype=_FP64)
    refresh = torch.randn(noise_dim, dtype=_FP64) * 0.02

    signal = torch.randn(3, signal_dim, dtype=_FP64)
    noise = torch.randn(3, noise_dim, dtype=_FP64)

    # W_tilde = M_in^-1 K M_out applied to c_in must equal mixing the
    # plaintext result with M_out.
    augmented_map = torch.zeros(
        signal_dim + noise_dim, signal_dim + noise_dim, dtype=_FP64
    )
    augmented_map[:signal_dim, :signal_dim] = weight
    augmented_map[:signal_dim, signal_dim:] = coupling
    augmented_map[signal_dim:, signal_dim:] = propagator
    deployed = (
        basis_in.dense_inverse() @ augmented_map @ basis_out.dense()
    )
    deployed_bias = torch.cat((bias, refresh)) @ basis_out.dense()

    mixed_in = basis_in.mix(torch.cat((signal, noise), dim=-1))
    observed = mixed_in @ deployed + deployed_bias
    expected = basis_out.mix(
        torch.cat(
            (
                signal @ weight + bias,
                signal @ coupling + noise @ propagator + refresh,
            ),
            dim=-1,
        )
    )
    assert torch.allclose(observed, expected, atol=1e-10)
