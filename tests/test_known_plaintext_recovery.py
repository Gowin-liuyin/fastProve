"""Tier-0 known-plaintext decoder recovery: threshold and exactness.

The algebra replicated here is the T5 attack of
``docs/noise_placement_explained.tex``; the production script is
``scripts/verify_known_plaintext_decoder_recovery.py``. The script stays
self-contained, so this test reimplements the same closed form on a small
deterministic instance (the repo's attack-test convention: a test must fail
when the attack stops working, not merely exercise the script).
"""

from __future__ import annotations

import torch

from fastprove.structured import generate_structured_basis

_FP64 = torch.float64


def _basis(signal_dim=16, noise_dim=4, block_size=4, seed=31):
    return generate_structured_basis(
        signal_dim,
        noise_dim,
        seed=seed,
        domain="kpa-test",
        block_size=block_size,
        dtype=_FP64,
    )


def _right_null_space(matrix: torch.Tensor, tolerance: float = 1e-9) -> torch.Tensor:
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=True)
    rank = int((singular > tolerance * max(1.0, float(singular.max()))).sum())
    return vh[rank:].T


def _recover_decoder(
    h_mat: torch.Tensor, c_mat: torch.Tensor, tolerance: float = 1e-9
) -> torch.Tensor | None:
    q_h, _ = torch.linalg.qr(h_mat)
    residual = c_mat - q_h @ (q_h.T @ c_mat)
    v = _right_null_space(residual, tolerance)
    if v.shape[1] < h_mat.shape[1]:
        return None
    av = torch.linalg.lstsq(h_mat, c_mat @ v).solution
    return v @ torch.linalg.pinv(av)


def _samples(basis, count: int, noise_scale: float, generator):
    h_mat = torch.randn(count, basis.signal_dim, generator=generator, dtype=_FP64)
    noise = (
        torch.randn(count, basis.noise_dim, generator=generator, dtype=_FP64)
        * noise_scale
    )
    return h_mat, basis.mix(torch.cat((h_mat, noise), dim=-1))


def _heldout_error(basis, decoder, noise_scale: float, generator) -> float:
    h_mat, c_mat = _samples(basis, 32, noise_scale, generator)
    target = h_mat
    denominator = float(target.norm())
    return float((c_mat @ decoder - target).norm()) / denominator


def test_recovery_fails_below_sample_threshold() -> None:
    basis = _basis()
    generator = torch.Generator().manual_seed(7)
    h_mat, c_mat = _samples(basis, basis.signal_dim + basis.noise_dim - 1, 1.0, generator)
    decoder = _recover_decoder(h_mat, c_mat)
    assert decoder is not None  # a rectangular left inverse exists ...
    error = _heldout_error(basis, decoder, 1.0, generator)
    assert error > 1e-3  # ... but it is wrong on fresh-noise states


def test_recovery_succeeds_at_threshold() -> None:
    basis = _basis()
    generator = torch.Generator().manual_seed(8)
    threshold = basis.signal_dim + basis.noise_dim + 1
    h_mat, c_mat = _samples(basis, threshold, 1.0, generator)
    decoder = _recover_decoder(h_mat, c_mat)
    assert decoder is not None
    assert _heldout_error(basis, decoder, 1.0, generator) < 1e-10


def test_recovered_decoder_exact_on_heldout_and_matches_true_L() -> None:
    """In the production D == d path the reachable set is the full ambient
    space, so the recovered map is the true ``L`` (not merely a map that
    agrees on reachable states). The expanded-concat divergence (0.33) belongs
    to the lifted D > d construction and is not reproducible here."""

    basis = _basis()
    generator = torch.Generator().manual_seed(9)
    h_mat, c_mat = _samples(basis, 4 * basis.total_dim, 1.0, generator)
    decoder = _recover_decoder(h_mat, c_mat)
    assert decoder is not None
    assert _heldout_error(basis, decoder, 1.0, generator) < 1e-12
    true_projection = basis.signal_projection()
    relative = float((decoder - true_projection).norm()) / float(
        true_projection.norm()
    )
    assert relative < 1e-12


def test_recovery_unaffected_by_free_noise_scale() -> None:
    """Free-cone noise buys zero hiding (theorem 4): 1e6x amplification still
    recovers the decoder; only FP64 cancellation grows (~1e-9)."""

    basis = _basis()
    generator = torch.Generator().manual_seed(10)
    h_mat, c_mat = _samples(basis, 4 * basis.total_dim, 1e6, generator)
    decoder = _recover_decoder(h_mat, c_mat)
    assert decoder is not None
    error = _heldout_error(basis, decoder, 1e6, generator)
    assert error < 1e-6
    true_projection = basis.signal_projection()
    assert float((decoder - true_projection).norm()) / float(
        true_projection.norm()
    ) < 1e-6


def test_recovery_on_exported_server_bundle_observations(tmp_path) -> None:
    """Attack observations produced by a real converted server bundle.

    The plaintext labels come from the public base checkpoint (the plain model
    object), the mixed states from the server bundle's deployed embedding
    table. This is the strongest local endpoint check available without a
    pretrained checkpoint; it is not a substitute for a task-level evaluation.
    """

    from fastprove.config import (
        AttentionConfig,
        EvaluationConfig,
        ModelConfig,
        ObfuscationConfig,
        PrototypeConfig,
        RuntimeConfig,
    )
    from fastprove.converter import ModelConverter, load_client, load_server_model
    from fastprove.layers.attention import AttentionMode
    from fastprove.models.plain import PlainTinyCausalLM

    model_config = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_sequence_length=32,
    )
    config = PrototypeConfig(
        model=model_config,
        obfuscation=ObfuscationConfig(
            hidden_noise_dim=4,
            value_noise_dim_per_head=1,
            max_condition_number=10.0,
            noise_propagation_gamma=0.5,
            refresh_mode="per_request",
            basis_block_size=12,
        ),
        attention=AttentionConfig(
            mode=AttentionMode.EXACT,
            tau_max=0.0,
            tau_error=0.0,
            alpha=0.8,
            preserve_top_k=4,
        ),
        runtime=RuntimeConfig(
            seed=41,
            request_id="kpa-bundle",
            device="cpu",
            activation_dtype="float32",
            debug_enabled=False,
        ),
        evaluation=EvaluationConfig(
            batch_size=1,
            sequence_length=4,
            sample_count=1,
            generation_tokens=1,
            warmup_runs=0,
            timed_runs=1,
        ),
    )
    plain = PlainTinyCausalLM(model_config, seed=41, debug_enabled=False).eval()
    ModelConverter().convert(
        plain,
        config,
        client_secret_dir=tmp_path / "client",
        server_model_dir=tmp_path / "server",
    )
    module = load_server_model(tmp_path / "server").eval()
    bundle = load_client(tmp_path / "client")

    tokens = torch.arange(model_config.vocab_size).reshape(-1, 1)
    with torch.no_grad():
        mixed = module.embedding(bundle.encode(tokens)).mixed.squeeze(1)
    h_mat = plain.embedding.weight.detach()[tokens.squeeze(1)].to(_FP64)
    c_mat = mixed.to(_FP64)
    # The server bundle stores the table in FP32, so the projected residual has
    # an FP32 floor (~1e-7 absolute, ~1e-6 relative to its leading singular
    # value) that must be treated as numerical zero. The rank cutoff is part of
    # the attacker's standard toolkit and is stated in the record.
    decoder = _recover_decoder(h_mat, c_mat, tolerance=1e-5)
    assert decoder is not None
    heldout_h = plain.embedding.weight.detach()[:16].to(_FP64)
    with torch.no_grad():
        heldout_c = (
            module.embedding(bundle.encode(torch.arange(16).reshape(-1, 1)))
            .mixed.squeeze(1)
            .to(_FP64)
        )
    # Accuracy is bounded by the FP32 storage of the bundle (relative floor
    # ~1e-7); the attack does not fail, it inherits the deployed dtype.
    assert float((heldout_c @ decoder - heldout_h).norm()) / float(
        heldout_h.norm()
    ) < 1e-6
