"""S1: configuration-driven noise scales and the single sigma conversion.

The deployed path must obtain its refresh scale from configuration through the
same conversion function the stage-A kernel uses; no caller may multiply a raw
magic number. Channel-0 scale is a conditioning knob only: theorem 4 gives it
exactly zero hiding, so these tests never assert privacy, only numerical
equivalence and accounting.
"""

from __future__ import annotations

import pytest
import torch

from fastprove.augmented_noise import (
    NoiseRefreshSpec,
    deployed_free_noise_sigma,
    free_noise_sigma,
)
from fastprove.layers.attention import AttentionMode
from fastprove.models.obfuscated import ObfuscatedTinyCausalLM
from fastprove.models.plain import PlainTinyCausalLM
from fastprove.seed import RequestContext
from fastprove.structured import AUXILIARY_MAGNITUDE_BOUND

from tests._fixtures import model_config, obfuscation_config


def _obfuscation(**overrides):
    return obfuscation_config(refresh_mode="per_request", **overrides)


def test_deployed_sigma_equals_energy_ratio_bit_exactly() -> None:
    for ratio in (0.0, 0.02, 0.03125, 0.1, 1e-6, 3.0):
        assert deployed_free_noise_sigma(ratio, 8) == ratio
        assert deployed_free_noise_sigma(ratio, 16) == ratio


def test_free_noise_sigma_matches_component_sigma() -> None:
    for noise_dim, calibration in ((8, 1.0), (16, 0.5), (4, 2.0)):
        spec = NoiseRefreshSpec(
            noise_dim=noise_dim, gamma=0.5, energy_ratio=0.7
        )
        assert spec.component_sigma(calibration) == free_noise_sigma(
            0.7, calibration, noise_dim
        )


def test_free_noise_sigma_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="energy_ratio"):
        free_noise_sigma(-0.1, 1.0, 8)
    with pytest.raises(ValueError, match="calibration_rms"):
        free_noise_sigma(0.1, float("nan"), 8)
    with pytest.raises(ValueError, match="noise_dim"):
        free_noise_sigma(0.1, 1.0, 0)


def test_config_rejects_invalid_scale_fields() -> None:
    with pytest.raises(ValueError, match="free_noise_energy_ratio"):
        _obfuscation(free_noise_energy_ratio=-1e-3)
    with pytest.raises(ValueError, match="signal_noise_coupling_scale"):
        _obfuscation(signal_noise_coupling_scale=float("inf"))


def test_config_rejects_energy_ratio_above_gram_bound() -> None:
    with pytest.raises(ValueError, match="FP32 Gram bound"):
        _obfuscation(free_noise_energy_ratio=AUXILIARY_MAGNITUDE_BOUND + 1.0)


def test_config_defaults_are_conservative() -> None:
    config = _obfuscation()
    assert config.free_noise_energy_ratio == 0.02
    assert config.signal_noise_coupling_scale == 0.0


def test_channel_zero_scale_changes_noise_state_not_signal() -> None:
    """Raising the channel-0 scale must leave the exact signal path intact."""

    dimensions = model_config(
        vocab_size=48,
        hidden_size=24,
        intermediate_size=48,
        max_sequence_length=24,
    )
    plain = PlainTinyCausalLM(dimensions, seed=611, debug_enabled=False)
    input_ids = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]])
    mask = torch.ones_like(input_ids, dtype=torch.bool)

    reference = plain(input_ids, token_mask=mask)
    logits = []
    increments = []
    for ratio in (0.02, 0.2):
        converted = ObfuscatedTinyCausalLM.from_plain(
            plain,
            obfuscation=_obfuscation(free_noise_energy_ratio=ratio),
            mode=AttentionMode.EXACT,
            approximation=None,
            seed=612,
            debug_enabled=False,
        )
        assert converted.module._free_noise_sigma == ratio
        assert converted.module.blocks[0]._free_noise_sigma == ratio
        codec = converted.token_codec
        with torch.no_grad():
            produced = converted.module(
                codec.encode(input_ids),
                token_mask=mask,
                request_context=RequestContext(613, "s1-scale"),
            )
            increments.append(
                converted.module._initial_refresh_out(
                    RequestContext(613, "s1-scale"), produced
                )
            )
        logits.append(produced[..., codec.permutation])

    for produced in logits:
        torch.testing.assert_close(produced, reference, atol=2e-4, rtol=2e-4)
        assert torch.equal(produced.argmax(dim=-1), reference.argmax(dim=-1))
    # Same underlying standard-normal draw, different sigma: the refresh
    # increments must differ and the norm ratio must track the scale ratio.
    assert not torch.equal(increments[0], increments[1])
    observed_ratio = float(increments[1].norm()) / float(increments[0].norm())
    assert abs(observed_ratio - 10.0) < 1e-3
