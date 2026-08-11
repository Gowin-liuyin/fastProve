from __future__ import annotations

import pytest
import torch

from fastprove.config import ModelConfig, ObfuscationConfig
from fastprove.layers.attention import AttentionMode
from fastprove.models.obfuscated import ObfuscatedTinyCausalLM
from fastprove.models.plain import PlainTinyCausalLM
from fastprove.seed import RequestContext


pytestmark = pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS is unavailable",
)


def test_exact_checkpoint_uses_mps_supported_runtime_dtype() -> None:
    config = ModelConfig(
        vocab_size=48,
        hidden_size=24,
        intermediate_size=48,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_sequence_length=24,
    )
    obfuscation = ObfuscationConfig(
        hidden_noise_dim=8,
        value_noise_dim_per_head=2,
        max_condition_number=10.0,
        noise_propagation_gamma=0.5,
        refresh_mode="per_request",
    )
    plain = PlainTinyCausalLM(config, seed=901, debug_enabled=True).to(
        device="mps", dtype=torch.float32
    )
    converted = ObfuscatedTinyCausalLM.from_plain(
        plain,
        obfuscation=obfuscation,
        mode=AttentionMode.EXACT,
        approximation=None,
        seed=902,
        debug_enabled=True,
    )
    obfuscated = converted.module.to(device="mps", dtype=torch.float32)
    tokens = torch.tensor([[1, 3, 5, 7, 9, 11]], device="mps")
    with torch.no_grad():
        plaintext = plain(tokens)
        exact = obfuscated(
            tokens,
            request_context=RequestContext(903, "mps-runtime-test"),
        )
    torch.mps.synchronize()
    assert exact.device.type == "mps"
    assert torch.isfinite(exact).all().item()
    torch.testing.assert_close(exact, plaintext, atol=2e-4, rtol=2e-4)

    # The real evaluator uses this explicitly gated path to collect layer
    # diagnostics.  It must not attempt an FP64 reduction on MPS.
    with torch.no_grad():
        debug_exact, debug_records = obfuscated.forward_debug(
            tokens,
            request_context=RequestContext(903, "mps-runtime-debug-test"),
        )
    torch.mps.synchronize()
    assert len(debug_records) == config.num_layers
    torch.testing.assert_close(debug_exact, plaintext, atol=2e-4, rtol=2e-4)
