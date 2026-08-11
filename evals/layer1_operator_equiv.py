"""Layer 1 — per-operator numerical equivalence (protocol §4.1).

Decode ĥ_ℓ = Decode_ℓ(c_ℓ) and report E_{ℓ,∞} and E_{ℓ,rel} for:

Embedding, ChainLinear, RMSNorm, Q/K, Value, Attention Output, SwiGLU,
Down Projection, LM Head.

Gate: FP32 max|ĥ − h| ≤ 1e-4 on small unit tests.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from fastprove.layers.linear import ChainLinear
from fastprove.layers.rmsnorm import rms_norm_fp32
from fastprove.seed import make_generator
from fastprove.state import MixedState, encode_debug
from fastprove.transforms import generate_transform

from .metrics_common import infinity_and_relative
from .model_factory import EvalModels, cast_activations
from .thresholds import FP32_OPERATOR_MAX_ABS_ERROR


def _chain_linear_identity_test(
    *,
    in_signal: int = 8,
    out_signal: int = 12,
    in_noise: int = 4,
    out_noise: int = 3,
    seed: int = 7,
) -> Dict[str, Any]:
    """Direct unit test of the ChainLinear affine identity (protocol gate)."""

    generator = make_generator(seed, "layer1-chain-linear")
    h = torch.randn(3, 5, in_signal, generator=generator, dtype=torch.float32)
    e = torch.randn(3, 5, in_noise, generator=generator, dtype=torch.float32)
    w = (
        torch.randn(in_signal, out_signal, generator=generator, dtype=torch.float32)
        * 0.2
    )
    b = torch.randn(out_signal, generator=generator, dtype=torch.float32) * 0.1
    c = (
        torch.randn(in_signal, out_noise, generator=generator, dtype=torch.float32)
        * 0.1
    )
    g = (
        torch.randn(in_noise, out_noise, generator=generator, dtype=torch.float32)
        * 0.2
    )
    xi = torch.randn(out_noise, generator=generator, dtype=torch.float32) * 0.05

    in_t = generate_transform(
        in_signal,
        in_noise,
        seed=seed,
        domain="l1-in",
        max_condition_number=10.0,
    )
    out_t = generate_transform(
        out_signal,
        out_noise,
        seed=seed + 1,
        domain="l1-out",
        max_condition_number=10.0,
    )
    layer = ChainLinear.from_math(
        weight_math=w,
        bias=b,
        coupling=c,
        propagator=g,
        in_transform=in_t,
        out_transform=out_t,
        refresh_mode="fixed_debug",
        fixed_refresh=xi,
        layer_id="layer1-chain",
    )

    from fastprove.state import decode_debug

    state_in = encode_debug(h, e, in_t, enabled=True)
    state_out = layer(state_in)
    h_hat, e_hat = decode_debug(state_out, out_t, enabled=True)
    y = h @ w + b
    e_expected = h @ c + e @ g + xi
    signal_err = infinity_and_relative(y, h_hat.to(dtype=y.dtype))
    noise_err = infinity_and_relative(e_expected, e_hat.to(dtype=e_expected.dtype))
    return {
        "max_absolute_error": signal_err["max_absolute_error"],
        "relative_l2_error": signal_err["relative_l2_error"],
        "noise_max_absolute_error": noise_err["max_absolute_error"],
        "signal_metrics": signal_err,
        "noise_metrics": noise_err,
        "gate_threshold": FP32_OPERATOR_MAX_ABS_ERROR,
        "gate_passed": signal_err["max_absolute_error"]
        <= FP32_OPERATOR_MAX_ABS_ERROR,
    }


@torch.no_grad()
def run_layer1(
    models: EvalModels,
    input_ids: torch.Tensor,
    *,
    token_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Compare plaintext vs decoded obfuscated checkpoints on identical inputs."""

    device = models.device
    dtype = models.dtype
    input_ids = input_ids.to(device)
    if token_mask is not None:
        token_mask = token_mask.to(device)

    modules: Dict[str, Any] = {}
    # Always run the isolated ChainLinear identity (FP32 gate).
    modules["chain_linear"] = _chain_linear_identity_test(
        seed=models.config.runtime.seed
    )

    if models.obfuscated is None:
        # Plaintext-only condition: operator equivalence is trivial / N/A.
        return {
            "modules": modules,
            "aggregate_max_absolute_error": modules["chain_linear"][
                "max_absolute_error"
            ],
            "note": "plaintext condition; no obfuscated decode comparison",
            "chain_linear": modules["chain_linear"],
        }

    plain = models.plain
    obf = models.obfuscated
    client = models.client
    ctx = models.request_context("layer1")

    # --- Embedding ---
    plain_emb = plain.embedding(input_ids)
    # Obfuscated initial mix: decode after embedding+noise encode inside LM.
    # We re-run the initial path: embedding is shared weight; noise may be zero
    # in structural mode.
    obf_emb = F.embedding(input_ids, obf.embedding_weight)
    initial_noise = (
        obf_emb.float() @ obf.initial_noise_coupling.float()
        + (
            obf.initial_fixed_refresh
            if obf.obfuscation.refresh_mode == "fixed_debug"
            else torch.zeros_like(obf.initial_fixed_refresh)
        )
    ).to(dtype=obf_emb.dtype)
    mixed0 = MixedState(
        obf._basis_mix(torch.cat((obf_emb, initial_noise), dim=-1)),
        obf.hidden_basis,
    )
    hat_emb, _ = client.decode_debug(mixed0)
    modules["embedding"] = infinity_and_relative(
        cast_activations(plain_emb, torch.float32),
        cast_activations(hat_emb, torch.float32),
    )

    # --- Full block debug comparison ---
    # Run the reference path in the declared condition dtype.  Convert only
    # metric operands to FP32; feeding FP32 activations into BF16 weights would
    # silently make the P0--P3 operator comparison invalid.
    plain_h = plain_emb.to(dtype=dtype)

    per_layer: List[Dict[str, Any]] = []
    state = mixed0
    for layer_index, (pblock, oblock) in enumerate(
        zip(plain.blocks, obf.blocks)
    ):
        p_out, p_dbg = pblock.forward_debug(plain_h, token_mask=token_mask)
        # Actually track plain_h across layers:
        o_out, o_dbg = oblock.forward_debug(
            state,
            token_mask=token_mask,
            request_context=ctx,
        )
        hat_final, hat_noise = client.decode_debug(o_out)

        layer_metrics = {
            "layer_index": layer_index,
            "attention_output": infinity_and_relative(
                p_dbg.attention_output.float(),
                o_dbg.attention_output.float(),
            ),
            "post_attention": infinity_and_relative(
                p_dbg.post_attention.float(),
                o_dbg.post_attention.float(),
            ),
            "final_output": infinity_and_relative(
                p_dbg.final_output.float(),
                hat_final.float(),
            ),
            "qk_score": dict(o_dbg.qk_score_error),
            "softmax": dict(o_dbg.softmax_error),
        }
        per_layer.append(layer_metrics)
        plain_h = p_out
        state = o_out

        modules.setdefault("attention_output", []).append(
            layer_metrics["attention_output"]
        )
        modules.setdefault("down_projection", []).append(
            layer_metrics["final_output"]
        )

    # Aggregate module-level max over layers.
    def _agg(items: List[Dict[str, float]]) -> Dict[str, float]:
        if not items:
            return {}
        return {
            "max_absolute_error": max(x["max_absolute_error"] for x in items),
            "relative_l2_error": max(x["relative_l2_error"] for x in items),
            "mean_absolute_error": sum(x.get("mean_absolute_error", 0.0) for x in items)
            / len(items),
        }

    if "attention_output" in modules and isinstance(
        modules["attention_output"], list
    ):
        modules["attention_output"] = _agg(modules["attention_output"])
    if "down_projection" in modules and isinstance(
        modules["down_projection"], list
    ):
        modules["down_projection"] = _agg(modules["down_projection"])

    # Q/K and Value via last layer debug (representative).
    if per_layer:
        last = per_layer[-1]
        modules["qk"] = last["qk_score"]
        modules["value_attention_path"] = last["attention_output"]
        modules["swiglu_via_final"] = last["final_output"]

    # --- LM Head ---
    plain_logits = plain(input_ids, token_mask=token_mask)
    obf_logits = obf(input_ids, token_mask=token_mask, request_context=ctx)
    from .model_factory import inverse_align_logits

    aligned = inverse_align_logits(obf_logits, models.vocab_permutation)
    modules["lm_head"] = infinity_and_relative(
        plain_logits.float(), aligned.float()
    )

    # RMSNorm: compare final norm inputs via decoded last state vs plain.
    plain_final_norm_in = plain_h  # after all blocks
    # Re-fetch plain final hidden properly:
    with torch.no_grad():
        ph = plain.embedding(input_ids)
        for block in plain.blocks:
            ph = block(ph, token_mask=token_mask)
        plain_final_norm_in = ph
    oh, _ = client.decode_debug(state)
    modules["rmsnorm_input"] = infinity_and_relative(
        plain_final_norm_in.float(), oh.float()
    )
    plain_normed = rms_norm_fp32(
        plain_final_norm_in, plain.final_norm_weight, plain.config.rms_epsilon
    )
    obf_normed = rms_norm_fp32(
        oh, obf.final_norm_weight, obf.config.rms_epsilon
    )
    modules["rmsnorm"] = infinity_and_relative(
        plain_normed.float(), obf_normed.float()
    )

    # Aggregate worst-case absolute error across signal modules.
    signal_keys = (
        "embedding",
        "attention_output",
        "down_projection",
        "rmsnorm",
        "lm_head",
        "chain_linear",
    )
    maxes = []
    for key in signal_keys:
        m = modules.get(key)
        if isinstance(m, dict) and "max_absolute_error" in m:
            maxes.append(float(m["max_absolute_error"]))
    aggregate = max(maxes) if maxes else float("nan")

    return {
        "modules": modules,
        "per_layer": per_layer,
        "aggregate_max_absolute_error": aggregate,
        "chain_linear": modules["chain_linear"],
        "gate_threshold": FP32_OPERATOR_MAX_ABS_ERROR,
        "gate_passed_fp32_chainlinear": modules["chain_linear"].get(
            "gate_passed", False
        ),
        "notes": list(models.notes),
    }
