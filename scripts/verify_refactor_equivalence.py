#!/usr/bin/env python3
"""Freeze and verify the pre-refactor numerical baseline.

This is the ground truth for the structure-only cleanup plan: every refactor
commit must keep the digests produced here bit-identical. The script never
writes experiment results; it only observes existing code paths with fixed
seeds and deterministic inputs.

Usage:
    python scripts/verify_refactor_equivalence.py
        Write results/raw/refactor_baseline_<git-sha>.json. Refuses to
        overwrite an existing baseline unless --force is given.

    python scripts/verify_refactor_equivalence.py --check
        Recompute every section and compare against the unique existing
        baseline file. Exit status 0 if identical, 1 otherwise.

    python scripts/verify_refactor_equivalence.py --check --baseline PATH
        Compare against an explicit baseline file.

    python scripts/verify_refactor_equivalence.py --only seed transforms
        Run only the named sections (development aid; --only with --check is
        allowed and compares only those sections).

Digest policy:
    * Tensors are hashed with sha256 over contiguous CPU bytes; shape and
      dtype are recorded alongside.
    * Floats are stored as JSON numbers, which round-trip exactly through
      Python's shortest repr.
    * Dataclasses, tuples, lists, and mappings are walked deterministically;
      mapping keys are sorted.
    * Volatile fields (timestamps, elapsed times, platform metadata) are
      excluded by construction, never by post-hoc normalization.

Requires the package to be installed (``pip install -e .``); this script does
not bootstrap sys.path.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import math
import platform
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from fastprove.augmented_noise import (
    NoiseRefreshSpec,
    apply_noise_refresh,
    deployed_free_noise_sigma,
    free_noise_sigma,
    generate_propagator,
    record_generator,
    sample_initial_noise,
    sample_refresh_noise,
)
from fastprove.config import load_config
from fastprove.conversion import convert_affine_chain
from fastprove.converter import ModelConverter, load_client, load_server_model
from fastprove.evaluation.accuracy import (
    compare_teacher_forced_metrics,
    greedy_generation_metrics,
    make_synthetic_token_batch,
)
from fastprove.evaluation.artifacts import build_run_record, validate_run_record
from fastprove.evaluation.metrics import (
    attention_distribution_metrics,
    tensor_error_metrics,
)
from fastprove.evaluation.report import _rows, _series_groups
from fastprove.layers.attention import ApproximationConfig, AttentionMode
from fastprove.layers.deployed import (
    build_deployed_attention,
    build_deployed_feed_forward,
    validate_auxiliary_budget,
)
from fastprove.layers.linear import ChainLinear
from fastprove.layers.moe import ObfuscatedMoE
from fastprove.layers.router import RouterConfig, RouterMode, StableRouter
from fastprove.models.obfuscated import (
    ObfuscatedDecoderBlock,
    ObfuscatedTinyCausalLM,
)
from fastprove.models.plain import PlainDecoderBlock, PlainTinyCausalLM
from fastprove.noise_budget import DualChannelBudget, semantic_hiding_bits
from fastprove.seed import RequestContext, derive_seed, make_generator
from fastprove.state import decode_debug, encode_debug
from fastprove.structured import (
    AUXILIARY_MAGNITUDE_BOUND,
    check_auxiliary_magnitude,
    generate_structured_basis,
    structured_condition_number,
)
from fastprove.transforms import (
    generate_orthogonal,
    generate_signed_permutation,
    generate_transform,
    generate_two_sided_transform,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "fastprove.refactor_equivalence.v1"
BASELINE_GLOB = "results/raw/refactor_baseline_*.json"

_SECTIONS: "Dict[str, Callable[[], Any]]" = {}


def section(function: Callable[[], Any]) -> Callable[[], Any]:
    _SECTIONS[function.__name__] = function
    return function


# ---------------------------------------------------------------------------
# canonicalization helpers


def _tensor_leaf(tensor: torch.Tensor) -> Dict[str, Any]:
    value = tensor.detach().to("cpu").contiguous()
    if value.dtype == torch.bfloat16:
        value = value.to(torch.float32)
    array = value.numpy()
    return {
        "kind": "tensor",
        "shape": list(value.shape),
        "dtype": str(value.dtype).replace("torch.", ""),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _canonical(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor_leaf(value)
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    raise TypeError("unsupported digest value: %r" % (type(value),))


def _tensor_max_abs_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        (left.detach().to(torch.float64) - right.detach().to(torch.float64))
        .abs()
        .max()
        .item()
    )


# ---------------------------------------------------------------------------
# sections: deterministic kernel primitives


@section
def seed() -> Any:
    context = RequestContext(52, "sample-7")
    other = RequestContext(52, "sample-8")
    generator = make_generator(7, "unit")
    draws = torch.rand(3, generator=generator)
    return {
        "derive_seed_1234_layer_2_attention": derive_seed(
            1234, "layer", 2, "attention"
        ),
        "request_seed_layer_1": context.seed_for("layer", 1),
        "request_seed_layer_1_other_request": other.seed_for("layer", 1),
        "make_generator_draws": draws,
    }


@section
def transforms() -> Any:
    with torch.random.fork_rng(devices=[]):
        dense = generate_transform(
            5,
            3,
            seed=17,
            domain="hidden",
            max_condition_number=4.0,
        )
        two_sided = generate_two_sided_transform(
            8,
            2,
            seed=17,
            domain="two-sided-test",
            max_condition_number=3.0,
            dtype=torch.float64,
        )
        orthogonal = generate_orthogonal(6, seed=8, domain="qk")
        signed = generate_signed_permutation(5, seed=11, domain="noise")
    return {
        "dense_family": {
            "matrix": dense.matrix,
            "inverse": dense.inverse,
            "condition_number": dense.condition_number,
            "fingerprint": dense.fingerprint,
            "descriptor": dense.descriptor,
        },
        "two_sided_family": {
            "matrix": two_sided.matrix,
            "inverse": two_sided.inverse,
            "condition_number": two_sided.condition_number,
            "fingerprint": two_sided.fingerprint,
            "descriptor": two_sided.descriptor,
        },
        "orthogonal_6": orthogonal,
        "signed_permutation_5": signed,
    }


@section
def structured() -> Any:
    basis = generate_structured_basis(
        64,
        8,
        seed=17,
        domain="free-noise-cone-check",
        block_size=8,
        max_condition_number=10.0,
        dtype=torch.float64,
    )
    free = basis.free_noise_basis()
    signal = torch.tensor([[1.0, -2.0, 3.0, 0.5]])
    noise = torch.tensor([[0.1, -0.2, 0.05, 0.0]])
    return {
        "perm_in": basis.perm_in,
        "scales": basis.scales,
        "blocks": basis.blocks,
        "perm_out": basis.perm_out,
        "gram_blocks": basis.gram_blocks,
        "signal_dim": basis.signal_dim,
        "noise_dim": basis.noise_dim,
        "total_dim": basis.total_dim,
        "block_size": basis.block_size,
        "block_count": basis.block_count,
        "descriptor": basis.descriptor,
        "condition_number": basis.condition_number,
        "recomputed_condition_number": structured_condition_number(basis),
        "fingerprint": basis.fingerprint,
        "free_noise_basis": free,
        "free_noise_dim": int(free.shape[0]),
        "signal_projection": basis.signal_projection(),
        "noise_projection": basis.noise_projection(),
        "signal_rows": basis.signal_rows(),
        "noise_rows": basis.noise_rows(),
        "dense": basis.dense(),
        "dense_inverse": basis.dense_inverse(),
        "auxiliary_magnitude_bound": AUXILIARY_MAGNITUDE_BOUND,
        "check_auxiliary_magnitude_ratio": check_auxiliary_magnitude(
            signal, noise, context="digest"
        ),
    }


@section
def augmented_noise() -> Any:
    spec = NoiseRefreshSpec(noise_dim=16, gamma=0.5, energy_ratio=0.7)
    generator = make_generator(5, "digest-noise")
    initial = sample_initial_noise(
        spec, calibration_rms=1.0, shape=(4, 16), generator=generator
    )
    refresh = sample_refresh_noise(
        spec, calibration_rms=1.0, shape=(4, 16), generator=generator
    )
    propagator = generate_propagator(spec, seed=9, domain="stationary")
    noise_state = torch.Generator().manual_seed(11)
    e0 = torch.randn(4, 16, generator=noise_state) * 0.05
    applied = apply_noise_refresh(e0, propagator, refresh)
    record_gen = record_generator(
        global_seed=7,
        key_epoch="k0",
        request_nonce="rq1",
        sample_id="s1",
        token_position=5,
        layer_id="L0",
        operation="attn",
        head_id=0,
    )
    sampled = torch.randn(4, generator=record_gen)
    return {
        "free_noise_sigma": free_noise_sigma(0.7, 0.5, 16),
        "deployed_free_noise_sigma_8": deployed_free_noise_sigma(0.02, 8),
        "deployed_free_noise_sigma_16": deployed_free_noise_sigma(0.02, 16),
        "component_sigma": spec.component_sigma(2.0),
        "initial_noise": initial,
        "refresh_noise": refresh,
        "propagator": propagator,
        "applied_refresh": applied,
        "record_generator_draws": sampled,
    }


@section
def noise_budget() -> Any:
    exact = load_config(ROOT / "configs" / "tiny_exact.yaml")
    approx = load_config(ROOT / "configs" / "tiny_approx.yaml")
    basis = generate_structured_basis(
        32,
        8,
        seed=11,
        domain="noise-budget-test",
        block_size=8,
        dtype=torch.float64,
    )
    return {
        "exact_manifest": DualChannelBudget.from_config(exact).manifest(),
        "approx_manifest": DualChannelBudget.from_config(approx).manifest(),
        "approx_manifest_with_basis": DualChannelBudget.from_config(
            approx, basis=basis
        ).manifest(),
        "semantic_hiding_bits_1e-5": semantic_hiding_bits(1e-5),
        "semantic_hiding_bits_0p577": semantic_hiding_bits(0.577),
        "semantic_hiding_bits_1": semantic_hiding_bits(1.0),
    }


@section
def chain_linear() -> Any:
    generator = torch.Generator().manual_seed(123)
    hidden = torch.randn(2, 3, 3, generator=generator)
    noise = torch.randn(2, 3, 2, generator=generator)
    weight_math = torch.randn(3, 4, generator=generator) * 0.2
    bias = torch.randn(4, generator=generator) * 0.1
    coupling = torch.randn(3, 3, generator=generator) * 0.1
    propagator = torch.randn(2, 3, generator=generator) * 0.2
    refresh = torch.randn(3, generator=generator) * 0.05
    in_transform = generate_transform(3, 2, seed=10, domain="linear-in")
    out_transform = generate_transform(4, 3, seed=20, domain="linear-out")
    layer = ChainLinear.from_math(
        weight_math=weight_math,
        bias=bias,
        coupling=coupling,
        propagator=propagator,
        in_transform=in_transform,
        out_transform=out_transform,
        refresh_mode="fixed_debug",
        fixed_refresh=refresh,
        layer_id="linear-0",
    )
    state = encode_debug(hidden, noise, in_transform, enabled=True)
    actual = layer(state)
    decoded_signal, decoded_noise = decode_debug(
        actual, out_transform, enabled=True
    )
    expected_signal = hidden @ weight_math + bias
    expected_noise = hidden @ coupling + noise @ propagator + refresh
    expected_mixed = torch.cat((expected_signal, expected_noise), dim=-1) @ (
        out_transform.matrix
    )
    per_request = ChainLinear.from_math(
        weight_math=weight_math,
        bias=bias,
        coupling=coupling,
        propagator=propagator,
        in_transform=in_transform,
        out_transform=out_transform,
        refresh_mode="per_request",
        fixed_refresh=None,
        layer_id="linear-0",
        refresh_scale=0.05,
    )
    first = per_request(state, request_context=RequestContext(7, "digest-a"))
    second = per_request(state, request_context=RequestContext(7, "digest-a"))
    other = per_request(state, request_context=RequestContext(7, "digest-b"))
    direct = convert_affine_chain(
        weight_math=weight_math,
        bias=bias,
        coupling=coupling,
        propagator=propagator,
        in_transform=in_transform,
        out_transform=out_transform,
        fixed_refresh=refresh,
    )
    return {
        "actual_mixed": actual.mixed,
        "decoded_signal": decoded_signal,
        "decoded_noise": decoded_noise,
        "expected_mixed": expected_mixed,
        "identity_residual": _tensor_max_abs_error(actual.mixed, expected_mixed),
        "per_request_same_context_equal": bool(
            torch.equal(first.mixed, second.mixed)
        ),
        "per_request_other_context_differs": bool(
            not torch.equal(first.mixed, other.mixed)
        ),
        "per_request_first": first.mixed,
        "affine_chain_weight_pt": direct.weight_pt,
        "affine_chain_bias_mixed": direct.bias_mixed,
    }


# ---------------------------------------------------------------------------
# sections: deployed weight builders


def _deployed_setup() -> Dict[str, Any]:
    hidden, noise, block = 64, 8, 8
    heads, kv_heads, head_dim, intermediate = 4, 2, 16, 96
    seed = 17
    torch.manual_seed(seed)
    basis = generate_structured_basis(
        hidden, noise, seed=seed, domain="block", block_size=block,
        dtype=torch.float64,
    )
    value_bases = tuple(
        generate_structured_basis(
            head_dim,
            2,
            seed=seed + 100 + head,
            domain="value",
            block_size=head_dim + 2,
            dtype=torch.float64,
        )
        for head in range(kv_heads)
    )
    value_noise = 2
    repeat = heads // kv_heads
    kv_index = torch.arange(kv_heads).repeat_interleave(repeat)
    gamma_attention = torch.rand(hidden, dtype=torch.float64) + 0.5
    gamma_ffn = torch.rand(hidden, dtype=torch.float64) + 0.5
    w_q = torch.randn(hidden, heads * head_dim, dtype=torch.float64) * 0.1
    w_k = torch.randn(hidden, kv_heads * head_dim, dtype=torch.float64) * 0.1
    w_v = torch.randn(hidden, kv_heads * head_dim, dtype=torch.float64) * 0.1
    w_o = torch.randn(heads * head_dim, hidden, dtype=torch.float64) * 0.1
    w_gate = torch.randn(hidden, intermediate, dtype=torch.float64) * 0.1
    w_up = torch.randn(hidden, intermediate, dtype=torch.float64) * 0.1
    w_down = torch.randn(intermediate, hidden, dtype=torch.float64) * 0.1
    coupling_value = (
        torch.randn(hidden + noise, kv_heads, value_noise, dtype=torch.float64)
        * 0.03
    )
    coupling_o = torch.randn(hidden, noise, dtype=torch.float64) * 0.02
    propagator_a = 0.5 * torch.eye(noise, dtype=torch.float64)
    auxiliary = torch.randn(value_noise, noise, dtype=torch.float64) * 0.08
    coupling_z = torch.randn(intermediate, noise, dtype=torch.float64) * 0.02
    coupling_d = torch.randn(hidden, noise, dtype=torch.float64) * 0.02
    propagator_f = 0.5 * torch.eye(noise, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed + 7)
    neuron_permutation = torch.randperm(intermediate, generator=generator)
    neuron_scale = torch.exp(
        (2 * torch.rand(intermediate, generator=generator, dtype=torch.float64) - 1)
        * 0.3
    )
    attention = build_deployed_attention(
        basis=basis,
        value_bases=value_bases,
        gamma_attention=gamma_attention,
        q_weight_math=w_q,
        k_weight_math=w_k,
        v_weight_math=w_v,
        o_weight_math=w_o,
        q_bias=None,
        k_bias=None,
        v_bias=None,
        value_signal_coupling=coupling_value,
        signal_noise_coupling=coupling_o,
        noise_propagator=propagator_a,
        auxiliary_to_hidden=auxiliary,
        fixed_refresh=None,
        kv_index=kv_index,
        head_dim=head_dim,
        dtype=torch.float64,
    )
    feed_forward = build_deployed_feed_forward(
        basis=basis,
        gamma_ffn=gamma_ffn,
        gate_weight_math=w_gate,
        up_weight_math=w_up,
        down_weight_math=w_down,
        neuron_permutation=neuron_permutation,
        neuron_scale=neuron_scale,
        swiglu_noise_coupling=coupling_z,
        down_noise_coupling=coupling_d,
        noise_propagator=propagator_f,
        fixed_refresh=None,
        dtype=torch.float64,
    )
    sample_signal = torch.randn(2, 6, hidden) * 0.5
    ratio = validate_auxiliary_budget(
        basis=basis,
        sample_signal=sample_signal,
        signal_noise_coupling=coupling_o,
        noise_propagator=propagator_a,
        context="digest",
    )
    return {
        "basis_fingerprint": basis.fingerprint,
        "attention": attention,
        "feed_forward": feed_forward,
        "auxiliary_budget_ratio": ratio,
    }


@section
def deployed_weights() -> Any:
    with torch.random.fork_rng(devices=[]):
        return _deployed_setup()


# ---------------------------------------------------------------------------
# sections: single block, all attention modes


def _block_config() -> Tuple[Any, Any]:
    from fastprove.config import ModelConfig, ObfuscationConfig

    model = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_sequence_length=32,
        rms_epsilon=1e-5,
        rope_theta=10000.0,
    )
    obfuscation = ObfuscationConfig(
        hidden_noise_dim=8,
        value_noise_dim_per_head=2,
        max_condition_number=10.0,
        noise_propagation_gamma=0.5,
        refresh_mode="fixed_debug",
        basis_block_size=8,
    )
    return model, obfuscation


def _block_inputs() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(303)
    hidden = torch.randn(2, 7, 32, generator=generator)
    noise = torch.randn(2, 7, 8, generator=generator)
    token_mask = torch.tensor(
        [
            [True, True, True, True, True, False, False],
            [True, True, True, True, True, True, True],
        ]
    )
    return hidden, noise, token_mask


def _convert_block(
    mode: AttentionMode,
    approximation: Optional[ApproximationConfig],
    refresh_mode: str = "fixed_debug",
    rho_noise_bound: float = 0.0,
):
    from fastprove.config import ObfuscationConfig

    model, obfuscation = _block_config()
    if refresh_mode != obfuscation.refresh_mode or rho_noise_bound != 0.0:
        obfuscation = ObfuscationConfig(
            hidden_noise_dim=obfuscation.hidden_noise_dim,
            value_noise_dim_per_head=obfuscation.value_noise_dim_per_head,
            max_condition_number=obfuscation.max_condition_number,
            noise_propagation_gamma=obfuscation.noise_propagation_gamma,
            refresh_mode=refresh_mode,
            basis_block_size=obfuscation.basis_block_size,
            rho_noise_bound=rho_noise_bound,
            free_noise_energy_ratio=obfuscation.free_noise_energy_ratio,
            signal_noise_coupling_scale=obfuscation.signal_noise_coupling_scale,
        )
    plain = PlainDecoderBlock(model, seed=101, layer_id=0, debug_enabled=True)
    converted = ObfuscatedDecoderBlock.from_plain(
        plain,
        obfuscation=obfuscation,
        mode=mode,
        approximation=approximation,
        seed=202,
        debug_enabled=True,
    )
    return plain, converted


def _block_digest(
    plain: PlainDecoderBlock,
    converted,
    hidden: torch.Tensor,
    noise: torch.Tensor,
    token_mask: torch.Tensor,
) -> Dict[str, Any]:
    context = RequestContext(404, "exact-block")
    plain_output, plain_debug = plain.forward_debug(hidden, token_mask)
    state = converted.client.encode_debug(hidden, noise)
    actual_state, debug = converted.module.forward_debug(
        state,
        token_mask=token_mask,
        request_context=context,
    )
    actual_signal, actual_noise = converted.client.decode_debug(actual_state)
    return {
        "plain_output": plain_output,
        "plain_debug": plain_debug,
        "debug": debug,
        "actual_signal": actual_signal,
        "actual_noise": actual_noise,
        "signal_max_abs_error": _tensor_max_abs_error(
            actual_signal, plain_output
        ),
    }


@section
def blocks() -> Any:
    model, _ = _block_config()
    hidden, noise, token_mask = _block_inputs()
    result: Dict[str, Any] = {}
    with torch.random.fork_rng(devices=[]):
        plain = PlainDecoderBlock(model, seed=101, layer_id=0, debug_enabled=True)
        plain_output, plain_debug = plain.forward_debug(hidden, token_mask)
        result["plaintext"] = {
            "plain_output": plain_output,
            "plain_debug": plain_debug,
        }
        for name, approximation in (
            ("exact", None),
            ("topk_preserving", ApproximationConfig(0.03, 0.1, 0.8, 2)),
            ("free_bounded", ApproximationConfig(0.03, 0.1, 0.8, 2)),
        ):
            mode = AttentionMode(name)
            plain_for_mode, converted = _convert_block(mode, approximation)
            result[name] = _block_digest(
                plain_for_mode, converted, hidden, noise, token_mask
            )
        # rho field and per-layer refresh samples on the per_request path
        _, converted_per = _convert_block(
            AttentionMode.EXACT, None, refresh_mode="per_request",
            rho_noise_bound=1e-5,
        )
        mixed_state = converted_per.client.encode_debug(hidden, noise)
        context = RequestContext(404, "exact-block")
        positions = torch.arange(hidden.shape[1])
        module = converted_per.module
        rho_attn = module._perturbed_rms_scale(
            mixed_state.mixed, context, "attn", positions
        )
        rho_ffn = module._perturbed_rms_scale(
            mixed_state.mixed, context, "ffn", positions
        )
        refresh_attn = module._refresh_out(
            module.deployed_attn_refresh_out, context, "attn", mixed_state.mixed
        )
        refresh_ffn = module._refresh_out(
            module.deployed_ffn_refresh_out, context, "ffn", mixed_state.mixed
        )
        result["per_request_rho_and_refresh"] = {
            "rho_attn": rho_attn,
            "rho_ffn": rho_ffn,
            "refresh_attn": refresh_attn,
            "refresh_ffn": refresh_ffn,
        }
        # fixed_debug refresh vectors for comparison
        _, converted_fixed = _convert_block(AttentionMode.EXACT, None)
        fixed_module = converted_fixed.module
        fixed_state = converted_fixed.client.encode_debug(hidden, noise)
        result["fixed_refresh"] = {
            "refresh_attn": fixed_module._refresh_out(
                fixed_module.deployed_attn_refresh_out,
                context,
                "attn",
                fixed_state.mixed,
            ),
            "refresh_ffn": fixed_module._refresh_out(
                fixed_module.deployed_ffn_refresh_out,
                context,
                "ffn",
                fixed_state.mixed,
            ),
        }
    return result


# ---------------------------------------------------------------------------
# sections: tiny LM end to end


def _lm_bundle(config_path: Path):
    config = load_config(config_path)
    seed = config.runtime.seed
    plain = PlainTinyCausalLM(
        config.model, seed=seed, debug_enabled=config.runtime.debug_enabled
    )
    approximation = None
    if config.attention.mode not in (AttentionMode.PLAINTEXT, AttentionMode.EXACT):
        approximation = ApproximationConfig(
            config.attention.tau_max,
            config.attention.tau_error,
            config.attention.alpha,
            config.attention.preserve_top_k,
        )
    converted = ObfuscatedTinyCausalLM.from_plain(
        plain,
        obfuscation=config.obfuscation,
        mode=config.attention.mode,
        approximation=approximation,
        seed=seed,
        debug_enabled=config.runtime.debug_enabled,
    )
    return plain, converted


def _lm_digest(plain, converted) -> Dict[str, Any]:
    codec = converted.token_codec
    input_ids = torch.tensor(
        [[1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]]
    )
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    context = RequestContext(503, "tiny-exact")
    plain_logits = plain(input_ids, token_mask=mask)
    encoded = codec.encode(input_ids)
    obfuscated_logits = converted.module(
        encoded, token_mask=mask, request_context=context
    )
    decoded_logits = obfuscated_logits[..., codec.permutation]
    plain_generated = plain.generate_greedy(input_ids, max_new_tokens=4)
    obfuscated_generated = codec.decode(
        converted.module.generate_greedy(
            encoded, max_new_tokens=4, request_context=context
        )
    )
    prefix = encoded[:, :4]
    cache_context = RequestContext(514, "tiny-cache")
    full_logits = converted.module(
        encoded, positions=torch.arange(encoded.shape[1]),
        request_context=cache_context,
    )
    _, cache = converted.module(
        prefix, positions=torch.arange(4),
        request_context=cache_context, use_cache=True,
    )
    step_logits, updated_cache = converted.module(
        encoded[:, 4:], positions=torch.tensor([4, 5]),
        request_context=cache_context, cache=cache, use_cache=True,
    )
    initial_refresh = converted.module._initial_refresh_out(
        context, converted.module.embedding(encoded).mixed
    )
    return {
        "plain_logits": plain_logits,
        "obfuscated_logits": obfuscated_logits,
        "decoded_logits": decoded_logits,
        "logits_max_abs_error": _tensor_max_abs_error(
            decoded_logits, plain_logits
        ),
        "logits_argmax_equal": bool(
            torch.equal(
                decoded_logits.argmax(dim=-1), plain_logits.argmax(dim=-1)
            )
        ),
        "plain_generated": plain_generated,
        "obfuscated_generated": obfuscated_generated,
        "greedy_equal": bool(torch.equal(plain_generated, obfuscated_generated)),
        "full_logits": full_logits,
        "step_logits": step_logits,
        "cache_step_max_abs_error": _tensor_max_abs_error(
            step_logits[:, -1], full_logits[:, -1]
        ),
        "cache": cache,
        "updated_cache": updated_cache,
        "initial_refresh": initial_refresh,
    }


@section
def tiny_lm() -> Any:
    with torch.random.fork_rng(devices=[]):
        exact_plain, exact_converted = _lm_bundle(
            ROOT / "configs" / "tiny_exact.yaml"
        )
        exact = _lm_digest(exact_plain, exact_converted)
    with torch.random.fork_rng(devices=[]):
        approx_plain, approx_converted = _lm_bundle(
            ROOT / "configs" / "tiny_approx.yaml"
        )
        approx = _lm_digest(approx_plain, approx_converted)
    return {"exact_config": exact, "approx_config": approx}


# ---------------------------------------------------------------------------
# sections: router and MoE


@section
def router_moe() -> Any:
    router_results: Dict[str, Any] = {}
    logits = torch.tensor([[5.0, 1.0, 4.0, 3.0]])
    permutation = torch.tensor([2, 0, 3, 1])
    exact_router = StableRouter(
        permutation=permutation,
        top_k=2,
        mode=RouterMode.EXACT,
        config=None,
        debug_enabled=True,
        layer_id="router-exact",
    )
    decision, debug = exact_router.forward_debug(
        logits, RequestContext(3, "router")
    )
    router_results["exact"] = {"decision": decision, "debug": debug}
    bounded_logits = torch.tensor([[4.0, 3.0, 1.0, -2.0], [3.0, 2.5, 2.0, 0.0]])
    config = RouterConfig(
        tau_max=0.2, tau_error=0.2, alpha=0.8, noisy_gate_weights=False
    )
    bounded_router = StableRouter(
        permutation=permutation,
        top_k=2,
        mode=RouterMode.MARGIN_BOUNDED,
        config=config,
        debug_enabled=True,
        layer_id="router-bounded",
    )
    decision, debug = bounded_router.forward_debug(
        bounded_logits, RequestContext(5, "router-bounded")
    )
    router_results["margin_bounded"] = {"decision": decision, "debug": debug}
    noisy_gate_router = StableRouter(
        permutation=permutation,
        top_k=2,
        mode=RouterMode.MARGIN_BOUNDED,
        config=RouterConfig(
            tau_max=0.2, tau_error=0.2, alpha=0.8, noisy_gate_weights=True
        ),
        debug_enabled=True,
        layer_id="router-bounded",
    )
    decision, debug = noisy_gate_router.forward_debug(
        bounded_logits, RequestContext(5, "router-bounded")
    )
    router_results["margin_bounded_noisy_gates"] = {
        "decision": decision,
        "debug": debug,
    }

    with torch.random.fork_rng(devices=[]):
        basis = generate_structured_basis(
            32, 4, seed=7, domain="moe", block_size=4, dtype=torch.float64
        )
        generator = torch.Generator().manual_seed(7)
        gamma = torch.rand(32, generator=generator) + 0.5
        router_w = torch.randn(32, 8, generator=generator) * 0.2
        expert_weights = [
            (
                torch.randn(32, 48, generator=generator) * 0.1,
                torch.randn(32, 48, generator=generator) * 0.1,
                torch.randn(48, 32, generator=generator) * 0.1,
            )
            for _ in range(8)
        ]
        permutation_moe = torch.randperm(8, generator=generator)
        hidden = torch.randn(
            2, 5, 32, generator=torch.Generator().manual_seed(9)
        )
        mixed = basis.mix(
            torch.cat(
                (
                    hidden,
                    torch.randn(
                        2,
                        5,
                        4,
                        generator=torch.Generator().manual_seed(3),
                    )
                    * 0.05,
                ),
                dim=-1,
            )
        )
        moe = ObfuscatedMoE(
            basis=basis,
            gamma_ffn=gamma,
            router_weight_math=router_w,
            plain_experts=expert_weights,
            expert_permutation=permutation_moe,
            top_k=2,
            router_mode=RouterMode.EXACT,
            router_config=None,
            seed=7,
            layer_id="block-0",
            debug_enabled=True,
            noise_propagation_gamma=0.5,
            refresh_mode="fixed_debug",
            signal_noise_coupling_scale=0.02,
            free_noise_energy_ratio=0.02,
        )
        _, moe_debug = moe.forward_debug(mixed, RequestContext(3, "moe"))
        moe_bounded = ObfuscatedMoE(
            basis=basis,
            gamma_ffn=gamma,
            router_weight_math=router_w,
            plain_experts=expert_weights,
            expert_permutation=permutation_moe,
            top_k=2,
            router_mode=RouterMode.MARGIN_BOUNDED,
            router_config=RouterConfig(
                tau_max=0.05,
                tau_error=0.05,
                alpha=0.5,
                noisy_gate_weights=True,
            ),
            seed=23,
            layer_id="block-0",
            debug_enabled=True,
            noise_propagation_gamma=0.5,
            refresh_mode="fixed_debug",
            signal_noise_coupling_scale=0.02,
            free_noise_energy_ratio=0.02,
        )
        _, moe_bounded_debug = moe_bounded.forward_debug(
            mixed, RequestContext(11, "moe-bounded")
        )
    return {
        "router": router_results,
        "moe_exact": {"debug": moe_debug},
        "moe_margin_bounded": {"debug": moe_bounded_debug},
    }


# ---------------------------------------------------------------------------
# sections: converter artifacts


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@section
def converter() -> Any:
    config = load_config(ROOT / "configs" / "tiny_approx.yaml")
    plain = PlainTinyCausalLM(
        config.model, seed=config.runtime.seed, debug_enabled=False
    )
    with torch.random.fork_rng(devices=[]):
        with tempfile.TemporaryDirectory() as tmp:
            client_dir = Path(tmp) / "client"
            server_dir = Path(tmp) / "server"
            manifest = ModelConverter().convert(
                plain,
                config,
                client_secret_dir=client_dir,
                server_model_dir=server_dir,
            )
            server_manifest = json.loads(
                (server_dir / "manifest.json").read_text(encoding="utf-8")
            )
            server_manifest.pop("converted_at_utc", None)
            runtime_config = json.loads(
                (server_dir / "runtime_config.json").read_text(encoding="utf-8")
            )
            noise_budget = json.loads(
                (server_dir / "noise_budget.json").read_text(encoding="utf-8")
            )
            client_manifest = json.loads(
                (client_dir / "manifest.json").read_text(encoding="utf-8")
            )
            from safetensors.torch import load_file

            weights_path = server_dir / "model.safetensors"
            if not weights_path.exists():
                weights_path = server_dir / "model.pt"
            if weights_path.suffix == ".safetensors":
                state_dict = load_file(str(weights_path))
            else:
                state_dict = torch.load(
                    weights_path, map_location="cpu", weights_only=True
                )
            bases = torch.load(
                client_dir / "bases.pt", map_location="cpu", weights_only=True
            )
            token_codec = torch.load(
                client_dir / "token_codec.pt",
                map_location="cpu",
                weights_only=True,
            )
            server_module = load_server_model(server_dir)
            bundle = load_client(client_dir)
            ids = torch.tensor([[1, 2, 3, 4]])
            encoded = bundle.encode(ids)
            logits = server_module(
                encoded, request_context=RequestContext(config.runtime.seed, "caps")
            )
            decoded_logits = bundle.decode_logits(logits)
            file_hashes = {
                "noise_budget.json": _file_digest(
                    server_dir / "noise_budget.json"
                ),
                "runtime_config.json": _file_digest(
                    server_dir / "runtime_config.json"
                ),
                "client_manifest.json": _file_digest(
                    client_dir / "manifest.json"
                ),
            }
    return {
        "conversion_manifest": {
            "weights_format": manifest.weights_format,
            "tied_embedding": manifest.tied_embedding,
            "server_state_keys": list(manifest.server_state_keys),
            "model_sha256": manifest.model_sha256,
            "runtime_config_sha256": manifest.runtime_config_sha256,
            "client_manifest_sha256": manifest.client_manifest_sha256,
        },
        "server_manifest": server_manifest,
        "runtime_config": runtime_config,
        "noise_budget": noise_budget,
        "client_manifest": client_manifest,
        "server_state_dict": state_dict,
        "server_state_key_list": sorted(state_dict.keys()),
        "client_bases": bases,
        "client_token_codec": token_codec,
        "roundtrip_encoded_tokens": encoded,
        "roundtrip_logits": logits,
        "roundtrip_decoded_logits": decoded_logits,
        "file_hashes": file_hashes,
    }


# ---------------------------------------------------------------------------
# sections: evaluation pipeline


def _report_fixture_metrics(tau: float) -> Dict[str, Any]:
    return {
        "plaintext": {
            "negative_log_likelihood": 1.0,
            "perplexity": math.e,
            "next_token_top1_accuracy": 0.5,
            "next_token_top5_accuracy": 0.9,
        },
        "obfuscated": {
            "negative_log_likelihood": 1.0 + tau,
            "perplexity": math.e + tau,
            "next_token_top1_accuracy": 0.5 - tau,
            "next_token_top5_accuracy": 0.9 - tau,
        },
        "agreement": {"next_token_top1_agreement": 1.0 - tau},
        "degradation": {
            "nll_absolute_increase": tau,
            "nll_relative_increase": tau,
            "perplexity_absolute_increase": tau,
            "perplexity_relative_increase": tau / math.e,
            "top1_absolute_drop": tau,
            "top1_relative_drop": tau / 0.5,
            "top5_absolute_drop": tau,
            "top5_relative_drop": tau / 0.9,
        },
        "greedy": {
            "greedy_token_exact_match": 1.0 - tau,
            "greedy_sequence_exact_match": 1.0 - tau,
        },
        "softmax": {
            "kl_divergence": tau,
            "js_divergence": tau / 2,
            "topk_overlap": 1.0 - tau,
            "rank_correlation": 1.0 - tau,
            "topk_changed_fraction": tau,
            "actual_noise_infinity_norm": tau,
            "zero_noise_query_fraction": 1.0 if tau == 0 else 0.0,
            "attention_output_relative_l2_error": tau,
        },
        "performance": {
            "conversion_time_seconds": 0.01,
            "prefill_latency_seconds": 0.02,
            "decode_tpot_seconds": 0.003,
            "tokens_per_second": 333.0,
            "peak_memory_bytes": 1024,
            "kv_cache_memory_bytes": 512,
        },
    }


def _report_fixture_record(tau: float = 0.01) -> Dict[str, Any]:
    record = build_run_record(
        run_id="digest-record",
        status="success",
        config={
            "mode": "free_bounded",
            "tau_max": tau,
            "alpha": 0.8,
            "preserve_top_k": 8,
            "stage": "full_evaluation",
            "evaluation": {
                "sequence_length": 128,
                "batch_size": 2,
            },
        },
        seed=52,
        model={
            "id": "unit-model",
            "revision": "revision-a",
            "weight_sha256": "weight-sha-a",
            "pretrained": False,
        },
        dataset={
            "id": "unit-data",
            "split": "test",
            "sample_ids_sha256": "sample-sha-a",
            "tokenized_inputs_sha256": "tokenized-sha-a",
            "meaningful_lm_evidence": False,
        },
        environment={
            "actual_device": "cpu",
            "activation_dtype": "float32",
            "batch_size": 2,
        },
        sample_count=4,
        metrics=_report_fixture_metrics(tau),
        elapsed_seconds=0.1,
    )
    record["_raw_source"] = "results/raw/attempt/eval.jsonl"
    record["_raw_line"] = 1
    record["timestamp_utc"] = "1970-01-01T00:00:00+00:00"
    return record


@section
def evaluation() -> Any:
    value = torch.tensor([[1.0, -2.0, 3.0]])
    error_metrics = tensor_error_metrics(value, value.clone() * 1.01)
    clean_probabilities = torch.tensor(
        [[[[0.6, 0.3, 0.1, 0.0], [0.5, 0.5, 0.0, 0.0]]]]
    )
    noisy_probabilities = torch.tensor(
        [[[[0.2, 0.3, 0.5, 0.0], [0.5, 0.5, 0.0, 0.0]]]]
    )
    clean_logits = torch.tensor(
        [[[[3.0, 2.0, 1.0, -torch.inf], [1.0, 1.0, -torch.inf, -torch.inf]]]]
    )
    noisy_logits = torch.tensor(
        [[[[1.0, 2.0, 3.0, -torch.inf], [1.0, 1.0, -torch.inf, -torch.inf]]]]
    )
    valid = torch.tensor(
        [[[[True, True, True, False], [True, True, False, False]]]]
    )
    logit_noise = torch.tensor(
        [[[[0.04, -0.02, -0.02, 0.0], [0.0, 0.0, 0.0, 0.0]]]]
    )
    margin = torch.tensor([[[[1.0], [0.0]]]])
    clean_output = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    noisy_output = torch.tensor([[[[1.1, 1.9], [3.0, 4.0]]]])
    distribution_metrics = attention_distribution_metrics(
        clean_probabilities=clean_probabilities,
        noisy_probabilities=noisy_probabilities,
        clean_logits=clean_logits,
        noisy_logits=noisy_logits,
        valid_mask=valid,
        noise=logit_noise,
        margin=margin,
        top_k=1,
        clean_output=clean_output,
        noisy_output=noisy_output,
    )
    tokens, identifiers = make_synthetic_token_batch(
        sample_count=5, sequence_length=8, vocab_size=32, seed=77
    )
    generator = torch.Generator().manual_seed(99)
    plaintext_logits = torch.randn(2, 5, 7, generator=generator)
    obfuscated_logits = plaintext_logits + 0.01 * torch.randn(
        2, 5, 7, generator=generator
    )
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    teacher_forced = compare_teacher_forced_metrics(
        plaintext_logits=plaintext_logits,
        obfuscated_logits=obfuscated_logits,
        input_ids=input_ids,
        token_mask=mask,
        bootstrap_replicates=50,
        bootstrap_seed=123,
    )
    generation = greedy_generation_metrics(
        plaintext_tokens=torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]]),
        obfuscated_tokens=torch.tensor([[1, 2, 3, 9], [2, 3, 4, 5]]),
        prompt_length=2,
    )
    record = _report_fixture_record()
    validate_run_record(record)
    digest_record = dict(record)
    rows = _rows([record])
    groups = _series_groups(rows)
    return {
        "tensor_error_metrics": error_metrics,
        "attention_distribution_metrics": distribution_metrics,
        "synthetic_token_batch_tokens": tokens,
        "synthetic_token_identifiers": identifiers,
        "teacher_forced_metrics": teacher_forced,
        "greedy_generation_metrics": generation,
        "run_record": digest_record,
        "report_rows": rows,
        "report_series_groups": groups,
    }


# ---------------------------------------------------------------------------
# runner plumbing


def _collect(section_names: List[str]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for name in section_names:
        function = _SECTIONS[name]
        results[name] = _canonical(function())
    return results


def _git_sha(short: bool = False) -> str:
    command = ["git", "rev-parse", "--short=12", "HEAD"]
    if not short:
        command = ["git", "rev-parse", "HEAD"]
    try:
        output = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit("failed to read git sha: %s" % (error,))
    return output.stdout.strip()


def _collect_mismatches(
    expected: Any,
    actual: Any,
    path: str,
    mismatches: List[str],
    limit: int,
) -> None:
    if len(mismatches) >= limit:
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                mismatches.append("%s.%s: unexpected key" % (path, key))
            elif key not in actual:
                mismatches.append("%s.%s: missing key" % (path, key))
            else:
                _collect_mismatches(
                    expected[key], actual[key], "%s.%s" % (path, key),
                    mismatches, limit,
                )
            if len(mismatches) >= limit:
                return
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            mismatches.append(
                "%s: length %d != %d" % (path, len(expected), len(actual))
            )
            return
        for index, (left, right) in enumerate(zip(expected, actual)):
            _collect_mismatches(
                left, right, "%s[%d]" % (path, index), mismatches, limit
            )
            if len(mismatches) >= limit:
                return
        return
    if type(expected) is not type(actual) or expected != actual:
        mismatches.append("%s: %r != %r" % (path, expected, actual))


def _write_baseline(payload: Dict[str, Any], output: Path, force: bool) -> None:
    if output.exists() and not force:
        raise SystemExit(
            "baseline already exists at %s; refusing to overwrite "
            "(pass --force only for a deliberate re-baseline)" % (output,)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    leaves = _count_leaves(payload["sections"])
    print("wrote baseline: %s" % (output,))
    print("sections: %d, digest leaves: %d" % (len(payload["sections"]), leaves))


def _count_leaves(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_count_leaves(item) for item in value.values())
    if isinstance(value, list):
        return sum(_count_leaves(item) for item in value)
    return 1


def _resolve_baseline(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    candidates = sorted(ROOT.glob(BASELINE_GLOB))
    if not candidates:
        raise SystemExit(
            "no baseline found under %s; run the script without --check first"
            % (ROOT / "results" / "raw",)
        )
    if len(candidates) > 1:
        raise SystemExit(
            "multiple baselines found (%s); pass --baseline explicitly"
            % ", ".join(str(path) for path in candidates)
        )
    return candidates[0]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare with baseline")
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(_SECTIONS),
        help="restrict to the named sections (development aid)",
    )
    parser.add_argument(
        "--list-sections", action="store_true", help="print section names"
    )
    args = parser.parse_args(argv)

    if args.list_sections:
        for name in _SECTIONS:
            print(name)
        return 0

    section_names = args.only if args.only else list(_SECTIONS)
    if args.check:
        baseline_path = _resolve_baseline(args.baseline)
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline.get("schema") != SCHEMA:
            raise SystemExit(
                "unexpected baseline schema: %r" % (baseline.get("schema"),)
            )
        expected_sections = baseline["sections"]
        if args.only:
            expected_sections = {
                key: expected_sections[key]
                for key in section_names
                if key in expected_sections
            }
        actual = _collect(section_names)
        mismatches: List[str] = []
        for name in section_names:
            if name not in expected_sections:
                mismatches.append("sections.%s: not present in baseline" % name)
                continue
            _collect_mismatches(
                expected_sections[name],
                actual.get(name),
                "sections.%s" % name,
                mismatches,
                limit=50,
            )
        if mismatches:
            print("EQUIVALENCE FAILED (%d mismatch(es))" % len(mismatches))
            for item in mismatches:
                print("  " + item)
            return 1
        print(
            "EQUIVALENCE OK: %d section(s) match %s"
            % (len(section_names), baseline_path.name)
        )
        return 0

    sections = _collect(section_names)
    payload = {
        "schema": SCHEMA,
        "git_sha": _git_sha(),
        "generated_at_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "sections": sections,
    }
    output = args.output
    if output is None:
        output = ROOT / "results" / "raw" / (
            "refactor_baseline_%s.json" % _git_sha(short=True)
        )
    _write_baseline(payload, output, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
