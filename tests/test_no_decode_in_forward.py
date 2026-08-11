"""Guard: the production forward path must not decode or invert.

Two complementary checks:

1. A *runtime sentinel* replaces the debug decode closure with a function that
   raises, then exercises every production entry point. This directly tests the
   property we care about ("forward never decodes") instead of approximating it,
   and it cannot be fooled by renaming a helper.
2. A *static* scan rejects linear-algebra inverses anywhere in the module and in
   the deployed-weight builders.

A purely static scan cannot distinguish "inside the ``return_debug`` branch of
``_run``" from "on the main path" without control-flow analysis, so the runtime
sentinel is the primary guard. An earlier static-only version of this file was
vacuous: it filtered every ``_debug_unmix`` mention away and therefore also
ignored a genuine decode injected into ``forward``.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch

from fastprove.config import ModelConfig, ObfuscationConfig
from fastprove.layers.attention import AttentionMode
from fastprove.models.obfuscated import ObfuscatedTinyCausalLM
from fastprove.models.plain import PlainTinyCausalLM
from fastprove.seed import RequestContext


class _DecodeReached(RuntimeError):
    """Raised by the sentinel when a decode happens on a production path."""


def _converted(debug_enabled: bool = True):
    config = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_sequence_length=32,
    )
    obfuscation = ObfuscationConfig(
        hidden_noise_dim=8,
        value_noise_dim_per_head=2,
        max_condition_number=10.0,
        noise_propagation_gamma=0.5,
        refresh_mode="per_request",
        basis_block_size=8,
    )
    plain = PlainTinyCausalLM(config, seed=5, debug_enabled=debug_enabled).eval()
    converted = ObfuscatedTinyCausalLM.from_plain(
        plain,
        obfuscation=obfuscation,
        mode=AttentionMode.EXACT,
        approximation=None,
        seed=5,
        debug_enabled=debug_enabled,
    )
    return converted.module.eval(), config


def _arm_sentinel(module) -> int:
    """Replace every debug decode closure with a raising sentinel.

    Returns the number of closures armed so the test can assert it actually
    patched something rather than silently patching nothing.
    """

    def sentinel(*_args, **_kwargs):
        raise _DecodeReached("decode reached a production path")

    armed = 0
    targets = [module] + list(module.blocks)
    for target in targets:
        for attribute in ("_debug_basis_unmix", "_basis_unmix"):
            if getattr(target, attribute, None) is not None:
                setattr(target, attribute, sentinel)
                armed += 1
    return armed


def test_sentinel_actually_patches_a_decode_closure() -> None:
    """Meta-test: the sentinel must find something to patch."""

    module, _ = _converted(debug_enabled=True)
    assert _arm_sentinel(module) > 0


def test_forward_does_not_decode() -> None:
    module, config = _converted(debug_enabled=True)
    _arm_sentinel(module)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    context = RequestContext(global_seed=5, request_id="sentinel")
    with torch.no_grad():
        module(input_ids, request_context=context)


def test_cached_forward_does_not_decode() -> None:
    module, config = _converted(debug_enabled=True)
    _arm_sentinel(module)
    input_ids = torch.randint(0, config.vocab_size, (1, 6))
    context = RequestContext(global_seed=5, request_id="sentinel-cache")
    with torch.no_grad():
        logits, cache = module(input_ids, request_context=context, use_cache=True)
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        module(
            token,
            positions=torch.tensor([6]),
            request_context=context,
            cache=cache,
            use_cache=True,
        )


def test_greedy_generation_does_not_decode() -> None:
    module, config = _converted(debug_enabled=True)
    _arm_sentinel(module)
    input_ids = torch.randint(0, config.vocab_size, (1, 6))
    context = RequestContext(global_seed=5, request_id="sentinel-greedy")
    with torch.no_grad():
        module.generate_greedy(
            input_ids, max_new_tokens=4, request_context=context
        )


def test_sentinel_fires_on_the_debug_path() -> None:
    """Reverse control: the debug path *does* decode, so the sentinel must fire.

    Without this, the three tests above could pass simply because the sentinel
    was never reachable at all.
    """

    module, config = _converted(debug_enabled=True)
    _arm_sentinel(module)
    input_ids = torch.randint(0, config.vocab_size, (1, 6))
    context = RequestContext(global_seed=5, request_id="sentinel-debug")
    with pytest.raises(_DecodeReached):
        with torch.no_grad():
            module.forward_debug(input_ids, request_context=context)


def test_debug_apis_are_rejected_when_debug_is_disabled() -> None:
    module, config = _converted(debug_enabled=False)
    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    context = RequestContext(global_seed=5, request_id="no-debug")
    with pytest.raises(PermissionError):
        module.forward_debug(input_ids, request_context=context)


def test_no_linear_algebra_inverse_anywhere_in_the_obfuscated_module() -> None:
    source = pathlib.Path("src/fastprove/models/obfuscated.py").read_text(
        encoding="utf-8"
    )
    forbidden = {"inv", "solve", "pinv", "lstsq"}
    offences = [
        node.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute) and node.attr in forbidden
    ]
    assert offences == [], "runtime inverse in obfuscated.py: %s" % offences


def test_deployed_builders_never_invert_at_runtime() -> None:
    source = pathlib.Path("src/fastprove/layers/deployed.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("build_"):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Attribute) and inner.attr in (
                    "inv",
                    "solve",
                    "pinv",
                    "lstsq",
                ):
                    raise AssertionError(
                        "%s calls a runtime inverse; conversion must use "
                        "dense_inverse() from the basis instead" % node.name
                    )
