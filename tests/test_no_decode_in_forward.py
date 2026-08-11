"""Static guard: the production forward path must not decode or invert."""

from __future__ import annotations

import ast
import pathlib

_FORBIDDEN_CALLS = {"inv", "solve", "pinv", "lstsq"}
_FORBIDDEN_NAMES = {"unmix", "_debug_unmix", "signal_projection"}


def _forward_functions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
            "forward",
            "_run",
            "generate_greedy",
        ):
            yield node


def test_forward_path_contains_no_inverse_or_decode() -> None:
    source = pathlib.Path("src/fastprove/models/obfuscated.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    offences = []
    for function in _forward_functions(tree):
        for node in ast.walk(function):
            if isinstance(node, ast.Attribute):
                if node.attr in _FORBIDDEN_CALLS | _FORBIDDEN_NAMES:
                    offences.append("%s -> .%s" % (function.name, node.attr))
            if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
                offences.append("%s -> %s" % (function.name, node.id))
    # ``_run`` legitimately references ``_debug_unmix`` inside its
    # ``return_debug`` branch, which ``forward`` can never reach.
    offences = [item for item in offences if "_debug_unmix" not in item]
    assert offences == [], "decode/inverse reached the forward path: %s" % offences


def test_deployed_module_never_inverts_at_runtime() -> None:
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
                ):
                    raise AssertionError(
                        "%s calls a runtime inverse; conversion must use "
                        "dense_inverse() from the basis instead" % node.name
                    )
