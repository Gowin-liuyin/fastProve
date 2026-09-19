"""Prompt corpus loading for pretrained comparison runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

DEFAULT_PROMPTS = (
    "The capital of France is",
    "In mathematics, the derivative of x squared is",
    "def fibonacci(n):",
    "Once upon a time in a small village,",
)

_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"


def load_prompt_texts(
    *,
    prompt_file: Optional[Path],
    sample_count: int,
    seed: int,
    real_scenario: bool,
) -> List[str]:
    """Load or synthesize prompt strings (at least sample_count)."""

    texts: List[str] = []
    if prompt_file is not None:
        path = Path(prompt_file)
        if not path.is_file():
            raise FileNotFoundError("prompt file not found: %s" % path)
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                texts.append(str(obj.get("text") or obj.get("prompt") or obj))
        elif path.suffix == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict):
                        texts.append(str(item.get("text") or item.get("prompt")))
            else:
                raise ValueError("JSON prompt file must be a list")
        else:
            texts = [
                line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
    def _build_real(n: int, s: int) -> List[str]:
        import importlib.util

        helper = _SCRIPTS_DIR / "build_real_scenario_prompts.py"
        spec = importlib.util.spec_from_file_location("real_prompts", helper)
        if spec is None or spec.loader is None:
            raise ImportError("cannot load build_real_scenario_prompts.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return list(mod.build_real_scenario_prompts(n, seed=s))

    if not texts and real_scenario:
        texts = _build_real(sample_count, seed)
    if not texts:
        texts = list(DEFAULT_PROMPTS)
    if len(texts) < sample_count and real_scenario:
        texts = _build_real(sample_count, seed)
    if len(texts) < sample_count:
        # Repeat with index tags rather than silent synthetic token IDs.
        base = list(texts)
        while len(texts) < sample_count:
            texts.append(base[len(texts) % len(base)] + " [dup-%d]" % len(texts))
    return texts[:sample_count]
