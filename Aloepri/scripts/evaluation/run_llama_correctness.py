from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_MAX_NEW_TOKENS, DEFAULT_SEED
from src.aloepri.evaluation import run_llama_correctness


DEFAULT_LLAMA_INSTRUCT_MODEL_DIR = "model/Llama-3.2-3B-Instruct"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Llama release-bundle correctness.")
    parser.add_argument("--bundle-dir", default="artifacts/llama_3_2_3b_instruct/release_bundle")
    parser.add_argument("--output-dir", default="outputs/evaluation/llama_3_2_3b_instruct/correctness")
    parser.add_argument("--profiles", nargs="+", default=["default", "reference"])
    parser.add_argument("--baseline-model-dir", default=DEFAULT_LLAMA_INSTRUCT_MODEL_DIR)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--correctness-atol", type=float, default=1e-4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_llama_correctness(
        bundle_dir=args.bundle_dir,
        output_dir=args.output_dir,
        profiles=tuple(args.profiles),
        baseline_model_dir=args.baseline_model_dir,
        dtype=args.dtype,
        device=args.device,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        correctness_atol=args.correctness_atol,
    )
    print(f"Saved Llama correctness evidence under {Path(args.output_dir)}")
    print(f"completion_status: {summary.get('completion_status')}")


if __name__ == "__main__":
    main()
