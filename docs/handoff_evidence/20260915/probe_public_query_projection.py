"""Tiny exported-bundle diagnostic; not the new two-sided COA benchmark.

Run from repo root with PYTHONPATH=src. The attack child receives public base
weights and actual exported server tensors only. This is input separation,
not an OS sandbox. Client material is used only by the parent for scoring.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import torch


def attack(input_path: Path, output_path: Path) -> None:
    """Solve D=P B using only public B and server-visible D and c."""
    inputs = torch.load(input_path, weights_only=True)
    assert set(inputs) == {"public_q_math", "public_gamma", "public_embedding",
                           "server_deployed_q", "server_embedding_table"}
    b = inputs["public_gamma"].double()[:, None] * inputs["public_q_math"].double()
    deployed = inputs["server_deployed_q"].double()
    projection = torch.linalg.solve(b.T, deployed.T).T
    recovered = inputs["server_embedding_table"].double() @ projection
    distances = torch.cdist(recovered, inputs["public_embedding"].double())
    predictions = distances.argmin(dim=-1)
    torch.save({"recovered": recovered, "predictions": predictions,
                "public_projection_condition": float(torch.linalg.cond(b)),
                "public_projection_rank": int(torch.linalg.matrix_rank(b)),
                "fit_residual_max": float((projection @ b - deployed).abs().max())},
               output_path)


def main() -> None:
    """Export a tiny existing structured model, run the attack, and score."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack-input", type=Path)
    parser.add_argument("--attack-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.attack_input is not None:
        if args.attack_output is None:
            parser.error("--attack-output is required with --attack-input")
        attack(args.attack_input, args.attack_output)
        return
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists():
        raise FileExistsError("preserve old records; choose a fresh --output")

    from fastprove.config import load_config
    from fastprove.converter import ModelConverter, load_client, load_server_model
    from fastprove.models.plain import PlainTinyCausalLM

    config = load_config(Path("configs/tiny_exact.yaml"))
    config = replace(config, obfuscation=replace(config.obfuscation, refresh_mode="per_request"),
                     runtime=replace(config.runtime, debug_enabled=False,
                                             device="cpu", activation_dtype="float32"))
    plain = PlainTinyCausalLM(config.model, seed=1, debug_enabled=False).eval()
    with tempfile.TemporaryDirectory(prefix="fastprove-handoff-probe-") as directory:
        root = Path(directory)
        ModelConverter().convert(plain, config, root / "client", root / "server")
        server = load_server_model(root / "server").eval()
        public_input = {
            "public_q_math": plain.blocks[0].q_proj.weight.detach().T.contiguous(),
            "public_gamma": plain.blocks[0].attention_norm_weight.detach(),
            "public_embedding": plain.embedding.weight.detach(),
            "server_deployed_q": server.blocks[0].deployed_q.detach(),
            "server_embedding_table": server.embedding.table.detach(),
        }
        torch.save(public_input, root / "attack_input.pt")
        subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        "--attack-input", str(root / "attack_input.pt"),
                        "--attack-output", str(root / "prediction.pt")], check=True)
        prediction = torch.load(root / "prediction.pt", weights_only=True)
        client = load_client(root / "client")
        truth = client.decode(torch.arange(config.model.vocab_size, dtype=torch.int64))
        expected = plain.embedding.weight.detach().double()[truth]
        recovered = prediction["recovered"]
        error = recovered - expected
        record = {
            "schema": "fastprove.handoff.public_query_projection.v1",
            "scope": "existing structured tiny deployment, NOT new two-sided basis",
            "observation_view": "cloud_coa public base weights plus server table and deployed_q",
            "attack_input_fields": sorted(public_input),
            "attack_uses_target_pairs_or_secret_basis": False,
            "isolation": "separate child process and explicit tensor input; no OS sandbox",
            "device": "cpu", "deployed_tensor_dtype": "float32",
            "attack_arithmetic_dtype": "float64", "model": "random PlainTinyCausalLM",
            "model_seed": 1, "conversion_seed": config.runtime.seed,
            "config_path": "configs/tiny_exact.yaml",
            "config_overrides": {"refresh_mode": "per_request", "debug_enabled": False,
                                 "device": "cpu", "activation_dtype": "float32"},
            "config_sha256": hashlib.sha256(Path("configs/tiny_exact.yaml").read_bytes()).hexdigest(),
            "hidden_size": config.model.hidden_size, "noise_dim": config.obfuscation.hidden_noise_dim,
            "vocabulary_size": config.model.vocab_size, "sample_count": len(truth),
            "observation_unit": "all rows of server-visible embedding table",
            "public_projection_rank": prediction["public_projection_rank"],
            "public_projection_condition": prediction["public_projection_condition"],
            "fit_residual_max": prediction["fit_residual_max"],
            "signal_max_abs_error": float(error.abs().max()),
            "signal_relative_l2": float(error.norm() / expected.norm()),
            "token_top1_fraction": float((prediction["predictions"] == truth).double().mean()),
            "torch_version": torch.__version__, "python": sys.version,
            "platform": platform.platform(),
            "limitation": "one synthetic key/model; no pretrained, transcript-only, or new two-sided attack claim",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
