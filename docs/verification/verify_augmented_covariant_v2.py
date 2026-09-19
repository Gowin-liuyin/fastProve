"""Small CPU numerical examples for PROOF_PACKAGE.md; not a privacy evaluation.

Uses only NumPy and the standard library. Output files are created exclusively.
The script deliberately has access to all toy matrices: its leakage checks
verify conditional identities, not attacks on an actual exported server bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def run(seed: int) -> list[dict]:
    """Return deterministic toy checks; each record distinguishes its dtype."""
    rng = np.random.default_rng(seed)
    records = []

    def check(name, value, bound, *, dtype="float64", relation="<="):
        value = float(value)
        passed = np.isfinite(value) and (
            value <= bound if relation == "<=" else value == bound
        )
        records.append({"name": name, "value": value, "bound": bound,
                        "relation": relation, "dtype": dtype, "passed": bool(passed)})

    def close(name, actual, expected, tolerance=1e-11, dtype="float64"):
        check(name, np.max(np.abs(actual - expected)), tolerance, dtype=dtype)

    def orth(n):
        return np.linalg.qr(rng.normal(size=(n, n)))[0]

    def basis(n):
        return orth(n) @ np.diag(np.linspace(0.8, 1.2, n)) @ orth(n)

    def softmax(x):
        exp = np.exp(x - np.max(x))
        return exp / exp.sum()

    def silu(x):
        return x / (1 + np.exp(-x))

    d, expanded, r, count = 4, 8, 2, 128
    n = expanded + r
    h, e = rng.normal(size=(count, d)), rng.normal(size=(count, r))
    phat = rng.normal(size=(d, expanded))
    qhat = np.linalg.pinv(phat)
    m = basis(n)
    inv = np.linalg.solve(m, np.eye(n))
    read = inv @ np.vstack((qhat, np.zeros((r, d))))
    noise_read = inv @ np.vstack((np.zeros((expanded, r)), np.eye(r)))
    c = np.concatenate((h @ phat, e), axis=1) @ m
    close("P1_expanded_concat_decode", c @ read, h)
    close("P1_expanded_concat_noise_read", c @ noise_read, e)
    check("P1_expanded_concat_cov_rank", np.linalg.matrix_rank(np.cov(c, rowvar=False)), d + r,
          relation="==")
    check("P1_expanded_concat_nullity", n - np.linalg.matrix_rank(np.cov(c, rowvar=False)),
          expanded - d, relation="==")
    fixed = np.concatenate((np.repeat(h[:1] @ phat, count, axis=0), e), axis=1) @ m
    check("P1_fixed_signal_cov_rank", np.linalg.matrix_rank(np.cov(fixed, rowvar=False),
                                                           tol=1e-10), r, relation="==")

    # Complement filling is an explicit comparison, not the primary requested route.
    _, _, vt = np.linalg.svd(qhat.T, full_matrices=True)
    complement = vt[d:]
    t = np.concatenate((phat, complement), axis=0)
    ma = orth(expanded)
    e_fill = rng.normal(size=(count, expanded - d))
    filled = (h @ phat + e_fill @ complement) @ ma
    close("P2_complement_annihilation", complement @ qhat, np.zeros((expanded - d, d)))
    close("P2_legacy_decode", filled @ ma.T @ qhat, h)
    close("P2_compact_equivalence", filled, np.concatenate((h, e_fill), axis=1) @ t @ ma)
    check("P2_filled_cov_rank", np.linalg.matrix_rank(np.cov(filled, rowvar=False)),
          expanded, relation="==")
    effective = np.block([[t, np.zeros((expanded, r))],
                          [np.zeros((r, expanded)), np.eye(r)]]) @ m
    full = np.concatenate((h @ phat + e_fill @ complement, e), axis=1) @ m
    close("P2_expanded_full_equivalence", full,
          np.concatenate((h, e_fill, e), axis=1) @ effective)
    close("P2_expanded_full_decode", full @ read, h)
    check("P2_expanded_full_cov_rank", np.linalg.matrix_rank(np.cov(full, rowvar=False)),
          n, relation="==")

    dout, expanded_out, rout = 3, 6, 2
    pout = rng.normal(size=(dout, expanded_out))
    qout = np.linalg.pinv(pout)
    mo = basis(expanded_out + rout)
    w, coupling = rng.normal(size=(d, dout)), rng.normal(size=(d, rout)) * 0.1
    g, bias = 0.5 * np.eye(r), rng.normal(size=(dout,))
    xi = rng.uniform(-0.2, 0.2, size=(count, rout))
    k = np.block([[qhat @ w @ pout, qhat @ coupling],
                  [np.zeros((r, expanded_out)), g]])
    deployed = np.linalg.solve(m, k) @ mo
    offset = np.r_[bias @ pout, np.zeros(rout)] @ mo
    expected = np.concatenate(((h @ w + bias) @ pout, h @ coupling + e @ g + xi), axis=1) @ mo
    close("P3_expanded_rectangular_affine_refresh",
          c @ deployed + offset + xi @ mo[expanded_out:], expected)
    # Actual FP32 arithmetic, with offline FP64 conversion cast to saved FP32 arrays.
    hf, ef, mf = h.astype("float32"), e.astype("float32"), m.astype("float32")
    cf = np.concatenate((hf @ phat.astype("float32"), ef), axis=1) @ mf
    out32 = cf @ deployed.astype("float32") + offset.astype("float32")
    out32 += xi.astype("float32") @ mo[expanded_out:].astype("float32")
    read_out = np.linalg.solve(mo, np.eye(expanded_out + rout)) @ np.vstack(
        (qout, np.zeros((rout, dout))))
    decoded32 = out32 @ read_out.astype("float32")
    plain32 = hf @ w.astype("float32") + bias.astype("float32")
    close("P3_actual_FP32_signal_gate", decoded32, plain32, 1e-5, "float32")

    gamma, steps, hmax, cmax, xmax = 0.5, 12, 1.0, 0.2, 0.1
    en = np.array([0.3, -0.4])
    initial = np.linalg.norm(en)
    for _ in range(steps):
        forcing = rng.normal(size=r)
        forcing *= hmax * cmax / np.linalg.norm(forcing)
        refresh = rng.normal(size=r)
        refresh *= xmax / np.linalg.norm(refresh)
        en = gamma * en[::-1] + forcing + refresh
    bound = gamma ** steps * initial + (hmax * cmax + xmax) * (1 - gamma ** steps) / (1 - gamma)
    check("P4_deterministic_noise_bound", np.linalg.norm(en), bound)
    pe = np.array([[0., -1.], [1., 0.]])
    covariance = (gamma * pe).T @ np.eye(r) @ (gamma * pe) + (1 - gamma ** 2) * np.eye(r)
    close("P4_covariance_identity", covariance, np.eye(r))

    mn = basis(n)
    update = np.tanh(h)
    cnoise = rng.normal(size=(d, r)) * 0.1
    residual_increment = np.concatenate((update @ phat,
                                         h @ cnoise + e @ (g - np.eye(r)) + xi), axis=1) @ mn
    reencode = np.block([[qhat @ phat, np.zeros((expanded, r))],
                         [np.zeros((r, expanded)), np.eye(r)]])
    correct_residual = c @ np.linalg.solve(m, reencode) @ mn + residual_increment
    target_residual = np.concatenate(((h + update) @ phat,
                                      h @ cnoise + e @ g + xi), axis=1) @ mn
    close("P5_expanded_cross_basis_residual", correct_residual, target_residual)
    check("P5_missing_basis_change_detected", float(np.max(np.abs(c + residual_increment - target_residual)) > 0.1),
          1, relation="==")

    rho = np.sqrt(np.mean(h * h, axis=1, keepdims=True) + 1e-5)
    rot, learned = orth(d), np.diag(rng.uniform(0.5, 1.5, size=d))
    rotated = h @ rot
    norm_projected = rotated / np.sqrt(np.mean(rotated ** 2, axis=1, keepdims=True) + 1e-5)
    close("P6_RMS_gamma_bias", norm_projected @ rot.T @ learned @ w + bias,
          h / rho @ learned @ w + bias)
    wrong = c / np.sqrt(np.mean(c ** 2, axis=1, keepdims=True) + 1e-5) @ read
    check("P6_whole_state_RMS_counterexample", float(np.max(np.abs(wrong - h / rho)) > 0.01),
          1, relation="==")
    delta = 1e-4
    numerator = h @ learned @ w
    error = np.linalg.norm(numerator / (rho + delta) - numerator / rho, axis=1)
    rho_bound = np.linalg.norm(numerator, axis=1) * delta / (rho[:, 0] * (rho[:, 0] - delta))
    check("P6_rho_perturbation_bound_excess", np.max(error - rho_bound), 1e-12)

    # GQA: 4 query heads share 2 KV heads, with one common post-RoPE transform per KV group.
    seq, heads, kv_heads, hd = 5, 4, 2, 4
    query = rng.normal(size=(seq, heads, hd))
    key = rng.normal(size=(seq, kv_heads, hd))
    value = rng.normal(size=(seq, kv_heads, hd))
    for position in range(seq):
        angle = position * 0.3
        rr = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rope = np.kron(np.eye(hd // 2), rr)
        query[position] = query[position] @ rope
        key[position] = key[position] @ rope
    rotations = [orth(hd) for _ in range(kv_heads)]
    value_bases = [basis(hd + 2) for _ in range(kv_heads)]
    value_noise = rng.normal(size=(seq, kv_heads, 2))
    score_error, value_error = 0., 0.
    for head in range(heads):
        kv = head // 2
        scores = query[:, head] @ key[:, kv].T / np.sqrt(hd)
        mixed_scores = (query[:, head] @ rotations[kv]) @ (key[:, kv] @ rotations[kv]).T / np.sqrt(hd)
        score_error = max(score_error, float(np.max(np.abs(scores - mixed_scores))))
        mv = value_bases[kv]
        cv = np.concatenate((value[:, kv], value_noise[:, kv]), axis=1) @ mv
        lv = np.linalg.solve(mv, np.eye(hd + 2))[:, :hd]
        for pos in range(seq):
            probs = softmax(scores[pos, :pos + 1])
            mixed_probs = softmax(mixed_scores[pos, :pos + 1])
            value_error = max(value_error, float(np.max(np.abs(
                (mixed_probs @ cv[:pos + 1]) @ lv - probs @ value[:pos + 1, kv]))))
    check("P7_RoPE_GQA_score_identity", score_error, 1e-11)
    check("P7_causal_mixed_value_identity", value_error, 1e-11)

    width = 7
    gate, up = rng.normal(size=(9, width)), rng.normal(size=(9, width))
    perm = np.eye(width)[:, rng.permutation(width)]
    scale = np.diag(rng.uniform(0.5, 1.5, size=width))
    down = rng.normal(size=(width, d))
    close("P8_SwiGLU_identity", (silu(gate @ perm) * (up @ scale @ perm)) @ perm.T @ np.linalg.solve(scale, down),
          (silu(gate) * up) @ down)
    route = np.array([2., 2., 1., 0.])
    physical = np.array([2, 1, 3, 0])  # physical index -> original expert id/priority
    original_choice = np.lexsort((np.arange(4), -route))[:2]
    physical_choice = np.lexsort((physical, -route[physical]))[:2]
    check("P8_router_tie_priority", float(np.array_equal(physical[physical_choice], original_choice)),
          1, relation="==")

    max_l1_excess, max_kl_excess, max_output_excess = 0., 0., 0.
    topk_failures, mask_failures, cases = 0, 0, 0
    for size in (1, 2, 5, 11):
        for requested in (0., 0.001, 0.03, 0.1, 1., 3.):
            for _ in range(4):
                s = rng.normal(size=size)
                raw = rng.uniform(-1., 1., size=size)
                raw -= raw.mean()
                noise = raw * (requested / np.max(np.abs(raw))) if np.max(np.abs(raw)) else raw
                noise = np.clip(noise, -requested, requested)
                p, q = softmax(s), softmax(s + noise)
                l1 = np.abs(q - p).sum()
                kl = max(float(p @ np.log(p / q)), float(q @ np.log(q / p)))
                vv = rng.normal(size=(size, 3))
                diameter = np.max(np.linalg.norm(vv[:, None] - vv[None, :], axis=-1))
                max_l1_excess = max(max_l1_excess, float(l1 - min(2., requested)))
                max_kl_excess = max(max_kl_excess, kl - requested ** 2 / 2)
                max_output_excess = max(max_output_excess, float(np.linalg.norm((q - p) @ vv)
                                                                - diameter * min(2., requested) / 2))
                masked = np.r_[s, -np.inf]
                masked[:-1] += noise
                mask_failures += int(not np.isneginf(masked[-1]))
                if size > 2:
                    order = np.argsort(-s, kind="stable")
                    margin = s[order[1]] - s[order[2]]
                    tau = min(requested, 0.8 * margin / 2)
                    topnoise = noise * (tau / requested) if requested else noise
                    topk_failures += int(set(np.argsort(-(s + topnoise))[:2]) != set(order[:2]))
                cases += 1
    check("P10_topk_failures", topk_failures, 0, relation="==")
    check("P10_mask_failures", mask_failures, 0, relation="==")
    check("P11_L1_bound_excess", max_l1_excess, 1e-12)
    check("P11_bidir_KL_bound_excess", max_kl_excess, 1e-12)
    check("P11_value_bound_excess", max_output_excess, 1e-12)
    check("P11_examples_checked", cases, 96, relation="==")

    public = rng.normal(size=(d, d + 3))
    dq = read @ public
    recovered = dq @ np.linalg.pinv(public)
    close("P13_public_projection_recovery", c @ recovered, h)
    noise_map = m[expanded:]
    signal_map = np.hstack((phat, np.zeros((d, r)))) @ m
    _, _, vt = np.linalg.svd(noise_map, full_matrices=True)
    null = vt[r:].T
    close("P13_noise_elimination", c @ null, h @ signal_map @ null)
    check("P13_remaining_signal_rank", np.linalg.matrix_rank(signal_map @ null), d,
          relation="==")
    close("P13_Gram_identity", c @ (read @ read.T) @ c.T, h @ h.T)
    full_aux_map = effective[d:]
    _, _, full_vt = np.linalg.svd(full_aux_map, full_matrices=True)
    full_null = full_vt[expanded - d + r:].T
    close("P13_full_auxiliary_elimination", full @ full_null,
          h @ effective[:d] @ full_null)
    check("P13_full_remaining_signal_rank",
          np.linalg.matrix_rank(effective[:d] @ full_null), d, relation="==")

    epsilon = 0.1  # well-resolved toy example; no finite-precision token recovery claim
    a = np.mean(h ** 2, axis=1)
    u = h / np.sqrt(a[:, None] + epsilon)
    qnorm = np.mean(u ** 2, axis=1)
    close("P14_RMS_epsilon_inversion", epsilon * qnorm / (1 - qnorm), a)
    noeps = h / np.sqrt(a[:, None])
    scaled = 2 * h / np.sqrt(np.mean((2 * h) ** 2, axis=1, keepdims=True))
    close("P14_epsilon_zero_normalized_collision", noeps, scaled)
    check("P14_residual_diff_despite_norm_collision",
          float(np.max(np.abs((h + noeps) - (2 * h + scaled))) > 0.1), 1, relation="==")
    return records


def main() -> int:
    """Run examples and optionally write a new immutable raw record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output exists; omit --output to replay without overwriting")
    records = run(args.seed)
    payload = {
        "schema": "fastprove.expanded_augmentation_v2.math_examples.v3",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed, "python": sys.version, "executable": sys.executable,
        "numpy": np.__version__, "platform": platform.platform(), "device": "cpu",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "meaningful_lm_evidence": False, "actual_bundle_attack": False,
        "scope": "expanded-concat and expanded-full numerical examples; not runtime or security acceptance",
        "status": "passed" if all(x["passed"] for x in records) else "failed",
        "checks": records,
    }
    output = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(output)
    print(output, end="")
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
