"""Numerical verification for the noise-ordering derivation (2026-09-19).

Question under test
-------------------
Given the augmented covariant construction ``c = [h P_hat, e] M``, is it better to
(A) inject LWE-style augmented-matrix noise *before* the covariant mixing, or
(B) lift covariantly first and then add a dynamic noise *outside* (on ``c``)?

All checks run in FP64 on small random instances. Nothing here is a
cryptographic proof; these are algebraic/statistical facts about the
construction, recorded so the written derivation is not asserted from memory.

Checks
------
T1  Order-equivalence (conjugation): [x,e]M + n == [x+dx, e+de]M with d = n M^-1.
T2  Semantic error of outside noise is exactly n L.
T3  Ker(L) is the exact-correctness noise cone; dim = N - d; and it EQUALS the
    expanded_full noise space span{[a B_c, e] M}. (Saturation.)
T4  Lossless recovery of h under known M for arbitrarily large in-Ker noise.
T5  Known-plaintext closed-form decoder recovery for expanded_concat and
    expanded_full (how many samples, and residual error).
T6  Outside noise over R: (a) secret A recovered at rate T^-1/2;
    (b) per-sample GLS decode error floor -> 0 for in-Ker noise, > 0 for
    outside noise, and proportional to the injected semantic noise.
T7  Nonlinear checkpoint is not affine-equivariant: no h-independent matrix Phi
    satisfies RMSNorm(h+delta) = RMSNorm(h) + delta Phi.
T8  Amplification: in-Ker noise leaves the semantic path exact through a
    deployed chain; semantic noise is amplified geometrically.
T9  SNR barrier arithmetic: bits per coordinate retained vs noise/signal ratio.

Usage
-----
    python docs/verification/verify_noise_ordering.py \
        --output results/raw/noise_ordering_analysis_20260919.json
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from typing import Any

import numpy as np

RNG_SEED = 20260919


# ---------------------------------------------------------------------------
# construction helpers (math layout: row vectors, Y = X W)
# ---------------------------------------------------------------------------


def orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-ish orthogonal matrix via QR of a Gaussian."""
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))[None, :]


def mixing_matrix(n: int, cond: float, rng: np.random.Generator) -> np.ndarray:
    """M = Q_L diag(s) Q_R with singular values in [1, cond]."""
    q_l = orthogonal(n, rng)
    q_r = orthogonal(n, rng)
    s = np.exp(np.linspace(0.0, np.log(cond), n))
    return q_l @ np.diag(s) @ q_r


@dataclass
class Scheme:
    """One instance of the augmented covariant construction."""

    d: int
    big_d: int
    r: int
    p_hat: np.ndarray  # d x D   AloePri lifting
    q_hat: np.ndarray  # D x d   right inverse, p_hat @ q_hat = I_d
    b_comp: np.ndarray  # (D-d) x D  complement-noise basis, b_comp @ q_hat = 0
    m: np.ndarray  # N x N  joint mixing
    m_inv: np.ndarray

    @property
    def n(self) -> int:
        return self.big_d + self.r

    @property
    def l_read(self) -> np.ndarray:
        """Semantic read matrix L = M^-1 [Q_hat; 0], shape N x d, with c L = h."""
        top = self.q_hat
        bottom = np.zeros((self.r, self.d))
        return self.m_inv @ np.vstack([top, bottom])

    @property
    def j_read(self) -> np.ndarray:
        """Noise read matrix J = M^-1 [0; I_r], shape N x r, with c J = e."""
        top = np.zeros((self.big_d, self.r))
        return self.m_inv @ np.vstack([top, np.eye(self.r)])

    def encode(
        self,
        h: np.ndarray,
        e: np.ndarray,
        a: np.ndarray | None = None,
    ) -> np.ndarray:
        """c = [h P_hat + a B_comp, e] M. a=None means expanded_concat."""
        x = h @ self.p_hat
        if a is not None:
            x = x + a @ self.b_comp
        return np.hstack([x, e]) @ self.m

    def decode(self, c: np.ndarray) -> np.ndarray:
        return c @ self.l_read


def make_scheme(d: int, big_d: int, r: int, cond: float, rng: np.random.Generator) -> Scheme:
    # P_hat as the first d rows of an orthogonal D x D matrix -> well conditioned,
    # Q_hat = P_hat^T exactly, and B_comp = remaining D-d rows (B_comp Q_hat = 0).
    basis = orthogonal(big_d, rng)
    p_hat = basis[:d, :]
    q_hat = p_hat.T
    b_comp = basis[d:, :]
    m = mixing_matrix(big_d + r, cond, rng)
    return Scheme(d, big_d, r, p_hat, q_hat, b_comp, m, np.linalg.inv(m))


def rel(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 error ||a-b|| / max(||b||, tiny)."""
    den = max(float(np.linalg.norm(b)), 1e-300)
    return float(np.linalg.norm(a - b) / den)


def principal_angle_max(u: np.ndarray, v: np.ndarray) -> float:
    """Largest principal angle (radians) between the column spaces of u and v."""
    qu, _ = np.linalg.qr(u)
    qv, _ = np.linalg.qr(v)
    s = np.linalg.svd(qu.T @ qv, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return float(np.arccos(s.min()))


def left_null_space(mat: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Basis (as rows) of {n : n @ mat == 0}."""
    u, s, _ = np.linalg.svd(mat, full_matrices=True)
    rank = int((s > tol * max(1.0, s.max())).sum())
    return u[:, rank:].T


def right_null_space(mat: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Basis (as columns) of {v : mat @ v == 0}."""
    _, s, vt = np.linalg.svd(mat, full_matrices=True)
    rank = int((s > tol * max(1.0, s.max())).sum())
    return vt[rank:, :].T


# ---------------------------------------------------------------------------
# T1 / T2 / T3 / T4
# ---------------------------------------------------------------------------


def t1_order_equivalence(sch: Scheme, rng: np.random.Generator) -> dict[str, Any]:
    h = rng.standard_normal(sch.d)
    e = rng.standard_normal(sch.r)
    n = rng.standard_normal(sch.n) * 3.0

    c_outside = sch.encode(h, e) + n  # mix first, then add noise outside
    delta = n @ sch.m_inv
    dx, de = delta[: sch.big_d], delta[sch.big_d :]
    c_inside = np.hstack([h @ sch.p_hat + dx, e + de]) @ sch.m  # noise first, then mix

    return {
        "claim": "mix-then-add-outside equals add-inside-then-mix with delta = n M^-1",
        "rel_error": rel(c_outside, c_inside),
    }


def t2_semantic_error_is_nL(sch: Scheme, rng: np.random.Generator) -> dict[str, Any]:
    h = rng.standard_normal(sch.d)
    e = rng.standard_normal(sch.r)
    n = rng.standard_normal(sch.n) * 0.7
    c = sch.encode(h, e)

    observed = sch.decode(c + n) - h
    predicted = n @ sch.l_read
    return {
        "claim": "Dec(c + n) - h == n L exactly",
        "rel_error": rel(observed, predicted),
        "semantic_error_norm": float(np.linalg.norm(observed)),
        "outside_noise_norm": float(np.linalg.norm(n)),
        "leverage_semantic_over_injected": float(
            np.linalg.norm(observed) / np.linalg.norm(n)
        ),
    }


def t3_kernel_saturation(sch: Scheme) -> dict[str, Any]:
    ker = left_null_space(sch.l_read)  # rows spanning {n : n L = 0}
    dim_ker = ker.shape[0]

    # expanded_full noise space: {[a B_comp, e] M}
    rows = []
    for i in range(sch.big_d - sch.d):
        a = np.zeros(sch.big_d - sch.d)
        a[i] = 1.0
        rows.append(np.hstack([a @ sch.b_comp, np.zeros(sch.r)]) @ sch.m)
    for i in range(sch.r):
        e = np.zeros(sch.r)
        e[i] = 1.0
        rows.append(np.hstack([np.zeros(sch.big_d), e]) @ sch.m)
    full_space = np.array(rows)

    angle = principal_angle_max(ker.T, full_space.T)
    return {
        "claim": (
            "dim Ker(L) = N - d = (D-d) + r, and Ker(L) equals the expanded_full "
            "noise space span{[a B_comp, e] M}; the free channel is saturated"
        ),
        "N": sch.n,
        "d": sch.d,
        "D": sch.big_d,
        "r": sch.r,
        "dim_ker_L": dim_ker,
        "expected_dim": sch.n - sch.d,
        "dim_expanded_full_noise_space": int(np.linalg.matrix_rank(full_space)),
        "max_principal_angle_rad": angle,
        "note_on_tolerance": (
            "arccos near 1 is ill-conditioned: angle 3e-8 rad corresponds to "
            "1 - cos = 5e-16, i.e. FP64 round-off"
        ),
        "subspaces_identical": bool(angle < 1e-6),
        "max_semantic_leak_from_ker_basis": float(np.abs(ker @ sch.l_read).max()),
        "max_semantic_leak_from_expanded_full_basis": float(
            np.abs(full_space @ sch.l_read).max()
        ),
    }


def t4_lossless_under_huge_inker_noise(sch: Scheme, rng: np.random.Generator) -> dict[str, Any]:
    h = rng.standard_normal(sch.d)
    out = {}
    for scale in (1e0, 1e6, 1e12):
        e = rng.standard_normal(sch.r) * scale
        a = rng.standard_normal(sch.big_d - sch.d) * scale
        c = sch.encode(h, e, a)
        out[f"noise_scale_{scale:.0e}"] = rel(sch.decode(c), h)
    return {
        "claim": (
            "in-Ker(L) noise of any magnitude adds zero uncertainty about h once M "
            "is known; residual is float round-off only"
        ),
        "rel_error_by_noise_scale": out,
    }


# ---------------------------------------------------------------------------
# T5 known-plaintext closed-form decoder recovery
# ---------------------------------------------------------------------------


def recover_decoder_kpa(
    h_mat: np.ndarray, c_mat: np.ndarray, nuisance_rank: int
) -> np.ndarray | None:
    """Closed-form decoder recovery from known (h, c) pairs.

    c_t = h_t A + nu_t B with rank(B) = nuisance_rank. Project out col(H),
    take the right null space (= ker B), then least-squares for A V.
    Returns L_hat with c L_hat = h, or None if the sample count is insufficient.
    """
    t_samples, d = h_mat.shape
    if t_samples <= d + nuisance_rank:
        return None
    q_h, _ = np.linalg.qr(h_mat)
    resid = c_mat - q_h @ (q_h.T @ c_mat)  # P_H^perp C
    v = right_null_space(resid, tol=1e-9)  # N x dim(ker B)
    if v.shape[1] < d:
        return None
    av, *_ = np.linalg.lstsq(h_mat, c_mat @ v, rcond=None)  # d x dim
    return v @ np.linalg.pinv(av)


def t5_known_plaintext_break(
    d: int, big_d: int, r: int, cond: float, rng: np.random.Generator
) -> dict[str, Any]:
    results = {}
    for variant in ("expanded_concat", "expanded_full"):
        sch = make_scheme(d, big_d, r, cond, rng)
        nuisance_rank = r if variant == "expanded_concat" else r + (big_d - d)
        per_variant = {
            "theoretical_min_samples": d + nuisance_rank + 1,
            "nuisance_rank": nuisance_rank,
            "trials": {},
        }
        for t_samples in (
            d + nuisance_rank - 1,
            d + nuisance_rank + 1,
            d + nuisance_rank + 8,
            4 * (d + nuisance_rank),
        ):
            h_mat = rng.standard_normal((t_samples, d))
            c_rows = []
            for i in range(t_samples):
                e = rng.standard_normal(r)
                a = (
                    rng.standard_normal(big_d - d)
                    if variant == "expanded_full"
                    else None
                )
                c_rows.append(sch.encode(h_mat[i], e, a))
            c_mat = np.array(c_rows)

            l_hat = recover_decoder_kpa(h_mat, c_mat, nuisance_rank)
            if l_hat is None:
                per_variant["trials"][f"T={t_samples}"] = {
                    "recovered": False,
                    "reason": "insufficient samples for rank condition",
                }
                continue

            # held-out targets with fresh noise
            h_test = rng.standard_normal((64, d))
            c_test = np.array(
                [
                    sch.encode(
                        h_test[i],
                        rng.standard_normal(r),
                        rng.standard_normal(big_d - d)
                        if variant == "expanded_full"
                        else None,
                    )
                    for i in range(64)
                ]
            )
            per_variant["trials"][f"T={t_samples}"] = {
                "recovered": True,
                "heldout_rel_error": rel(c_test @ l_hat, h_test),
                "decoder_rel_error_vs_true_L": rel(l_hat, sch.l_read),
            }
        results[variant] = per_variant
    return {
        "claim": (
            "with known/chosen plaintext, the exact semantic decoder is recovered in "
            "closed form from O(d + nuisance_rank) samples; complement noise raises "
            "the sample count but not the asymptotic cost"
        ),
        "cost_model": "O(T N^2) for QR + SVD; no search, no training",
        "results": results,
    }


# ---------------------------------------------------------------------------
# T6 outside noise: key recovery rate vs per-sample error floor
# ---------------------------------------------------------------------------


def t6_outside_noise_statistics(
    d: int, big_d: int, r: int, cond: float, rng: np.random.Generator
) -> dict[str, Any]:
    sch = make_scheme(d, big_d, r, cond, rng)
    a_true = sch.p_hat @ sch.m[: sch.big_d, :]  # d x N, so that c = h A + nuisance
    sigma_n = 0.05

    def sample(t_samples: int, outside: bool) -> tuple[np.ndarray, np.ndarray]:
        h_mat = rng.standard_normal((t_samples, d))
        c_rows = []
        for i in range(t_samples):
            c = sch.encode(h_mat[i], rng.standard_normal(r))
            if outside:
                c = c + rng.standard_normal(sch.n) * sigma_n
            c_rows.append(c)
        return h_mat, np.array(c_rows)

    # (a) secret recovery error vs T, with outside noise present
    key_recovery = {}
    for t_samples in (64, 256, 1024, 4096, 16384):
        h_mat, c_mat = sample(t_samples, outside=True)
        a_hat, *_ = np.linalg.lstsq(h_mat, c_mat, rcond=None)
        key_recovery[f"T={t_samples}"] = {
            "rel_error_A_hat": rel(a_hat, a_true),
            "times_sqrt_T": rel(a_hat, a_true) * np.sqrt(t_samples),
        }

    # (b) per-sample decode error floor with the true M known
    decode_floor = {}
    for outside in (False, True):
        h_mat, c_mat = sample(2048, outside=outside)
        err = rel(c_mat @ sch.l_read, h_mat)
        decode_floor["outside_noise" if outside else "in_ker_noise_only"] = {
            "rel_decode_error": err,
        }

    # (c) decode error floor scales linearly in sigma_n
    linearity = {}
    for s in (1e-4, 1e-3, 1e-2, 1e-1):
        h_mat = rng.standard_normal((512, d))
        c_mat = np.array(
            [
                sch.encode(h_mat[i], rng.standard_normal(r))
                + rng.standard_normal(sch.n) * s
                for i in range(512)
            ]
        )
        err = rel(c_mat @ sch.l_read, h_mat)
        linearity[f"sigma_n={s:.0e}"] = {
            "rel_decode_error": err,
            "error_over_sigma": err / s,
        }

    return {
        "claim": (
            "outside noise does NOT hide the secret (A recovered at rate T^-1/2 with "
            "no modular reduction), but it does create a per-sample error floor that "
            "is exactly linear in the injected noise"
        ),
        "sigma_n_for_key_recovery": sigma_n,
        "key_recovery_vs_T": key_recovery,
        "per_sample_decode_floor": decode_floor,
        "floor_linear_in_sigma": linearity,
    }


# ---------------------------------------------------------------------------
# T7 nonlinear checkpoint non-equivariance
# ---------------------------------------------------------------------------


def rmsnorm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    denom = np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps)
    return x / denom


def t7_nonlinear_non_equivariance(
    sch: Scheme, rng: np.random.Generator
) -> dict[str, Any]:
    d = sch.d
    out: dict[str, Any] = {
        "claim": (
            "exact covariance across a nonlinear checkpoint requires an h-independent "
            "matrix Phi with F(h+delta) = F(h) + delta Phi; no such Phi exists"
        )
    }

    for delta_scale in (1e-1, 1e-3):
        deltas, targets = [], []
        for _ in range(4000):
            h = rng.standard_normal(d)
            delta = rng.standard_normal(d) * delta_scale
            deltas.append(delta)
            targets.append(rmsnorm(h + delta) - rmsnorm(h))
        deltas = np.array(deltas)
        targets = np.array(targets)
        phi, *_ = np.linalg.lstsq(deltas, targets, rcond=None)
        resid = targets - deltas @ phi
        out[f"varying_h_delta_scale_{delta_scale:.0e}"] = {
            "best_fit_relative_residual": float(
                np.linalg.norm(resid) / np.linalg.norm(targets)
            ),
        }

    # by contrast, in-Ker noise is invisible to the checkpoint
    h = rng.standard_normal(d)
    e = rng.standard_normal(sch.r)
    c = sch.encode(h, e)
    ker = left_null_space(sch.l_read)
    n = ker.T @ rng.standard_normal(ker.shape[0]) * 50.0
    out["in_ker_noise_invisible_to_checkpoint"] = {
        "rel_error_F_of_decoded": rel(rmsnorm(sch.decode(c + n)), rmsnorm(h)),
        "in_ker_noise_norm": float(np.linalg.norm(n)),
    }
    return out


# ---------------------------------------------------------------------------
# T8 amplification through a deployed chain
# ---------------------------------------------------------------------------


def t8a_linear_chain_conserves_relative_error(
    d: int, big_d: int, r: int, cond: float, depth: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Purely linear deployed chain: c <- c + c W_tilde, semantic h <- h + h W.

    Result to record: in-Ker noise stays exact, and a generic outside perturbation
    keeps roughly CONSTANT relative error. A linear chain does not amplify; the
    amplification seen in real models comes from the nonlinear checkpoints (t8b).
    """
    sch = make_scheme(d, big_d, r, cond, rng)
    gamma = 0.5

    layers = []
    for _ in range(depth):
        w = rng.standard_normal((d, d)) / np.sqrt(d)
        perm = rng.permutation(r)
        g = gamma * np.eye(r)[perm]
        k = np.zeros((sch.n, sch.n))
        k[: sch.big_d, : sch.big_d] = sch.q_hat @ w @ sch.p_hat
        k[sch.big_d :, sch.big_d :] = g
        layers.append((w, sch.m_inv @ k @ sch.m))

    h0 = rng.standard_normal(d)
    e0 = rng.standard_normal(r)
    a0 = rng.standard_normal(big_d - d)
    ker = left_null_space(sch.l_read)

    def run(noise: np.ndarray) -> list[float]:
        h = h0.copy()
        c = sch.encode(h0, e0, a0) + noise
        errs = []
        for w, w_tilde in layers:
            h = h + h @ w
            c = c + c @ w_tilde
            errs.append(float(np.linalg.norm(sch.decode(c) - h) / np.linalg.norm(h)))
        return errs

    eps_rel = 1e-5
    base = np.linalg.norm(sch.encode(h0, e0, a0))
    n_ker = ker.T @ rng.standard_normal(ker.shape[0])
    n_ker = n_ker / np.linalg.norm(n_ker) * eps_rel * base
    n_gen = rng.standard_normal(sch.n)
    n_gen = n_gen / np.linalg.norm(n_gen) * eps_rel * base

    err_ker = run(n_ker)
    err_gen = run(n_gen)
    return {
        "claim": (
            "through a purely linear deployed chain, in-Ker(L) noise stays exact and a "
            "generic outside perturbation keeps an approximately CONSTANT relative "
            "semantic error: linear propagation does not amplify"
        ),
        "depth": depth,
        "injected_relative_noise": eps_rel,
        "in_ker_rel_semantic_error_first_last": [err_ker[0], err_ker[-1]],
        "generic_rel_semantic_error_first_last": [err_gen[0], err_gen[-1]],
        "generic_rel_semantic_error_by_depth": err_gen,
        "linear_amplification_last_over_first": err_gen[-1] / err_gen[0],
    }


def t8b_nonlinear_amplification(
    d: int, depth: int, seq: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Plaintext transformer-like stack; measure how a relative delta_h at the input
    propagates to relative logit error at the output, as a function of depth.

    Block: RMSNorm -> single-head softmax attention (causal) -> residual
           -> RMSNorm -> SwiGLU -> residual.
    This is a random-weight surrogate, not a pretrained model; it isolates the
    amplification mechanism, it is not a task-accuracy measurement.
    """
    d_ff = 2 * d
    blocks = []
    for _ in range(depth):
        blocks.append(
            {
                "wq": rng.standard_normal((d, d)) / np.sqrt(d),
                "wk": rng.standard_normal((d, d)) / np.sqrt(d),
                "wv": rng.standard_normal((d, d)) / np.sqrt(d),
                "wo": rng.standard_normal((d, d)) / np.sqrt(d),
                "wg": rng.standard_normal((d, d_ff)) / np.sqrt(d),
                "wu": rng.standard_normal((d, d_ff)) / np.sqrt(d),
                "wd": rng.standard_normal((d_ff, d)) / np.sqrt(d_ff),
            }
        )
    w_head = rng.standard_normal((d, 64)) / np.sqrt(d)

    causal = np.triu(np.full((seq, seq), -np.inf), k=1)

    def silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-x))

    def softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=-1, keepdims=True)
        ex = np.exp(x)
        return ex / ex.sum(axis=-1, keepdims=True)

    def forward(x: np.ndarray, upto: int) -> np.ndarray:
        for blk in blocks[:upto]:
            h = rmsnorm(x)
            q, k, v = h @ blk["wq"], h @ blk["wk"], h @ blk["wv"]
            s = q @ k.T / np.sqrt(d) + causal
            x = x + (softmax(s) @ v) @ blk["wo"]
            h = rmsnorm(x)
            x = x + (silu(h @ blk["wg"]) * (h @ blk["wu"])) @ blk["wd"]
        return x

    x0 = rng.standard_normal((seq, d))
    out: dict[str, Any] = {}
    for eps_rel in (1e-5, 1e-3):
        delta = rng.standard_normal((seq, d))
        delta = delta / np.linalg.norm(delta) * eps_rel * np.linalg.norm(x0)
        by_depth = []
        for upto in range(1, depth + 1):
            clean = forward(x0, upto)
            dirty = forward(x0 + delta, upto)
            logit_err = rel(dirty @ w_head, clean @ w_head)
            by_depth.append(
                {
                    "depth": upto,
                    "rel_hidden_error": rel(dirty, clean),
                    "rel_logit_error": logit_err,
                    "amplification_vs_input": logit_err / eps_rel,
                }
            )
        out[f"input_rel_delta_{eps_rel:.0e}"] = {
            "by_depth": by_depth,
            "final_amplification": by_depth[-1]["amplification_vs_input"],
        }
    return {
        "claim": (
            "a semantic perturbation delta_h IS amplified by the nonlinear stack "
            "(RMSNorm / softmax / SwiGLU); the amplification grows with depth"
        ),
        "surrogate": "random-weight transformer-like stack, NOT a pretrained model",
        "d": d,
        "depth": depth,
        "seq": seq,
        "results": out,
    }


# ---------------------------------------------------------------------------
# T9 SNR barrier arithmetic
# ---------------------------------------------------------------------------


def t9_snr_barrier() -> dict[str, Any]:
    """bits retained per coordinate for a Gaussian channel h -> h + delta."""
    rows = []
    for ratio in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.577, 1.0, 10.0):
        bits = 0.5 * np.log2(1.0 + 1.0 / ratio**2)
        rows.append({"sigma_delta_over_sigma_h": ratio, "bits_retained_per_coord": float(bits)})
    return {
        "claim": (
            "information-theoretic hiding needs sigma_delta comparable to sigma_h; "
            "bits_retained = 0.5 log2(1 + SNR), SNR = (sigma_h/sigma_delta)^2"
        ),
        "table": rows,
        "sigma_for_1_bit_retained": 0.5773502691896258,
    }


def t10_modulus_is_what_creates_hardness(
    d: int, n: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Same secret, same noise, two rings.

    Over R:      c_t = u_t M + n_t         -> ordinary least squares recovers M.
    Over Z_q:    c_t = (u_t M + n_t) mod q -> the same estimator fails.

    This isolates the source of LWE hardness: it is the modular wrap-around, not
    the presence of noise. A float-GEMM obfuscation pipeline has no wrap-around.
    """
    q = 3329  # ML-KEM modulus, used only as a representative small prime
    m_true = rng.integers(0, q, size=(d, n)).astype(np.float64)
    sigma = 2.0  # "small" noise relative to q

    out: dict[str, Any] = {}
    for ring in ("real", "mod_q"):
        rows = []
        for t_samples in (64, 256, 1024, 4096):
            u = rng.integers(0, q, size=(t_samples, d)).astype(np.float64)
            noise = np.round(rng.standard_normal((t_samples, n)) * sigma)
            c = u @ m_true + noise
            if ring == "mod_q":
                c = np.mod(c, q)
            m_hat, *_ = np.linalg.lstsq(u, c, rcond=None)
            rows.append(
                {
                    "T": t_samples,
                    "rel_error_M_hat": rel(m_hat, m_true),
                    "rounded_entries_exactly_correct_fraction": float(
                        (np.mod(np.round(m_hat), q) == m_true).mean()
                    ),
                }
            )
        out[ring] = rows
    return {
        "claim": (
            "least squares recovers the secret over R but not over Z_q; hardness comes "
            "from modular reduction, not from adding noise"
        ),
        "q": q,
        "sigma_noise": sigma,
        "d": d,
        "n": n,
        "results": out,
        "caveat": (
            "this is a demonstration that the estimator breaks, not a hardness proof; "
            "actual LWE hardness rests on lattice reductions"
        ),
    }


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--big-d", type=int, default=48)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--cond", type=float, default=3.0)
    parser.add_argument("--depth", type=int, default=28)
    args = parser.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    sch = make_scheme(args.d, args.big_d, args.r, args.cond, rng)

    report = {
        "generated": "2026-09-19",
        "seed": RNG_SEED,
        "dtype": "float64",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "geometry": {
            "d_semantic": args.d,
            "D_lifted": args.big_d,
            "r_concat_noise": args.r,
            "N_ambient": args.big_d + args.r,
            "cond_M": args.cond,
            "cond_M_measured": float(np.linalg.cond(sch.m)),
        },
        "T1_order_equivalence": t1_order_equivalence(sch, rng),
        "T2_semantic_error_is_nL": t2_semantic_error_is_nL(sch, rng),
        "T3_kernel_saturation": t3_kernel_saturation(sch),
        "T4_lossless_under_huge_inker_noise": t4_lossless_under_huge_inker_noise(sch, rng),
        "T5_known_plaintext_break": t5_known_plaintext_break(
            args.d, args.big_d, args.r, args.cond, rng
        ),
        "T6_outside_noise_statistics": t6_outside_noise_statistics(
            args.d, args.big_d, args.r, args.cond, rng
        ),
        "T7_nonlinear_non_equivariance": t7_nonlinear_non_equivariance(sch, rng),
        "T8a_linear_chain_conserves_relative_error": t8a_linear_chain_conserves_relative_error(
            args.d, args.big_d, args.r, args.cond, args.depth, rng
        ),
        "T8b_nonlinear_amplification": t8b_nonlinear_amplification(
            args.d, args.depth, 16, rng
        ),
        "T9_snr_barrier": t9_snr_barrier(),
        "T10_modulus_is_what_creates_hardness": t10_modulus_is_what_creates_hardness(
            16, 16, rng
        ),
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"written: {args.output}")
    print(text)


if __name__ == "__main__":
    main()
