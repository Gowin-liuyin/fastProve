"""Standalone FP64 verification of every algebraic identity used by docs/IMPLEMENTATION_PLAN.md.

Run:  python docs/design_identity_check.py

This script depends only on torch. It does not import fastprove, so it can be
run before, during, and after the refactor as an independent oracle. Every
printed error must be <= 1e-12 (pure FP64 round-off). A failure here means the
plan's algebra is wrong; a failure in the repo tests while this passes means the
implementation deviated from the plan.
"""

from __future__ import annotations

import math

import torch

F64 = torch.float64
EPS = 1e-5


# --------------------------------------------------------------------------
# structured basis:  M = P1 @ D @ B @ P2
# --------------------------------------------------------------------------
def make_structured_basis(seed: int, signal_dim: int, noise_dim: int, block: int):
    total = signal_dim + noise_dim
    if total % block != 0:
        raise ValueError("block must divide signal_dim + noise_dim")
    count = total // block
    gen = torch.Generator().manual_seed(seed)
    perm_in = torch.randperm(total, generator=gen)
    perm_out = torch.randperm(total, generator=gen)
    half = 0.5 * math.log(10.0)
    scales = torch.exp((2 * torch.rand(total, generator=gen, dtype=F64) - 1) * half)
    blocks = []
    for _ in range(count):
        raw = torch.randn(block, block, generator=gen, dtype=F64)
        q, upper = torch.linalg.qr(raw)
        signs = torch.sign(torch.diagonal(upper))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        blocks.append(q * signs.unsqueeze(0))
    blocks = torch.stack(blocks)
    identity = torch.eye(total, dtype=F64)
    dense = (
        identity[perm_in]
        @ torch.diag(scales)
        @ torch.block_diag(*list(blocks))
        @ identity[perm_out]
    )
    return {
        "dense": dense,
        "inverse": torch.linalg.inv(dense),
        "perm_in": perm_in,
        "perm_out": perm_out,
        "scales": scales,
        "blocks": blocks,
        "block": block,
        "count": count,
        "signal_dim": signal_dim,
        "noise_dim": noise_dim,
    }


def block_apply(x: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
    count, block, _ = blocks.shape
    segments = x.reshape(*x.shape[:-1], count, block)
    mixed = torch.einsum("...mi,mij->...mj", segments, blocks)
    return mixed.reshape(*x.shape)


def fast_mix(basis: dict, augmented: torch.Tensor) -> torch.Tensor:
    inverse_perm_in = torch.argsort(basis["perm_in"])
    gathered = augmented[..., inverse_perm_in] * basis["scales"]
    mixed = block_apply(gathered, basis["blocks"])
    return mixed[..., torch.argsort(basis["perm_out"])]


def fast_unmix(basis: dict, mixed: torch.Tensor) -> torch.Tensor:
    gathered = mixed[..., basis["perm_out"]]
    unblocked = block_apply(gathered, basis["blocks"].transpose(-1, -2).contiguous())
    return (unblocked / basis["scales"])[..., basis["perm_in"]]


def signal_gram_blocks(basis: dict) -> torch.Tensor:
    projection = basis["inverse"][:, : basis["signal_dim"]]
    gram = projection @ projection.T
    permuted = gram[basis["perm_out"]][:, basis["perm_out"]]
    block, count = basis["block"], basis["count"]
    off = permuted.clone()
    for index in range(count):
        lo, hi = index * block, (index + 1) * block
        off[lo:hi, lo:hi] = 0
    if float(off.abs().max()) > 1e-12:
        raise AssertionError("signal Gram is not block diagonal in perm_out order")
    return torch.stack(
        [
            permuted[index * block : (index + 1) * block, index * block : (index + 1) * block]
            for index in range(count)
        ]
    )


def signal_norm_squared(basis: dict, mixed: torch.Tensor, gram_blocks: torch.Tensor) -> torch.Tensor:
    gathered = mixed[..., basis["perm_out"]]
    count, block = basis["count"], basis["block"]
    segments = gathered.reshape(*mixed.shape[:-1], count, block)
    return torch.einsum("...mi,mij,...mj->...", segments, gram_blocks, segments)


def report(name: str, value: float) -> None:
    status = "OK  " if value <= 1e-12 else "FAIL"
    print("[%s] %-46s %.3e" % (status, name, value))
    if value > 1e-12:
        raise AssertionError(name)


def main() -> None:
    torch.manual_seed(0)
    d, r, block = 64, 8, 8
    n = d + r
    d_head, r_head, heads, kv_heads = 16, 2, 4, 2
    d_ff = 96
    batch, seq = 2, 6

    basis = make_structured_basis(7, d, r, block)
    dense, inverse = basis["dense"], basis["inverse"]
    projection = inverse[:, :d]
    noise_read = inverse[:, d:]
    top, bottom = dense[:d], dense[d:]
    gram_blocks = signal_gram_blocks(basis)

    print("kappa(M) = %.4f" % float(torch.linalg.cond(dense)))

    signal = torch.randn(batch, seq, d, dtype=F64)
    noise = torch.randn(batch, seq, r, dtype=F64)
    augmented = torch.cat((signal, noise), dim=-1)
    mixed = augmented @ dense

    report("fast_mix == augmented @ M", float((fast_mix(basis, augmented) - mixed).abs().max()))
    report("fast_unmix == mixed @ M^-1", float((fast_unmix(basis, mixed) - augmented).abs().max()))
    report("h == c @ P", float((mixed @ projection - signal).abs().max()))
    report("e == c @ N", float((mixed @ noise_read - noise).abs().max()))
    report(
        "||h||^2 == blockwise Gram",
        float((signal_norm_squared(basis, mixed, gram_blocks) - signal.pow(2).sum(-1)).abs().max()),
    )

    gamma_attention = torch.rand(d, dtype=F64) + 0.5
    gamma_ffn = torch.rand(d, dtype=F64) + 0.5
    w_q = torch.randn(d, heads * d_head, dtype=F64) * 0.1
    w_k = torch.randn(d, kv_heads * d_head, dtype=F64) * 0.1
    w_v = torch.randn(d, kv_heads * d_head, dtype=F64) * 0.1
    w_o = torch.randn(heads * d_head, d, dtype=F64) * 0.1
    w_gate = torch.randn(d, d_ff, dtype=F64) * 0.1
    w_up = torch.randn(d, d_ff, dtype=F64) * 0.1
    w_down = torch.randn(d_ff, d, dtype=F64) * 0.1
    kv_index = torch.tensor([0, 0, 1, 1])

    def rms_scale(hidden: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(hidden.pow(2).mean(-1, keepdim=True) + EPS)

    def normalize(hidden: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return hidden / rms_scale(hidden) * gamma

    def split_heads(flat: torch.Tensor, count: int) -> torch.Tensor:
        return flat.view(batch, seq, count, d_head).transpose(1, 2)

    def causal_softmax(scores: torch.Tensor) -> torch.Tensor:
        length = scores.shape[-1]
        mask = torch.tril(torch.ones(length, length, dtype=torch.bool))
        return torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)

    def plaintext_block(hidden: torch.Tensor):
        normalized = normalize(hidden, gamma_attention)
        q = split_heads(normalized @ w_q, heads)
        k = split_heads(normalized @ w_k, kv_heads)
        v = split_heads(normalized @ w_v, kv_heads)
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k[:, kv_index]) / math.sqrt(d_head)
        probabilities = causal_softmax(scores)
        context = torch.einsum("bhqk,bhkd->bhqd", probabilities, v[:, kv_index])
        delta = context.transpose(1, 2).reshape(batch, seq, heads * d_head) @ w_o
        post_attention = hidden + delta
        normalized_ffn = normalize(post_attention, gamma_ffn)
        activated = torch.nn.functional.silu(normalized_ffn @ w_gate) * (normalized_ffn @ w_up)
        return post_attention + activated @ w_down, probabilities, scores, delta, activated

    value_bases = [make_structured_basis(21 + head, d_head, r_head, r_head + d_head) for head in range(kv_heads)]
    value_mix = torch.stack([item["dense"] for item in value_bases])
    value_unmix = torch.stack([item["inverse"] for item in value_bases])

    coupling_value = torch.randn(n, kv_heads, r_head, dtype=F64) * 0.03
    coupling_attention = torch.randn(d, r, dtype=F64) * 0.02
    propagator_attention = torch.diag(torch.full((r,), 0.5, dtype=F64))
    aux_to_hidden = torch.randn(r_head, r, dtype=F64) * 0.08
    refresh_attention = torch.randn(r, dtype=F64) * 0.02

    generator = torch.Generator().manual_seed(9)
    neuron_perm = torch.randperm(d_ff, generator=generator)
    neuron_scale = torch.exp((2 * torch.rand(d_ff, generator=generator, dtype=F64) - 1) * 0.3)
    coupling_swiglu = torch.randn(d_ff, r, dtype=F64) * 0.02
    coupling_down = torch.randn(d, r, dtype=F64) * 0.02
    propagator_ffn = torch.diag(torch.full((r,), 0.5, dtype=F64))
    refresh_ffn = torch.randn(r, dtype=F64) * 0.02

    deployed_q = projection @ (gamma_attention[:, None] * w_q)
    deployed_k = projection @ (gamma_attention[:, None] * w_k)
    deployed_value = torch.stack(
        [
            torch.cat(
                (
                    projection @ (gamma_attention[:, None] * w_v[:, head * d_head : (head + 1) * d_head]),
                    coupling_value[:, head],
                ),
                dim=-1,
            )
            @ value_mix[head]
            for head in range(kv_heads)
        ]
    )
    # Maximally fused attention output: the per-KV-head Value unmix, W_O, the
    # residual re-mix (M_top), the signal noise coupling (C_O @ M_bot) and the
    # auxiliary-to-hidden term all collapse into ONE matrix per QUERY head.
    # Consequence: the clean attention context O is never materialized on the
    # server path, and the deployed block has the plaintext GEMM count.
    identity_r = torch.eye(r, dtype=F64)
    deployed_attention_out = torch.stack(
        [
            value_unmix[kv_index[head]][:, :d_head] @ w_o[head * d_head : (head + 1) * d_head] @ (top + coupling_attention @ bottom)
            + (1.0 / heads) * value_unmix[kv_index[head]][:, d_head:] @ aux_to_hidden @ bottom
            for head in range(heads)
        ]
    )
    deployed_noise_out_attention = (propagator_attention - identity_r) @ bottom
    deployed_refresh_attention = refresh_attention @ bottom

    deployed_gate = projection @ (gamma_ffn[:, None] * w_gate[:, neuron_perm])
    deployed_up = projection @ (gamma_ffn[:, None] * (w_up[:, neuron_perm] * neuron_scale[neuron_perm][None, :]))
    down_math = (1.0 / neuron_scale[neuron_perm])[:, None] * w_down[neuron_perm]
    # Fully fused down path: Pf^T Df^-1 W_d, the residual re-mix, the down noise
    # coupling and the SwiGLU noise coupling collapse into ONE deployed matrix.
    deployed_out_ffn = down_math @ (top + coupling_down @ bottom) + coupling_swiglu @ bottom
    deployed_noise_out_ffn = (propagator_ffn - identity_r) @ bottom
    deployed_refresh_ffn = refresh_ffn @ bottom

    def obfuscated_block(state: torch.Tensor):
        scale = torch.sqrt(signal_norm_squared(basis, state, gram_blocks).clamp_min(0).unsqueeze(-1) / d + EPS)
        q = split_heads(state @ deployed_q / scale, heads)
        k = split_heads(state @ deployed_k / scale, kv_heads)
        # Single GEMM per KV head. Dividing the whole product by rho also
        # divides the auxiliary columns, so the deployed noise definition is
        # e_V := (c @ C_V) / rho. That is a design choice, not an error: any
        # deterministic function of (c, rho) is a valid auxiliary state.
        mixed_value = torch.einsum("btn,hne->bhte", state, deployed_value) / scale.unsqueeze(1)
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k[:, kv_index]) / math.sqrt(d_head)
        probabilities = causal_softmax(scores)
        mixed_context = torch.einsum("bhqk,bhkd->bhqd", probabilities, mixed_value[:, kv_index])
        # The mixed context is consumed directly; no Value unmix, no clean O.
        carried = state @ noise_read
        post_attention = (
            state
            + torch.einsum("bhqd,hdn->bqn", mixed_context, deployed_attention_out)
            + carried @ deployed_noise_out_attention
            + deployed_refresh_attention
        )
        scale_ffn = torch.sqrt(signal_norm_squared(basis, post_attention, gram_blocks).clamp_min(0).unsqueeze(-1) / d + EPS)
        activated = torch.nn.functional.silu(post_attention @ deployed_gate / scale_ffn) * (
            post_attention @ deployed_up / scale_ffn
        )
        carried_ffn = post_attention @ noise_read
        final = (
            post_attention
            + activated @ deployed_out_ffn
            + carried_ffn @ deployed_noise_out_ffn
            + deployed_refresh_ffn
        )
        return final, probabilities, scores, mixed_context, activated, post_attention

    reference, ref_probabilities, ref_scores, ref_delta, ref_activated = plaintext_block(signal)
    result, probabilities, scores, mixed_context, activated, post_attention = obfuscated_block(mixed)

    report("QK scores match plaintext", float((ref_scores - scores).abs().max()))
    report("softmax probabilities match", float((ref_probabilities - probabilities).abs().max()))
    report("post-attention decodes to plaintext", float((post_attention @ projection - (signal + ref_delta)).abs().max()))
    report("z' == z Df Pf", float((activated - (ref_activated * neuron_scale)[..., neuron_perm]).abs().max()))
    report("z' Wd' == z Wd", float((activated @ down_math - ref_activated @ w_down).abs().max()))
    report("block output decodes to plaintext", float((result @ projection - reference).abs().max()))

    carried_ffn = post_attention @ noise_read
    expected_final_noise = (
        carried_ffn @ propagator_ffn
        + (ref_activated @ w_down) @ coupling_down
        + activated @ coupling_swiglu
        + refresh_ffn
    )
    report("final noise matches e' = e G + dH C + z' Cz + xi", float((result @ noise_read - expected_final_noise).abs().max()))

    print("\nall identities verified in FP64")


if __name__ == "__main__":
    main()
