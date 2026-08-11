"""Layer 2 — Attention Softmax ranking and distribution (protocol §4.2).

Most important layer for the covariant scheme. Per (layer, head, valid row):

* ε^S = max |S̃ − S|
* m = S_(1) − S_(2)
* Theorem: top-1 guaranteed safe when m > 2ε

HARD GATE: rank-flip rate = 0 on the release validation set.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .metrics_common import (
    infinity_and_relative,
    js_divergence,
    kl_divergence,
    margin_risk_rate,
    relative_l2_error,
    score_max_abs_error,
    top1_margin,
    topk_set_overlap,
    total_variation,
)
from .model_factory import EvalModels


@torch.no_grad()
def run_layer2(
    models: EvalModels,
    input_ids: torch.Tensor,
    *,
    token_mask: Optional[torch.Tensor] = None,
    top_k: int = 4,
) -> Dict[str, Any]:
    """Compute attention ranking / distribution metrics vs plaintext."""

    device = models.device
    input_ids = input_ids.to(device)
    if token_mask is not None:
        token_mask = token_mask.to(device)

    if models.obfuscated is None:
        return {
            "rank_flip_rate": 0.0,
            "top1_match": 1.0,
            "topk_overlap": 1.0,
            "causal_mask_match": 1.0,
            "note": "plaintext condition; identity metrics",
            "per_layer": [],
        }

    plain = models.plain
    obf = models.obfuscated
    ctx = models.request_context("layer2")
    codec = models.token_codec

    # Teacher-forced single forward with debug on both paths.
    plain_hidden = plain.embedding(input_ids)
    # Run obfuscated full LM debug for attention internals.
    _, obf_debugs = obf.forward_debug(
        codec.encode(input_ids),
        token_mask=token_mask,
        request_context=ctx,
    )

    per_layer: List[Dict[str, Any]] = []
    all_top1: List[float] = []
    all_topk: List[float] = []
    all_flip: List[float] = []
    all_kl: List[float] = []
    all_tv: List[float] = []
    all_js: List[float] = []
    all_score_max: List[float] = []
    all_score_rel: List[float] = []
    all_risk: List[float] = []
    all_av_rel: List[float] = []
    causal_ok = 1.0

    hidden = plain_hidden
    for layer_index, (pblock, o_dbg) in enumerate(
        zip(plain.blocks, obf_debugs)
    ):
        p_out, p_dbg = pblock.forward_debug(hidden, token_mask=token_mask)
        hidden = p_out

        # Plain scores vs obfuscated clean (exact-mode) logits.
        plain_scores = p_dbg.qk_scores.float()
        plain_probs = p_dbg.probabilities.float()
        plain_valid = p_dbg.valid_mask
        # Broadcast valid if needed.
        if plain_valid.shape != plain_scores.shape:
            plain_valid = torch.broadcast_to(plain_valid, plain_scores.shape)

        # Obfuscated exact-mode clean scores should match plain QK.
        obf_clean = o_dbg.clean_logits.float()
        obf_noisy = o_dbg.noisy_logits.float()
        obf_probs_clean = o_dbg.clean_probabilities.float()
        obf_probs_noisy = o_dbg.noisy_probabilities.float()
        obf_valid = o_dbg.valid_mask
        if obf_valid.shape != obf_clean.shape:
            obf_valid = torch.broadcast_to(obf_valid, obf_clean.shape)

        # Align shapes if GQA expansion differs slightly.
        if plain_scores.shape != obf_clean.shape:
            # Use obfuscated internal plain-vs-transformed from debug when shapes mismatch.
            score_ref = obf_clean
            score_act = obf_noisy
            valid = obf_valid
            probs_ref = obf_probs_clean
            probs_act = obf_probs_noisy
            note = "shape mismatch plain/obf; using internal clean vs noisy"
        else:
            # Primary comparison: plaintext scores vs obfuscated clean (exact path).
            score_ref = plain_scores
            score_act = obf_clean
            valid = plain_valid & obf_valid
            probs_ref = plain_probs
            probs_act = obf_probs_clean
            note = "plaintext vs obfuscated exact clean scores"

        # Causal mask consistency: invalid positions stay -inf / zero prob.
        if plain_scores.shape == obf_clean.shape:
            plain_masked = ~plain_valid
            obf_masked = ~obf_valid
            if plain_masked.shape == obf_masked.shape:
                causal_ok = min(
                    causal_ok,
                    float((plain_masked == obf_masked).float().mean().item()),
                )

        eps_s = score_max_abs_error(score_ref, score_act, valid)
        margins = top1_margin(score_ref, valid)
        risk = margin_risk_rate(margins, eps_s)
        top1, topk, flip = topk_set_overlap(
            score_ref, score_act, k=top_k, valid=valid
        )
        # Also check noisy vs clean if approximate noise present.
        if not torch.equal(obf_clean, obf_noisy):
            top1_n, topk_n, flip_n = topk_set_overlap(
                obf_clean, obf_noisy, k=top_k, valid=obf_valid
            )
            flip = max(flip, flip_n)
            top1 = min(top1, top1_n)
            topk = min(topk, topk_n)

        score_err = infinity_and_relative(
            score_ref.masked_fill(~valid, 0.0),
            score_act.masked_fill(~valid, 0.0),
        )
        kl = kl_divergence(probs_ref, probs_act, valid=valid)
        js = js_divergence(probs_ref, probs_act, valid=valid)
        tv = total_variation(probs_ref, probs_act, valid=valid)
        av_rel = relative_l2_error(
            o_dbg.clean_attention_output.float(),
            o_dbg.attention_output.float(),
        )
        # Prefer plain vs obf attention output when available.
        if p_dbg.attention_output.shape == o_dbg.attention_output.shape:
            av_rel = relative_l2_error(
                p_dbg.attention_output.float(),
                o_dbg.attention_output.float(),
            )

        layer_rec = {
            "layer_index": layer_index,
            "score_max_absolute_error": score_err["max_absolute_error"],
            "score_relative_error": score_err["relative_l2_error"],
            "softmax_kl": kl,
            "softmax_js": js,
            "total_variation": tv,
            "top1_match": top1,
            "topk_overlap": topk,
            "rank_flip_rate": flip,
            "margin_risk_rate_m_le_2eps": risk,
            "av_relative_error": av_rel,
            "mean_margin": float(margins.float().mean().item()),
            "mean_eps_s": float(eps_s.float().mean().item()),
            "note": note,
            "qk_score_error_internal": dict(o_dbg.qk_score_error),
        }
        per_layer.append(layer_rec)
        all_top1.append(top1)
        all_topk.append(topk)
        all_flip.append(flip)
        all_kl.append(kl)
        all_tv.append(tv)
        all_js.append(js)
        all_score_max.append(score_err["max_absolute_error"])
        all_score_rel.append(score_err["relative_l2_error"])
        all_risk.append(risk)
        all_av_rel.append(av_rel)

    def _mean(xs: List[float], default: float = 0.0) -> float:
        return float(sum(xs) / len(xs)) if xs else default

    return {
        "score_max_absolute_error": max(all_score_max) if all_score_max else 0.0,
        "score_relative_error": _mean(all_score_rel),
        "softmax_kl": _mean(all_kl),
        "softmax_js": _mean(all_js),
        "total_variation": _mean(all_tv),
        "top1_match": min(all_top1) if all_top1 else 1.0,
        "topk_overlap": min(all_topk) if all_topk else 1.0,
        "rank_flip_rate": max(all_flip) if all_flip else 0.0,
        "margin_risk_rate_m_le_2eps": _mean(all_risk),
        "av_relative_error": _mean(all_av_rel),
        "causal_mask_match": causal_ok,
        "per_layer": per_layer,
        "top_k": top_k,
        "hard_gate_rank_flip_zero": (max(all_flip) if all_flip else 0.0) == 0.0,
    }
