"""First-mismatch diagnostics for exact-mode comparisons."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def _first_argmax_mismatch(
    plaintext_logits: torch.Tensor,
    obfuscated_logits: torch.Tensor,
) -> Optional[Dict[str, Any]]:
    plain_predictions = plaintext_logits.argmax(dim=-1)
    obfuscated_predictions = obfuscated_logits.argmax(dim=-1)
    locations = torch.nonzero(
        plain_predictions != obfuscated_predictions, as_tuple=False
    )
    if locations.numel() == 0:
        return None
    batch_index = int(locations[0, 0].item())
    position = int(locations[0, 1].item())
    row = plaintext_logits[batch_index, position].float()
    top_count = min(2, row.numel())
    top_values = torch.topk(row, top_count).values
    margin = (
        float((top_values[0] - top_values[1]).item())
        if top_count == 2
        else None
    )
    return {
        "batch_index": batch_index,
        "position": position,
        "plaintext_token": int(
            plain_predictions[batch_index, position].item()
        ),
        "obfuscated_token": int(
            obfuscated_predictions[batch_index, position].item()
        ),
        "plaintext_top2_margin": margin,
    }


def _first_token_mismatch(
    plaintext_tokens: torch.Tensor,
    obfuscated_tokens: torch.Tensor,
    *,
    prompt_length: int,
) -> Optional[Dict[str, int]]:
    differences = torch.nonzero(
        plaintext_tokens[:, prompt_length:]
        != obfuscated_tokens[:, prompt_length:],
        as_tuple=False,
    )
    if differences.numel() == 0:
        return None
    batch_index = int(differences[0, 0].item())
    generation_step = int(differences[0, 1].item())
    position = prompt_length + generation_step
    return {
        "batch_index": batch_index,
        "generation_step": generation_step,
        "plaintext_token": int(
            plaintext_tokens[batch_index, position].item()
        ),
        "obfuscated_token": int(
            obfuscated_tokens[batch_index, position].item()
        ),
    }
