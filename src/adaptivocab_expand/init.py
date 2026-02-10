from __future__ import annotations

from typing import Dict, List

import torch


def safe_decode_single(tok, token_id: int) -> str:
    # safest way to get the surface form for a single token
    return tok.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def softmax_weights(k: int, alpha: float, sign: float, device) -> torch.Tensor:
    # pos=0..k-1 (index shift does not matter for softmax)
    pos = torch.arange(k, device=device, dtype=torch.float32)
    logits = sign * alpha * pos
    return torch.softmax(logits, dim=0)


def exp_init_from_parts(part_rows: torch.Tensor, alpha: float, mode: str) -> torch.Tensor:
    """AdaptiVocab exponential init.

    mode:
      - "input": weights increase exponentially (emphasize last token)
      - "output": weights decrease exponentially (emphasize first token)
    """
    k = part_rows.shape[0]
    if k == 1:
        return part_rows[0].clone()

    sign = +1.0 if mode == "input" else -1.0
    w = softmax_weights(k, alpha=alpha, sign=sign, device=part_rows.device).to(part_rows.dtype)
    return (w[:, None] * part_rows).sum(dim=0)


def gather_rows_with_buffer(
    weight: torch.Tensor,
    ids: List[int],
    buf_ids: List[int],
    buf_tensor: torch.Tensor,
    buf_index: Dict[int, int],
) -> torch.Tensor:
    """Gather rows from weight, but use buf_tensor for ids that were buffered."""
    device = weight.device
    dtype = weight.dtype
    n = len(ids)
    d = weight.shape[1]
    out = torch.empty((n, d), device=device, dtype=dtype)

    pos_buf, idx_buf = [], []
    pos_nbuf, ids_nbuf = [], []
    for p, tid in enumerate(ids):
        bi = buf_index.get(int(tid), -1)
        if bi >= 0:
            pos_buf.append(p)
            idx_buf.append(bi)
        else:
            pos_nbuf.append(p)
            ids_nbuf.append(int(tid))

    if pos_nbuf:
        t_ids = torch.tensor(ids_nbuf, device=device, dtype=torch.long)
        out[torch.tensor(pos_nbuf, device=device)] = weight.index_select(0, t_ids)

    if pos_buf:
        b_ids = torch.tensor(idx_buf, device=device, dtype=torch.long)
        out[torch.tensor(pos_buf, device=device)] = buf_tensor.index_select(0, b_ids)

    return out
