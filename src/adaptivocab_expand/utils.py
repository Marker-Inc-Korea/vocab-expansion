from __future__ import annotations

import torch


def parse_dtype(s: str):
    s = s.lower()
    if s == "auto":
        return "auto"
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def resize_embeddings(model, new_size: int, mean_resizing: bool = False):
    """Resize token embeddings. Tries to keep transformers warnings quiet via mean_resizing=False."""
    try:
        model.resize_token_embeddings(new_size, mean_resizing=mean_resizing)
    except TypeError:
        model.resize_token_embeddings(new_size)
