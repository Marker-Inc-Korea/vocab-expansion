from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .init import exp_init_from_parts, gather_rows_with_buffer, safe_decode_single


def parse_core_special_tokens(arg: Optional[str]) -> List[str]:
    default = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|fim_pad|>",
        "<|quad_start|>",
        "<|quad_end|>",
    ]
    if not arg:
        return default
    toks = [t.strip() for t in arg.split(",") if t.strip()]
    return toks if toks else default


def check_config_files(output_dir: Path, tok) -> Tuple[bool, bool]:
    cfg_path = output_dir / "config.json"
    gen_path = output_dir / "generation_config.json"

    ok_cfg = True
    ok_gen = True

    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        bos = cfg.get("bos_token_id")
        pad = cfg.get("pad_token_id")
        eos = cfg.get("eos_token_id")
        vsz = cfg.get("vocab_size")
        eot = tok.convert_tokens_to_ids("<|endoftext|>")
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        if eot is not None and eot >= 0:
            ok_cfg &= (bos == eot) and (pad == eot)
        if im_end is not None and im_end >= 0:
            ok_cfg &= (eos == im_end)
        ok_cfg &= (vsz == len(tok))
    else:
        ok_cfg = False

    if gen_path.exists():
        gen = json.loads(gen_path.read_text(encoding="utf-8"))
        gbos = gen.get("bos_token_id")
        gpad = gen.get("pad_token_id")
        geos = gen.get("eos_token_id")
        eot = tok.convert_tokens_to_ids("<|endoftext|>")
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        if eot is not None and eot >= 0:
            ok_gen &= (gbos == eot) and (gpad == eot)
        if im_end is not None and im_end >= 0:
            if isinstance(geos, list) and len(geos) >= 1:
                ok_gen &= (geos[0] == im_end) and (geos[-1] == eot)
            else:
                ok_gen &= (geos == im_end)
    else:
        ok_gen = False

    return ok_cfg, ok_gen


@torch.no_grad()
def verify_alignment(
    *,
    output_dir: str,
    base_tokenizer,
    base_vocab: Dict[str, int],
    aligned_model,  # already in memory (post-build)
    buf_ids: List[int],
    buf_index: Dict[int, int],
    buf_in: Optional[torch.Tensor],
    buf_out: Optional[torch.Tensor],
    mismatches: List[Tuple[str, int, int]],
    alpha: float,
    seed: int = 0,
    max_shared: int = 2000,
    max_new: int = 200,
    core_special_tokens: Optional[str] = None,
) -> bool:
    """Lightweight verification; returns True if checks pass."""
    rng = random.Random(seed)

    out_dir = Path(output_dir)
    aligned_tok = AutoTokenizer.from_pretrained(str(out_dir), use_fast=True)
    aligned_vocab = aligned_tok.get_vocab()

    # Config id alignment
    ok_cfg, ok_gen = check_config_files(out_dir, aligned_tok)

    # Shared-token row equality check (sample)
    W_in = aligned_model.get_input_embeddings().weight
    W_out = aligned_model.get_output_embeddings().weight
    atol = 5e-3 if W_in.dtype in (torch.float16, torch.bfloat16) else 1e-6

    core = set(parse_core_special_tokens(core_special_tokens))
    forced = set([t for (t, _, _) in mismatches]).union(core)

    common = list(set(base_vocab).intersection(aligned_vocab))
    rest = [t for t in common if t not in forced]
    sample_n = min(max_shared, len(rest))
    sampled = rng.sample(rest, sample_n) if sample_n > 0 else []
    to_check = list(set(sampled).union(forced))

    max_diff = 0.0
    for t in to_check:
        bi = int(base_vocab[t])
        ai = int(aligned_vocab[t])
        exp_in = buf_in[buf_index[bi]] if (buf_in is not None and bi in buf_index) else W_in[bi]
        exp_out = buf_out[buf_index[bi]] if (buf_out is not None and bi in buf_index) else W_out[bi]
        din = float((exp_in.float() - W_in[ai].float()).abs().max().item())
        dout = float((exp_out.float() - W_out[ai].float()).abs().max().item())
        max_diff = max(max_diff, din, dout)

    ok_shared = max_diff <= atol

    # New-token init check (sample cosine)
    new_tokens = [t for t in aligned_vocab if t not in base_vocab]
    sample_new = rng.sample(new_tokens, min(max_new, len(new_tokens))) if new_tokens else []
    ok_new = True
    for t in sample_new:
        tid = int(aligned_vocab[t])
        surface = safe_decode_single(aligned_tok, tid)
        part_ids = base_tokenizer.encode(surface, add_special_tokens=False)
        if not part_ids:
            continue

        parts_in = (
            gather_rows_with_buffer(W_in, part_ids, buf_ids, buf_in, buf_index)
            if buf_in is not None
            else W_in.index_select(0, torch.tensor(part_ids, device=W_in.device))
        )
        parts_out = (
            gather_rows_with_buffer(W_out, part_ids, buf_ids, buf_out, buf_index)
            if buf_out is not None
            else W_out.index_select(0, torch.tensor(part_ids, device=W_out.device))
        )

        exp_in = exp_init_from_parts(parts_in, alpha=alpha, mode="input").float()
        exp_out = exp_init_from_parts(parts_out, alpha=alpha, mode="output").float()
        act_in = W_in[tid].float()
        act_out = W_out[tid].float()

        cos_in = float(F.cosine_similarity(exp_in, act_in, dim=0).item())
        cos_out = float(F.cosine_similarity(exp_out, act_out, dim=0).item())
        if min(cos_in, cos_out) < 0.98:
            # heuristic; not a strict correctness criterion
            ok_new = False
            break

    return bool(ok_cfg and ok_gen and ok_shared and ok_new)
