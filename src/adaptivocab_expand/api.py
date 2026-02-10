from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ExpansionOptions, ExpansionReport
from .init import exp_init_from_parts, gather_rows_with_buffer, safe_decode_single
from .special_tokens import sync_qwen3_special_ids
from .utils import parse_dtype, resize_embeddings
from .verify import verify_alignment


@torch.no_grad()
def expand_and_save(
    *,
    base_model: str,
    base_tokenizer: str,
    expanded_tokenizer: str,
    output_dir: str,
    options: Optional[ExpansionOptions] = None,
) -> ExpansionReport:
    """Expand+align a HF causal LM to an expanded tokenizer, then save to output_dir."""
    options = options or ExpansionOptions()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizers
    base_tok = AutoTokenizer.from_pretrained(
        base_tokenizer, use_fast=True, trust_remote_code=options.trust_remote_code
    )
    ext_tok = AutoTokenizer.from_pretrained(
        expanded_tokenizer, use_fast=True, trust_remote_code=options.trust_remote_code
    )

    base_vocab: Dict[str, int] = base_tok.get_vocab()
    ext_vocab: Dict[str, int] = ext_tok.get_vocab()

    common = set(base_vocab).intersection(ext_vocab)
    mismatches: List[Tuple[str, int, int]] = [
        (t, base_vocab[t], ext_vocab[t]) for t in common if base_vocab[t] != ext_vocab[t]
    ]
    mismatches.sort(key=lambda x: x[1])

    new_tokens: List[Tuple[str, int]] = [(t, ext_vocab[t]) for t in ext_vocab if t not in base_vocab]
    new_tokens.sort(key=lambda x: x[1])

    # Load model
    torch_dtype = parse_dtype(options.dtype)
    kwargs = dict(low_cpu_mem_usage=True, trust_remote_code=options.trust_remote_code)
    if torch_dtype != "auto":
        kwargs["torch_dtype"] = torch_dtype
    if options.device_map:
        kwargs["device_map"] = options.device_map

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    model.eval()

    W_in = model.get_input_embeddings().weight
    out_emb = model.get_output_embeddings()
    if out_emb is None:
        raise RuntimeError("model.get_output_embeddings() is None; cannot align lm_head.")
    W_out = out_emb.weight

    tied = (W_in.data_ptr() == W_out.data_ptr())

    old_V = int(W_in.shape[0])
    new_V = int(len(ext_tok))

    # Resize
    resize_embeddings(model, new_V, mean_resizing=False)
    W_in = model.get_input_embeddings().weight
    W_out = model.get_output_embeddings().weight

    # Buffer base rows that may be overwritten but needed as sources
    overwritten_ids = {ext_id for _, ext_id in new_tokens if ext_id < old_V}
    src_ids = {bi for _, bi, _ in mismatches}
    buf_ids = sorted(overwritten_ids.union(src_ids))
    buf_index = {tid: i for i, tid in enumerate(buf_ids)}
    buf_in = (
        W_in.index_select(0, torch.tensor(buf_ids, device=W_in.device, dtype=torch.long)).clone()
        if buf_ids
        else None
    )
    buf_out = (
        W_out.index_select(0, torch.tensor(buf_ids, device=W_out.device, dtype=torch.long)).clone()
        if buf_ids
        else None
    )

    mean_in = W_in[: min(old_V, W_in.shape[0])].mean(dim=0).clone()
    mean_out = W_out[: min(old_V, W_out.shape[0])].mean(dim=0).clone()

    # Init new tokens
    inited = 0
    skipped = 0
    for tok_str, ext_id in new_tokens:
        surface = safe_decode_single(ext_tok, ext_id)
        part_ids = base_tok.encode(surface, add_special_tokens=False)

        if not part_ids:
            W_in[ext_id].copy_(mean_in)
            if not tied:
                W_out[ext_id].copy_(mean_out)
            else:
                W_out[ext_id].copy_(W_in[ext_id])
            skipped += 1
            continue

        parts_in = (
            gather_rows_with_buffer(W_in, part_ids, buf_ids, buf_in, buf_index)
            if buf_in is not None
            else W_in.index_select(0, torch.tensor(part_ids, device=W_in.device, dtype=torch.long))
        )
        vec_in = exp_init_from_parts(parts_in, alpha=options.alpha, mode="input")
        W_in[ext_id].copy_(vec_in)

        if not tied:
            parts_out = (
                gather_rows_with_buffer(W_out, part_ids, buf_ids, buf_out, buf_index)
                if buf_out is not None
                else W_out.index_select(0, torch.tensor(part_ids, device=W_out.device, dtype=torch.long))
            )
            vec_out = exp_init_from_parts(parts_out, alpha=options.alpha, mode="output")
            W_out[ext_id].copy_(vec_out)
        else:
            W_out[ext_id].copy_(W_in[ext_id])

        inited += 1

    # Remap shared mismatched tokens (copy by token string)
    for tok_str, base_id, ext_id in mismatches:
        if base_id in buf_index and buf_in is not None and buf_out is not None:
            bi = buf_index[base_id]
            W_in[ext_id].copy_(buf_in[bi])
            if not tied:
                W_out[ext_id].copy_(buf_out[bi])
            else:
                W_out[ext_id].copy_(W_in[ext_id])
        else:
            W_in[ext_id].copy_(W_in[base_id])
            if not tied:
                W_out[ext_id].copy_(W_out[base_id])
            else:
                W_out[ext_id].copy_(W_in[ext_id])

    # Sync special tokens in config (optional)
    if options.special_token_strategy == "qwen3":
        sync_qwen3_special_ids(model, ext_tok)
    elif options.special_token_strategy == "none":
        model.config.vocab_size = len(ext_tok)
    else:
        raise ValueError(f"Unknown special_token_strategy: {options.special_token_strategy}")

    # Save
    model.save_pretrained(out_dir, safe_serialization=True)
    model.generation_config.save_pretrained(out_dir)
    ext_tok.save_pretrained(out_dir)

    # Verify (optional)
    verify_ok = None
    if options.verify:
        verify_ok = verify_alignment(
            output_dir=str(out_dir),
            base_tokenizer=base_tok,
            base_vocab=base_vocab,
            aligned_model=model,
            buf_ids=buf_ids,
            buf_index=buf_index,
            buf_in=buf_in,
            buf_out=buf_out,
            mismatches=mismatches,
            alpha=options.alpha,
            seed=options.seed,
            max_shared=options.max_shared,
            max_new=options.max_new,
            core_special_tokens=options.core_special_tokens,
        )

    return ExpansionReport(
        base_vocab_size=len(base_tok),
        expanded_vocab_size=len(ext_tok),
        shared_tokens=len(common),
        mismatched_shared_tokens=len(mismatches),
        new_tokens=len(new_tokens),
        new_tokens_inited=inited,
        new_tokens_fallback_mean=skipped,
        embeddings_tied=bool(tied),
        output_dir=str(out_dir),
        verify_passed=verify_ok,
    )
