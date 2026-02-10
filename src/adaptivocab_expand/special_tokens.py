from __future__ import annotations

from transformers import GenerationConfig


def sync_qwen3_special_ids(model, tok):
    """Sync Qwen3 special token ids.

    Convention used in many Qwen chat checkpoints:
      bos/pad: <|endoftext|>
      eos    : <|im_end|>
      generation eos: [<|im_end|>, <|endoftext|>]
    """
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if eot is None or eot < 0:
        raise RuntimeError("Expanded tokenizer missing <|endoftext|>")
    if im_end is None or im_end < 0:
        raise RuntimeError("Expanded tokenizer missing <|im_end|>")

    model.config.bos_token_id = int(eot)
    model.config.pad_token_id = int(eot)
    model.config.eos_token_id = int(im_end)
    model.config.vocab_size = len(tok)

    if getattr(model, "generation_config", None) is None:
        model.generation_config = GenerationConfig.from_model_config(model.config)

    model.generation_config.bos_token_id = int(eot)
    model.generation_config.pad_token_id = int(eot)
    model.generation_config.eos_token_id = [int(im_end), int(eot)]
