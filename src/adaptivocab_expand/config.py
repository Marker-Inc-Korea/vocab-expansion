from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


SpecialTokenStrategy = Literal["none", "qwen3"]


@dataclass
class ExpansionOptions:
    """Options for vocab expansion alignment.

    alpha:
        Controls the exponential weighting strength. The AdaptiVocab paper uses exp(±2i);
        default alpha=2.0 matches that. (Softmax is invariant to index offset.)
    dtype:
        "auto" | "fp16" | "bf16" | "fp32"
    device_map:
        e.g. "auto" or None.
    trust_remote_code:
        Passed to Hugging Face loaders.
    special_token_strategy:
        Built-in: "qwen3" (bos/pad=<|endoftext|>, eos=<|im_end|>).
        Use "none" if you want to keep model config unchanged.
    verify:
        Run verification after saving.
    """

    alpha: float = 2.0
    dtype: str = "auto"
    device_map: Optional[str] = None
    trust_remote_code: bool = False
    special_token_strategy: SpecialTokenStrategy = "qwen3"
    verify: bool = True

    # Verification knobs (keep reasonable defaults)
    seed: int = 0
    max_shared: int = 2000
    max_new: int = 200
    top_k_shared: int = 20
    top_k_new: int = 20
    top_k_skipped_new: int = 20
    core_special_tokens: Optional[str] = None  # comma-separated list


@dataclass
class ExpansionReport:
    base_vocab_size: int
    expanded_vocab_size: int
    shared_tokens: int
    mismatched_shared_tokens: int
    new_tokens: int
    new_tokens_inited: int
    new_tokens_fallback_mean: int
    embeddings_tied: bool
    output_dir: str
    verify_passed: Optional[bool] = None
