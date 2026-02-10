from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .api import expand_and_save
from .config import ExpansionOptions


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="adaptivocab-expand", description="Align model to expanded tokenizer.")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--base-tokenizer", required=True)
    ap.add_argument("--expanded-tokenizer", required=True)
    ap.add_argument("--output-dir", required=True)

    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--special-token-strategy", default="qwen3", choices=["qwen3", "none"])

    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-shared", type=int, default=2000)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--core-special-tokens", type=str, default=None)
    ap.add_argument("--json", action="store_true", help="Print report as JSON")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)

    opts = ExpansionOptions(
        alpha=args.alpha,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        special_token_strategy=args.special_token_strategy,
        verify=not args.no_verify,
        seed=args.seed,
        max_shared=args.max_shared,
        max_new=args.max_new,
        core_special_tokens=args.core_special_tokens,
    )

    report = expand_and_save(
        base_model=args.base_model,
        base_tokenizer=args.base_tokenizer,
        expanded_tokenizer=args.expanded_tokenizer,
        output_dir=args.output_dir,
        options=opts,
    )

    if args.json:
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    else:
        print(report)
