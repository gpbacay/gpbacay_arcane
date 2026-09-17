#!/usr/bin/env python3
"""Interactive chat loop for ArcaneSmallLanguageModel.

The 100m preset is randomly initialized until you pretrain it. Replies will
be noise unless you pass --weights / --tokenizer from a training run.
Use --preset tiny for a faster CPU loop.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpbacay_arcane.language_model import ArcaneSmallLanguageModel
from gpbacay_arcane.tokenization import BASE_VOCAB, EOS_ID, BytePairTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with an ARCANE small language model")
    parser.add_argument("--preset", default="tiny", choices=["tiny", "100m"])
    parser.add_argument("--weights", default=None, help="Path to saved .weights.h5")
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer JSON from training")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def _print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def main():
    args = parse_args()
    print(f"Building ARCANE SLM ({args.preset})...")
    if args.preset == "100m":
        print("This allocates ~100M weights and is slow on CPU.")
    model = ArcaneSmallLanguageModel.from_preset(args.preset)
    model.build_model()
    if args.weights:
        model.load_weights(args.weights)
        print(f"loaded weights: {args.weights}")
    else:
        print("No --weights given: replies will be untrained noise.")

    if args.tokenizer and os.path.exists(args.tokenizer):
        tokenizer = BytePairTokenizer.load(args.tokenizer)
    else:
        tokenizer = BytePairTokenizer(vocab_size=max(model.slm_config.vocab_size, BASE_VOCAB))

    print("Type a message. Commands: /reset  /quit")
    history = ""
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in {"/quit", "/exit"}:
            break
        if user == "/reset":
            history = ""
            print("(context cleared)")
            continue
        history = f"{history}{user}\n" if history else user
        prompt_ids = tokenizer.encode(history, add_bos=True)
        out_ids = model.generate(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_id=EOS_ID,
            allowed_token_ids=tokenizer.generation_ids(printable_only=True),
        )
        reply = tokenizer.decode(out_ids[len(prompt_ids) :]).strip() or "(empty)"
        history = f"{history}{reply}\n"
        _print(f"ARCANE: {reply}")


if __name__ == "__main__":
    main()
