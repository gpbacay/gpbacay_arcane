#!/usr/bin/env python3
"""Train or smoke-test an ARCANE causal small language model.

Default preset is the ~100M decoder. Use ``--preset tiny`` for a fast
forward/backward check. This script trains next-token prediction; it does
not download a web-scale corpus. Point ``--text-file`` at your own data
for a real pretrain run.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpbacay_arcane.language_model import (
    ArcaneSLMConfig,
    ArcaneSmallLanguageModel,
    make_causal_dataset,
)
from gpbacay_arcane.tokenization import EOS_ID, BytePairTokenizer


def load_text(path: str, max_chars: int) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()[:max_chars]


def download_shakespeare(dest: str) -> str:
    import requests

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not os.path.exists(dest):
        print(f"Downloading Tiny Shakespeare to {dest}...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(dest, "w", encoding="utf-8") as f:
            f.write(response.text)
    return dest


def parse_args():
    parser = argparse.ArgumentParser(description="Train an ARCANE small language model")
    parser.add_argument("--preset", default="100m", choices=["tiny", "100m"])
    parser.add_argument("--text-file", default=None, help="UTF-8 corpus. Defaults to Tiny Shakespeare.")
    parser.add_argument("--max-chars", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many batches (smoke test).")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None, help="Window stride. Defaults to seq_len.")
    parser.add_argument("--byte-level", action="store_true", help="Skip BPE merges; use raw UTF-8 bytes.")
    parser.add_argument("--checkpoint", default="arcane_slm.weights.h5")
    parser.add_argument("--tokenizer-path", default="arcane_slm_tokenizer.json")
    parser.add_argument("--generate", default="To be, or not to be", help="Prompt after training.")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--build-only", action="store_true", help="Build, print param count, exit.")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {}
    if args.seq_len is not None:
        overrides["seq_len"] = args.seq_len
    if args.preset == "tiny":
        overrides.setdefault("dropout_rate", 0.0)

    config = ArcaneSLMConfig.from_preset(args.preset, **overrides)
    print("=== ARCANE Small Language Model ===")
    print(f"preset: {args.preset}")
    print(f"config: {config.to_dict()}")
    print(f"trainable estimate: {config.estimate_trainable_parameters():,}")

    model = ArcaneSmallLanguageModel(config)
    print("Building model (this allocates weights)...")
    model.build_model()
    trainable = int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
    total = int(model.count_params())
    print(f"trainable parameters: {trainable:,}")
    print(f"total parameters (incl. plastic kernels): {total:,}")

    if args.build_only:
        return

    text_path = args.text_file
    if text_path is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        bundled = os.path.join(data_dir, "shakespeare_small.txt")
        if os.path.exists(bundled):
            text_path = bundled
        else:
            text_path = download_shakespeare("shakespeare.txt")
    text = load_text(text_path, args.max_chars)
    print(f"corpus: {text_path} ({len(text):,} chars)")

    tokenizer = BytePairTokenizer(vocab_size=config.vocab_size)
    if not args.byte_level:
        tokenizer.train([text], max_chars=min(len(text), args.max_chars))
    tokenizer.save(args.tokenizer_path)
    print(f"tokenizer merges: {len(tokenizer.merges)}  saved: {args.tokenizer_path}")

    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    print(f"tokens: {len(token_ids):,}")
    stride = args.stride if args.stride is not None else config.seq_len
    dataset = make_causal_dataset(
        token_ids,
        seq_len=config.seq_len,
        batch_size=args.batch_size,
        stride=stride,
    )
    if args.max_steps:
        dataset = dataset.take(args.max_steps)

    model.compile_model(learning_rate=args.learning_rate)
    model.fit(dataset, epochs=args.epochs)

    if not args.skip_generate:
        seed = tokenizer.encode(args.generate, add_bos=True)
        generated = model.generate(
            seed,
            max_new_tokens=40,
            temperature=0.8,
            top_k=40,
            eos_id=EOS_ID,
            allowed_token_ids=tokenizer.generation_ids(printable_only=True),
        )
        sample = tokenizer.decode(generated)
        encoding = sys.stdout.encoding or "utf-8"
        print("=== sample ===")
        print(sample.encode(encoding, errors="replace").decode(encoding, errors="replace"))

    os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)) or ".", exist_ok=True)
    model.save_weights(args.checkpoint)
    print(f"saved weights: {args.checkpoint}")


if __name__ == "__main__":
    main()
