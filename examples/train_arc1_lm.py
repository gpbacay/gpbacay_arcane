#!/usr/bin/env python3
"""Train the ARC 1 language model (``Arc1LanguageModel``) on raw text.

Plain next-token cross-entropy, no teacher needed, so it can use the whole
local corpus instead of the few hundred windows a CPU teacher dump yields.
Uses the same Qwen vocab adapter as distillation, so the chat server loads the
result unchanged. Warm-starts from existing weights when present.

    python examples/train_arc1_lm.py --time-budget-min 60
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpbacay_arcane.arc1 import Arc1Config, Arc1LanguageModel
from gpbacay_arcane.distillation import WarmupCosine
from gpbacay_arcane.language_model import make_causal_dataset
from gpbacay_arcane.qwen_vocab import QwenVocabAdapter
from gpbacay_arcane.tokenization import EOS_ID


def parse_args():
    p = argparse.ArgumentParser(description="Train Arc1LanguageModel on a text corpus")
    p.add_argument("--text-file", default="data/tinystories_valid.txt")
    p.add_argument("--config", default="Models/arc1_lm.config.json")
    p.add_argument("--checkpoint", default="Models/arc1_lm.weights.h5")
    p.add_argument("--vocab-adapter", default="Models/qwen_vocab_adapter.json")
    p.add_argument("--seq-len", type=int, default=128, help="Training window (<= config seq_len).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--peak-lr", type=float, default=2e-3)
    p.add_argument("--time-budget-min", type=float, default=60.0)
    p.add_argument("--fresh", action="store_true", help="Ignore existing weights.")
    return p.parse_args()


def encode_corpus(path: str, adapter: QwenVocabAdapter) -> np.ndarray:
    """Token ids for the corpus, cached beside it (encoding 19 MB takes a while)."""
    cache = path + ".arc1lm.npy"
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        return np.load(cache)
    with open(path, encoding="utf-8") as f:
        stories = f.read().split("<|endoftext|>")
    ids = []
    for story in stories:
        if story.strip():
            ids.extend(adapter.encode(story.strip(), add_bos=True, add_eos=True))
    arr = np.asarray(ids, dtype=np.int32)
    np.save(cache, arr)
    return arr


class TimeBudget(tf.keras.callbacks.Callback):
    def __init__(self, minutes: float):
        super().__init__()
        self.start = time.time()
        self.deadline = self.start + minutes * 60

    def on_train_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            rate = (batch + 1) / max(time.time() - self.start, 1e-6)
            print(f"step {batch:>6}  loss {logs['loss']:.3f}  {rate:.2f} steps/s", flush=True)
        if time.time() >= self.deadline:
            self.model.stop_training = True


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = Arc1Config.from_dict(json.load(f))
    model = Arc1LanguageModel(cfg).build_model()
    if os.path.exists(args.checkpoint) and not args.fresh:
        model.load_weights(args.checkpoint)
        print(f"warm start: {args.checkpoint}")
    print(f"Arc1LanguageModel: {model.count_params():,} params (~{model.count_params() * 4 / 1e6:.0f} MB)")

    adapter = QwenVocabAdapter.load(args.vocab_adapter)
    ids = encode_corpus(args.text_file, adapter)
    print(f"corpus: {ids.size:,} tokens")
    split = int(ids.size * 0.98)
    seq = min(args.seq_len, cfg.seq_len)
    train = make_causal_dataset(ids[:split], seq, args.batch_size)
    val = make_causal_dataset(ids[split:], seq, args.batch_size, shuffle=False).take(20)
    steps = int(train.cardinality()) * args.epochs

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            WarmupCosine(args.peak_lr, warmup_steps=min(200, steps // 10), total_steps=steps),
            weight_decay=0.01, clipnorm=1.0,
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    print(f"training {steps:,} steps of {args.batch_size}x{seq} (budget {args.time_budget_min} min)")
    model.fit(train, epochs=args.epochs, validation_data=val, callbacks=[TimeBudget(args.time_budget_min)],
              verbose=0)
    val_loss = model.evaluate(val, verbose=0)
    print(f"held-out loss {val_loss:.3f}  ppl {np.exp(val_loss):.1f}")
    model.save_weights(args.checkpoint)
    print(f"saved {args.checkpoint}")

    for prompt in ("Once upon a time", "hi"):
        out = model.generate(adapter.encode(prompt, add_bos=True), max_new_tokens=60, temperature=0.6,
                             top_k=20, eos_id=EOS_ID, allowed_token_ids=adapter.generation_ids())
        print(f"\n>>> {prompt}\n{adapter.decode(out)}")


if __name__ == "__main__":
    main()
