#!/usr/bin/env python3
"""Train ARC 1's decision heads, calibrate them, and evaluate end to end.

Trains noul (tool relevance / presence / booleans), span (copied arguments),
choice (enums), and echo embeddings on synthetic tool-calling and extraction
data, samples a ladder depth per step, fits per-head temperatures on held-out
values, and writes weights, config (with calibration), tokenizer, and metrics.

  python examples/train_arc1.py --preset arc1-tiny --steps 3000
  python examples/train_arc1.py --eval-only            # re-evaluate saved weights
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from gpbacay_arcane.arc1 import Arc1Config, Arc1Model
from gpbacay_arcane.arc1_codec import Arc1Codec
from gpbacay_arcane.arc1_data import build_tool_library, sample_extract_example, sample_tool_example
from gpbacay_arcane.arc1_train import TrainConfig, calibrate, full_evaluation, save_artifacts, train
from gpbacay_arcane.tokenization import BASE_VOCAB, BytePairTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Train ARC 1 automation model")
    p.add_argument("--preset", default="arc1-tiny", choices=["arc1-tiny", "arc1"])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--n-tool", type=int, default=12, help="Tool-calling examples per step")
    p.add_argument("--n-extract", type=int, default=4, help="Extraction examples per step")
    p.add_argument("--n-embed", type=int, default=8, help="Embedding paraphrase pairs per step")
    p.add_argument("--out-dir", default="Models")
    p.add_argument("--resume", action="store_true", help="Continue from saved weights")
    p.add_argument("--eval-only", action="store_true", help="Skip training; calibrate + evaluate saved weights")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--eval-examples", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def make_tokenizer(config: Arc1Config) -> BytePairTokenizer:
    tokenizer = BytePairTokenizer(vocab_size=max(config.vocab_size, BASE_VOCAB))
    if config.vocab_size > BASE_VOCAB + 64:
        # BPE merges over a sample of the synthetic corpus (arc1 preset only).
        rng, lib = random.Random(0), build_tool_library()
        texts = [sample_tool_example(rng, lib).user for _ in range(3000)]
        texts += [sample_extract_example(rng).text for _ in range(1000)]
        texts += [f"TOOL {t.name}: {d}" for t in lib.values() for d in t.descriptions]
        tokenizer.train(["\n".join(texts)], max_chars=200_000)
    return tokenizer


def main():
    args = parse_args()
    config = Arc1Config.from_preset(args.preset)
    prefix = os.path.join(args.out_dir, f"arc1_{args.preset.replace('-', '_')}")
    os.makedirs(args.out_dir, exist_ok=True)

    tok_path = prefix + "_tokenizer.json"
    if (args.resume or args.eval_only) and os.path.exists(tok_path):
        tokenizer = BytePairTokenizer.load(tok_path)
    else:
        tokenizer = make_tokenizer(config)

    model = Arc1Model(config)
    model.build_model()
    print("=== ARC 1 ===")
    print(f"preset: {args.preset}  parameters: {model.count_params():,}")
    print(f"ladder: { {d: config.ladder_block_indices(d) for d in config.ladder_depths} }")
    if args.build_only:
        return

    if args.eval_only:
        model.load_weights(prefix + ".weights.h5")
        train_info = {}
    else:
        tc = TrainConfig(
            steps=args.steps,
            learning_rate=args.learning_rate,
            n_tool=args.n_tool,
            n_extract=args.n_extract,
            n_embed=args.n_embed,
            seed=args.seed,
        )
        tokenizer.save(tok_path)
        train_info = train(model, tokenizer, tc, prefix, resume=args.resume)

    lib = build_tool_library()
    calibration = calibrate(model, Arc1Codec(tokenizer, config.seq_len), lib)
    print(f"[arc1] calibration: {json.dumps(calibration)}", flush=True)
    evaluation = full_evaluation(model, tokenizer, n_tool=args.eval_examples, depths=sorted(set(config.ladder_depths)))
    print(json.dumps(evaluation, indent=2), flush=True)

    metrics = {
        "preset": args.preset,
        "parameters": int(model.count_params()),
        "calibration": calibration,
        "evaluation": evaluation,
        "train": {k: v for k, v in train_info.items() if k != "history"},
        "history_tail": train_info.get("history", [])[-5:],
    }
    paths = save_artifacts(model, tokenizer, prefix, metrics)
    for k, v in paths.items():
        print(f"saved {k}: {v}")


if __name__ == "__main__":
    main()
