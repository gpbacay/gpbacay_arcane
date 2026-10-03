#!/usr/bin/env python3
"""Train ARC 1 (Resonant Schema Binding), calibrate it, and evaluate end to end.

Trains the fire (tool fires / optional argument present / boolean), anchor
(grounded copy pointer), and select (enum / classification label) readouts plus
utterance embeddings on synthetic tool-calling, extraction, and classification data. A binding cycle count is
sampled per step, readout temperatures are fitted on held-out values, and
weights, config (with calibration), tokenizer, and metrics are written.

  python examples/train_arc1.py --preset arc1-tiny --steps 4000
  python examples/train_arc1.py --eval-only            # re-evaluate saved weights
  python examples/train_arc1.py --teacher-cache data/arc1_teacher   # distilled A -> B -> C recipe
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# oneDNN speeds up batched training (~1.8x) but slows single-request inference (~2x) at
# this size, so evaluation, whose latency numbers should match serving, runs without it.
if "--eval-only" in sys.argv:
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from gpbacay_arcane.arc1 import Arc1Config, Arc1Model
from gpbacay_arcane.arc1_codec import Arc1Codec, tool_text
from gpbacay_arcane.arc1_data import build_tool_library, sample_extract_example, sample_tool_example
from gpbacay_arcane.arc1_train import TrainConfig, calibrate, full_evaluation, save_artifacts, train
from gpbacay_arcane.tokenization import BASE_VOCAB, BytePairTokenizer
from gpbacay_arcane.tools import Arc1Agent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINC = os.path.join(ROOT, "data", "external", "clinc150_data_full.json")


def parse_args():
    p = argparse.ArgumentParser(description="Train ARC 1 automation model")
    p.add_argument("--preset", default="arc1-tiny", choices=["arc1-tiny", "arc1"])
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--n-tool", type=int, default=16, help="Tool-calling examples per step")
    p.add_argument("--n-extract", type=int, default=6, help="Extraction examples per step")
    p.add_argument("--n-embed", type=int, default=8, help="Embedding paraphrase pairs per step")
    p.add_argument("--n-classify", type=int, default=6, help="Classification examples per step")
    p.add_argument("--out-dir", default="Models")
    p.add_argument("--resume", action="store_true", help="Continue from saved weights")
    p.add_argument("--eval-only", action="store_true", help="Skip training; calibrate + evaluate saved weights")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--skip-eval", action="store_true", help="Train and save only; run --eval-only afterwards")
    p.add_argument("--eval-examples", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--teacher-cache", default=None,
                   help="Teacher embeddings dir (gpbacay_arcane/arc1_distill.py): stages A, B, then --steps of stage C")
    p.add_argument("--steps-a", type=int, default=3000, help="Stage A: contrastive request->tool + distillation")
    p.add_argument("--steps-b", type=int, default=1000, help="Stage B: stage A with teacher-mined hard negatives")
    p.add_argument("--start-stage", default="A", choices=["A", "B", "C"], help="Resume a distilled run at this stage")
    p.add_argument("--distill-weight", type=float, default=0.5, help="Stage C weight of the teacher distillation loss")
    p.add_argument("--replay-weight", type=float, default=0.5, help="Stage C weight of the contrastive replay loss")
    return p.parse_args()


def make_tokenizer(config: Arc1Config) -> BytePairTokenizer:
    tokenizer = BytePairTokenizer(vocab_size=max(config.vocab_size, BASE_VOCAB))
    if config.vocab_size > BASE_VOCAB + 64:
        # BPE merges over a sample of the synthetic corpus (arc1 preset only).
        rng, lib = random.Random(0), build_tool_library()
        texts = [sample_tool_example(rng, lib).user for _ in range(3000)]
        texts += [sample_extract_example(rng).text for _ in range(1000)]
        texts += [tool_text(t.name, d) for t in lib.values() for d in t.descriptions]
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
    print(f"architecture: Resonant Schema Binding  layers={config.num_layers}  cycles={config.binding_cycles}")
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
            n_classify=args.n_classify,
            seed=args.seed,
        )
        tokenizer.save(tok_path)
        if args.teacher_cache:
            from gpbacay_arcane.arc1_distill import DistillConfig, train_distilled
            dc = DistillConfig(steps_a=args.steps_a, steps_b=args.steps_b,
                               distill_weight_c=args.distill_weight, replay_weight=args.replay_weight)
            train_info = train_distilled(model, tokenizer, tc, dc, args.teacher_cache, prefix, args.start_stage)
        else:
            train_info = train(model, tokenizer, tc, prefix, resume=args.resume)
        if args.skip_eval:
            model.save_weights(prefix + ".weights.h5")
            print(f"saved weights: {prefix}.weights.h5 (run --eval-only to calibrate + evaluate)")
            return

    lib = build_tool_library()
    calibration = calibrate(model, Arc1Codec(tokenizer, config.seq_len, config.schema_len), lib)
    print(f"[arc1] calibration: {json.dumps(calibration)}", flush=True)
    evaluation = full_evaluation(model, tokenizer, n_tool=args.eval_examples,
                                 cycles_list=sorted({1, config.binding_cycles}))
    if os.path.exists(CLINC):  # real utterances, intents no training corpus contained
        from gpbacay_arcane.arc1_distill import evaluate_real_intents
        agent = Arc1Agent(model, tokenizer)
        evaluation["real_intents"] = {f"cycles_{c}": evaluate_real_intents(agent, CLINC, n=600, cycles=c)
                                      for c in sorted({1, config.binding_cycles})}
    print(json.dumps(evaluation, indent=2), flush=True)

    metrics = {
        "preset": args.preset,
        "architecture": "Resonant Schema Binding",
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
