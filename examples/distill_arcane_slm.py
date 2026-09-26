#!/usr/bin/env python3
"""Distil dumped Qwen2.5-0.5B predictions into an ARCANE small language model.

Reads the TFRecord shards written by ``dump_qwen_logits.py`` and trains the
student on ``alpha * CE + (1 - alpha) * T^2 * KL(teacher || student)``.

Example::

    python examples/distill_arcane_slm.py \
        --shards "data/qwen_shards/*.tfrecord" \
        --preset distill --steps 20000 --batch-size 8

``--arch arc1`` distils into ``Arc1LanguageModel`` (ARC 1's perception stack
run causally, ~10M params) instead; it trains in minutes on CPU::

    python examples/distill_arcane_slm.py --arch arc1 --steps 1500

``--baseline-transformer`` swaps in a parameter-matched vanilla transformer
student instead. Run both on identical shards: if ARCANE tracks the baseline's
KL-to-teacher curve its mechanisms are free, and if it plateaus higher you have
measured exactly what they cost.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpbacay_arcane.distillation import ArcaneDistiller, read_distill_dataset
from gpbacay_arcane.language_model import ArcaneSLMConfig, ArcaneSmallLanguageModel
from gpbacay_arcane.tokenization import EOS_ID


def parse_args():
    p = argparse.ArgumentParser(description="Distil Qwen2.5-0.5B into ARCANE")
    p.add_argument("--shards", default="data/qwen_shards/*.tfrecord")
    p.add_argument("--meta", default=None, help="meta.json from the dump (defaults beside shards)")
    from gpbacay_arcane.language_model import SLM_PRESETS

    p.add_argument("--arch", default="slm", choices=["slm", "arc1"],
                   help="slm = ArcaneSmallLanguageModel, arc1 = Arc1LanguageModel.")
    p.add_argument("--preset", default=None,
                   help=f"slm: {sorted(SLM_PRESETS)} (default distill); arc1: arc1-lm (default)")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--peak-lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.4, help="Weight on the hard-label CE term.")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--holdout-shards", type=int, default=1,
                   help="Shards reserved for validation (0 = evaluate on training data).")
    p.add_argument("--time-budget-min", type=float, default=None,
                   help="Stop cleanly after this many minutes, saving what is trained.")
    p.add_argument("--checkpoint", default=None,
                   help="default Models/arcane_slm_distilled.weights.h5 (slm) or Models/arc1_lm.weights.h5 (arc1)")
    p.add_argument("--history", default=None)
    p.add_argument("--vocab-adapter", default="Models/qwen_vocab_adapter.json")
    p.add_argument("--warm-start-embedding", action="store_true",
                   help="Initialise the embedding from Qwen's, PCA-projected to d_model.")
    p.add_argument("--baseline-transformer", action="store_true",
                   help="Train a parameter-matched vanilla transformer control instead.")
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override linear-attention chunk size. seq_len collapses tf.scan to "
        "one iteration (identical output; typically faster on CPU).",
    )
    p.add_argument("--generate", default=None, help="Sample from this prompt after training.")
    args = p.parse_args()
    arc1 = args.arch == "arc1"
    args.preset = args.preset or ("arc1-lm" if arc1 else "distill")
    stem = "Models/arc1_lm" if arc1 else "Models/arcane_slm_distilled"
    args.checkpoint = args.checkpoint or f"{stem}.weights.h5"
    args.history = args.history or ("Models/arc1_lm_history.json" if arc1 else "Models/arcane_slm_distill_history.json")
    return args


def build_baseline(cfg: ArcaneSLMConfig) -> tf.keras.Model:
    """Vanilla pre-norm transformer with the same width/depth/vocab, as a control."""
    from gpbacay_arcane.mechanisms import CausalSoftmaxSelfAttention, RMSNorm

    class BaselineTransformer(tf.keras.Model):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.embed = tf.keras.layers.Embedding(cfg.vocab_size, cfg.d_model)
            self.blocks = []
            for i in range(cfg.num_layers):
                self.blocks.append(
                    (
                        CausalSoftmaxSelfAttention(
                            cfg.d_model, cfg.num_heads, cfg.dropout_rate,
                            use_rope=True, max_position=max(cfg.seq_len, 2048),
                            name=f"attn_{i}",
                        ),
                        tf.keras.layers.Dense(cfg.d_model * cfg.ffn_mult, activation="gelu"),
                        tf.keras.layers.Dense(cfg.d_model),
                        RMSNorm(name=f"norm_{i}"),
                    )
                )
            self.final_norm = RMSNorm(name="final_norm")

        def call(self, token_ids, training=False):
            x = self.embed(token_ids)
            for attn, up, down, norm in self.blocks:
                x = attn(x, training=training)
                x = norm(x + down(up(x)))
            x = self.final_norm(x)
            return tf.matmul(x, self.embed.embeddings, transpose_b=True)

    return BaselineTransformer(cfg)


def _save_config(checkpoint_path: str, cfg: ArcaneSLMConfig) -> str:
    """Write the student geometry beside its weights so the API can rebuild it.

    A distilled checkpoint matches no named preset once vocab_size and seq_len
    come from the dump, so serving needs the exact config.
    """
    base = checkpoint_path
    for suffix in (".weights.h5", ".h5", ".keras"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    path = base + ".config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    return path


def main():
    args = parse_args()
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("mixed precision: mixed_float16")

    meta_path = args.meta or os.path.join(os.path.dirname(args.shards), "meta.json")
    if not os.path.exists(meta_path):
        raise SystemExit(
            f"missing {meta_path}. Run examples/dump_qwen_logits.py first to produce shards."
        )
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    seq_len, top_k, vocab_size = meta["seq_len"], meta["top_k"], meta["vocab_size"]
    print(f"=== ARCANE <- Qwen distillation ===")
    print(f"shards: {args.shards}")
    print(f"teacher: {meta['model_id']} | windows {meta['windows']:,} | top_k {top_k}")
    print(f"seq_len {seq_len} | student vocab {vocab_size:,}")

    overrides = {"vocab_size": vocab_size, "seq_len": seq_len}
    if args.chunk_size is not None:
        overrides["chunk_size"] = args.chunk_size
    if args.arch == "arc1":
        from gpbacay_arcane.arc1 import Arc1Config, Arc1LanguageModel

        overrides.pop("chunk_size", None)
        cfg = Arc1Config.from_preset(args.preset, **overrides)
        student = Arc1LanguageModel(cfg)
        label = f"arc1-lm ({args.preset})"
    elif args.baseline_transformer:
        cfg = ArcaneSLMConfig.from_preset(args.preset, **overrides)
        student = build_baseline(cfg)
        label = "baseline-transformer"
    else:
        cfg = ArcaneSLMConfig.from_preset(args.preset, **overrides)
        student = ArcaneSmallLanguageModel(cfg)
        label = f"arcane-{args.preset}"
    student(tf.zeros((1, seq_len), dtype=tf.int32), training=False)
    trainable = int(np.sum([tf.keras.backend.count_params(w) for w in student.trainable_weights]))
    print(f"student: {label} | {trainable:,} trainable params (~{trainable * 4 / 1e6:.0f} MB fp32)")

    if args.warm_start_embedding and not args.baseline_transformer:
        from gpbacay_arcane.qwen_vocab import QwenVocabAdapter, project_qwen_embeddings

        adapter = QwenVocabAdapter.load(args.vocab_adapter)
        print("warm-starting embedding from the teacher (PCA projection)...")
        from transformers import AutoModelForCausalLM

        teacher = AutoModelForCausalLM.from_pretrained(
            meta["model_id"], low_cpu_mem_usage=True
        )
        # torch 2.0.1 built against numpy 1.x raises "Numpy is not available"
        # under numpy 2.x, so avoid the Tensor.numpy() bridge.
        weights = np.asarray(
            teacher.get_input_embeddings().weight.detach().cpu().tolist(),
            dtype=np.float32,
        )
        del teacher
        init = project_qwen_embeddings(weights, adapter, cfg.d_model)
        student.token_embedding.set_weights([init])
        print(f"  embedding warm-started from {weights.shape} -> {init.shape}")

    # Hold shards out so the reported perplexity is not measured on training data.
    all_shards = sorted(glob.glob(args.shards))
    if not all_shards:
        raise SystemExit(f"no shards matched {args.shards}")
    holdout = min(args.holdout_shards, max(len(all_shards) - 1, 0))
    train_files = all_shards[: len(all_shards) - holdout] if holdout else all_shards
    val_files = all_shards[len(all_shards) - holdout :] if holdout else all_shards
    if holdout:
        print(f"shards: {len(train_files)} train / {len(val_files)} held out for eval")
    else:
        print(f"shards: {len(train_files)} (eval runs on training data -- no holdout)")

    dataset = read_distill_dataset(train_files, seq_len, top_k, args.batch_size, shuffle=True)
    eval_ds = read_distill_dataset(val_files, seq_len, top_k, args.batch_size, shuffle=False)

    distiller = ArcaneDistiller(
        student,
        peak_lr=args.peak_lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        temperature=args.temperature,
    )

    history = []
    step = 0
    started = time.time()
    running = np.zeros(3)
    seen = 0
    print(f"\ntraining for {args.steps:,} steps (alpha={args.alpha}, T={args.temperature})\n")
    while step < args.steps:
        for inputs, labels, t_ids, t_vals, mask in dataset:
            total, ce, kd = distiller.train_step(inputs, labels, t_ids, t_vals, mask)
            running += [float(total), float(ce), float(kd)]
            seen += 1
            step += 1
            if step % args.log_every == 0:
                loss, ce_m, kd_m = running / seen
                lr = distiller.current_lr()
                rate = step / max(time.time() - started, 1e-6)
                print(
                    f"step {step:>6}/{args.steps}  loss {loss:.4f}  ce {ce_m:.4f} "
                    f"(ppl {np.exp(min(ce_m, 20)):8.2f})  kd {kd_m:.4f}  lr {lr:.2e}  "
                    f"{rate:.2f} steps/s",
                    flush=True,
                )
                running[:] = 0
                seen = 0
            if step % args.eval_every == 0 or step == args.steps:
                metrics = distiller.evaluate(eval_ds, max_batches=args.eval_batches)
                metrics["step"] = step
                history.append(metrics)
                print(
                    f"  [eval] step {step}  loss {metrics['loss']:.4f}  "
                    f"ce {metrics['ce']:.4f}  ppl {metrics['ppl']:.2f}  "
                    f"kd(teacher) {metrics['kd']:.4f}",
                    flush=True,
                )
                os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)) or ".", exist_ok=True)
                student.save_weights(args.checkpoint)
                _save_config(args.checkpoint, cfg)
            if args.time_budget_min is not None:
                elapsed_min = (time.time() - started) / 60.0
                if elapsed_min >= args.time_budget_min:
                    print(f"\ntime budget reached ({elapsed_min:.1f} min); stopping at step {step}")
                    metrics = distiller.evaluate(eval_ds, max_batches=args.eval_batches)
                    metrics["step"] = step
                    history.append(metrics)
                    print(
                        f"  [eval] step {step}  loss {metrics['loss']:.4f}  "
                        f"ce {metrics['ce']:.4f}  ppl {metrics['ppl']:.2f}  "
                        f"kd(teacher) {metrics['kd']:.4f}"
                    )
                    os.makedirs(
                        os.path.dirname(os.path.abspath(args.checkpoint)) or ".", exist_ok=True
                    )
                    student.save_weights(args.checkpoint)
                    _save_config(args.checkpoint, cfg)
                    step = args.steps
                    break
            if step >= args.steps:
                break

    os.makedirs(os.path.dirname(os.path.abspath(args.history)) or ".", exist_ok=True)
    with open(args.history, "w", encoding="utf-8") as f:
        json.dump({"student": label, "config": cfg.to_dict(), "history": history}, f, indent=2)
    config_path = _save_config(args.checkpoint, cfg)
    print(f"\nsaved weights:  {args.checkpoint}")
    print(f"saved config:   {config_path}")
    print(f"saved history:  {args.history}")
    if not args.baseline_transformer:
        print(
            "\nTo chat with it:\n"
            f"  set SLM_CONFIG_PATH={config_path}\n"
            f"  set SLM_WEIGHTS_PATH={args.checkpoint}\n"
            f"  set SLM_VOCAB_ADAPTER={args.vocab_adapter}\n"
            "  cd arcane-docs-web && npm run dev:with-slm    -> /docs/chat"
        )

    if args.generate and not args.baseline_transformer:
        from gpbacay_arcane.qwen_vocab import QwenVocabAdapter

        adapter = QwenVocabAdapter.load(args.vocab_adapter)
        seed = adapter.encode(args.generate, add_bos=True)
        out = student.generate(seed, max_new_tokens=60, temperature=0.8, top_k=40, eos_id=EOS_ID)
        print("\n=== sample ===")
        print(adapter.decode(out))


if __name__ == "__main__":
    main()
