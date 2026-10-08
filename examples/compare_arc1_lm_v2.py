#!/usr/bin/env python3
"""Reproducible ARC 1 LM v1/v2 and optional LFM2.5-230M comparison.

The default run is offline and compares built architecture, parameter memory,
prefill latency, and one-token decode latency.  ``--with-lfm`` downloads and
benchmarks LiquidAI/LFM2.5-230M through Transformers.  Quality is intentionally
not reported for an untrained v2 model; pass ``--v2-weights`` to evaluate a
learned checkpoint in a separate downstream benchmark.

    python examples/compare_arc1_lm_v2.py --json Models/arc1_lm_v2_compare.json
    python examples/compare_arc1_lm_v2.py --with-lfm --lengths 64 256
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf

from gpbacay_arcane.arc1 import Arc1LanguageModel


LFM25_230M_SPEC = {
    "parameters": 230_000_000,
    "layers": 14,
    "hidden_size": 1024,
    "vocab_size": 65_536,
    "conv_blocks": 8,
    "attention_blocks": 6,
    "query_heads": 16,
    "kv_heads": 8,
    "context": 32_768,
    "source": "https://huggingface.co/LiquidAI/LFM2.5-230M",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lengths", type=int, nargs="+", default=[64, 256])
    p.add_argument("--runs", type=int, default=7)
    p.add_argument("--v1-weights", default="Models/arc1_lm.weights.h5")
    p.add_argument("--v2-weights", default=None)
    p.add_argument("--with-lfm", action="store_true")
    p.add_argument("--lfm-model", default="LiquidAI/LFM2.5-230M")
    p.add_argument("--json", default=None)
    return p.parse_args()


def median_ms(fn, runs):
    fn()
    values = []
    for _ in range(runs):
        started = time.perf_counter()
        fn()
        values.append(1000.0 * (time.perf_counter() - started))
    return statistics.median(values)


def arc1_result(preset, weights, lengths, runs):
    model = Arc1LanguageModel.from_preset(preset).build_model()
    learned = bool(weights and os.path.exists(weights))
    if learned:
        model.load_weights(weights)
    cfg = model.slm_config
    row = {
        "parameters": int(model.count_params()),
        "fp32_parameter_mb": model.count_params() * 4 / 1e6,
        "layers": cfg.num_layers,
        "hidden_size": cfg.d_model,
        "vocab_size": cfg.vocab_size,
        "context": cfg.seq_len,
        "learned_checkpoint": learned,
        "prefill_ms": {},
        "decode_ms": None,
    }
    for length in lengths:
        if length > cfg.seq_len:
            continue
        ids = tf.constant(np.random.default_rng(7).integers(4, cfg.vocab_size, (1, length)), tf.int32)
        row["prefill_ms"][str(length)] = median_ms(lambda: model(ids, training=False), runs)
    prompt_len = min(64, cfg.seq_len - 2)
    ids = tf.constant(np.random.default_rng(8).integers(4, cfg.vocab_size, (1, prompt_len)), tf.int32)
    if cfg.lm_architecture == "hybrid":
        _, state = model.prefill(ids)
        token = tf.constant([5], tf.int32)
        row["decode_ms"] = median_ms(lambda: model.decode_step(token, state), runs)
        row["decode_mode"] = "cached"
    else:
        token = tf.constant([[5]], tf.int32)
        extended = tf.concat([ids, token], axis=1)
        row["decode_ms"] = median_ms(lambda: model(extended, training=False), runs)
        row["decode_mode"] = "full_recompute"
    return row


def lfm_result(model_id, lengths, runs):
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit("--with-lfm requires torch and transformers>=5.0") from exc
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    except Exception as exc:
        raise SystemExit(
            "Could not load LFM2.5. Install transformers>=5.0 and retry; "
            f"underlying error: {exc}"
        ) from exc
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    row = dict(LFM25_230M_SPEC)
    row.update({"parameters": int(params), "fp32_parameter_mb": params * 4 / 1e6,
                "prefill_ms": {}, "decode_mode": "cached"})
    with torch.inference_mode():
        for length in lengths:
            ids = torch.randint(4, model.config.vocab_size, (1, length))
            row["prefill_ms"][str(length)] = median_ms(lambda: model(ids, use_cache=False), runs)
        ids = torch.randint(4, model.config.vocab_size, (1, min(64, max(lengths))))
        first = model(ids, use_cache=True)
        token = torch.tensor([[5]])
        cache = first.past_key_values
        row["decode_ms"] = median_ms(lambda: model(token, past_key_values=cache, use_cache=True), runs)
    return row


def main():
    args = parse_args()
    results = {
        "arc1_lm_v1": arc1_result("arc1-lm", args.v1_weights, args.lengths, args.runs),
        "arc1_lm_v2": arc1_result("arc1-lm-v2", args.v2_weights, args.lengths, args.runs),
        "lfm2_5_230m": dict(LFM25_230M_SPEC),
    }
    if args.with_lfm:
        results["lfm2_5_230m"] = lfm_result(args.lfm_model, args.lengths, args.runs)
    print(json.dumps(results, indent=2))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    if not results["arc1_lm_v2"]["learned_checkpoint"]:
        print("\nNOTE: v2 has random weights; these are architecture/runtime results, not a quality claim.")


if __name__ == "__main__":
    main()
