#!/usr/bin/env python3
"""Export an ARC 1 ladder slice to SavedModel and optional int8 TFLite.

Signatures (token_ids: int32 [batch, seq], right-padded with 0):
  decide  -> noul [B], choice [B], span_start [B, T], span_end [B, T]  (raw logits;
             divide by config.json "calibration" temperatures before sigmoid/softmax)
  logits  -> LM logits [B, T, V]
  embed   -> L2-normalised embedding [B, D] (mean over non-pad tokens)

Example:
  python examples/export_arc1.py --config Models/arc1_arc1_tiny.config.json       --weights Models/arc1_arc1_tiny.weights.h5 --layers 2 --out Models/arc1_export
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpbacay_arcane.arc1 import Arc1Config, Arc1Model


def parse_args():
    p = argparse.ArgumentParser(description="Export ARC 1 for on-device deployment")
    p.add_argument("--preset", default="arc1-tiny", choices=["arc1-tiny", "arc1"])
    p.add_argument("--config", default=None)
    p.add_argument("--weights", default=None)
    p.add_argument("--layers", type=int, default=None, help="Ladder depth to activate")
    p.add_argument("--out", default="Models/arc1_export")
    p.add_argument("--tflite", action="store_true", help="Also write int8 TFLite if possible")
    return p.parse_args()


def main():
    args = parse_args()
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            config = Arc1Config.from_dict(json.load(f))
    else:
        config = Arc1Config.from_preset(args.preset)

    if args.layers is not None:
        config = config.with_depth(args.layers)

    model = Arc1Model(config)
    model.build_model()
    if args.weights and os.path.exists(args.weights):
        model.load_weights(args.weights)
        print(f"loaded {args.weights}")
    else:
        print("WARNING: no --weights given; exporting an untrained model")

    os.makedirs(args.out, exist_ok=True)
    spec = [tf.TensorSpec([None, None], tf.int32, name="token_ids")]

    @tf.function(input_signature=spec)
    def serve_decide(token_ids):
        out = model.decide(token_ids, training=False)
        return {k: out[k] for k in ("noul", "choice", "span_start", "span_end")}

    @tf.function(input_signature=spec)
    def serve_logits(token_ids):
        return {"logits": model(token_ids, training=False)}

    @tf.function(input_signature=spec)
    def serve_embed(token_ids):
        return {"embedding": model.embed_text(token_ids, training=False)}

    saved = os.path.join(args.out, "saved_model")
    tf.saved_model.save(
        model,
        saved,
        signatures={
            "serving_default": serve_decide,
            "decide": serve_decide,
            "logits": serve_logits,
            "embed": serve_embed,
        },
    )
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"SavedModel -> {saved}")
    print(f"active_depth={config.resolve_depth()} blocks={config.ladder_block_indices()}")

    if args.tflite:
        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(saved)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            tflite_path = os.path.join(args.out, "arc1_int8.tflite")
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            print(f"TFLite -> {tflite_path} ({len(tflite_model):,} bytes)")
        except Exception as exc:  # noqa: BLE001
            print(f"TFLite export skipped: {exc}")


if __name__ == "__main__":
    main()
