#!/usr/bin/env python3
"""Export ARC 1 (Resonant Schema Binding) to SavedModel and optional int8 TFLite.

Signatures (int32 ids are right-padded with 0; D = config "d_model"):
  schema  token_ids [S, Ts]                       -> engram [S, D]
          Run once per tool / parameter / option text and cache the result.
  decide  utter_ids [1, T], engram_a [P, D], engram_b [P, D], roles [P]
          -> fire [P], anchor_start [P, T], anchor_end [P, T], select_q [P, D], select_k [P, D]
          Raw logits; divide by config.json "calibration" temperatures before
          sigmoid (fire) / softmax (anchor, select). Enum logit = select_q . select_k / sqrt(D).
  embed   token_ids [B, T]                         -> embedding [B, D] (L2-normalised)

In the TFLite file the signature keys are signature_wrapper_serve_decide / _schema / _embed.

Example:
  python examples/export_arc1.py --config Models/arc1_arc1_tiny.config.json \
      --weights Models/arc1_arc1_tiny.weights.h5 --cycles 2 --out Models/arc1_export --tflite
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpbacay_arcane.arc1 import Arc1Config, Arc1Model


def parse_args():
    p = argparse.ArgumentParser(description="Export ARC 1 for on-device deployment")
    p.add_argument("--preset", default="arc1-tiny", choices=["arc1-tiny", "arc1"])
    p.add_argument("--config", default=None)
    p.add_argument("--weights", default=None)
    p.add_argument("--cycles", type=int, default=None, help="Binding cycles baked into the export")
    p.add_argument("--out", default="Models/arc1_export")
    p.add_argument("--tflite", action="store_true", help="Also write int8 TFLite if possible")
    p.add_argument("--rcn", default=None, help="Also write an .rcn container to this path")
    p.add_argument("--quant", default="rq4", choices=["f16", "rq8", "rq4"], help=".rcn weight format")
    p.add_argument("--tokenizer", default=None, help="Tokenizer JSON to embed in the .rcn file")
    return p.parse_args()


def main():
    args = parse_args()
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            config = Arc1Config.from_dict(json.load(f))
    else:
        config = Arc1Config.from_preset(args.preset)
    if args.cycles is not None:
        config = config.with_cycles(args.cycles)

    model = Arc1Model(config)
    model.build_model()
    if args.weights and os.path.exists(args.weights):
        model.load_weights(args.weights)
        print(f"loaded {args.weights}")
    else:
        print("WARNING: no --weights given; exporting an untrained model")

    d = config.d_model
    cycles = config.resolve_cycles()

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.int32, name="token_ids")])
    def serve_schema(token_ids):
        return {"engram": model.schema_engrams(token_ids, training=False)}

    @tf.function(input_signature=[
        tf.TensorSpec([1, None], tf.int32, name="utter_ids"),
        tf.TensorSpec([None, d], tf.float32, name="engram_a"),
        tf.TensorSpec([None, d], tf.float32, name="engram_b"),
        tf.TensorSpec([None], tf.int32, name="roles"),
    ])
    def serve_decide(utter_ids, engram_a, engram_b, roles):
        out = model.decide(utter_ids, tf.zeros_like(roles), engram_a, engram_b, roles,
                           cycles=cycles, training=False)
        return {k: out[k] for k in ("fire", "anchor_start", "anchor_end", "select_q", "select_k")}

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.int32, name="token_ids")])
    def serve_embed(token_ids):
        return {"embedding": model.embed_text(token_ids, training=False)}

    os.makedirs(args.out, exist_ok=True)
    saved = os.path.join(args.out, "saved_model")
    tf.saved_model.save(
        model,
        saved,
        signatures={
            "serving_default": serve_decide,
            "decide": serve_decide,
            "schema": serve_schema,
            "embed": serve_embed,
        },
    )
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"SavedModel -> {saved}  (binding cycles={cycles})")

    if args.rcn:
        from gpbacay_arcane.rcn import save_rcn
        from gpbacay_arcane.tokenization import BytePairTokenizer

        tok = BytePairTokenizer.load(args.tokenizer) if args.tokenizer else None
        info = save_rcn(model, tok, args.rcn, quant=args.quant, cycles=args.cycles)
        print(f"RCN -> {args.rcn} ({info['bytes']:,} bytes, {info['quant']}, tokenizer={'yes' if tok else 'no'})")

    if args.tflite:
        try:
            # Convert the reloaded SavedModel's concrete functions so the weights are frozen
            # into the flatbuffer. Converting from the SavedModel path, or from the live Keras
            # model, leaves them as uninitialised resource variables and every output is NaN.
            loaded = tf.saved_model.load(saved)
            fns = [loaded.signatures[k] for k in ("decide", "schema", "embed")]
            converter = tf.lite.TFLiteConverter.from_concrete_functions(fns, loaded)
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
