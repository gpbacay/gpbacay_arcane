#!/usr/bin/env python3
"""Export a trained ARC 1 LM as a weight-quantized TFLite model.

This exports the full-logit forward pass at a fixed sequence length. It is a
portable correctness/size artifact and a prefill runtime; the TensorFlow chat
server should continue to use ``Arc1LanguageModel.decode_step`` for cached
token-by-token generation.

Example::

    python examples/export_arc1_lm.py \
      --config Models/arc1_lm_100m.config.json \
      --weights Models/arc1_lm_100m.weights.h5 \
      --quant dynamic-int8 --seq-len 256 \
      --out Models/arc1_lm_100m_export
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpbacay_arcane.arc1 import Arc1Config, Arc1LanguageModel


def parse_args():
    parser = argparse.ArgumentParser(description="Export and validate an ARC 1 LM TFLite model")
    parser.add_argument("--preset", default="arc1-lm-100m")
    parser.add_argument("--config", default=None)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--vocab-adapter", default=None)
    parser.add_argument("--out", default="Models/arc1_lm_100m_export")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--quant", choices=["dynamic-int8", "float16", "none"], default="dynamic-int8")
    parser.add_argument("--validation-runs", type=int, default=5)
    parser.add_argument("--max-mean-error", type=float, default=0.15)
    parser.add_argument("--min-top1-agreement", type=float, default=0.90)
    return parser.parse_args()


def _load_config(args) -> Arc1Config:
    if args.config:
        with open(args.config, encoding="utf-8") as handle:
            return Arc1Config.from_dict(json.load(handle))
    return Arc1Config.from_preset(args.preset)


def convert_to_tflite(model, seq_len: int, quant: str) -> bytes:
    @tf.function(input_signature=[tf.TensorSpec([1, seq_len], tf.int32, name="token_ids")])
    def serve(token_ids):
        return {"logits": model(token_ids, training=False)}

    # Freeze first.  Passing a subclassed Keras model as ``trackable_obj`` to
    # the TF 2.19 converter can leave resource reads unresolved and produce an
    # otherwise valid-looking FlatBuffer whose logits are all NaN.
    concrete = convert_variables_to_constants_v2(serve.get_concrete_function())
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    if quant == "dynamic-int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quant == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def validate_tflite(model, payload: bytes, seq_len: int, vocab_size: int, runs: int) -> dict:
    interpreter = tf.lite.Interpreter(model_content=payload)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError("expected exactly one TFLite input and output")
    rng = np.random.default_rng(7)
    token_ids = rng.integers(4, vocab_size, size=(1, seq_len), dtype=np.int32)
    reference = model(token_ids, training=False).numpy()

    def invoke():
        interpreter.set_tensor(inputs[0]["index"], token_ids)
        interpreter.invoke()
        return interpreter.get_tensor(outputs[0]["index"])

    candidate = invoke()
    if not np.isfinite(reference).all():
        raise RuntimeError("TensorFlow reference produced non-finite logits")
    if not np.isfinite(candidate).all():
        raise RuntimeError("TFLite model produced non-finite logits")
    timings = []
    for _ in range(max(runs, 1)):
        started = time.perf_counter()
        candidate = invoke()
        timings.append(1000.0 * (time.perf_counter() - started))
    return {
        "mean_abs_logit_error": float(np.mean(np.abs(reference - candidate))),
        "max_abs_logit_error": float(np.max(np.abs(reference - candidate))),
        "top1_agreement": float(np.mean(np.argmax(reference, axis=-1) == np.argmax(candidate, axis=-1))),
        "median_prefill_ms": float(statistics.median(timings)),
    }


def main():
    args = parse_args()
    if not os.path.exists(args.weights):
        raise SystemExit(f"trained weights are required: {args.weights}")
    config = _load_config(args)
    if args.seq_len < 1 or args.seq_len > config.seq_len:
        raise SystemExit(f"--seq-len must be in [1, {config.seq_len}]")

    model = Arc1LanguageModel(config).build_model()
    model.load_weights(args.weights)
    print(f"loaded {args.weights} ({model.count_params():,} parameters)")
    payload = convert_to_tflite(model, args.seq_len, args.quant)
    metrics = validate_tflite(
        model, payload, args.seq_len, config.vocab_size, args.validation_runs,
    )
    if metrics["mean_abs_logit_error"] > args.max_mean_error:
        raise SystemExit(
            f"quantized mean logit error {metrics['mean_abs_logit_error']:.4f} exceeds "
            f"{args.max_mean_error:.4f}"
        )
    if metrics["top1_agreement"] < args.min_top1_agreement:
        raise SystemExit(
            f"quantized top-1 agreement {metrics['top1_agreement']:.2%} is below "
            f"{args.min_top1_agreement:.2%}"
        )

    os.makedirs(args.out, exist_ok=True)
    model_path = os.path.join(args.out, f"arc1_lm_{args.quant}.tflite")
    with open(model_path, "wb") as handle:
        handle.write(payload)
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)
    if args.vocab_adapter:
        shutil.copy2(args.vocab_adapter, os.path.join(args.out, "vocab_adapter.json"))

    report = {
        "format": "tflite",
        "quantization": args.quant,
        "parameters": int(model.count_params()),
        "bytes": len(payload),
        "sequence_length": args.seq_len,
        **metrics,
    }
    with open(os.path.join(args.out, "validation.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f"saved {model_path}")


if __name__ == "__main__":
    main()
