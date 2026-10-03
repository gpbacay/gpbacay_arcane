#!/usr/bin/env python3
"""Export ARC 1 to ONNX for the browser (node/web.mjs, onnxruntime-web).

Writes one graph, the same single forward pass Arc1Agent runs (Arc1Model.decide_joint):
  inputs   ids [N, T] int32 (row 0 utterance, rows 1.. uncached schema texts),
           bank [C, D] float32, probe_a [P] int32, probe_b [P] int32 (-1 = none), roles [P] int32
  outputs  fire [P], anchor_start [P, T], anchor_end [P, T], select_q [P, D], select_k [P, D],
           new_engrams [N-1, D], embedding [1, D]
plus arc1.json (config with calibration, tokenizer merges). Then checks ONNX against TensorFlow.

    pip install tf2onnx onnxruntime
    python examples/export_arc1_onnx.py [--model path.rcn] [--out node/web]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
import tf2onnx
from tf2onnx import utils

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpbacay_arcane.arc1_codec import pad_batch
from gpbacay_arcane.rcn import ARC1_TINY, load_rcn

OUTPUTS = ["fire", "anchor_start", "anchor_end", "select_q", "select_k", "new_engrams", "embedding"]


def erfc_handler(ctx, node, name, args):
    """tf2onnx has no Erfc (exact GELU uses it); erfc(x) = 1 - erf(x)."""
    erf = ctx.make_node("Erf", [node.input[0]])
    one = ctx.make_const(utils.make_name("one"), np.array(1, np.float32))
    node.type = "Sub"
    ctx.replace_inputs(node, [one.output[0], erf.output[0]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=ARC1_TINY)
    ap.add_argument("--out", default="node/web")
    args = ap.parse_args()

    model, tok, _ = load_rcn(args.model)
    cfg = model.arc1_config
    d = cfg.d_model

    spec = [
        tf.TensorSpec([None, None], tf.int32, name="ids"),
        tf.TensorSpec([None, d], tf.float32, name="bank"),
        tf.TensorSpec([None], tf.int32, name="probe_a"),
        tf.TensorSpec([None], tf.int32, name="probe_b"),
        tf.TensorSpec([None], tf.int32, name="roles"),
    ]

    @tf.function(input_signature=spec)
    def joint(ids, bank, probe_a, probe_b, roles):
        out = model.decide_joint(ids, bank, probe_a, probe_b, roles, training=False)
        field, mask = model.perceive(ids[:1], training=False)  # ponytail: perceives row 0 twice; fold into decide_joint if latency matters
        out["embedding"] = model.pooled_embedding(field, mask)
        return {k: out[k] for k in OUTPUTS}

    os.makedirs(args.out, exist_ok=True)
    onnx_path = os.path.join(args.out, "arc1.onnx")
    tf2onnx.convert.from_function(
        joint, input_signature=spec, opset=17, output_path=onnx_path,
        custom_op_handlers={"Erfc": (erfc_handler, ())},
    )
    with open(os.path.join(args.out, "arc1.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg.to_dict(), "cycles": cfg.resolve_cycles(),
                   "tokenizer": {"vocab_size": tok.vocab_size, "merges": [list(m) for m in tok.merges]}}, f)
    print(f"ONNX -> {onnx_path} ({os.path.getsize(onnx_path):,} bytes)")

    # Parity: a tool call with a cached bank row, a context probe, and an enum option.
    import onnxruntime as ort

    texts = ["what's the weather in Tokyo tomorrow?", "get weather. Get the current weather.", "city (string). City name",
             "unit (string, optional). Unit", "celsius"]
    ids = pad_batch([[3] + tok.encode(t) for t in texts], max_len=cfg.seq_len)
    bank = model.schema_engrams(tf.constant(pad_batch([[3] + tok.encode("fahrenheit")])), training=False).numpy()
    feeds = {"ids": ids, "bank": bank, "probe_a": np.array([1, 2, 3, 4, 0], np.int32),
             "probe_b": np.array([-1, 1, 1, 3, 3], np.int32), "roles": np.array([0, 1, 3, 4, 4], np.int32)}
    ref = joint(**{k: tf.constant(v) for k, v in feeds.items()})
    want = [ref[k].numpy() for k in OUTPUTS]
    got = ort.InferenceSession(onnx_path).run(OUTPUTS, feeds)
    for k, w, g in zip(OUTPUTS, want, got):
        live = w > -1e8  # masked anchor logits are -1e9 in both; compare real values only
        err = float(np.abs(w[live] - g[live]).max())
        print(f"  {k:13s} {str(w.shape):12s} max|tf-onnx| = {err:.2e}")
        # The graded spike rounds, so TF kernels with different reduction orders (and ONNX) can flip a
        # value sitting on a .5 boundary: ~1e-5 usually, up to ~6e-2 on a flip (logits span +-20).
        # A broken conversion is off by O(1).
        assert w.shape == g.shape and err < 1e-1, k
    print("parity ok")


if __name__ == "__main__":
    main()
