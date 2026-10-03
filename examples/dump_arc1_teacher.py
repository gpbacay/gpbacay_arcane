#!/usr/bin/env python3
"""Embed the ARC 1 distillation corpus with a frozen teacher (Qwen3-Embedding-0.6B).

Teacher side only (torch + transformers>=4.51, no TensorFlow). Reads
``texts.json`` written by ``python -m gpbacay_arcane.arc1_distill`` and writes
``teacher.npy`` (float16, L2-normalised, Matryoshka-truncated to ``--dim``).
All texts are embedded without an instruction prefix: the prefix would more than
double the tokens of these short texts, and CPU time is the bottleneck.

  python examples/dump_arc1_teacher.py --corpus data/arc1_teacher
  python examples/dump_arc1_teacher.py --corpus data/arc1_teacher --limit 1000   # speed check
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/arc1_teacher")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="Embed only the first N texts and report speed")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"], help="bfloat16 halves RAM")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    with open(os.path.join(a.corpus, "texts.json"), encoding="utf-8") as f:
        meta = json.load(f)
    inputs = meta["texts"]
    if a.limit:
        inputs = inputs[: a.limit]

    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    model = AutoModel.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).eval()
    # Length-sorted batches waste far less compute on padding.
    order = np.argsort([len(s) for s in inputs], kind="stable")
    out = np.zeros((len(inputs), a.dim), dtype=np.float16)
    partial = os.path.join(a.corpus, "teacher.partial.npz")
    start = 0
    if not a.limit and os.path.exists(partial):  # resume an interrupted dump
        saved = np.load(partial)
        out, start = saved["out"], int(saved["next"])
        print(f"[teacher] resuming at {start:,}", flush=True)
    t0 = time.time()
    with torch.inference_mode():
        for b in range(start, len(order), a.batch_size):
            idx = order[b: b + a.batch_size]
            enc = tok([inputs[i] for i in idx], padding=True, truncation=True, max_length=a.max_length, return_tensors="pt")
            hidden = model(**enc).last_hidden_state[:, -1]  # left padding: last position is the last token
            vec = F.normalize(hidden[:, : a.dim], dim=-1)
            out[idx] = vec.float().numpy().astype(np.float16)
            done = b + len(idx)
            if not a.limit and (b // a.batch_size) % 100 == 99:
                np.savez(partial, out=out, next=done)
            if (b // a.batch_size) % 50 == 0 or done == len(order):
                rate = (done - start) / (time.time() - t0)
                print(f"[teacher] {done:,}/{len(order):,} texts  {rate:.1f}/s  eta {(len(order) - done) / rate / 60:.1f}m", flush=True)
    if a.limit:
        print(f"[teacher] speed check only ({a.limit} texts); nothing written")
        return
    np.save(os.path.join(a.corpus, "teacher.npy"), out)
    if os.path.exists(partial):
        os.remove(partial)
    print(f"[teacher] wrote {os.path.join(a.corpus, 'teacher.npy')} {out.shape} in {(time.time() - t0) / 60:.1f}m")


if __name__ == "__main__":
    main()
