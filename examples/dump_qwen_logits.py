#!/usr/bin/env python3
"""Dump Qwen2.5-0.5B top-k logits as TFRecord shards for ARCANE distillation.

Runs the teacher in PyTorch and writes sparse predictions in *student* id space,
so the training side never needs torch. Two passes over the corpus:

  1. build (or load) the trimmed student vocabulary from Qwen's BPE
  2. forward the teacher over causal windows, keep top-k per position

Example::

    python examples/dump_qwen_logits.py \
        --text-file data/corpus.txt --out-dir data/qwen_shards \
        --vocab-size 32000 --seq-len 512 --top-k 64

The teacher is Apache-2.0 licensed, so the dumped predictions and anything
distilled from them are yours to release.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpbacay_arcane.distillation import write_shard
from gpbacay_arcane.qwen_vocab import QWEN_MODEL_ID, QwenVocabAdapter, load_qwen_tokenizer
from gpbacay_arcane.tokenization import BOS_ID, EOS_ID


def parse_args():
    p = argparse.ArgumentParser(description="Dump Qwen teacher logits for distillation")
    p.add_argument("--text-file", required=True, help="UTF-8 corpus to distil over")
    p.add_argument("--out-dir", default="data/qwen_shards")
    p.add_argument("--model-id", default=QWEN_MODEL_ID)
    p.add_argument("--vocab-adapter", default="Models/qwen_vocab_adapter.json")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--windows-per-shard", type=int, default=512)
    p.add_argument("--max-windows", type=int, default=None, help="Cap total windows (CPU runs).")
    p.add_argument("--max-chars", type=int, default=None)
    p.add_argument("--stride", type=int, default=None, help="Window stride. Defaults to seq_len.")
    p.add_argument("--device", default=None, help="cuda / cpu. Auto-detected by default.")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--threads", type=int, default=None, help="Torch CPU threads.")
    p.add_argument("--time-budget-min", type=float, default=None,
                   help="Stop cleanly after this many minutes, flushing what is done.")
    p.add_argument("--rebuild-vocab", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(torch, args.dtype)
    print(f"=== Qwen teacher dump ===\nmodel: {args.model_id}\ndevice: {device} ({args.dtype})")
    if device == "cpu":
        print("NOTE: CPU inference is slow. Use --max-windows to keep the run bounded.")

    with open(args.text_file, encoding="utf-8") as f:
        text = f.read()
    if args.max_chars:
        text = text[: args.max_chars]
    print(f"corpus: {args.text_file} ({len(text):,} chars)")

    tokenizer = load_qwen_tokenizer(args.model_id)

    # --- student vocabulary ------------------------------------------------
    if os.path.exists(args.vocab_adapter) and not args.rebuild_vocab:
        adapter = QwenVocabAdapter.load(args.vocab_adapter, tokenizer=tokenizer)
        print(f"vocab adapter: loaded {args.vocab_adapter} (vocab_size={adapter.vocab_size})")
    else:
        print("building trimmed student vocabulary from corpus frequencies...")
        adapter = QwenVocabAdapter.build(
            [text], vocab_size=args.vocab_size, model_id=args.model_id, tokenizer=tokenizer
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.vocab_adapter)) or ".", exist_ok=True)
        adapter.save(args.vocab_adapter)
        print(
            f"vocab adapter: kept {adapter.vocab_size:,} of {adapter.qwen_vocab_size:,} "
            f"Qwen ids -> {args.vocab_adapter}"
        )

    # --- windows -----------------------------------------------------------
    qwen_ids = tokenizer.encode(text)
    student_ids = adapter.map_qwen_ids(qwen_ids)
    keep_mask = adapter.qwen_to_student[np.asarray(qwen_ids)] >= 0
    coverage = float(keep_mask.mean())
    print(f"tokens: {len(qwen_ids):,} | vocab coverage: {coverage:.4%} of occurrences mapped")

    seq_len = args.seq_len
    stride = args.stride or seq_len
    starts = list(range(0, max(len(qwen_ids) - seq_len - 1, 0), stride))
    if args.max_windows:
        starts = starts[: args.max_windows]
    if not starts:
        raise SystemExit(f"corpus too short: need > {seq_len + 1} tokens")
    print(f"windows: {len(starts):,} of length {seq_len} (stride {stride})")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    model.to(device).eval()
    torch.set_num_threads(args.threads or (os.cpu_count() or 8))

    # Teacher columns for Qwen ids that survived the trim; everything else is
    # dropped before top-k so the student is never asked to model a symbol it
    # has no embedding row for.
    kept_qwen = adapter.student_to_qwen.copy()
    valid_student = np.where(kept_qwen >= 0)[0]
    valid_qwen = kept_qwen[valid_student]
    gather_idx = torch.tensor(valid_qwen, dtype=torch.long, device=device)
    student_cols = torch.tensor(valid_student, dtype=torch.long, device=device)

    # Materialising all 151,936 logits costs ~622 MB per (4, 256) batch in fp32
    # and dominates runtime on a memory-constrained box. We only ever need the
    # kept columns, so slice the LM head once and project the backbone's hidden
    # states straight onto it.
    backbone = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    use_restricted_head = backbone is not None and lm_head is not None
    w_kept_t = None
    if use_restricted_head:
        with torch.no_grad():
            w_kept_t = lm_head.weight.index_select(0, gather_idx).t().contiguous()
        print(
            f"restricted LM head: {tuple(w_kept_t.shape)} "
            f"(from {tuple(lm_head.weight.shape)}) -- skips "
            f"{adapter.qwen_vocab_size - len(valid_qwen):,} unused columns"
        )
    else:
        print("WARNING: could not isolate the LM head; falling back to full logits")

    def to_numpy(tensor, dtype):
        """torch -> numpy without the .numpy() bridge.

        torch 2.0.1 is built against numpy 1.x and its ``Tensor.numpy()`` raises
        "Numpy is not available" under numpy 2.x. Going through ``tolist()``
        sidesteps the broken bridge; the arrays here are small next to a forward
        pass, so the extra copy costs nothing measurable.
        """
        return np.asarray(tensor.detach().cpu().tolist(), dtype=dtype)

    def kept_logits_for(tokens):
        """(B, T, V_kept) teacher logits over the student's vocabulary."""
        if use_restricted_head:
            hidden = backbone(tokens).last_hidden_state
            return torch.matmul(hidden.float(), w_kept_t.float())
        return model(tokens).logits.float().index_select(-1, gather_idx)

    os.makedirs(args.out_dir, exist_ok=True)
    buf_in, buf_lab, buf_ids, buf_val = [], [], [], []
    shard_idx = total = 0
    started = time.time()

    def flush():
        nonlocal shard_idx, buf_in, buf_lab, buf_ids, buf_val
        if not buf_in:
            return
        path = os.path.join(args.out_dir, f"shard_{shard_idx:05d}.tfrecord")
        write_shard(
            path,
            np.stack(buf_in),
            np.stack(buf_lab),
            np.stack(buf_ids),
            np.stack(buf_val),
        )
        size_mb = os.path.getsize(path) / 1e6
        print(f"  wrote {path} ({len(buf_in)} windows, {size_mb:.1f} MB)")
        shard_idx += 1
        buf_in, buf_lab, buf_ids, buf_val = [], [], [], []

    with torch.no_grad():
        for b0 in range(0, len(starts), args.batch_size):
            batch_starts = starts[b0 : b0 + args.batch_size]
            windows = np.stack([qwen_ids[s : s + seq_len] for s in batch_starts])
            tokens = torch.tensor(windows, dtype=torch.long, device=device)
            kept_logits = kept_logits_for(tokens)             # (B, T, V_kept)
            k = min(args.top_k, kept_logits.shape[-1])
            vals, idx = torch.topk(kept_logits, k=k, dim=-1)
            # idx indexes the kept table; translate to student ids.
            student_topk = student_cols[idx]

            x = adapter.map_qwen_ids(windows)
            y = np.stack(
                [adapter.map_qwen_ids(qwen_ids[s + 1 : s + seq_len + 1]) for s in batch_starts]
            )
            buf_in.extend(x)
            buf_lab.extend(y)
            buf_ids.extend(to_numpy(student_topk, np.int32))
            buf_val.extend(to_numpy(vals, np.float16))
            total += len(batch_starts)

            if len(buf_in) >= args.windows_per_shard:
                flush()
            if args.time_budget_min is not None:
                elapsed_min = (time.time() - started) / 60.0
                if elapsed_min >= args.time_budget_min:
                    print(f"  time budget reached ({elapsed_min:.1f} min); stopping early")
                    break
            if total % (args.batch_size * 10) == 0 or total == len(starts):
                rate = total / max(time.time() - started, 1e-6)
                eta = (len(starts) - total) / max(rate, 1e-6)
                print(
                    f"  {total}/{len(starts)} windows  {rate:.2f} win/s  eta {eta/60:.1f} min",
                    flush=True,
                )
    flush()

    meta = {
        "model_id": args.model_id,
        "seq_len": seq_len,
        "top_k": args.top_k,
        "vocab_size": adapter.vocab_size,
        "windows": total,
        "shards": shard_idx,
        "stride": stride,
        "vocab_adapter": args.vocab_adapter,
        "vocab_coverage": coverage,
    }
    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\ndone: {total:,} windows in {shard_idx} shards -> {args.out_dir}")
    print(f"meta: {meta_path}")


if __name__ == "__main__":
    main()
