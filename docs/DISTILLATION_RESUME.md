# Resuming the Qwen2.5-0.5B → ARCANE distillation

Status as of **2026-09-19 04:43**. Written for whoever picks this up next.
Architecture rationale lives in [DISTILLATION.md](DISTILLATION.md). This file
is *where things are and how to continue*.

---

## TL;DR

The teacher dump is done. Student training is **in progress** on this CPU box
(`distill-small`, 8.8M params, `chunk_size=256`). Do **not** redo the dump.
Do **not** start a second training job — one `python examples/distill_arcane_slm.py`
is already running (PID around 13020).

First held-out eval is in (step 250). Training continues toward the 120-minute budget.

```
  [eval] step 250  loss 2.4811  ce 3.6463  ppl 38.33  kd(teacher) 0.4261
```

Held-out ppl **38** vs train ppl **23** — some overfitting, as expected on 92k tokens.

As of step 150 the student was already learning:

```
step     25/4000  loss 5.2359  ce 8.6241 (ppl  5564.08)  kd 0.7443  0.08 steps/s
step     50/4000  loss 4.3461  ce 7.0492 (ppl  1151.92)  kd 0.6360  0.10 steps/s
step     75/4000  loss 3.6752  ce 5.8409 (ppl   344.09)  kd 0.5579  0.11 steps/s
step    100/4000  loss 3.2328  ce 5.1040 (ppl   164.68)  kd 0.4963  0.12 steps/s
step    125/4000  loss 2.8593  ce 4.4652 (ppl    86.94)  kd 0.4472  0.12 steps/s
step    150/4000  loss 2.6033  ce 4.0697 (ppl    58.54)  kd 0.4064  0.12 steps/s
```

Steady state is **~8.3 s/step** (0.12 steps/s), not the 16.7 s/step from the
previous run — that figure was an average that still included graph tracing.
A 120-minute `--time-budget-min` buys ~800–900 steps after compile, ~25–28
epochs over the 256 training windows. First checkpoint is written at
**step 250** (`--eval-every 250`).

---

## What is on disk right now

| Artifact | Path | State |
|---|---|---|
| Corpus | `data/tinystories_valid.txt` | 19.4 MB TinyStories |
| Vocab adapter | `Models/qwen_vocab_adapter.json` | 8,000 ids, **99.95%** coverage |
| Teacher shards | `data/qwen_shards/*.tfrecord` | **3 shards, 360 windows, 92,160 tokens** |
| Dump metadata | `data/qwen_shards/meta.json` | seq_len 256, top_k 32, vocab 8000 |
| Student checkpoint | `Models/arcane_slm_distilled.weights.h5` | written at step 250 (re-saved every eval) |
| Student config | `Models/arcane_slm_distilled.config.json` | on disk (`distill-small`, vocab 8000, seq 256) |
| Training log | `logs/distill.log` | UTF-16 (PowerShell `Tee-Object`); live copy also in the training terminal |
| History | `Models/arcane_slm_distill_history.json` | written at process exit |

**The teacher dump is the expensive part and it is already paid for** (~42 min
of CPU). Do not redo it unless you want more tokens.

---

## What changed since 03:45

The previous blocker was “16.7 s/step, no checkpoint”. Three things were done
before restarting training:

1. **`--chunk-size` CLI** on `examples/distill_arcane_slm.py`.
2. **`distill-small` default `chunk_size=256`** (one linear-attention chunk,
   identical output — `test_chunk_size_does_not_change_output`).
3. **CPU fast path** in `CausalLinearSelfAttention`: when the static sequence
   length fits in one chunk, skip `tf.scan` entirely
   (`test_single_chunk_fast_path_matches_scan`).

Isolated bench (`scratchpad/bench_step.py`) on this machine:

```
chunk=256 (single-chunk fast path)   7.36 s/step
chunk=64  (4-iteration tf.scan)      7.39 s/step
```

So `tf.scan` was **not** the 16.7 s/step. That number was compile + first
steps. Real training settled at **8.3 s/step**, matching the bench.

Hardware is still Intel UHD / TF 2.20 CPU-only / torch 2.0.1+cpu. No CUDA.

---

## If the current job is still running

Leave it. Checkpoints write at every eval (250, 500, …) and when the 120 min
budget fires. Tail:

```powershell
Get-Content logs\distill.log -Wait -Tail 20    # may be UTF-16
```

Watch `ppl` and `kd`. Both have been falling. A first held-out eval appears
next to `kd(teacher)` at step 250.

The command that is running:

```bash
python -W ignore -u examples/distill_arcane_slm.py \
    --shards "data/qwen_shards/*.tfrecord" \
    --preset distill-small \
    --steps 4000 --batch-size 8 --peak-lr 1e-3 --warmup-steps 100 \
    --alpha 0.4 --temperature 2.0 \
    --log-every 25 --eval-every 250 --eval-batches 8 \
    --holdout-shards 1 --time-budget-min 120 \
    --checkpoint Models/arcane_slm_distilled.weights.h5 \
    --history Models/arcane_slm_distill_history.json \
    --vocab-adapter Models/qwen_vocab_adapter.json \
    --generate "Once upon a time"
```

On PowerShell do **not** pipe through `Tee-Object` for a new run — it
UTF-16-encodes the log and can delay flushes. Prefer:

```bat
cmd /c "set TF_CPP_MIN_LOG_LEVEL=3&& python -W ignore -u examples/distill_arcane_slm.py ... >> logs\distill.log 2>&1"
```

---

## If the job died before a checkpoint

Restart with the same command. There is nothing to resume from until
`Models/arcane_slm_distilled.weights.h5` exists. The dump is still valid.

---

## To chat with it once a checkpoint exists

The API auto-discovers `Models/arcane_slm_distilled.*`:

```bash
cd arcane-docs-web
npm run dev:with-slm
# then open /docs/chat
```

To be explicit:

```powershell
$env:SLM_CONFIG_PATH   = "Models/arcane_slm_distilled.config.json"
$env:SLM_WEIGHTS_PATH  = "Models/arcane_slm_distilled.weights.h5"
$env:SLM_VOCAB_ADAPTER = "Models/qwen_vocab_adapter.json"
cd arcane-docs-web; npm run dev:with-slm
```

```bash
curl http://127.0.0.1:8001/health
# expect "trained": true, "distilled": true, "parameters": 8773924
```

`"trained": false` means it did not find weights. The Next.js route at
`src/app/api/slm-chat/route.ts` only proxies to `127.0.0.1:8001`.

---

## To get a genuinely good model (different hardware)

On a GPU box, nothing about the commands changes — only the scale. More
teacher tokens is the biggest quality lever.

```bash
pip install -r requirements-distill.txt

python examples/dump_qwen_logits.py \
    --text-file data/tinystories_valid.txt \
    --out-dir data/qwen_shards_big \
    --vocab-size 32000 --seq-len 512 --top-k 64 \
    --batch-size 16 --rebuild-vocab

python examples/distill_arcane_slm.py \
    --shards "data/qwen_shards_big/*.tfrecord" \
    --preset distill --steps 20000 --batch-size 8 --mixed-precision \
    --warm-start-embedding

python examples/distill_arcane_slm.py \
    --shards "data/qwen_shards_big/*.tfrecord" \
    --baseline-transformer --steps 20000 --batch-size 8
```

**Run the baseline.** With 92K tokens and a fresh corpus, most of any
improvement comes from the data, not the soft targets. The ARCANE-vs-baseline
KL-to-teacher curve on identical shards is the only thing that isolates what
the neuromimetic mechanisms actually cost or buy.

---

## Environment gotchas that cost time here

### torch 2.0.1 cannot export to numpy 2.4

`Tensor.numpy()` raises `RuntimeError: Numpy is not available`. Worked around
via `.tolist()` in the dump and warm-start paths. The `_ARRAY_API not found`
traceback from `numexpr` / pandas on every TensorFlow import is noisy and
harmless — TF still loads.

### PowerShell `Tee-Object` writes UTF-16 logs

Redirect with `cmd /c "... >> logs\distill.log 2>&1"` instead.

### Memory is the binding constraint

~8 GB total. Never run the dump and training concurrently. Close spare
browsers before a run. The current 8.8M student fits; the 145M `distill`
preset does not, on this box.

---

## Honest expectation setting

**92,160 tokens for an 8.8M-parameter model is roughly 1,000x less data than
conventional scaling guidance.** Even with training finishing the 120-minute
budget, expect short, repetitive, TinyStories-flavoured English — not
conversation. The bar is “looks like a language model, not noise.”

The durable deliverable is the pipeline. More teacher tokens is the biggest
quality lever.

---

## Related files

- [docs/DISTILLATION.md](DISTILLATION.md) — architecture changes, measurements, loss
- [gpbacay_arcane/distillation.py](../gpbacay_arcane/distillation.py) — KD loss, shards, trainer
- [gpbacay_arcane/qwen_vocab.py](../gpbacay_arcane/qwen_vocab.py) — vocabulary adapter
- [gpbacay_arcane/mechanisms.py](../gpbacay_arcane/mechanisms.py) — chunked / single-chunk linear attention
- [examples/dump_qwen_logits.py](../examples/dump_qwen_logits.py) — teacher side
- [examples/distill_arcane_slm.py](../examples/distill_arcane_slm.py) — student side
- [examples/serve_slm_api.py](../examples/serve_slm_api.py) — chat backend
- [tests/test_distillation.py](../tests/test_distillation.py), [tests/test_qwen_vocab.py](../tests/test_qwen_vocab.py)
