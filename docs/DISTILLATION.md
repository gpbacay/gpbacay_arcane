# Distilling Qwen2.5-0.5B into the ARCANE SLM

Qwen2.5-0.5B is Apache-2.0, so the teacher, the dumped predictions, and anything
distilled from them are all yours to release.

This document covers what had to change in ARCANE before the distillation was
feasible, and how to run it.

---

## Quick start

```bash
pip install -r requirements-distill.txt      # teacher side only (torch + transformers)

# 1. Dump teacher predictions (PyTorch). Writes TFRecord shards + meta.json.
python examples/dump_qwen_logits.py \
    --text-file data/corpus.txt \
    --out-dir data/qwen_shards \
    --vocab-size 32000 --seq-len 512 --top-k 64

# 2. Distil into ARCANE (TensorFlow, no torch needed).
python examples/distill_arcane_slm.py \
    --shards "data/qwen_shards/*.tfrecord" \
    --preset distill --steps 20000 --batch-size 8

# 3. The control that makes the result interpretable.
python examples/distill_arcane_slm.py \
    --shards "data/qwen_shards/*.tfrecord" \
    --baseline-transformer --steps 20000 --batch-size 8
```

---

## Why the old architecture could not be distilled

### The attention memory wall

`CausalLinearSelfAttention` built the prefix state with a per-token outer product:

```python
kv    = tf.expand_dims(keys, -1) * tf.expand_dims(values, -2)   # (b, h, s, dh, dh)
kv_cs = tf.cumsum(kv, axis=2)
```

At the `100m` preset (b=2, h=12, s=256, dh=64) that is **100.7 MB per tensor,
2.42 GB across 12 layers**, retained for backprop — for a batch of *two*. At
seq_len 1024 it is 9.66 GB.

It is now computed chunkwise: the `(d_head, d_head)` state is carried across
chunks by `tf.scan` and materialised once per chunk rather than once per token.

| seq_len | old (cumsum) | new (chunked) |
|--------:|-------------:|--------------:|
|     256 |      2.42 GB |       0.024 GB |
|    1024 |      9.66 GB |       0.080 GB |
|    4096 |     38.65 GB |       0.307 GB |

**102x less activation memory at seq_len 256.** This is an optimisation, not a
change of semantics — `test_chunked_matches_cumsum_reference` checks the chunked
output against the original cumsum formulation across five chunk/length
combinations (max error ~2.4e-7, i.e. float32 round-off).

### Tokenizer mismatch

Logit distillation needs teacher and student to score the same symbols. Qwen's
vocabulary is 151,936 tokens; adopting it wholesale would put
`151936 x 768 = 116.7M` parameters into the embedding table alone.

`QwenVocabAdapter` keeps the most frequent Qwen ids in the corpus and remaps them
into a dense student range, preserving ARCANE's special-token layout:

```
0 = PAD   1 = EOS   2 = UNK   3 = BOS   4.. = kept Qwen ids
```

Qwen's own end-of-text folds onto `EOS`. The dump script reports what fraction of
token *occurrences* survive the trim, which is the number to watch — at 32k it is
normally well above 99% on ordinary text.

This also sidesteps a scaling problem: the built-in `BytePairTokenizer.train` is
`O(vocab x corpus)` in pure Python and does not survive a distillation-sized
corpus.

### Framework split

The teacher is PyTorch, the student is TensorFlow. They never share a graph.
`dump_qwen_logits.py` writes **sparse top-k logits in student id space** to
TFRecord; training replays them with no torch dependency. Dense logits are not an
option — 151,936 floats per position is terabytes for any real corpus, while
top-64 over a trimmed 32k vocabulary is a few GB.

---

## Architecture changes

All of these are opt-in flags on `ArcaneSLMConfig`. The `100m` preset is
unchanged and every existing test still passes; the `distill` preset turns them on.

### Hybrid attention (`softmax_every`)

Linear attention compresses the entire prefix into a fixed `d_head x d_head`
state, so it cannot do exact key lookup — which is exactly what a softmax teacher
is good at. `softmax_every=4` makes every 4th layer full softmax attention:

```
layer:      0 1 2 3  4 5 6 7  8 9 10 11
attention:  L L L S  L L L S  L L L  S
```

Three of twelve layers pay quadratic cost; the rest stay linear.

### KV decay (`use_decay`)

`tf.cumsum` weights all history equally forever — the denominator grows linearly
in `t`, so later tokens get progressively more averaged. A learned per-head
forget factor `gamma = sigmoid(logit)` (init ~0.982) gives
`KV_t = gamma * KV_{t-1} + phi(k_t) v_t`.

Worth knowing: `ResonantSequenceMixer` *also* pulls every token toward the causal
prefix mean. Before this change there were two independent prefix-averaging
mechanisms stacked in every block, which is a strong low-pass bias against a
sharp teacher distribution.

### RoPE (`use_rope`)

The old `PositionalEncodingLayer` baked a non-trainable sinusoidal buffer at
`max_position = seq_len`, freezing context length with no extrapolation. RoPE
matches the teacher's positional geometry and is applied **before** the
`elu(x)+1` feature map so the linear-attention denominator stays positive.

When `use_rope` is on the model skips the additive table entirely — applying both
would double-count position. `test_rope_model_skips_additive_positional_encoding`
guards this.

### FFN fixes

Two measured defects, at `d_model=768` with LayerNorm'd unit-scale input:

| | before | after |
|---|---:|---:|
| FFN branch rms / input rms | 0.117 | **0.242** |
| output range | -0.170 … +0.796 | -1.018 … +1.058 |
| positive mass vs negative | 1.58x | **1.02x** |

1. **`ffn_out_activation=None`.** A GELU on the FFN *down*-projection floors
   negatives at -0.17 while leaving positives unbounded, so every FFN write into
   the residual stream was biased positive. Standard transformer FFNs leave this
   projection linear.

2. **`gate_normalize=True`.** `DenseGSER`'s spike gate applied an *absolute*
   threshold (0.4) to unnormalised post-GELU activations — it has no idea what
   scale the activations are on, and attenuated the branch ~6x at init. The gate
   is now scale-relative with a learnable per-channel threshold, so the layer
   sets its own sparsity. The spiking character is preserved.

3. **`ffn_mult=4`** (from 2). FFN width is where factual knowledge lands, and
   Qwen's ratio is 4864/896 ≈ 5.43.

### Other

- **`norm_type="rms"`** — matches the teacher, one less distribution mismatch.
- **`reweight_centered=True`** — the semantic reweighting gate was `sigmoid(...)`
  in (0, 1), which can only attenuate and halves the branch at init. The centred
  form spans (0, 2) with unit gain at init.

---

## Parameter budget

| preset | params | attention | notes |
|---|---:|---|---|
| `100m`    | 102,618,636 | `LLLLLLLLLLLL` | unchanged |
| `distill` | 145,147,509 | `LLLSLLLSLLLS` | `ffn_mult=4`, seq_len 512 |

`distill` is ~145M, not 100M — the wider FFN costs ~42M. To hold ~100M instead,
prefer width over depth: `num_layers=8` with `ffn_mult=4` lands near 105M. For
absorbing knowledge from a teacher, the wider FFN is the better trade.

`estimate_trainable_parameters()` is exact — `test_parameter_estimate_matches_built_model`
asserts it against a real build.

---

## Loss

```
L = alpha * CE(student, hard labels) + (1 - alpha) * T^2 * KL(teacher || student)
```

Defaults are `alpha=0.4`, `T=2.0`. Both distributions are renormalised over the
teacher's top-k support, so the KL is proper (two distributions on a shared
support), and the `T^2` factor keeps the KD gradient comparable to CE as
temperature moves.

Padding positions are masked out of both terms.

---

## Reading the result

**Most of the gain will come from the data, not the soft targets.** The SLM was
previously trained on 200k characters of Tiny Shakespeare; any real corpus is a
large upgrade on its own. Run `--baseline-transformer` on the identical shards or
you will be measuring the corpus rather than the distillation.

The comparison that is actually informative is the KL-to-teacher curve of ARCANE
against the parameter-matched vanilla transformer:

- **ARCANE tracks the baseline** — its mechanisms are free, and you have a
  neuromimetic model at transformer parity.
- **ARCANE plateaus higher** — you have measured what resonance, bioplasticity
  and depth-wise AttnRes cost in capacity, which is a far more useful result than
  the checkpoint.

The ceiling either way is the teacher. A ~145M student with mostly-linear
attention will not match Qwen2.5-0.5B.

---

## Hardware note

This repo's environment has `torch 2.0.1+cpu`. The teacher dump runs on CPU but
slowly — use `--max-windows` to bound it. Training the student needs a GPU
regardless; that, not the dump, is the real bottleneck.
