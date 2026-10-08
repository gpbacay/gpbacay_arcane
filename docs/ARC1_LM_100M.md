# ARC 1 LM 100M

`arc1-lm-100m` scales the ARC 1 LM v2 block design without replacing it. It has
94,681,648 parameters, 16 layers (11 gated short-convolution and five GQA), a
640-wide residual stream, two KV heads, a 16,000-token vocabulary, and a 4,096
token serving context.

The preset and export tooling do not include trained weights. A randomly
initialised or briefly trained checkpoint is not a chat model.

## 1. Prepare teacher shards

Use dialogue text formatted consistently with `User:` and `Assistant:`. The
assistant-only mask avoids spending student capacity imitating prompts.

```powershell
python examples/dump_qwen_logits.py `
  --text-file data/arc1_chat_corpus.txt `
  --out-dir data/arc1_100m_qwen_shards `
  --model-id Qwen/Qwen2.5-0.5B-Instruct `
  --vocab-adapter Models/qwen_vocab_adapter_16k.json `
  --vocab-size 16000 `
  --rebuild-vocab `
  --seq-len 256 `
  --stride 128 `
  --top-k 64 `
  --assistant-only `
  --device cuda `
  --dtype float16
```

Use a held-out shard and verify that vocabulary occurrence coverage is close to
100%. The shard length controls training memory; it no longer reduces the
model's saved 4,096-token serving context.

## 2. Distil the student

```powershell
python examples/distill_arcane_slm.py `
  --arch arc1 `
  --preset arc1-lm-100m `
  --shards "data/arc1_100m_qwen_shards/*.tfrecord" `
  --meta data/arc1_100m_qwen_shards/meta.json `
  --steps 20000 `
  --batch-size 4 `
  --alpha 0.3 `
  --temperature 2.0 `
  --warm-start-embedding `
  --vocab-adapter Models/qwen_vocab_adapter_16k.json `
  --checkpoint Models/arc1_lm_100m.weights.h5 `
  --history Models/arc1_lm_100m_history.json
```

Continue an interrupted or staged run with the same arguments plus `--resume`
and without `--warm-start-embedding`. Resume reloads model weights; the optimizer
schedule intentionally restarts.

Do not judge the model from training loss alone. Maintain held-out suites for
chat instruction following, factual QA, arithmetic, short reasoning, safety,
and generation repetition. Compare against ARC 1 LM v2 and the teacher.

## 3. Quantize only after quality is acceptable

```powershell
python examples/export_arc1_lm.py `
  --config Models/arc1_lm_100m.config.json `
  --weights Models/arc1_lm_100m.weights.h5 `
  --vocab-adapter Models/qwen_vocab_adapter_16k.json `
  --quant dynamic-int8 `
  --seq-len 256 `
  --out Models/arc1_lm_100m_export
```

The exporter refuses a missing checkpoint and validates logit error and top-1
agreement before writing the artifact. This TFLite signature is a fixed-window
full forward pass suitable for prefill and portability testing. The TensorFlow
chat server remains faster for generation because it uses ARC 1's incremental
convolution, KV, resonance, and engram caches.
