# Qwen3-0.6B Q4 GGUF to ARC 1 LM v2 Plan

## Objective

Improve ARC 1 LM v2 by using `Qwen3-0.6B` in `Q4_K_M` GGUF format as an
offline response teacher. The workflow must run on a CPU-only laptop, keep peak
memory low, preserve the existing ARC 1 LM v2 checkpoint, and produce a small,
fast INT8 deployment artifact.

LittleBit-2 is not part of this workflow. Its SVD, Joint-ITQ, and
quantization-aware training process is designed for compressing the teacher
itself and is not practical on the target CPU-only machine. A GGUF teacher run
through llama.cpp is a better fit.

## Target machine

- CPU: Intel Core i5-1240P, 12 cores and 16 logical processors
- RAM: 8 GB
- Accelerator: none
- Student framework: TensorFlow 2.20, CPU only
- Existing ARC 1 LM v2 checkpoint: approximately 145 MB
- ARC 1 LM v2 configuration: 12 hybrid convolution/GQA layers, width 256,
  8,000-token vocabulary, and 2,048-token configured context

Close browsers, development servers, and other memory-intensive programs during
teacher generation and student training. Windows paging would make both stages
substantially slower.

## Expected duration

| Milestone | Elapsed time | Hands-on time |
|---|---:|---:|
| Benchmark and working smoke pipeline | 1 day | 2-4 hours |
| Evaluated 300-500 example proof of concept | 1-2 days | 3-5 hours |
| First useful 5,000-example candidate | 3-5 days | 4-7 hours |
| Evaluated 10,000-example candidate | 5-8 days | 5-8 hours |

Most elapsed time is unattended teacher generation and student training. These
ranges must be revised after the initial machine-specific benchmark.

## Architecture of the workflow

```text
Qwen3-0.6B Q4_K_M GGUF
        |
        | llama.cpp, non-thinking generation
        v
Resumable prompt/response JSONL corpus
        |
        | filtering, deduplication, train/validation split
        v
8K Qwen vocabulary adapter and encoded ARC corpus
        |
        | assistant-only cross-entropy plus replay data
        v
ARC 1 LM v2 candidate checkpoint
        |
        | fixed held-out evaluation
        v
Dynamic INT8 export
```

GGUF is used for response generation, not full logit distillation. Extracting
per-position logits over Qwen's complete vocabulary through llama.cpp would be
memory- and bandwidth-heavy and would lose the restricted-output-head
optimization already present in `examples/dump_qwen_logits.py`.

## Phase 1: establish baselines

### Tasks

1. Record the current ARC 1 LM v2 responses on a fixed evaluation prompt set.
2. Record cached decoding latency, peak RAM, repetition rate, and output quality.
3. Install or validate a llama.cpp-compatible Python runtime.
4. Download a trusted `Qwen3-0.6B-Q4_K_M.gguf` artifact.
5. Benchmark Qwen prompt processing and generation on the target laptop.
6. Benchmark ARC 1 LM v2 training for 50-100 steps on a tiny corpus.

### Measurements

- Qwen prompt tokens per second
- Qwen generated tokens per second
- ARC training seconds per step
- Peak resident memory for each process
- Estimated time for 500, 5,000, and 10,000 examples
- Estimated time for the planned number of student updates

### Initial pass criteria

- No sustained paging or out-of-memory failure
- Approximately 10 or more Qwen generation tokens per second
- A 500-step ARC experiment can complete overnight
- The generated answers are coherent with thinking disabled

If these criteria are not met, reduce context length, batch size, output length,
and concurrent processes before continuing.

## Phase 2: build the teacher-data generator

Add a new script:

```text
examples/generate_qwen3_gguf_corpus.py
```

The generator must:

- load Qwen3-0.6B Q4_K_M through llama.cpp;
- use Qwen3's chat template;
- disable thinking and remove any residual `<think>` blocks;
- write one JSON object per example incrementally;
- flush output frequently and resume safely after interruption;
- record prompt, response, seed, model identity, and generation settings;
- reject empty, malformed, and excessively repetitive responses;
- impose configurable input and output length limits;
- never overwrite completed examples silently.

Suggested initial generation settings:

| Setting | Initial value |
|---|---:|
| Context | 512-2,048 tokens |
| Maximum response | 128 tokens |
| Temperature | 0.7 |
| Top-p | 0.8 |
| Top-k | 20 |
| Thinking | Disabled |
| Threads | Benchmark 8, 10, and 12 |
| Seed | Recorded per example |

Avoid greedy decoding because Qwen3 can become repetitive under that setting.

## Phase 3: create the smoke corpus

Generate 300-500 examples before committing to a large run. Use a mixture close
to the expected production workload:

| Category | Share |
|---|---:|
| Conversation and instruction following | 30% |
| Factual questions | 20% |
| Summarization and rewriting | 15% |
| Structured responses | 15% |
| Elementary reasoning and arithmetic | 10% |
| ARCANE-specific questions | 10% |

The percentages are starting points. Replace them with measured production needs
when those needs are known.

### Corpus validation

- Remove exact and near duplicates.
- Remove teacher self-identification and unrelated boilerplate.
- Reject incomplete responses.
- Keep most answers below 150 tokens.
- Reserve 10% of examples as a fixed validation set before training.
- Do not allow validation prompts or answers into the training split.

## Phase 4: prepare the student vocabulary

Build the vocabulary adapter from the complete prompt-and-response corpus before
encoding the training data.

Start with the existing 8,000-token vocabulary budget. Measure token-occurrence
coverage rather than only the number of unique tokens represented.

Acceptance target:

- Preferred occurrence coverage: at least 99%
- Minimum acceptable occurrence coverage: 98%

If coverage is below 98%, inspect the missing tokens first. Increase the
vocabulary to 12,000 only if the missing mass affects ordinary output rather
than rare artifacts. A larger vocabulary increases student parameters and makes
the new checkpoint incompatible with the current 8K configuration.

## Phase 5: train the smoke candidate

Warm-start from the existing ARC 1 LM v2 checkpoint. Do not replace or modify the
baseline artifact.

Suggested output names:

```text
Models/arc1_lm_v2_qwen3_smoke.config.json
Models/arc1_lm_v2_qwen3_smoke.weights.h5
Models/arc1_lm_v2_qwen3_smoke_history.json
```

Initial training configuration:

- Sequence length: 128
- Batch size: 1, increasing to 2 only if memory permits
- Loss: assistant-only next-token cross-entropy
- Initialization: current ARC 1 LM v2 weights
- Learning rate: conservative fine-tuning rate
- Checkpointing: every few hundred steps
- Validation: every few hundred steps
- Replay: a small sample of the original ARC training corpus
- Early stopping: enabled based on held-out loss and generation quality

Replay data protects against improving Qwen-style chat while erasing capabilities
already present in ARC 1 LM v2.

## Phase 6: evaluate the smoke candidate

Compare the baseline and candidate using identical prompts and decoding settings.
Do not select a model from training loss alone.

Evaluate:

- short instruction following;
- factual question answering;
- elementary arithmetic;
- ARCANE documentation questions;
- requested output formatting;
- multi-turn consistency;
- incomplete and nonsensical output rate;
- phrase and token repetition;
- cached decode latency;
- peak RAM.

### Promotion gate

Proceed to the full corpus only if the smoke candidate produces a clear held-out
quality improvement without unacceptable regression in repetition, latency, or
existing ARC-specific behavior.

If it fails, modify the dataset and repeat the small experiment. Do not scale a
failed corpus to thousands of examples.

## Phase 7: generate and train the full candidate

### First full run

1. Generate 5,000 curated examples.
2. Deduplicate and filter them.
3. Preserve a fixed 10% held-out split.
4. Rebuild and validate the vocabulary adapter.
5. Fine-tune from the original ARC 1 LM v2 baseline.
6. Evaluate against both the baseline and smoke candidate.

Move to 10,000 examples only if the 5,000-example learning and validation curves
show continued improvement. Prefer additional task diversity over repeated
paraphrases.

Approximate CPU-only elapsed-time budget:

| Operation | Expected time |
|---|---:|
| Generate 300-500 examples | 1-3 hours |
| Train smoke candidate | 3-10 hours |
| Generate 5,000 examples | 8-24 hours |
| Train 5,000-example candidate | 12-48 hours |
| Evaluation and one corrective run | 8-24 hours |

## Phase 8: export and validate

Quantize only after selecting the best FP32 checkpoint.

Use the existing ARC exporter with dynamic INT8, adapted to the new artifact
names if necessary. Keep the FP32 checkpoint as the training source and use the
INT8 artifact for deployment.

Validate:

- export completes without unsupported-operation fallback errors;
- logit error stays within the exporter's accepted tolerance;
- top-1 agreement with FP32 remains acceptable;
- held-out generations remain qualitatively equivalent;
- cached decoding remains faster than full recomputation;
- deployment memory is materially lower than FP32.

## Deliverables

- `examples/generate_qwen3_gguf_corpus.py`
- Versioned prompt source files
- Resumable teacher-response JSONL files
- Corpus validation and split report
- Vocabulary coverage report
- Smoke candidate weights, configuration, and history
- Full candidate weights, configuration, and history
- Baseline-versus-candidate evaluation report
- Dynamic INT8 deployment export
- Reproduction commands and measured runtime table

## Stop conditions

Stop or revise the experiment if any of the following occurs:

- Qwen generation causes sustained paging on the 8 GB machine.
- Smoke training cannot complete overnight.
- Vocabulary occurrence coverage remains below 98%.
- The candidate improves training loss but not held-out generation.
- Repetition or incomplete-answer rates worsen materially.
- Existing ARC-specific behavior regresses beyond the agreed tolerance.
- The 5,000-example candidate has plateaued, making a 10,000-example run
  unlikely to justify its CPU time.

## Recommended execution order

1. Complete the one-hour hardware benchmark.
2. Implement and test resumable teacher generation.
3. Run the 300-500-example smoke experiment.
4. Evaluate it before generating more data.
5. Run the 5,000-example experiment.
6. Expand to 10,000 only when measurements justify it.
7. Export the winning model to INT8 and preserve all baseline artifacts.

