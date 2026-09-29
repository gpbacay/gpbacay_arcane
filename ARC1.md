# Stop Using LLMs for If-Else Decisions: Meet ARC 1

*A 1.3M-parameter, open-source, neuromimetic model that makes grounded, typed decisions in 7.8 milliseconds on a standard CPU.*

Every modern software architecture contains a hidden absurdity: we are constantly asking multi-billion-parameter language models to make simple choices. Software needs to know which API tool should fire, which support queue gets a ticket, or what date and dollar amount sit inside an invoice snippet. To answer, an autoregressive model burns significant compute generating prose token by token, and then a fragile parser tries to extract the payload from the text.

The whole process is slow, expensive, and fundamentally unreliable. A model designed to generate language will eventually invent values that never existed in the prompt, so teams compensate with retries, validation layers, guardrails, and bloated cloud budgets. Most automation does not need eloquence. It needs a correct, typed decision, delivered instantly, with an honest confidence score.

## Enter ARC 1: decisions without generation

ARC 1 is an open-source, neuromimetic decision model built for structured classification, extraction, and routing. Instead of predicting the next token, it evaluates a request with learned probes that resonate with the input through spiking dynamics until they settle into a stable state. That mechanism is drawn from how biological neurons align and fire, and it turns text into a typed, grounded decision in a single forward pass.

- **Ultra-light footprint:** 1,320,067 parameters, packed into a single file of 769 KB.
- **Instant inference:** a median of 7.8 milliseconds on a CPU, with no GPU.
- **Grounded by design:** names, numbers, and strings are copied from the input, never generated, so extraction cannot invent a value.
- **Permissive open source:** MIT license, free for commercial use, fully inspectable.

## How it works: Resonant Schema Binding

Rather than looping through tokens one at a time, ARC 1 runs one unified pass.

- **Bidirectional perception:** a stack of perception blocks reads the message exactly once and builds a single representation of the whole utterance.
- **Schema engrams:** every element of your schema, whether a tool, a parameter, an enumerated option, or a label, is encoded by the same blocks into a compact vector called an engram, then cached after first use.
- **Spiking settlement:** each engram acts as a probe that resonates with the message through a biologically inspired spiking gate. All probes settle in parallel over a few quick cycles.

Once settled, the model produces four readouts. A firing signal decides whether a tool applies or an optional argument is present. An anchor signal points to exact start and end positions in the source text, which is what keeps extraction grounded. A selection signal handles categorical choices and classification. A pooled embedding supports semantic retrieval. Each readout has a temperature fitted on held-out data, so the confidence it reports can be trusted, and the number of resonance cycles is a single dial that trades accuracy for compute using the same weights.

Why call it neuromimetic? Its building blocks come from ARCANE, a library of layers that mimic biological principles: spiking dynamics with leak and threshold, resonant alignment, and an engram-style lexical memory. ARC 1 is inspired by these principles, not a simulation of a brain. It trains with standard gradient methods and runs on ordinary hardware.

## Proven benchmarks and precision

On withheld evaluation data, the base arc1-tiny checkpoint showed high accuracy and reliable calibration.

- **99.3% tool selection:** the correct action identified across 300 held-out cases.
- **93.0% exact tool calls:** the complete call, with every argument right, formed correctly.
- **100% correct refusals:** when no tool applied, it declined every time, which matters because a wrongly triggered action usually costs more than a missed one.
- **98.6% extraction F1:** across 150 held-out records, with 95.3% of records entirely correct.
- **100% intent retrieval at rank one:** for embedding-based lookup.
- **0.31% expected calibration error:** on the firing decision, down from 0.48% before temperature fitting, so stated confidence tracks real accuracy closely enough to route uncertain cases to a person.

## The .rcn container: one file, 769 KB

Shipping a model usually means juggling weights, vocabularies, and runtime configs. ARC 1 packs all of it into a single .rcn file built for fast, in-place loading.

- **Self-contained:** the architecture, per-readout calibration, tokenizer, a baked device profile, and the quantized weights all live in one file, with a checksum on the data.
- **Inspect without loading:** a fixed 128-byte header describes the model, so a runtime can read a file's details without touching the weights.
- **No parsing pass:** tensors are aligned and stored in their final layout, then memory-mapped and read in place.
- **Extreme compression:** the original Keras weights are 5.47 MB. The half-precision .rcn is 2.65 MB, the 8-bit version is 1.43 MB, and the 4-bit version is 0.77 MB. Exact-call accuracy is 90.7% in the first three and 91.3% in the 4-bit file, and extraction F1 is 98.2% against 98.1%, so a roughly sevenfold size reduction costs no meaningful accuracy.
- **Runs where you do:** it loads directly in Python and Node, converts to ONNX for browsers, and has a TFLite export for mobile. It also converts back to weights and a tokenizer when you want to fine-tune.

## Flexible and scalable

Tools, fields, and labels arrive at request time as a schema, so ARC 1 is not locked to a fixed task list. The download is deliberately a base model, meant to be fine-tuned on your own tools, records, and categories. Because the source is open, that means running the included scripts on a small dataset, then packing the result into a new .rcn with a single export command. The same design scales across sizes, from the sub-megabyte checkpoint to a full-size decision model with six layers, and to a causal variant of about 10 million parameters and 40 MB that powers the project's documentation chat. Adapting ARC 1 means retraining a small model, not provisioning a large one.

## The competitive landscape

ARC 1 reflects a broader shift away from prompt-wrapped generative models toward dedicated, lightweight decision layers. Two recent projects sit nearby.

- **Versus large encoders such as Laya AI:** Laya publishes a 421-million-parameter model with an 808 MB download and a reported 32.8 ms latency on a GPU. It does not offer tool calling or structured extraction. By those published figures, ARC 1 is about 319 times smaller in parameters, over 1,000 times smaller as a download, and roughly four times faster on a CPU than Laya is on a GPU. The tasks differ, so this compares footprint and speed, not accuracy on a shared benchmark.
- **Versus hosted decision engines such as TypeSafe AI's Jev:** Jev is a proprietary, hosted System One model that returns typed decisions with calibrated probabilities, in limited early access, priced per input token, with calls typically around 100 ms including the network round trip. ARC 1 aims at the same class of problem but runs entirely on your own hardware, with no per-call fee and no data leaving your infrastructure. Their accuracy figures are not directly comparable, and this post does not claim either is more accurate. The difference is ownership: Jev is a service to call, while ARC 1 is a model you can inspect, fine-tune, and keep.

## Operational boundaries

ARC 1 is an intentional specialist, not a conversational engine, and it comes with clear limits.

- **Fine-tuning required for new schemas:** on tools never seen in training, selection accuracy drops to roughly 40%, so it is meant to be fine-tuned on your project's actual actions.
- **Labels should match the text:** classification is strongest when labels tie to words in the input (100% on intent). It is weaker on paraphrases unlike its training data, at about 52% on support-ticket routing and 27% on product topics, because it has no pretrained language knowledge.
- **Capacity bounds:** inputs are capped at a 160-token window, and it makes at most one call per tool per request.
- **Measure your cycles:** more resonance cycles did not improve accuracy on this checkpoint, so tune the dial by measurement.

## Get started

ARC 1 is available now, under the MIT license, with no sign-in. Clone the gpbacay_arcane repository from GitHub for the architecture, training pipelines, calibration routines, and export tooling, and download the pre-quantized arc1-tiny.rcn checkpoint from the project documentation site at /docs/arc-1. If you find it useful, a star on the repository is appreciated, and entirely optional. Then put grounded, sub-10-millisecond decisions on your own hardware.

*Sources for the comparison: Laya AI's published figures at laya.convaiinnovations.com, checked September 2026; Jev details from TypeSafe AI coverage, including the Wikipedia entry for Jev (AI model) and the DataCamp and MindStudio write-ups, September 2026. Jev is in early access and its details may change.*
