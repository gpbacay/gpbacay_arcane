# ARC 1: A Neuromimetic Model That Decides by Resonance

ARC 1 is an open-source, neuromimetic decision model. Instead of generating text one token at a time, it lets learned probes resonate with the input through spiking dynamics until they settle, a mechanism drawn from how biological neurons align and fire, and it turns text into a typed, grounded, calibrated decision in a single forward pass. With 1.32 million parameters, it runs at a median of 7.8 milliseconds on an ordinary CPU, selects the right tool 99.3 percent of the time on held-out data, and ships as a single self-contained .rcn file of about three quarters of a megabyte. It is released under the MIT license: free to download, free to use commercially, and free to modify, with the full source, training code, and weights available to anyone.

## The problem: decisions wrapped in prose

Software constantly asks language models to make small decisions: which tool should run, which team should receive a ticket, which fields does this message contain. To answer, a model with billions of parameters generates a paragraph, and a parser then tries to recover the answer from it. The process is slow and expensive, and it is unreliable, because a model that writes text can also invent a value that was never in the input. Teams compensate with retries, guardrails, and larger budgets.

Most automation does not need eloquence. It needs a correct answer, delivered quickly, with an honest estimate of how far to trust it. ARC 1 was built for exactly that. It does not generate text, so it cannot fabricate it. It returns function calls, extracted records, classification labels, or embeddings, and it reports how confident it is in each.

## Open source and free

ARC 1 is open in every sense that matters to a team adopting it. The code is MIT licensed, so there is no usage fee, no per-call pricing, no account requirement to run it, and no restriction on commercial use. The model runs on your own hardware, which means your data never leaves your machine and there is no service to depend on or to be discontinued. The license itself attaches no conditions, and the source code on GitHub is open to everyone. The one thing to know is about convenience, not licensing: the prebuilt arc1-tiny.rcn checkpoint on the documentation site is free, but it asks you to sign in with GitHub and star the repository before the download starts, a small way of supporting the project. Nothing stops you from training your own model from the repository. The repository includes everything needed to reproduce and extend the model: the architecture, the tokenizer, the training and calibration scripts, the evaluation suite, and export tools for a compact file, ONNX for browsers, and TFLite for mobile.

## How it works

ARC 1 uses a technique called Resonant Schema Binding, which is what lets it decide in one pass. The message is read exactly once by a stack of bidirectional perception blocks, producing a single representation of the whole utterance. Every element of the task schema, whether a tool, a parameter, an enumerated option, or a label, is encoded by the same blocks into a compact vector called an engram, which is cached after first use. Each engram then becomes a probe that resonates with the message over a few cycles through a spiking gate, and all probes settle in parallel. Because the request and any new schema are perceived together, one request costs one forward pass, however many tools or fields it carries.

The settled probes are read out four ways. A firing signal decides whether a tool applies or an optional argument is present. An anchor signal points to a start and end position in the user's own words, so names, numbers, and strings are copied from the input, never generated. A selection signal chooses among options or labels. A pooled embedding supports semantic retrieval. Each readout has a temperature fitted on held-out data, so the reported confidence is calibrated, and the number of resonance cycles is a single dial that trades accuracy against compute using the same weights. This is what makes ARC 1 neuromimetic: its building blocks come from ARCANE, a library of layers that mimic biological principles, including spiking dynamics with leak and threshold, resonant alignment, and an engram-style lexical memory. It is inspired by these principles, not a simulation of a brain, and it trains with standard gradient methods and runs on ordinary hardware.

## Accuracy

The results below are for the arc1-tiny checkpoint, evaluated on data withheld from training. On 300 held-out tool-calling cases, the model selected the correct tool 99.3 percent of the time and produced the complete, exactly correct call 93 percent of the time. When no tool was appropriate, it correctly declined in every case, which matters because a wrongly triggered action is usually costlier than a missed one. Across 150 held-out records, structured extraction reached a field-level F1 of 98.6 percent, with 95.3 percent of records entirely correct. Intent classification scored 100 percent, and embedding retrieval of intent reached 100 percent at rank one.

Calibration is a first-class result. On the firing decision, temperature fitting reduced expected calibration error from 0.48 percent to 0.31 percent, so stated confidence tracks real accuracy closely enough to set a threshold that sends uncertain cases to a person.

## Speed and lightness

Because the model decides instead of generating, there is nothing to wait for. Median latency is 7.8 milliseconds with three resonance cycles and 9.4 milliseconds with one, on a CPU. The model has 1,320,067 parameters, and its smallest distribution, a single .rcn file, is 769,344 bytes. The footprint lets ARC 1 run in a browser, a mobile application, a serverless function, or an edge device, with no GPU and no network call.

## The .rcn format

ARC 1 has its own model container, the .rcn file, designed so that one file is the whole model. It holds the architecture, the calibration for every readout, the tokenizer, a baked device profile, and the weights, so there is no separate config, vocabulary, or calibration file to lose or mismatch. A fixed 128-byte header describes the model, which means a runtime can inspect a file without touching the weights, and the file ends with a checksum of its data. Tensors are aligned and stored in their final layout, then memory-mapped and read in place, so loading involves no parsing pass.

The format also carries its own compression. Weights can be stored in half precision, in 8-bit blocks, or in 4-bit blocks, each with a small scale value per group of 32 weights, while small tensors such as norms, biases, and gates stay in half precision because quantizing them saves nothing and costs accuracy. The measured trade-off is unusually favorable. The original Keras weights are 5.47 megabytes at full precision; the half-precision .rcn is 2.65 megabytes, the 8-bit .rcn is 1.43 megabytes, and the 4-bit .rcn is 0.77 megabytes. Exact-call accuracy is 90.7 percent in the first three and 91.3 percent in the 4-bit file, and extraction F1 is 98.2 percent in the first three and 98.1 percent in the 4-bit file, so a roughly sevenfold reduction in size costs no meaningful accuracy.

An .rcn file loads directly in both Python and Node, and it converts to ONNX for use in a browser, where the number of resonance cycles is fixed at export time. The format is for running the model, but it is not a one-way door: a file can be converted back into weights and a tokenizer for further fine-tuning, and a fine-tuned model is packed into a new .rcn with a single export command. The checkpoint offered for download, arc1-tiny.rcn, uses the 4-bit format and includes its tokenizer and calibration.

## Flexible and scalable

Tools, fields, and labels arrive at request time as a schema, so ARC 1 is not tied to a fixed task list. The distributed checkpoint is deliberately a base model, meant to be fine-tuned on your own tools, records, and categories, and because the source is open, fine-tuning is a matter of running the included scripts on a small dataset. The design also scales across sizes, from the sub-megabyte checkpoint to a full-size decision model with six layers, and to a causal variant of about 10 million parameters and 40 megabytes that powers the project's documentation chat. Adapting ARC 1 to a new use case means retraining a small model, not provisioning a large one.

## How it differs from existing systems

Two recent projects also step away from text generation, and the comparison shows where ARC 1 stands.

Laya AI publishes an encoder-based model of 421 million parameters, with a download of about 808 megabytes and a reported latency of 32.8 milliseconds on a GPU. Its own figures list an expected calibration error of 8.1 percent on its tasks, and it offers neither tool calling nor structured extraction. By those published figures, ARC 1 is about 319 times smaller in parameters and over 1,000 times smaller as a download, runs on a CPU roughly four times faster than Laya's GPU time, and adds tool calling and extraction with values copied from the input. The tasks differ, so this compares footprint and speed, not accuracy on a shared benchmark.

Jev, from TypeSafe AI, is closest in spirit. It is a non-autoregressive System One model that returns typed decisions with calibrated probabilities, aimed at the same class of problem. It is a proprietary hosted service in limited early access, priced per input token, with typical calls completing in around 100 milliseconds including the round trip. ARC 1 is open source and free, runs where the application runs, and has a measured local latency under 10 milliseconds, with no per-call fee, no network dependency, and no data leaving the machine. Jev's accuracy figures are not directly comparable to ARC 1's, and this document does not claim that either outperforms the other on quality. The difference is ownership: Jev is a service to call, while ARC 1 is a model you can inspect, fine-tune, and keep.

## Limits

ARC 1 is a specialist, and its limits are measurable. On tools it never saw during training, tool selection falls to about 40 percent, so it is strongest when fine-tuned on the tools it will serve. Classification is excellent when labels relate to words in the text, as with intent and sentiment, but weaker on paraphrases unlike its training data: about 52 percent on support-ticket routing and 27 percent on product topics, because the model has no pretrained language knowledge. It makes at most one call per tool per request, and long inputs are truncated to a 160-token window. More resonance cycles did not improve accuracy on this checkpoint, so the setting should be tuned by measurement, not assumed.

## Get started

ARC 1 is available now. The source is in the gpbacay_arcane repository on GitHub under the MIT license, the base checkpoint, arc1-tiny.rcn, can be downloaded from the documentation site, and the full technical report and fine-tuning guide are at /docs/arc-1. A model that decides in milliseconds, runs on a laptop, costs nothing, and can be made your own is now a download away.

Sources for the comparison: Laya AI's published figures at laya.convaiinnovations.com, checked September 2026; Jev details from TypeSafe AI coverage, including the Wikipedia entry for Jev (AI model) and the DataCamp and MindStudio write-ups, September 2026. Jev is in early access and its details may change.
