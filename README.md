# ARCANE

**Augmented Reconstruction of Consciousness through Artificial Neural Evolution**

A Python library for building neuromimetic AI models inspired by biological neural principles. ARCANE provides researchers and developers with biologically-plausible neural layers, models, and training mechanisms that bridge neuroscience and artificial intelligence.

## What is ARCANE?

ARCANE is a comprehensive Python library that enables you to build, train, and deploy neuromimetic AI models. Unlike traditional deep learning frameworks, ARCANE incorporates biological neural principles such as:

- **Neural Resonance**: Bi-directional prototype alignment between ResonantGSER layers, plus local per-example alignment in PredictiveResonantLayer. Optional inference-time BCM plasticity.
- **Spiking Neural Dynamics**: LIF-style leak plus graded spikes: signed spike counts against a self-scaling threshold, trained end to end.
- **Hebbian / BCM Learning**: Dual-weight plastic kernels (`BioplasticDenseLayer`, `HebbianHomeostaticNeuroplasticity`).
- **Homeostatic Plasticity**: Activity-dependent gain and plastic-kernel scaling.
- **Hierarchical Processing**: Multi-level ResonantGSER stacks with `set_higher_layer` / `set_lower_layer` (Keras 3-safe).

The library provides ready-to-use models, customizable neural layers, and training callbacks that make it easy to experiment with biologically-inspired AI architectures.

## Key Features

### Biological Neural Layers
- **ResonantGSER**: Spiking neural dynamics with reservoir computing and hierarchical resonance.
- **PredictiveResonantLayer**: Local predictive resonance RNN with optional stateful alignment for inference-time adaptation.
- **BioplasticDenseLayer**: Hebbian learning with homeostatic plasticity; optional inference-time plasticity.
- **Hierarchical Resonance**: Multi-level neural architectures with bi-directional feedback.
- **Neural Reservoir Computing**: Dynamic temporal processing with configurable parameters.
- **Relational Concept Graph Reasoning**: Unified mechanism for concept extraction and relational reasoning.
- **Linear Self-Attention**: Katharopoulos kernel attention, $O(n d^2)$ in sequence length (not $QK^{\top}$).

### Ready-to-Use Models
- **ArcaneSmallLanguageModel**: Causal decoder LM (~100M with the `100m` preset, ~145M with the distillation-tuned `distill` preset).
- **HierarchicalResonanceFoundationModel**: Advanced model with multi-level resonance hierarchy and deliberative reasoning.
- **NeuromimeticSemanticModel**: Standard neuromimetic model with biological learning rules for general tasks.
- **Custom Architecture Support**: Build your own models using individual layers.

### Training and Generation Tools
- **Neural Resonance Callbacks**: Orchestrate the "thinking phase" during training.
- **Multi-Temperature Generation**: Conservative, balanced, and creative text generation modes.
- **Dynamic Self-Modeling**: Adaptive reservoir sizing during training.
- **CLI Tools**: Command-line utilities for model management and information.

### Research-Focused Design
- **Biologically-Plausible**: Grounded in neuroscience principles.
- **Highly Configurable**: Extensive parameter control for experimentation.
- **Extensible Architecture**: Easy to add new layers and mechanisms.
- **Performance Monitoring**: Built-in callbacks for tracking neural dynamics.

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.12+

### Install from PyPI (Recommended)

```bash
pip install gpbacay-arcane
```

### Install from Source

```bash
git clone https://github.com/gpbacay/gpbacay_arcane.git
cd gpbacay_arcane
pip install -e .
```

## Quick Start

### Basic Usage

```python
from gpbacay_arcane import NeuromimeticSemanticModel

# Create a simple neuromimetic model
model = NeuromimeticSemanticModel(vocab_size=1000)
model.build_model()
model.compile_model()

# Generate text (requires a trained tokenizer)
generated = model.generate_text(
    seed_text="artificial intelligence",
    tokenizer=your_tokenizer,
    max_length=50,
    temperature=0.8
)
```

### Advanced Usage with Resonance

```python
from gpbacay_arcane import HierarchicalResonanceFoundationModel, NeuralResonanceCallback

# Create an advanced model with biological neural principles
model = HierarchicalResonanceFoundationModel(
    vocab_size=3000,
    seq_len=32,
    hidden_dim=128,
    num_resonance_levels=4
)

model.build_model()
model.compile_model(learning_rate=3e-4)

# Train with neural resonance (biological "thinking phase")
resonance_callback = NeuralResonanceCallback(resonance_cycles=10)
model.model.fit(X_train, y_train, callbacks=[resonance_callback])

# Generate text with different creativity levels
generated = model.generate_text(
    seed_text="the nature of consciousness",
    tokenizer=tokenizer,
    temperature=0.8  # 0.6=conservative, 0.9=balanced, 1.2=creative
)
```

### 100M Small Language Model

```python
from gpbacay_arcane import ArcaneSmallLanguageModel, BytePairTokenizer

model = ArcaneSmallLanguageModel.from_preset("100m")
model.build_model()
model.compile_model(learning_rate=3e-4)
print(model.count_params())  # ~100M trainable + bioplastic kernels

# Smoke-train: python examples/train_arcane_slm.py --preset tiny --max-steps 2
# Full 100M architecture: python examples/train_arcane_slm.py --preset 100m --build-only
# Chat (untrained until you pass --weights): python examples/chat_arcane_slm.py --preset tiny
```

### ARC 1 — Automation Foundation Model

ARC 1 is an ultra-compact, non-autoregressive "System 1" decision model. In a single forward pass
it turns text into calibrated, typed output (tool calls, extracted records, classification labels,
or embeddings) in milliseconds. It never generates prose. Its architecture, **Resonant Schema
Binding**, is built from ARCANE mechanisms:

1. **Perceive once**: the utterance is read a single time by bidirectional `Arc1PerceptionBlock`s
   (`FieldAttention` → `ResonantChannelMixer` → `ConceptEngram` → `FieldResonance`).
2. **Schema engrams**: every tool, argument, enum option, and label is pooled into one vector by the
   same blocks. New schema texts are perceived in the same batch as the utterance, then cached by
   `SchemaMemory`, so a request is always exactly one forward pass.
3. **Resonant binding**: each engram is a probe that resonates with the utterance field for a few
   shared-weight cycles through a GSER spiking gate (`ResonantBinding`). All probes bind in parallel.
4. **Readouts**: `fire` (tool applies / optional argument present / boolean), `anchor` (start/end
   pointer that copies strings and numbers from the user's words), `select` (enum option or
   classification label), embedding.

A request is one forward pass. Every readout has a temperature fitted on held-out data, so
`confidence` is calibrated. `cycles` trades accuracy for speed with the same weights.

```python
from gpbacay_arcane import Arc1Agent, Arc1Config, Arc1Model, BytePairTokenizer, ToolParam, ToolSpec
import json

config = Arc1Config.from_dict(json.load(open("Models/arc1_arc1_tiny.config.json")))
model = Arc1Model(config).build_model()
model.load_weights("Models/arc1_arc1_tiny.weights.h5")
agent = Arc1Agent(model, BytePairTokenizer.load("Models/arc1_arc1_tiny_tokenizer.json"))

weather = ToolSpec("get_weather", "Get the current weather for a city.", [ToolParam("city", description="City name")])
agent.run("is it raining in Tokyo?", tools=[weather], cycles=2)
# {"function_calls": [{"name": "get_weather", "arguments": {"city": "Tokyo"}}], "confidence": ..., "latency_ms": ...}

agent.classify("my card was charged twice", ["billing", "technical support", "sales"],
               task="Route the support ticket to the right team.",
               descriptions={"billing": "charges, invoices, refunds"})  # hints are optional
# {"label": "billing", "confidence": ..., "distribution": {...}, "latency_ms": ...}
```

```bash
python examples/train_arc1.py --preset arc1-tiny --steps 4000   # train + calibrate + evaluate (plain recipe)
python examples/serve_arc1_api.py                                # port 8002: /run /extract /classify /embed
python examples/export_arc1.py --config Models/arc1_arc1_tiny.config.json     --weights Models/arc1_arc1_tiny.weights.h5 --cycles 2 --tflite
# Docs sandbox: cd arcane-docs-web && npm run dev:with-arc1  →  /docs/arc-1
```

Held-out metrics are written to `Models/arc1_arc1_tiny.metrics.json`: unseen argument values, tools
never seen in training, extraction field F1, classification accuracy (held-out wordings and unseen
tool intents), real-utterance intent accuracy, latency, and calibration error before and after
temperature fitting. Limits: one call per tool per request, long text is truncated to `seq_len`, and
the only language knowledge is what was distilled from a small text encoder (below), so classification
is still weakest on wordings unlike its training data and it has no world knowledge.

**Training recipe (distilled).** The shipped `arc1-tiny` is trained in three stages with a frozen
[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) as an offline teacher. The
teacher is not part of the model and not needed at inference: the exported model keeps the same
architecture, tokenizer, size, and speed.

1. **A, contrastive pre-training**: an InfoNCE loss pulls each request toward its tool or intent
   engram (35k request→tool/intent pairs from the synthetic tools, CLINC150, and Banking77), while
   ARC 1's request and engram vectors learn to match the teacher's similarity structure.
2. **B, hard negatives**: the same, with each positive's nearest wrong tools or intents under the
   teacher added as negatives.
3. **C, multi-task**: the usual fire/anchor/select/embed losses, with the distillation loss and
   40% contrastive replay from A mixed in so the paraphrase knowledge is not forgotten, then a
   continuation at a lower learning rate and lighter auxiliary losses.

On 600 real CLINC150 utterances for intents no training corpus contained (5 labels, chance 20%) this
lifted accuracy from 39% to 58%, and unseen-tool classification from 39% to 70%. The cost is about
6 points of exact tool calls and extracted-record accuracy against the previous checkpoint
(see [ARC1.md](ARC1.md) for the full comparison).

```bash
# Retrain (CPU: about 1 day). Teacher vectors need torch and transformers>=4.51 in a separate env.
mkdir -p data/external   # CLINC150 + Banking77: the two curl commands are in data/DATA_LICENSE.md
python -m gpbacay_arcane.arc1_distill --out data/arc1_teacher
python examples/dump_arc1_teacher.py --corpus data/arc1_teacher    # ~75 min on a laptop CPU
python examples/train_arc1.py --teacher-cache data/arc1_teacher --out-dir Models/distill --skip-eval
# continue stage C from that result at a lower learning rate with lighter auxiliary losses
mkdir Models/distill2
cp Models/distill/arc1_arc1_tiny.weights.h5 Models/distill2/arc1_arc1_tiny.stageB.weights.h5
cp Models/distill/arc1_arc1_tiny_tokenizer.json Models/distill2/
python examples/train_arc1.py --teacher-cache data/arc1_teacher --out-dir Models/distill2 --start-stage C \
    --steps 2500 --learning-rate 1e-3 --distill-weight 0.1 --replay-weight 0.2 --seed 1 --skip-eval
python examples/train_arc1.py --eval-only --out-dir Models/distill2
python examples/compare_arc1.py old=Models/arc1_arc1_tiny new=Models/distill2/arc1_arc1_tiny
```

### Grounded Graph of Thought — Document Knowledge Graph and Verified Reasoning

The document graph is a Python port of [graph-of-thought](https://github.com/gpbacay/graph-of-thought).
Headings become nodes, linked by structure, by explicit cross-references ("see Configuration") and by shared
terms. Search uses BM25 seeds followed by a bounded graph expansion, and every hit says how it was reached.
Retrieval needs no API keys or vector database. Saved graphs use the same JSON format as the Node package.

```python
from gpbacay_arcane import DocumentGraph, GroundedGraphOfThought, got_tools

graph = DocumentGraph()
graph.add_document(markdown, "User Guide")       # same title/doc_id again replaces it
graph.search("config fails")                     # hits with score, hops, via, edgeType, path
context = graph.retrieve("config fails")         # prompt-ready "### Title [node-id]" blocks

# Graph-of-Thoughts (Besta et al., 2023) with the LLM judge replaced by a check against the graph.
# `llm` is any prompt -> text callable. Default plan: 1 to 4 LLM calls.
result = GroundedGraphOfThought(graph, llm=my_llm).reason("My app cannot reach the database. What should I check?")
result["answer"], result["claims"], result["unsupported"], result["citations"]

tools = got_tools(graph)                         # Arcane ToolSpecs; schema_dict() for any tool-calling API
```

Every thought's sentences are checked against the document graph when the thought is created
(`graph.support(sentence, node_ids)`, IDF-weighted term coverage, no LLM call). That check:

- **ranks candidates** in place of an LLM scoring step;
- **gates merges and rewrites**: a merge or rewrite that lowers the answer's grounding is discarded;
- **drives retrieval**: each unsupported sentence is used as a graph query before any rewrite, and only
  sentences still unsupported go back to the LLM;
- **decides citations**: only sections that back a supported sentence are cited.

Supported sentences are weighted by how relevant their sections are to the question, so an answer
grounded in off-topic sections loses to an on-topic one, and an answer with nothing checkable ranks last.
The check is lexical: paraphrases score lower and a negated claim still matches unless you pass
`verify=`, any `(claim, evidence) -> entailment probability` function such as an NLI model. `embed=` takes any
`text -> vector` function for semantic links; arc1-tiny's embeddings are too weak for this, so use a
sentence-embedding model. `got_tools` is read-only unless you pass `writable=True`. The bundled `arc1-tiny`
is not trained to route these tools, so drive them with an LLM, or fine-tune ARC 1 before you hand them
to `Arc1Agent`.

#### Hippocampus — a fast-learning memory for any decision model

Named for the brain's hippocampus, which learns single episodes at once and lets the slower neocortex use
them, Hippocampus is a memory of decided examples next to a model, with no change to the model's weights.
RAG lets you use an LLM on your own knowledge without fine-tuning it. Hippocampus does the same for a System 1
decision model: instead of training ARC 1 on your tools and labels, give it a few decided examples. A
model like this can't read retrieved text, so Hippocampus retrieves *decisions*. Examples live in a
`DocumentGraph`. At request time the most similar ones vote for their labels, weighted by search score,
and the vote is combined with the model's own probabilities. A tool example can also carry its arguments.
It then works as a pattern ("rate {title} {stars} stars"), and a request that matches it gets its argument
values copied from its own words. Examples take effect on the next request. Labels with no examples fall
back to the model alone.

```python
from gpbacay_arcane import Hippocampus, load_arc1

hippocampus = Hippocampus(load_arc1())                                   # or any model, or None for memory only
hippocampus.remember("I was charged twice this month", "billing")    # a class label...
hippocampus.remember("rate Dune 4 stars", "rate_movie", {"title": "Dune", "stars": 4})  # ...or a tool call
hippocampus.remember("switch bluetooth off", "toggle_bluetooth", {"enabled": False})
hippocampus.remember("thanks, that's all", None)                     # None = no tool applies

hippocampus.classify("why is my card charged again?", ["billing", "shipping"])  # + "neighbors" (the evidence)
hippocampus.run("rate spirited away 5 stars", tools=my_tools)        # rate_movie(title="spirited away", stars=5)
hippocampus.react("charged again?")                                  # the memory's vote alone, no model call
hippocampus.forget("thanks, that's all")                             # re-remembering a text replaces it
```

- **Model-agnostic.** `model` can be `None` (memory only; abstains with `label=None` when nothing similar
  is stored), any `(text, labels) -> {label: probability}` callable (a classifier, or an LLM asked for
  probabilities), or an object with `classify(text, labels, **kw)` returning a `"distribution"`, like
  `Arc1Agent`. For tool calling, the model needs `run(prompt, tools=, tool_prior=, execute=)` returning
  `function_calls`; `Arc1Agent.run` combines the prior with its own firing probability by noisy-OR.
  Arguments copied from a matching pattern replace the model's for that tool, and a call the model held
  back is added; `confidence` is then `None`, since the model's calibrated probability no longer describes
  the call. A model call that copies words a matched pattern explains is dropped (a word belongs to one
  argument). `memory_arguments` names the tools whose arguments came from memory.
- **Dynamic.** `remember` / `forget` take effect immediately. Labels and tools are given per request, and
  `to_json` / `from_json` save the memory, arguments included, with the graph (examples are documents with
  ids starting `memory-`).
- **Scalable.** Inserts don't scan the memory. The default graph skips semantic links, which on CLINC150
  lowered accuracy and cost 15 ms per insert at 15k examples. Lookups touch only examples that share words
  with the request.

`python examples/benchmark_hippocampus.py --shots 1 2 5 10` (arc1-tiny, same weights, no fine-tuning).
CLINC150 intents held out of ARC 1's training, real text, 5 labels:

| Examples per label | ARC 1 alone | Memory alone | Hippocampus |
|---|---|---|---|
| none | 56.5% | – | – |
| 1 | – | 64.7% | **80.5%** |
| 5 | – | 87.3% | **90.5%** |
| 10 | – | 92.0% | **93.3%** |

The held-out tool split (tools ARC 1 never trained on), 300 requests. Fully correct means the right tools
with every argument right:

| Examples per tool | Right tool | Fully correct, labels only | Fully correct, with arguments | Arguments right |
|---|---|---|---|---|
| none (ARC 1) | 55.7% | 42.3% | – | 18.6% |
| 1 | 77.0% | 50.3% | **64.0%** | 53.3% |
| 2 | 83.7% | 52.0% | **70.7%** | 64.1% |
| 5 | 93.7% | 54.7% | **86.0%** | 83.5% |
| 10 | 99.3% | 54.7% | **99.3%** | 100% |

For comparison, arc1-tiny's fully correct calls on the tools it *was* trained on: 86.0% (a different test set). Refusals stayed at
100% throughout. Caveat: these held-out tools are tested with the same sentence patterns used to write the
examples, only with new values, so by 10 examples the memory has seen every phrasing. Real requests vary
more, and a phrasing no example covers falls back to ARC 1's own argument copying. Store varied examples.

Memory only, all 150 CLINC150 intents with 15,000 examples: 80.6% (150-way), 0.13 ms per insert, 3.4 ms
median / 11 ms p90 per decision on a busy laptop CPU. Retrieval is lexical; ARC 1's embeddings as
`DocumentGraph(embed=...)` did not meaningfully help on CLINC150.

### Distilling Qwen2.5-0.5B into ARCANE

The `distill` preset is tuned as a distillation target for a softmax teacher:
hybrid attention (3 of 12 layers softmax), RoPE, learned KV decay, a wider FFN
and RMSNorm. See [docs/DISTILLATION.md](docs/DISTILLATION.md) for the full
rationale and measurements.

```python
from gpbacay_arcane import ArcaneSLMConfig

cfg = ArcaneSLMConfig.from_preset("distill")
print(cfg.estimate_trainable_parameters())  # 145,147,509
print("".join("S" if k == "softmax" else "L" for k in cfg.attention_types()))
# LLLSLLLSLLLS
```

```bash
pip install -r requirements-distill.txt

# 1. Dump teacher top-k logits to TFRecord (PyTorch side)
python examples/dump_qwen_logits.py --text-file data/corpus.txt     --out-dir data/qwen_shards --seq-len 512 --top-k 64

# 2. Distil into ARCANE (TensorFlow side, no torch needed)
python examples/distill_arcane_slm.py --shards "data/qwen_shards/*.tfrecord"     --preset distill --steps 20000

# 3. Parameter-matched control -- run this or you are measuring the corpus,
#    not the distillation
python examples/distill_arcane_slm.py --shards "data/qwen_shards/*.tfrecord"     --baseline-transformer --steps 20000
```

Qwen2.5-0.5B is Apache-2.0, so the teacher, the dumped predictions and the
distilled student are all yours to release.

## Documentation Portal

ARCANE comes with a dedicated documentation web application built with Next.js, providing in-depth explanations of the underlying mechanisms and research papers.

To run the documentation portal locally:

```bash
cd arcane-docs-web
npm install
npm run dev
```

The portal will be available at `http://localhost:3000`.

## Available Models

ARCANE provides three main model classes for different use cases:

### ArcaneSmallLanguageModel
Causal decoder language model (~100M with `from_preset("100m")`). Uses causal linear attention, DenseGSER, bioplastic projection, token-parallel resonance, and AttentionResidual. Best for:
- Next-token language modeling
- Scaling ARCANE layers to SLM size
- Autoregressive generation

### HierarchicalResonanceFoundationModel
Advanced model with multi-level neural resonance, temporal coherence, and attention fusion. Best for:
- Complex reasoning tasks
- Research applications
- When training stability is crucial
- Maximum biological accuracy

### NeuromimeticSemanticModel
Standard neuromimetic model with biological learning rules. Best for:
- General NLP tasks
- Faster training and inference
- Balanced performance and biological plausibility
- Prototyping and experimentation

## Available Layers

| Layer | Description |
|-------|-------------|
| `GSER` | Gated spiking elastic reservoir; recurrent weights scaled to `spectral_radius` |
| `DenseGSER` | Dense map with leak-controlled spike gating and optional conceptual gate (not a reservoir) |
| `ResonantGSER` | Hierarchical resonant RNN; closed-form EMA toward a top-down prototype |
| `PredictiveResonantLayer` | Local predictive resonance; alignment is per-example in RNN state |
| `BioplasticDenseLayer` | Dual kernel; optional inference-time BCM + homeostasis on `plastic_kernel` |
| `HebbianHomeostaticNeuroplasticity` | Trainable dense + Hebbian plastic kernel and homeostatic gain |
| `RelationalConceptModeling` | Multi-head self-attention wrapper |
| `RelationalGraphAttentionReasoning` | Self-attention plus pooled classifier |
| `RelationalConceptGraphReasoning` | Stacked MHA with residual/norm; not a graph network |
| `MultiheadLinearSelfAttentionKernalization` | Katharopoulos linear attention (`Kernalization` is a historical spelling) |
| `CausalLinearSelfAttention` | Causal prefix (cumsum) variant of Katharopoulos attention |
| `ResonantSequenceMixer` | Token-parallel closed-form resonance with a causal running-mean prototype |
| `ArcaneDecoderBlock` | SLM block: causal attention + DenseGSER + bioplastic + resonance |
| `AttentionResidual` | Softmax over depth of prior block outputs (AttnRes) |
| `LatentTemporalCoherence` | Mean-pool then linear projection |
| `SpatioTemporalSummarization` | Local GLU + sequence summary (softmax over time when weighted) |
| `PositionalEncodingLayer` | Sinusoidal positional encoding added to the sequence |

## CLI Commands

```bash
# Show library information
gpbacay-arcane-about

# List available models
gpbacay-arcane-list-models

# List available layers
gpbacay-arcane-list-layers

# Show version
gpbacay-arcane-version
```

## Performance and Benchmarks

Exploratory Tiny Shakespeare run (15k chars, 10 epochs, **unequal parameter counts**). Treat as a smoke comparison, not a Transformer result.

| Model | Val Accuracy | Val Loss | Training Time | Parameters |
|-------|--------------|----------|---------------|------------|
| Traditional Deep LSTM | 9.50% | 6.85 | ~45s | ~195K |
| ARCANE Neuromimetic | 10.20% | 6.42 | ~58s | ~220K |
| ARCANE Hierarchical Resonance | 11.25% | 6.15 | ~95s | ~385K |

### Unit tests

```bash
python -m pytest tests/test_mechanism_correctness.py tests/test_activations.py tests/test_resonant_gser.py tests/test_homeostatic_plasticity.py tests/test_arcane_slm.py -q
```

## Project Structure

```
gpbacay_arcane/
├── gpbacay_arcane/          # Core library
│   ├── __init__.py          # Module exports
│   ├── activations.py       # Neuromimetic activations
│   ├── callbacks.py         # Training callbacks
│   ├── cli_commands.py      # CLI interface
│   ├── foundational_models.py # Foundation model architectures
│   ├── language_model.py    # Causal ARCANE small language model (100m / distill)
│   ├── distillation.py      # Top-k KD loss, TFRecord shards, distillation trainer
│   ├── qwen_vocab.py        # Qwen BPE -> compact student vocabulary adapter
│   ├── tokenization.py      # Byte-level BPE tokenizer
│   ├── layers.py            # High-level neural layers
│   ├── mechanisms.py        # Core neural mechanisms
│   ├── models.py            # Standard models
│   └── ollama_integration.py # Ollama integration
├── arcane-docs-web/         # Documentation web portal (Next.js)
├── examples/                # Usage examples
│   ├── arcane_foundational_model.py
│   ├── create_foundation_model.py
│   ├── train_arcane_slm.py
│   ├── dump_qwen_logits.py
│   ├── distill_arcane_slm.py
│   ├── train_hierarchical_resonance.py
│   ├── train_neuromimetic_sm.py
│   └── test_hierarchical_resonance_comparison.py
├── tests/                   # Unit tests (see test_mechanism_correctness.py)
├── docs/                    # Research and technical documentation
│   ├── NEURAL_RESONANCE.md
│   ├── RESONANT_GSER.md
│   ├── PREDICTIVE_RESONANT_LAYER.md
│   ├── HOMEOSTATIC_PLASTICITY.md
│   ├── INFERENCE_TIME_RESONANCE.md
│   └── ACTIVATIONS.md
├── data/                    # Sample datasets
│   └── shakespeare_small.txt
├── setup.py                 # Package configuration
├── requirements.txt         # Dependencies
└── README.md
```

## Contributing

We welcome contributions to advance neuromimetic AI:
1. Research: Novel biological neural mechanisms.
2. Engineering: Performance optimizations and scaling.
3. Applications: Domain-specific implementations.
4. Documentation: Tutorials and examples.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Neuroscience Research: Inspired by biological brain principles.
- Reservoir Computing: Building on echo state network principles.
- Hebbian Learning: Based on Donald Hebb's fundamental work.
- Open Source Community: Built with TensorFlow and Python.

## Contact

- Author: Gianne P. Bacay
- Email: giannebacay2004@gmail.com
- Project: [GitHub Repository](https://github.com/gpbacay/gpbacay_arcane)

---

**"Neurons that fire together, wire together, and now they learn together."**

*ARCANE - Building the future of biologically-inspired AI, one neural connection at a time.*

