# ARCANE

**Augmented Reconstruction of Consciousness through Artificial Neural Evolution**

**The first library for making true System 1 AI models through adaptive inference, grounded decisions, neural memory, and biologically inspired model components.**

ARCANE 5.0 is a Python library for models that make fast, bounded decisions rather than defaulting to open-ended token generation. A System 1 model in ARCANE has a declared output space, an explicit abstention policy, inspectable evidence, optional memory of prior decisions, and controlled inference-time adaptation.

“System 1” is an operational engineering profile, not a claim of consciousness or biological equivalence. The biological mechanisms in ARCANE are testable design inspirations whose value must be established on the target task.

## ARCANE 5.0

ARCANE 5.0 unifies four capabilities behind one deployment contract:

- **Adaptive inference** — refine transient state or bounded fast weights against a measurable objective, with trust regions and rollback.
- **Grounded decisions** — return a label, action, field, score, or abstention constrained to a declared schema.
- **Neural memory** — teach decisions from examples immediately and expose the recalled evidence without retraining model weights.
- **Biologically inspired components** — compose feedback, resonance, spiking dynamics, Hebbian/BCM plasticity, reservoirs, and homeostasis.

The `System1` API composes these pieces while ARC 1, Hippocampus, Graph of Thought, and every neural layer remain available independently.

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
- **HierarchicalResonanceFoundationModel**: Research model with a configurable multi-level resonance hierarchy.
- **NeuromimeticSemanticModel**: Compact wrapper for semantic-model prototyping.
- **Custom Architecture Support**: Build your own models using individual layers.

### Training and Generation Tools
- **Explicit Resonance APIs**: Refine the current input with per-example targets; the legacy callback stages only a previous-batch prototype.
- **Test-Time Adaptation**: Optional entropy, consistency, reconstruction, masked-token, or custom objectives with bounded updates and rollback.
- **Multi-Temperature Generation**: Conservative, balanced, and creative text generation modes.
- **Dynamic Self-Modeling**: Adaptive reservoir sizing during training.
- **CLI Tools**: Command-line utilities for model management and information.

### Research-Focused Design
- **Biologically Inspired**: Implements testable analogies to several neural mechanisms.
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

### Build a System 1 decision model

```python
from gpbacay_arcane import DecisionSpace, System1

# Use ARC 1, a TensorFlow classifier, or any callable that returns scores.
def route(text, labels):
    return {
        "billing": 0.72 if "charged" in text.lower() else 0.10,
        "shipping": 0.18,
        "other": 0.10,
    }

support = System1(
    predictor=route,
    decisions=DecisionSpace(
        ["billing", "shipping", "other"],
        min_confidence=0.60,
        min_margin=0.15,
    ),
    memory_weight=0.65,
)

# Neural memory changes the next decision without fine-tuning the predictor.
support.remember("my card was charged twice", "billing")

decision = support.decide("why was I charged again?")
print(decision.label, decision.confidence, decision.source)
print(decision.evidence)       # recalled examples behind the result
print(decision.adaptation)     # bounded-update diagnostics, when enabled
```

Every call returns a `GroundedDecision`. Its label is always in the declared `DecisionSpace`, or `None` when the model abstains. See [`docs/ARCANE_5_ARCHITECTURE.md`](docs/ARCANE_5_ARCHITECTURE.md) for the contract and migration guide.

### Advanced Usage with Resonance

```python
from gpbacay_arcane import HierarchicalResonanceFoundationModel

# Create a configurable resonant hierarchy
model = HierarchicalResonanceFoundationModel(
    vocab_size=3000,
    seq_len=32,
    hidden_dim=128,
    num_resonance_levels=4
)

model.build_model()
model.compile_model(learning_rate=3e-4)

# Standard task training
model.model.fit(X_train, y_train)

# Train task weights and feedback projections with explicit alignment loss
model.train_feedback_step(X_train[:32], labels=y_train[:32])

# Refine the current batch while keeping learned weights fixed
predictions = model.predict_with_resonance(X_test, resonance_cycles=4)

# Optional objective-driven adaptation; the checkpoint is restored afterward
adaptation = model.adapt_at_inference(
    X_test,
    variables="all",
    objective="entropy",
    steps=2,
    max_update_norm=1e-3,
    episodic=True,
)

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

ARC 1 is an ultra-compact, non-autoregressive "System 1" decision model. In one bounded inference
it turns text into calibrated, typed output (tool calls, extracted records, classification labels,
or embeddings) in milliseconds. It never generates prose. Its architecture, **Resonant Schema
Binding**, is built from ARCANE mechanisms:

1. **Aligned perception**: bidirectional `Arc1PerceptionBlock`s read the utterance, then bounded RSAA
   cycles align adjacent hierarchy states and re-evaluate the current input with fixed weights.
   The blocks contain
   (`FieldAttention` → `ResonantChannelMixer` → `ConceptEngram` → `FieldResonance`).
2. **Schema engrams**: every tool, argument, enum option, and label is pooled into one vector by the
   same blocks. New schema texts are perceived in the same batch as the utterance, then cached by
   `SchemaMemory`, so all examples remain independent inside one batched computation graph.
3. **Resonant binding**: each engram is a probe that resonates with the utterance field for a few
   shared-weight cycles through a GSER spiking gate (`ResonantBinding`). All probes bind in parallel.
4. **Readouts**: `fire` (tool applies / optional argument present / boolean), `anchor` (start/end
   pointer that copies strings and numbers from the user's words), `select` (enum option or
   classification label), embedding.

A request has no autoregressive decoding loop. The `arc1-tiny` and `arc1` presets expose up to two
state-only RSAA cycles; `rsaa_cycles`, `rsaa_factor`, and `rsaa_epsilon` bound alignment and expose convergence.
The current checkpoint defaults to zero after a same-checkpoint ablation found lower aggregate quality with RSAA enabled.
The complete result is recorded in [`Models/arc1_rsaa_comparison.json`](Models/arc1_rsaa_comparison.json).
Every readout has a temperature fitted on held-out data, so `confidence` is calibrated. Binding
`cycles` independently trades accuracy for speed with the same weights. Pre-5.0 configs load with
RSAA disabled for checkpoint compatibility.

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
- **Decides what to remember (Python).** `observe(text, label, trust=1.0)` is the automatic alternative to
  `remember`: give it the outcome (a user correction, a tool that ran, a ground truth) and it stores the
  example only if the memory and model don't already decide it confidently, judged over `labels=` (default:
  every label in the memory; `System1.observe` uses its decision space). `observe` never overwrites an
  example you `remember`ed. `trust` weighs the signal;
  evidence for the same text and label accumulates, is cancelled by evidence for another label, and commits
  at `threshold` (integrate-and-fire), so a weak or noisy signal has to repeat before it becomes memory.
  With the defaults it commits at once. `leak` (off by default) makes old evidence fade by that share per
  call; `max_pending` bounds the candidates waiting for more evidence. `max_size` caps the memory by
  forgetting the oldest `observe`d example; examples you `remember` are never evicted. `remember` and `forget` stay as
  the manual override. It needs an outcome signal. The JavaScript port has it too (`await hippocampus.observe(...)`).
- **Learns from a stronger model (`teacher=`).** With no person in the loop, a memory can only echo its model:
  storing its own confident predictions lowered accuracy in our tests, so there is no such mode. Give
  `Hippocampus(model, teacher=fn)` a stronger, slower model instead (an LLM, a bigger classifier) as
  `fn(text, labels) -> label | {label: probability} | None`. When `classify` is under `confident` and the memory
  alone isn't decisive (three neighbors holding 80% of the vote), it asks the teacher, returns the answer
  (`source="teacher"`) and remembers it, so the next similar request is decided here without the teacher.
  A teacher answer under `confident` isn't trusted, a teacher that raises leaves the memory's own decision
  with `teacher_error`, and a text you `remember`ed is never put to it. The teacher is a plain callable in
  Python and JavaScript; no LLM client is bundled.
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

`python examples/benchmark_hippocampus.py --policy` compares `remember` alone with `observe` (memory only,
CLINC150 150-way, 3,000 training examples in a shuffled stream, mean of 3 seeds, 1,000 test utterances):

| Training stream | `remember` | `observe` |
|---|---|---|
| Clean labels | 3,000 examples, 74.0% | 2,513 examples, 74.2% |
| Clean, budget of 1,500 examples | random subset: 68.3% | `max_size=1500`: 69.2% |
| 20% wrong labels, each text seen 3 times | 3,000 examples, 71.0% | 1,977 examples, 70.9% (`trust=0.5`, `threshold=0.9`) |

`observe` keeps accuracy with about 16% fewer examples on clean labels, and does a little better than a random
subset under a memory budget (ahead in all 3 seeds, by about a point). On noisy labels it matches `remember`
with 34% fewer examples; it has no accuracy advantage there, so don't expect it to filter bad feedback.
Scoring examples by how often a lookup agreed with the outcome did no better than evicting the oldest, so
eviction is plain FIFO. `leak` is off because it has to match the gap between repeats of a text: on this
stream `leak=0.01` stored 32 examples at 10.5% accuracy and a tuned `5e-5` scored 70.2%, below no leak at all.
The old `remember` path scores the same as before this change (74.0% vs 74.5% on the clean stream).

`python examples/benchmark_hippocampus.py --teacher` measures the teacher (ARC 1 + memory, all 29 CLINC150
held-out intents as labels, a stream of 1,740 unlabeled utterances, 600 test utterances afterwards with the
teacher switched off, mean of 3 seeds). **The teacher is simulated** (right with probability 100% or 80%, errors
at random); no real LLM was run:

| Hand-labeled per intent | No teacher | Teacher always right | Teacher 80% right | Every stream item labeled by hand |
|---|---|---|---|---|
| 1 | 62.0% | **94.0%**, asked 787 times (45%) | **92.3%**, asked 1,144 times (66%) | 93.5% |
| 5 | 81.7% | **93.6%**, asked 710 times (41%) | **92.4%**, asked 971 times (56%) | 93.7% |

The memory reaches the hand-labeled ceiling while the teacher is asked about under half of the stream, and an
80%-right teacher still gets within about a point, because the vote outweighs its mistakes. Without the
decisive-memory check the teacher was asked 98% of the time for the same accuracy. A real LLM's mistakes are
likely correlated (it confuses the same similar intents), which this does not model, so treat 92% as a best case
for a teacher of that accuracy.

#### Occipital — the visual pathway of Hippocampus

Hippocampus recalls examples by whole words, so a request with a typo or another word form ("parcle",
"refunded") gets no vote from an example it obviously matches. Occipital reads the way the brain's visual word
form area does, by letters: it indexes every remembered example by letter 3- and 4-grams and fuses those scores
with the word search before the neighbors vote. It changes which examples vote, not the model, so every System 1
model behind the memory (ARC 1, a classifier, an LLM router) decides better.

```python
from gpbacay_arcane import Hippocampus, Occipital, load_arc1

hippocampus = Hippocampus(load_arc1(), occipital=Occipital())   # or any model, or None for memory only
hippocampus.remember("my parcel never arrived", "shipping")
hippocampus.classify("my parcle nevr arived", ["billing", "shipping"])   # found through its letters
```

`python examples/benchmark_hippocampus.py --shots 1 5 10 --occipital` (same protocol as the tables above):

| | Words only | + Occipital |
|---|---|---|
| ARC 1 + memory, CLINC150 unseen intents, 1 / 5 / 10 examples | 80.5% / 90.5% / 93.5% | **81.3% / 92.2% / 94.5%** |
| Memory alone, 150 intents, 15,000 examples | 80.1% | **83.2%** |
| Median decision at 15,000 examples | 4.2 ms | 16 ms |

With typos in the requests (`node/bench-occipital-arc1.mjs`) the default settings gain about as much, and
memory-only abstention on CLINC150's out-of-scope requests stays within half a point. `min_coverage` (default 0.3)
trades abstention for accuracy (at 0.1, ARC 1 gains 5 to 7 points on typos), and `max_df` (default 0.1) trades accuracy for speed; both are documented in `occipital.py`. For
tool calling it doesn't help: tool choice and exact calls were unchanged, and with one example per tool it made
ARC 1 fire a tool on 1.2% of requests that need none, so leave it off for a tool memory. In Node.js, Occipital
can also store a whole memory as QR-code frames of an MP4 ([node/README.md](node/README.md#occipital-the-visual-pathway-of-hippocampus)).

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

The portal's ARC 1 demo and chatbot always serve the latest trained ARC 1, with Hippocampus and Occipital:

- The demo calls `examples/serve_arc1_api.py` (Render), which loads `Models/arc1_arc1_tiny.*` straight from the
  repository on every deploy.
- The chatbot runs ARC 1 in the browser from `node/web/arc1.onnx`. That file and the bundled
  `gpbacay_arcane/models/arc1-tiny.rcn` are built from the same trained files by `python examples/publish_arc1.py`.
- `.github/workflows/publish-arc1.yml` runs that script whenever the trained files change and commits the result.
  The portal's build refuses a browser model older than the weights.
- To confirm both run the same model, compare `/health`'s `weights` with `source.sha256` in `node/web/arc1.json`.

## Available Models

ARCANE provides the `System1` composition API plus model classes for different use cases:

### System1
Deployment wrapper for any classifier or ARCANE model. It applies a closed decision schema, confidence and margin thresholds, neural-memory recall, optional bounded adaptation, and returns evidence-rich `GroundedDecision` objects.

### ArcaneSmallLanguageModel
Causal decoder language model (~100M with `from_preset("100m")`). Uses causal linear attention, DenseGSER, bioplastic projection, token-parallel resonance, and AttentionResidual. Best for:
- Next-token language modeling
- Scaling ARCANE layers to SLM size
- Autoregressive generation

### HierarchicalResonanceFoundationModel
Research model with multi-level neural resonance, temporal coherence, and attention fusion. Best for:
- Resonance cycle-count ablations
- Feedback-projection experiments
- Test-time adaptation research
- Studying hierarchical recurrent state refinement

### NeuromimeticSemanticModel
Compact neuromimetic model wrapper. Best for:
- Small semantic-model experiments
- Lower-depth baselines
- Prototyping optional plasticity and homeostasis

## Available Layers

| Layer | Description |
|-------|-------------|
| `GSER` | Gated spiking elastic reservoir; recurrent weights scaled to `spectral_radius` |
| `DenseGSER` | Dense map with leak-controlled spike gating and optional conceptual gate (not a reservoir) |
| `ResonantGSER` | Hierarchical resonant RNN; closed-form alignment toward per-example top-down targets |
| `PredictiveResonantLayer` | Local predictive resonance; alignment is per-example in RNN state |
| `HierarchicalHomeostaticController` | Bounded multi-level activity regulation and hierarchy signals for plastic layers |
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
1. Research: Controlled evaluations of biologically inspired mechanisms.
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

*ARCANE — experimental, biologically inspired neural components with measurable behavior.*

