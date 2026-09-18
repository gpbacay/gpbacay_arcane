# ARCANE

**Augmented Reconstruction of Consciousness through Artificial Neural Evolution**

A Python library for building neuromimetic AI models inspired by biological neural principles. ARCANE provides researchers and developers with biologically-plausible neural layers, models, and training mechanisms that bridge neuroscience and artificial intelligence.

## What is ARCANE?

ARCANE is a comprehensive Python library that enables you to build, train, and deploy neuromimetic AI models. Unlike traditional deep learning frameworks, ARCANE incorporates biological neural principles such as:

- **Neural Resonance**: Bi-directional prototype alignment between ResonantGSER layers, plus local per-example alignment in PredictiveResonantLayer. Optional inference-time BCM plasticity.
- **Spiking Neural Dynamics**: LIF-style leak, threshold, and subtractive reset with a straight-through estimator so spikes can train.
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

