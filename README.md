# A.R.C.A.N.E.

**Augmented Reconstruction of Consciousness through Artificial Neural Evolution**

A Python library for building neuromimetic AI models inspired by biological neural principles. A.R.C.A.N.E. provides researchers and developers with biologically-plausible neural layers, models, and training mechanisms that bridge neuroscience and artificial intelligence.

## What is A.R.C.A.N.E.?

A.R.C.A.N.E. is a comprehensive Python library that enables you to build, train, and deploy neuromimetic AI models. Unlike traditional deep learning frameworks, A.R.C.A.N.E. incorporates biological neural principles such as:

- **Neural Resonance**: Bi-directional state alignment between neural layers, enabling **Inference-Time State Adaptation** and **Inference-Time Learning**
- **Spiking Neural Dynamics**: Realistic neuron behavior with leak rates and thresholds
- **Hebbian Learning**: "Neurons that fire together, wire together" plasticity rules
- **Homeostatic Plasticity**: Self-regulating neural activity for stable representations
- **Hierarchical Processing**: Multi-level neural architectures for complex reasoning

The library provides ready-to-use models, customizable neural layers, and training callbacks that make it easy to experiment with biologically-inspired AI architectures.

## Key Features

### 🧠 Biological Neural Layers
- **ResonantGSER**: Spiking neural dynamics with reservoir computing and spectral radius control
- **BioplasticDenseLayer**: Hebbian learning with homeostatic plasticity regulation
- **Hierarchical Resonance**: Multi-level neural architectures with bi-directional feedback
- **Neural Reservoir Computing**: Dynamic temporal processing with configurable parameters

### 🏗️ Ready-to-Use Models
- **HierarchicalResonanceFoundationModel**: Advanced model with multi-level resonance hierarchy
- **NeuromimeticSemanticModel**: Standard neuromimetic model with biological learning rules
- **Custom Architecture Support**: Build your own models using individual layers

### ⚡ Training & Generation Tools
- **Neural Resonance Callbacks**: Orchestrate the "thinking phase" during training
- **Multi-Temperature Generation**: Conservative, balanced, and creative text generation modes
- **Dynamic Self-Modeling**: Adaptive reservoir sizing during training
- **CLI Tools**: Command-line utilities for model management and information

### 🔬 Research-Focused Design
- **Biologically-Plausible**: Grounded in neuroscience principles
- **Highly Configurable**: Extensive parameter control for experimentation
- **Extensible Architecture**: Easy to add new layers and mechanisms
- **Performance Monitoring**: Built-in callbacks for tracking neural dynamics

## Installation

### Prerequisites
- Python 3.11+
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

### Installation

```bash
pip install gpbacay-arcane
```

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

## Usage

### Complete Training Example

```python
import numpy as np
from gpbacay_arcane import NeuromimeticSemanticModel
from tensorflow.keras.preprocessing.text import Tokenizer # Or any other data preprocessor

# 1. Prepare your semantic data
semantic_data = "your training text here..." # Or other multi-modal data

# 2. Create and train tokenizer/data preprocessor
tokenizer = Tokenizer(num_words=1000, oov_token="<UNK>") # Example for text
tokenizer.fit_on_texts([semantic_data])

# 3. Initialize the neuromimetic model
model = NeuromimeticSemanticModel(
    vocab_size=len(tokenizer.word_index) + 1, # Adjust vocab_size based on data type
    seq_len=16,
    embed_dim=32,
    hidden_dim=64
)

# 4. Build and compile the model
neuromimetic_model = model.build_model()
model.compile_model(learning_rate=1e-3)

# 5. Generate semantic output after training
generated_output = model.generate_text(
    seed_text="artificial intelligence is", # Or other initial semantic input
    tokenizer=tokenizer,
    max_length=50,
    temperature=0.8  # 0.6=conservative, 0.9=balanced, 1.2=creative
)
print(f"Generated: {generated_output}")
```

### Using Individual Neural Layers

```python
from gpbacay_arcane.layers import ResonantGSER, BioplasticDenseLayer
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# Build custom architecture with neuromimetic layers
inputs = Input(shape=(16, 32))  # (sequence_length, embedding_dim)

# Hierarchical Resonant Layer with spiking dynamics
resonant_layer = ResonantGSER(
    units=64,
    spectral_radius=0.9,
    leak_rate=0.1,
    spike_threshold=0.35,
    activation='gelu',
    resonance_factor=0.1
)(inputs)

# Hebbian learning layer
hebbian_layer = BioplasticDenseLayer(
    units=128,
    learning_rate=1e-3,
    target_avg=0.11,
    homeostatic_rate=8e-5,
    activation='gelu'
)(resonant_layer)

# Create custom model
custom_model = Model(inputs=inputs, outputs=hebbian_layer)
```

### Training with Neural Resonance

```python
from gpbacay_arcane.callbacks import NeuralResonanceCallback, DynamicSelfModelingReservoirCallback

# 1. Add resonance callback to synchronize hierarchical layers
resonance_cb = NeuralResonanceCallback(resonance_cycles=5)

# 2. Add self-modeling callback for structural adaptation
modeling_cb = DynamicSelfModelingReservoirCallback(
    reservoir_layer=your_resonant_layer,
    performance_metric='accuracy',
    target_metric=0.98,
    growth_rate=10
)

model.fit(X_train, y_train, callbacks=[resonance_cb, modeling_cb])
```

### Multi-Temperature Semantic Generation

```python
# Conservative semantic generation (coherent, precise)
conservative = model.generate_text(
    seed_text="machine learning",
    tokenizer=tokenizer,
    temperature=0.6,
    max_length=30
)

# Balanced semantic generation (diverse yet coherent)
balanced = model.generate_text(
    seed_text="machine learning",
    tokenizer=tokenizer,
    temperature=0.9,
    max_length=30
)

# Creative semantic generation (exploratory, novel)
creative = model.generate_text(
    seed_text="machine learning",
    tokenizer=tokenizer,
    temperature=1.2,
    max_length=30
)
```

## Available Models

A.R.C.A.N.E. provides two main model classes for different use cases:

### HierarchicalResonanceFoundationModel
Advanced model with multi-level neural resonance, temporal coherence, and attention fusion. Best for:
- Complex reasoning tasks
- Research applications
- When training stability is crucial
- Maximum biological accuracy

```python
from gpbacay_arcane import HierarchicalResonanceFoundationModel, NeuralResonanceCallback

model = HierarchicalResonanceFoundationModel(
    vocab_size=5000,
    seq_len=32,
    hidden_dim=128,
    num_resonance_levels=4
)
model.build_model()
model.compile_model()

# Use neural resonance training
resonance_cb = NeuralResonanceCallback(resonance_cycles=10)
model.model.fit(X_train, y_train, callbacks=[resonance_cb])
```

### NeuromimeticSemanticModel
Standard neuromimetic model with biological learning rules. Best for:
- General NLP tasks
- Faster training and inference
- Balanced performance and biological plausibility
- Prototyping and experimentation

```python
from gpbacay_arcane import NeuromimeticSemanticModel

model = NeuromimeticSemanticModel(vocab_size=1000)
model.build_model()
model.compile_model()
```

### Ollama Integration (Optional)

Create a neuromimetic foundation model by combining Ollama's llama3.2:1b with A.R.C.A.N.E.'s biological neural mechanisms:

```bash
# Install optional dependencies
pip install ollama sentence-transformers

# Install and pull Ollama model
# Download Ollama from: https://ollama.ai
ollama pull llama3.2:1b

# Create the foundation model
python examples/create_foundation_model.py
```

## Understanding Neural Resonance

### What is Neural Resonance?

Neural Resonance is A.R.C.A.N.E.'s core innovation - a biologically-inspired training mechanism that mimics how the brain processes information. Unlike traditional feed-forward networks, Neural Resonance introduces a **"Thinking Phase"** where:

1. **Higher layers project feedback** (expectations) downward to lower layers
2. **Lower layers harmonize** their internal states to match those expectations
3. **Multiple resonance cycles** align the entire hierarchy before weight updates

This process mirrors the brain's predictive coding, enabling more stable and biologically-plausible learning.

### Key Features of Neural Resonance

| Feature | Description |
|---------|-------------|
| **Prospective Configuration** | Neural activities optimized before weight updates |
| **Bi-directional Feedback** | Higher layers send expectations to lower layers |
| **Temporal Coherence** | Distills temporal dynamics into coherence vectors |
| **Attention Fusion** | Multi-pathway aggregation with self-attention |
| **BCM Metaplasticity** | Adaptive learning thresholds |

### When to Use Neural Resonance

✅ **Best for:**
- Complex reasoning tasks requiring deliberation
- Training stability in deep networks
- Biologically-plausible learning dynamics
- Research applications prioritizing generalization

⚠️ **Trade-offs:**
- Slower training due to resonance cycles (~2x vs traditional methods)
- Higher memory usage for state tracking
- Best suited for medium-sized models

## Architecture & Components

### Model Architecture

```
Input (16 tokens)
→ Embedding (32 dim)
→ ResonantGSER₁ (64 units, ρ=0.9, leak=0.1, Resonance)
→ LayerNorm + Dropout
→ ResonantGSER₂ (64 units, ρ=0.8, leak=0.12, Resonance)
→ LSTM (64 units, temporal processing)
→ [Global Pool LSTM + Global Pool GSER₂]
→ Feature Fusion (128 features)
→ BioplasticDenseLayer (128 units, Hebbian learning)
→ Dense Processing (64 units)
→ Output (vocab_size, softmax)
```

### Core Neural Layers

1. **ResonantGSER**:
   - Combines reservoir computing with spiking neural dynamics
   - Spectral radius control for memory vs. dynamics balance
   - Leak rate and spike threshold for biological realism
   - **[Detailed Resonance Documentation](docs/NEURAL_RESONANCE.md)**

2. **BioplasticDenseLayer**:
   - Implements Hebbian learning ("neurons that fire together, wire together")
   - Homeostatic plasticity for activity regulation
   - Adaptive weight updates based on neural activity

3. **Feature Fusion**:
   - Multiple neural pathways combined
   - LSTM for sequential processing
   - Global pooling for feature extraction

### Model Comparison

| Architecture | Accuracy | Stability | Speed | Parameters |
|--------------|----------|-----------|-------|------------|
| Traditional LSTM | ★★☆☆ | ★★☆☆ | ★★★★ | ~195K |
| Neuromimetic (Standard) | ★★★☆ | ★★★☆ | ★★★☆ | ~220K |
| Hierarchical Resonance | ★★★★ | ★★★★ | ★★☆☆ | ~385K |

## Available Layers

| Layer | Description |
|-------|-------------|
| `GSER` | Gated Spiking Elastic Reservoir with dynamic reservoir sizing |
| `DenseGSER` | Dense layer with spiking dynamics and gating mechanisms |
| `ResonantGSER` | Hierarchical resonant layer with bi-directional feedback |
| `BioplasticDenseLayer` | Hebbian learning with homeostatic plasticity |
| `HebbianHomeostaticNeuroplasticity` | Simplified Hebbian learning layer |
| `RelationalConceptModeling` | Multi-head attention for concept extraction |
| `RelationalGraphAttentionReasoning` | Graph attention for relational reasoning |
| `RelationalConceptGraphReasoning` | Unified relational reasoning with configurable outputs |
| `MultiheadLinearSelfAttentionKernalization` | Linear attention with kernel approximation |
| `LatentTemporalCoherence` | Temporal coherence distillation |
| `PositionalEncodingLayer` | Sinusoidal positional encoding |

## Available Activations

| Activation | Description |
|------------|-------------|
| `resonant_spike` | STATEFUL: LIF potential with top-down resonance modulation |
| `homeostatic_gelu` | SELF-REGULATING: GELU with dynamic gain adjustment |
| `adaptive_softplus` | STABLE: Tunable saturation for biological fidelity |
| `NeuromimeticActivation` | Wrapper for easy Keras/TensorFlow integration |

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

## Performance & Benchmarks

### Benchmark Results

Comprehensive testing on the Tiny Shakespeare dataset shows A.R.C.A.N.E. models outperform traditional approaches:

| Model | Val Accuracy | Val Loss | Training Time | Parameters |
|-------|--------------|----------|---------------|------------|
| Traditional Deep LSTM | 9.50% | 6.85 | ~45s | ~195K |
| **A.R.C.A.N.E. Neuromimetic** | 10.20% | 6.42 | ~58s | ~220K |
| **A.R.C.A.N.E. Hierarchical Resonance** | **11.25%** | **6.15** | ~95s | ~385K |

### Key Advantages

- **18.4% relative improvement** in validation accuracy over traditional LSTM
- **Lowest loss variance** (0.0142) indicating stable training
- **Smallest train/val gap** (0.048) showing reduced overfitting
- **Biologically-plausible learning** with neural resonance

### Text Generation Quality

A.R.C.A.N.E. models produce more coherent and contextually appropriate text:

- **Temperature Control**: 0.6 (conservative), 0.9 (balanced), 1.2 (creative)
- **Multi-modal Support**: Text, with extensions for other modalities
- **Nucleus Sampling**: High-quality generation with configurable diversity

### Running Benchmarks

```bash
# Run the comprehensive model comparison
python examples/test_hierarchical_resonance_comparison.py
```

This benchmark compares traditional LSTM vs A.R.C.A.N.E. models on training stability, generation quality, and performance metrics.

## Research Applications

A.R.C.A.N.E. serves researchers in multiple fields:

### Computational Neuroscience
- Study biological neural principles in artificial systems
- Investigate spiking neural dynamics and Hebbian learning
- Research homeostatic plasticity mechanisms

### Cognitive Modeling
- Model human-like learning and memory processes
- Explore hierarchical information processing
- Study neural resonance in decision-making

### Neuromorphic Computing
- Develop brain-inspired AI architectures
- Research energy-efficient neural processing
- Advance spiking neural network technology

### AI Safety & Interpretability
- Build more interpretable neural models
- Study controllable generation mechanisms
- Research stable training dynamics

## Scientific Contributions

### Novel Mechanisms
- **Neural Resonance**: Bi-directional state alignment in deep networks
- **Hierarchical Processing**: Multi-level neural architectures
- **Biological Learning Rules**: Hebbian and homeostatic plasticity
- **Prospective Learning**: Activity refinement before weight updates

### Research Impact
A.R.C.A.N.E. advances the field of biologically-inspired AI by providing:
- Open-source implementation of cutting-edge neural mechanisms
- Reproducible benchmarks for neuromimetic model comparison
- Extensible framework for neuroscience research
- Bridge between theoretical neuroscience and practical AI applications

## Running the Comparison Test

To run the comprehensive model comparison benchmark:

```bash
python examples/test_hierarchical_resonance_comparison.py
```

This test compares:
1. **Traditional Deep LSTM** - 4-layer stacked LSTM baseline
2. **Neuromimetic (Standard)** - `NeuromimeticSemanticModel` with 2-level resonance
3. **Hierarchical Resonance** - `HierarchicalResonanceFoundationModel` with multi-level hierarchy

The test outputs:
- Training and validation accuracy/loss
- Training time comparison
- Text generation samples
- Training dynamics analysis (convergence speed, stability metrics)

## Project Structure

```
gpbacay_arcane/
├── gpbacay_arcane/          # Core library
│   ├── __init__.py          # Module exports
│   ├── layers.py            # Neural network layers
│   ├── models.py            # Model architectures
│   ├── callbacks.py         # Training callbacks
│   ├── cli_commands.py      # CLI interface
│   └── ollama_integration.py # Ollama integration
├── examples/                # Usage examples
│   ├── train_neuromimetic_lm.py       # Training script
│   ├── create_foundation_model.py     # Ollama integration
│   ├── arcane_foundational_model.py   # Foundation model demo
│   └── test_hierarchical_resonance_comparison.py  # Model comparison benchmark
├── tests/                   # Test files
├── docs/                    # Documentation
│   └── NEURAL_RESONANCE.md  # Detailed resonance documentation
├── data/                    # Sample data
│   └── shakespeare_small.txt # Tiny Shakespeare dataset
├── Models/                  # Saved models directory
├── setup.py                 # Package configuration
├── requirements.txt         # Dependencies
└── README.md
```

## Contributing

We welcome contributions to advance neuromimetic AI:

1. **Research**: Novel biological neural mechanisms
2. **Engineering**: Performance optimizations and scaling
3. **Applications**: Domain-specific implementations
4. **Documentation**: Tutorials and examples

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- **Neuroscience Research**: Inspired by decades of brain research
- **Reservoir Computing**: Building on echo state network principles  
- **Hebbian Learning**: Following Donald Hebb's groundbreaking work
- **Open Source Community**: TensorFlow and Python ecosystems

## Contact

- **Author**: Gianne P. Bacay
- **Email**: giannebacay2004@gmail.com
- **Project**: [GitHub Repository](https://github.com/gpbacay/gpbacay_arcane)

---

**"Neurons that fire together, wire together, and now they learn together."**

*A.R.C.A.N.E. - Building the future of biologically-inspired AI, one neural connection at a time.*
