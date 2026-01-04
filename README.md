# A.R.C.A.N.E. - Neuromimetic Semantic Foundation Model

**Augmented Reconstruction of Consciousness through Artificial Neural Evolution**

A revolutionary neuromimetic semantic foundation model that incorporates biological neural principles including hierarchical resonance, spiking neural dynamics, Hebbian learning, and homeostatic plasticity.

## What Makes This Unique

This is the **world's first neuromimetic semantic foundation model** that bridges neuroscience and artificial intelligence to enable advanced semantic understanding:

- **Hierarchical Neural Resonance**: Bi-directional state alignment for Latent Space Reasoning and Direct Semantic Optimization
- **ResonantGSER Layers**: Spiking neural dynamics with Unified Multi-Modal Semantic Space integration and Non-Autoregressive Semantic Prediction for Efficiency
- **BioplasticDenseLayer**: Hebbian learning for Abstraction of Surface-Level Conceptual Variability and Direct Semantic Optimization  
- **Homeostatic Regulation**: Activity-dependent neural regulation for stable Semantic Representation
- **Temporal Integration**: Sequential processing via LSTM and spiking dynamics for Latent Space Reasoning in temporal contexts
- **Advanced Semantic Generation**: Multiple creativity levels and sampling strategies for Non-Autoregressive Semantic Prediction for Efficiency

## Features

### Biological Neural Principles
- **Neural Resonance**: Real-time state harmonization between hierarchical layers for Latent Space Reasoning and Direct Semantic Optimization
- **Prospective Alignment**: Neural activity refinement before synaptic weight updates for Direct Semantic Optimization
- **Spiking Neural Networks**: Realistic neuron behavior with leak rates and thresholds, contributing to Non-Autoregressive Semantic Prediction for Efficiency
- **Hebbian Learning**: "Neurons that fire together, wire together" for Abstraction of Surface-Level Conceptual Variability
- **Homeostatic Plasticity**: Self-regulating neural activity for stable Semantic Representation
- **Reservoir Computing**: Dynamic temporal processing within a Unified Multi-Modal Semantic Space

### Advanced Semantic Capabilities
- **Multi-temperature Semantic Generation**: Conservative, balanced, and creative modes
- **Nucleus Sampling**: High-quality semantic generation
- **Context-aware Processing**: Enhanced semantic understanding across diverse modalities
- **Adaptive Creativity**: Temperature-controlled output diversity

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

### Hierarchical Resonance Foundation Model (Recommended)

```python
from gpbacay_arcane import HierarchicalResonanceFoundationModel, NeuralResonanceCallback

# Initialize the deep resonance model
model = HierarchicalResonanceFoundationModel(
    vocab_size=3000,
    seq_len=32,
    hidden_dim=128,
    num_resonance_levels=4,  # 4-level hierarchy
    resonance_factor=0.15,
    use_temporal_coherence=True,
    use_attention_fusion=True
)

# Build and compile
model.build_model()
model.compile_model(learning_rate=3e-4)

# Train with Neural Resonance (the "Thinking Phase")
resonance_cb = NeuralResonanceCallback(resonance_cycles=10)
model.model.fit(X_train, y_train, callbacks=[resonance_cb])

# Generate text
generated = model.generate_text(
    seed_text="the nature of",
    tokenizer=tokenizer,
    max_length=50,
    temperature=0.8
)
```

### Basic Neuromimetic Model

```python
from gpbacay_arcane import NeuromimeticSemanticModel

# Initialize the model
model = NeuromimeticSemanticModel(vocab_size=1000)
model.build_model()
model.compile_model()

# Generate semantic output (requires trained tokenizer/processor)
generated_output = model.generate_text(
    seed_text="artificial intelligence",
    tokenizer=tokenizer,
    max_length=50,
    temperature=0.8
)
print(generated_output)
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

## Hierarchical Neural Resonance

The **HierarchicalResonanceFoundationModel** implements a revolutionary bi-directional neural architecture for deliberative "System 2" reasoning.

### What is Neural Resonance?

Unlike traditional feed-forward networks that process inputs in a single pass, Neural Resonance introduces a **"Thinking Phase"** where:

1. **Higher layers project feedback** (expectations) downward to lower layers
2. **Lower layers harmonize** their internal states to match those expectations
3. **Multiple resonance cycles** align the entire hierarchy before weight updates

This mimics the brain's predictive coding mechanism, where perception involves iterative top-down predictions and bottom-up error correction.

### Key Features

| Feature | Description |
|---------|-------------|
| **Prospective Configuration** | Neural activities optimized before weight updates for Direct Semantic Optimization |
| **Bi-directional Feedback** | Higher layers send expectations to lower layers |
| **Cross-level Skip Connections** | Multi-scale information flow |
| **Temporal Coherence** | Distills temporal dynamics into coherence vectors for Latent Space Reasoning |
| **Attention Fusion** | Multi-pathway aggregation with self-attention |
| **BCM Metaplasticity** | Bienenstock-Cooper-Munro sliding threshold learning |

### Training with Resonance Cycles

```python
from gpbacay_arcane import (
    HierarchicalResonanceFoundationModel,
    NeuralResonanceCallback,
    DynamicSelfModelingReservoirCallback
)

# Create the model with 4 resonance levels
model = HierarchicalResonanceFoundationModel(
    vocab_size=5000,
    seq_len=32,
    hidden_dim=128,
    num_resonance_levels=4,
    resonance_factor=0.15
)
model.build_model()
model.compile_model()

# Neural Resonance Callback - orchestrates the "Thinking Phase"
# More cycles = deeper deliberation but slower training
resonance_cb = NeuralResonanceCallback(
    resonance_cycles=10,  # 5-15 recommended
    learning_rate=0.01
)

# Optional: Dynamic reservoir adaptation
reservoir_cb = DynamicSelfModelingReservoirCallback(
    reservoir_layer=model.get_resonant_layers()[0],
    performance_metric='accuracy',
    target_metric=0.95,
    growth_rate=10
)

# Train with resonance
model.model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    callbacks=[resonance_cb, reservoir_cb]
)

# View detailed model information
model.summary()
```

### When to Use Hierarchical Resonance

✅ **Use when:**
- You need deliberative reasoning over complex patterns
- Training stability in very deep networks is important
- You want biologically-plausible learning dynamics
- Interpretability of internal representations matters
- You prioritize generalization over training speed

⚠️ **Consider trade-offs:**
- Training is slower due to resonance cycles (~2x compared to traditional LSTM)
- Higher memory usage for internal state tracking
- Best suited for medium-sized models and datasets
- Higher parameter count due to multi-level hierarchy

### Benchmark Evidence

Based on comprehensive testing (see `examples/test_hierarchical_resonance_comparison.py`):
- **+18.4% relative improvement** in validation accuracy over Traditional LSTM
- **Lowest loss variance** (0.0142) indicating training stability
- **Smallest train/val gap** (0.048) showing reduced overfitting

### Model Information and Statistics

```python
# Get model architecture information
model_info = model.get_model_info()
print(f"Model: {model_info['name']}")
print(f"Features: {model_info['features']}")
print(f"Parameters: {model_info['parameters']}")

# Access bioplastic layer statistics
for layer in model.model.layers:
    if hasattr(layer, 'get_plasticity_stats'):
        stats = layer.get_plasticity_stats()
        print(f"Average activity: {stats['avg_activity'].mean():.3f}")
        print(f"Synaptic density: {stats['synaptic_density']:.3f}")
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

## Architecture

### Model Components

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

### Key Innovations

1. **Neural Resonance (ResonantGSER)**:
   - Bi-directional state alignment via feedback projections
   - Prospective neural synchronization before synaptic updates
   - Minimizes internal representation divergence for stable learning
   - **[Detailed Resonance Documentation](docs/NEURAL_RESONANCE.md)**

2. **GSER (Gated Spiking Elastic Reservoir)**:
   - Combines reservoir computing with spiking neural dynamics
   - Spectral radius control for memory vs. dynamics tradeoff
   - Leak rate and spike threshold for biological realism

3. **BioplasticDenseLayer**:
   - Implements Hebbian learning rule
   - Homeostatic plasticity for activity regulation
   - Adaptive weight updates based on neural activity

4. **Feature Fusion Architecture**:
   - Multiple neural pathways combined
   - LSTM for sequential processing
   - Global pooling for feature extraction

## Available Models

| Model | Description | Resonance Levels | Use Case |
|-------|-------------|------------------|----------|
| `HierarchicalResonanceFoundationModel` | Advanced model with multi-level resonance hierarchy, temporal coherence, and attention fusion | 3-4 | Complex reasoning tasks, research |
| `NeuromimeticLanguageModel` | Standard neuromimetic model with ResonantGSER and Hebbian learning | 2 | General NLP tasks, balanced performance |

### Model Comparison Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODEL PERFORMANCE COMPARISON                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Architecture                │ Accuracy │ Stability │ Speed  │ Parameters  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Traditional Deep LSTM       │ ★★☆☆     │ ★★☆☆      │ ★★★★   │ ~195K       │
│  Neuromimetic (Standard)     │ ★★★☆     │ ★★★☆      │ ★★★☆   │ ~220K       │
│  Hierarchical Resonance      │ ★★★★     │ ★★★★      │ ★★☆☆   │ ~385K       │
└─────────────────────────────────────────────────────────────────────────────┘
```

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
| `MultiheadLinearSelfAttentionKernalization` | Linear attention with kernel approximation |
| `LatentTemporalCoherence` | Temporal coherence distillation |
| `PositionalEncodingLayer` | Sinusoidal positional encoding |

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

## Performance

### Model Comparison Study

A comprehensive benchmark was conducted comparing three model architectures on the Tiny Shakespeare dataset (15,000 characters, 10 epochs):

| Model | Val Accuracy | Val Loss | Training Time | Parameters |
|-------|--------------|----------|---------------|------------|
| Traditional Deep LSTM | 9.50% | 6.85 | ~45s | ~195K |
| **Neuromimetic (Standard)** | 10.20% | 6.42 | ~58s | ~220K |
| **Hierarchical Resonance** | **11.25%** | **6.15** | ~95s | ~385K |

### Key Findings

1. **Superior Generalization**: The Hierarchical Resonance model achieved **18.4% relative improvement** in validation accuracy over traditional deep LSTM
2. **Stability**: Resonance models show lower validation loss variance, indicating more stable training
3. **Deliberative Processing**: Multiple resonance cycles enable "System 2" reasoning before weight updates
4. **Trade-off**: Higher accuracy comes with increased training time due to resonance cycles

### Training Dynamics

| Model | Convergence (90% final) | Loss Variance | Train/Val Gap |
|-------|------------------------|---------------|---------------|
| Traditional LSTM | Epoch 2 | 0.0234 | 0.082 |
| Neuromimetic (Standard) | Epoch 3 | 0.0189 | 0.065 |
| Hierarchical Resonance | Epoch 4 | **0.0142** | **0.048** |

**Observation**: The Hierarchical Resonance model shows the **lowest validation loss variance** and **smallest train/val gap**, indicating better generalization and reduced overfitting.

### Text Generation Quality

Sample generations for prompt "the king" (T=0.8):
- **Traditional LSTM**: Repetitive patterns, limited vocabulary
- **Neuromimetic**: Improved coherence, better word relationships
- **Hierarchical Resonance**: Most diverse vocabulary, contextually appropriate phrases

### Temperature Settings
- **Conservative (T=0.6)**: Coherent, predictable outputs
- **Balanced (T=0.9)**: Rich vocabulary, creative phrasing
- **Creative (T=1.2)**: Diverse, experimental language

### Architecture Complexity

| Architecture | Resonance Levels | Temporal Coherence | Attention Fusion |
|--------------|------------------|-------------------|------------------|
| Traditional LSTM | 0 | ❌ | ❌ |
| Neuromimetic | 2 | ❌ | ❌ |
| Hierarchical Resonance | 3-4 | ✅ | ✅ |

## Research Applications

This library serves as a foundation for research in:

- **Computational Neuroscience**: Studying biological neural principles for Semantic Processing
- **Cognitive Modeling**: Understanding semantic representation and consciousness
- **Neuromorphic Computing**: Brain-inspired AI architectures
- **AI Safety**: Interpretable and controllable semantic models

## Scientific Significance

### Novel Contributions

1. **First Neuromimetic Semantic Model**: Bridges neuroscience and AI for semantic engineering
2. **Hierarchical Neural Resonance**: Novel state alignment mechanism for deep models
3. **Prospective Learning**: Activity refinement before weight updates
4. **Biological Learning Rules**: Hebbian plasticity integrated with spiking dynamics
5. **Self-Modeling Reservoirs**: Structural neurogenesis and synaptic pruning

### Publications & Citations

This work represents groundbreaking research suitable for:
- **Nature Machine Intelligence**
- **Neural Networks**
- **IEEE Transactions on Neural Networks**
- **Conference on Neural Information Processing Systems (NeurIPS)**

## Running the Comparison Test

To run the comprehensive model comparison benchmark:

```bash
python examples/test_hierarchical_resonance_comparison.py
```

This test compares:
1. **Traditional Deep LSTM** - 4-layer stacked LSTM baseline
2. **Neuromimetic (Standard)** - `NeuromimeticLanguageModel` with 2-level resonance
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

**"Neurons that fire together, wire together, and now they write together."**

*A.R.C.A.N.E. represents the future of biologically-inspired artificial intelligence - where neuroscience meets artificial intelligence to create truly conscious-like semantic AI systems.*
