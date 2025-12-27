# A.R.C.A.N.E. - Neuromimetic Language Foundation Model

**Augmented Reconstruction of Consciousness through Artificial Neural Evolution**

A revolutionary neuromimetic language foundation model that incorporates biological neural principles including hierarchical resonance, spiking neural dynamics, Hebbian learning, and homeostatic plasticity.

## What Makes This Unique

This is the **world's first neuromimetic language foundation model** that bridges neuroscience and natural language processing:

- **Hierarchical Neural Resonance**: Bi-directional state alignment for prospective neural synchronization.
- **ResonantGSER Layers**: Spiking neural dynamics with reservoir computing and feedback projections.
- **BioplasticDenseLayer**: Hebbian learning and synaptic plasticity  
- **Homeostatic Regulation**: Activity-dependent neural regulation
- **Temporal Integration**: Sequential processing via LSTM and spiking dynamics
- **Advanced Text Generation**: Multiple creativity levels and sampling strategies

## Features

### Biological Neural Principles
- **Neural Resonance**: Real-time state harmonization between hierarchical layers.
- **Prospective Alignment**: Neural activity refinement before synaptic weight updates.
- **Spiking Neural Networks**: Realistic neuron behavior with leak rates and thresholds.
- **Hebbian Learning**: "Neurons that fire together, wire together."
- **Homeostatic Plasticity**: Self-regulating neural activity.
- **Reservoir Computing**: Dynamic temporal processing.

### Advanced Language Capabilities
- **Multi-temperature Generation**: Conservative, balanced, and creative modes
- **Nucleus Sampling**: High-quality text generation
- **Context-aware Processing**: 16-token sequence understanding
- **Adaptive Creativity**: Temperature-controlled output diversity

## Installation & Setup

### Prerequisites
- Python 3.11+
- TensorFlow 2.12+
- (Optional) Django 4.2+ for documentation interface

### Installation Methods

#### Option 1: Install from PyPI (Recommended)
```bash
pip install gpbacay-arcane
```

#### Option 2: Install from Source
```bash
git clone https://github.com/yourusername/gpbacay_arcane.git
cd gpbacay_arcane
pip install -e .
```

#### Basic Usage
```python
from gpbacay_arcane import NeuromimeticLanguageModel

# Initialize the model
model = NeuromimeticLanguageModel(vocab_size=1000)
model.build_model()
model.compile_model()

# Generate text (requires trained tokenizer)
generated_text = model.generate_text(
    seed_text="artificial intelligence",
    max_length=50,
    temperature=0.8
)
print(generated_text)
```

## Usage

### Core Python Package

The `gpbacay-arcane` package provides the neuromimetic language model implementation:

#### Complete Training and Usage Example
```python
import numpy as np
from gpbacay_arcane import NeuromimeticLanguageModel
from tensorflow.keras.preprocessing.text import Tokenizer

# 1. Prepare your text data
text_data = "your training text here..."

# 2. Create and train tokenizer
tokenizer = Tokenizer(num_words=1000, oov_token="<UNK>")
tokenizer.fit_on_texts([text_data])

# 3. Initialize the neuromimetic model
model = NeuromimeticLanguageModel(
    vocab_size=len(tokenizer.word_index) + 1,
    seq_len=16,
    embed_dim=32,
    hidden_dim=64
)

# 4. Build and compile the model
neuromimetic_model = model.build_model()
model.compile_model(learning_rate=1e-3)

# 5. Generate text after training
generated_text = model.generate_text(
    seed_text="artificial intelligence is",
    tokenizer=tokenizer,
    max_length=50,
    temperature=0.8  # 0.6=conservative, 0.9=balanced, 1.2=creative
)
print(f"Generated: {generated_text}")
```

#### Using Individual Neural Layers
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

#### Training with Neural Resonance
```python
# python train_neuromimetic_lm.py

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

**Note**: All trained models are automatically saved to the `/Models` folder for easy organization and access.

### Advanced Features

#### Multi-Temperature Text Generation
```python
# Conservative generation (coherent, safe)
conservative = model.generate_text(
    seed_text="machine learning",
    tokenizer=tokenizer,
    temperature=0.6,
    max_length=30
)

# Balanced generation (creative but coherent)
balanced = model.generate_text(
    seed_text="machine learning",
    tokenizer=tokenizer,
    temperature=0.9,
    max_length=30
)

# Creative generation (diverse, experimental)
creative = model.generate_text(
    seed_text="machine learning",
    tokenizer=tokenizer,
    temperature=1.2,
    max_length=30
)
```

#### Model Information and Statistics
```python
# Get model architecture information
model_info = model.get_model_info()
print(f"Model: {model_info['name']}")
print(f"Features: {model_info['features']}")
print(f"Parameters: {model_info['parameters']}")

# Access bioplastic layer statistics (if using BioplasticDenseLayer)
for layer in model.model.layers:
    if hasattr(layer, 'get_plasticity_stats'):
        stats = layer.get_plasticity_stats()
        print(f"Average activity: {stats['avg_activity'].mean():.3f}")
        print(f"Synaptic density: {stats['synaptic_density']:.3f}")
```

### Web Interface (Documentation Only)

A Django web interface is included for **documentation and demonstration purposes only**. The actual functionality is accessed through the Python package:

```bash
# Run documentation interface (optional)
cd arcane_project
python manage.py runserver
# Visit http://localhost:8000 for demonstrations
```

**Note**: The web interface is for showcasing the model's capabilities. For production use, integrate the `gpbacay-arcane` package directly into your Python applications.

### Create A.R.C.A.N.E. Foundation Model with Ollama

**Create a neuromimetic foundation small language model** by combining Ollama's llama3.2:1b with A.R.C.A.N.E.'s biological neural mechanisms:

```bash
# Install dependencies
pip install ollama sentence-transformers

# Install and pull Ollama model
# Download Ollama from: https://ollama.ai
ollama pull llama3.2:1b

# Create the foundation model
python create_foundation_model.py
```

This creates a single, comprehensive neuromimetic foundation model that:
- Combines Ollama's pre-trained knowledge with biological neural dynamics
- Features spiking neural networks and Hebbian learning
- Automatically saves to `/Models` folder
- Provides a foundation for neuromimetic language modeling

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
   - Bi-directional state alignment via feedback projections.
   - Prospective neural synchronization before synaptic updates.
   - Minimizes internal representation divergence for stable learning.
   - **[Detailed Resonance Documentation](docs/NEURAL_RESONANCE.md)**

2. **GSER (Gated Spiking Elastic Reservoir)**:
   - Combines reservoir computing with spiking neural dynamics.
   - Spectral radius control for memory vs. dynamics tradeoff.
   - Leak rate and spike threshold for biological realism.

3. **BioplasticDenseLayer**:
   - Implements Hebbian learning rule
   - Homeostatic plasticity for activity regulation
   - Adaptive weight updates based on neural activity

4. **Feature Fusion Architecture**:
   - Multiple neural pathways combined
   - LSTM for sequential processing
   - Global pooling for feature extraction

## Performance

### Training Results
Recent benchmarks demonstrate that the **Resonant A.R.C.A.N.E.** model outperforms traditional deep architectures in both accuracy and stability across different domains.

#### MNIST (10 Epochs)
- **Test Accuracy**: **98.89%** (vs 98.76% for Deep LSTM)
- **Test Loss**: **0.0391** (vs 0.0432 for Deep LSTM)

#### Tiny Shakespeare (15 Epochs)
- **Validation Accuracy**: **12.34%** (vs 10.69% for Deep LSTM)
- **Validation Loss**: **5.9818** (vs 6.7895 for Deep LSTM)

### Key Metrics
- **Stability**: Superior convergence in deep hierarchies due to prospective alignment.
- **Perplexity**: Highly competitive for small-scale foundation models.
- **Model Size**: ~500K parameters.
- **Inference Speed**: Fast execution after training phase.

### Text Generation Quality
- **Conservative (T=0.6)**: Coherent, safe outputs
- **Balanced (T=0.9)**: Rich vocabulary, creative phrasing
- **Creative (T=1.2)**: Diverse, experimental language

## Deployment

### Production Deployment

The application is production-ready with support for:
- **Heroku**: One-click deployment
- **Railway**: Simple git-based deployment  
- **Render**: Automatic scaling
- **Vercel**: Serverless deployment

See [deploy.md](deploy.md) for detailed deployment instructions.

### Environment Configuration
```bash
# Required environment variables
SECRET_KEY=your-django-secret-key
DEBUG=False
CUSTOM_DOMAIN=your-domain.com

# Optional database (defaults to SQLite)
DATABASE_URL=postgres://user:pass@host:port/db
```

## Research Applications

This model serves as a foundation for research in:

- **Computational Neuroscience**: Studying biological neural principles
- **Cognitive Modeling**: Understanding language and consciousness
- **Neuromorphic Computing**: Brain-inspired AI architectures
- **AI Safety**: Interpretable and controllable language models

## Scientific Significance

### Novel Contributions

1. **First Neuromimetic Language Model**: Bridges neuroscience and NLP.
2. **Hierarchical Neural Resonance**: Novel state alignment mechanism for deep models. **[Read the technical whitepaper](docs/NEURAL_RESONANCE.md)**.
3. **Prospective Learning**: Activity refinement before weight updates (Prospective Configuration).
4. **Biological Learning Rules**: Hebbian plasticity integrated with spiking dynamics.
5. **Self-Modeling Reservoirs**: Structural neurogenesis and synaptic pruning.

### Publications & Citations

This work represents groundbreaking research suitable for:
- **Nature Machine Intelligence**
- **Neural Networks**
- **IEEE Transactions on Neural Networks**
- **Conference on Neural Information Processing Systems (NeurIPS)**

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
- **Open Source Community**: TensorFlow, Django, and Python ecosystems

## Contact

- **Author**: Gianne P. Bacay
- **Email**: giannebacay2004@gmail.com
- **Project**: [GitHub Repository](https://github.com/gpbacay/gpbacay_arcane)

---

**"Neurons that fire together, wire together, and now they write together."**

*A.R.C.A.N.E. represents the future of biologically-inspired artificial intelligence - where neuroscience meets natural language processing to create truly conscious-like AI systems.*
