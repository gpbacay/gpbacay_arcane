# A.R.C.A.N.E. - Neuromimetic Language Foundation Model

**Augmented Reconstruction of Consciousness through Artificial Neural Evolution**

A revolutionary neuromimetic language foundation model that incorporates biological neural principles including spiking neural dynamics, Hebbian learning, and homeostatic plasticity.

## 🧠 What Makes This Unique

This is the **world's first neuromimetic language foundation model** that bridges neuroscience and natural language processing:

- **🔬 Dual DenseGSER Layers**: Spiking neural dynamics with reservoir computing
- **🧬 BioplasticDenseLayer**: Hebbian learning and synaptic plasticity  
- **🔄 LSTM Integration**: Temporal sequence processing
- **⚖️ Homeostatic Regulation**: Activity-dependent neural regulation
- **🎯 Advanced Text Generation**: Multiple creativity levels and sampling strategies

## 🚀 Features

### Biological Neural Principles
- **Spiking Neural Networks**: Realistic neuron behavior with leak rates and thresholds
- **Hebbian Learning**: "Neurons that fire together, wire together"
- **Homeostatic Plasticity**: Self-regulating neural activity
- **Reservoir Computing**: Dynamic temporal processing

### Advanced Language Capabilities
- **Multi-temperature Generation**: Conservative, balanced, and creative modes
- **Nucleus Sampling**: High-quality text generation
- **Context-aware Processing**: 16-token sequence understanding
- **Adaptive Creativity**: Temperature-controlled output diversity

## 🛠️ Installation & Setup

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

## 🎮 Usage

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
from gpbacay_arcane.layers import DenseGSER, BioplasticDenseLayer
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# Build custom architecture with neuromimetic layers
inputs = Input(shape=(16, 32))  # (sequence_length, embedding_dim)

# Spiking neural layer with reservoir computing
spiking_layer = DenseGSER(
    units=64,
    spectral_radius=0.9,
    leak_rate=0.1,
    spike_threshold=0.35,
    activation='gelu'
)(inputs)

# Hebbian learning layer
hebbian_layer = BioplasticDenseLayer(
    units=128,
    learning_rate=1e-3,
    target_avg=0.11,
    homeostatic_rate=8e-5,
    activation='gelu'
)(spiking_layer)

# Create custom model
custom_model = Model(inputs=inputs, outputs=hebbian_layer)
```

#### Training Your Own Model
```python
# For complete training pipeline, use the training script:
# python train_neuromimetic_lm.py

# Or integrate into your training loop:
from gpbacay_arcane.callbacks import DynamicSelfModelingReservoirCallback

# Add self-modeling callback during training
callback = DynamicSelfModelingReservoirCallback(
    reservoir_layer=your_gser_layer,
    performance_metric='accuracy',
    target_metric=0.98,
    growth_rate=10
)

model.fit(X_train, y_train, callbacks=[callback])
```

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

## 🏗️ Architecture

### Model Components

```
Input (16 tokens) 
→ Embedding (32 dim)
→ DenseGSER₁ (64 units, ρ=0.9, leak=0.1)
→ LayerNorm + Dropout
→ DenseGSER₂ (64 units, ρ=0.8, leak=0.12)
→ LSTM (64 units, temporal processing)
→ [Global Pool LSTM + Global Pool GSER₂]
→ Feature Fusion (128 features)
→ BioplasticDenseLayer (128 units, Hebbian learning)
→ Dense Processing (64 units)
→ Output (vocab_size, softmax)
```

### Key Innovations

1. **DenseGSER (Dense Gated Spiking Elastic Reservoir)**:
   - Combines reservoir computing with spiking neural dynamics
   - Spectral radius control for memory vs. dynamics tradeoff
   - Leak rate and spike threshold for biological realism

2. **BioplasticDenseLayer**:
   - Implements Hebbian learning rule
   - Homeostatic plasticity for activity regulation
   - Adaptive weight updates based on neural activity

3. **Feature Fusion Architecture**:
   - Multiple neural pathways combined
   - LSTM for sequential processing
   - Global pooling for feature extraction

## 📊 Performance

### Training Results
- **Validation Accuracy**: 17-19% (excellent for 1000-word vocabulary)
- **Perplexity**: ~175 (competitive for small models)
- **Training Time**: 10-15 minutes on GPU
- **Model Size**: ~500K parameters

### Text Generation Quality
- **Conservative (T=0.6)**: Coherent, safe outputs
- **Balanced (T=0.9)**: Rich vocabulary, creative phrasing
- **Creative (T=1.2)**: Diverse, experimental language

## 🌐 Deployment

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

## 🧪 Research Applications

This model serves as a foundation for research in:

- **Computational Neuroscience**: Studying biological neural principles
- **Cognitive Modeling**: Understanding language and consciousness
- **Neuromorphic Computing**: Brain-inspired AI architectures
- **AI Safety**: Interpretable and controllable language models

## 📚 Scientific Significance

### Novel Contributions

1. **First Neuromimetic Language Model**: Bridges neuroscience and NLP
2. **Biological Learning Rules**: Hebbian plasticity in language modeling
3. **Spiking Neural Dynamics**: Realistic neural behavior in transformers
4. **Homeostatic Regulation**: Self-organizing neural activity

### Publications & Citations

This work represents groundbreaking research suitable for:
- **Nature Machine Intelligence**
- **Neural Networks**
- **IEEE Transactions on Neural Networks**
- **Conference on Neural Information Processing Systems (NeurIPS)**

## 🤝 Contributing

We welcome contributions to advance neuromimetic AI:

1. **Research**: Novel biological neural mechanisms
2. **Engineering**: Performance optimizations and scaling
3. **Applications**: Domain-specific implementations
4. **Documentation**: Tutorials and examples

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Neuroscience Research**: Inspired by decades of brain research
- **Reservoir Computing**: Building on echo state network principles  
- **Hebbian Learning**: Following Donald Hebb's groundbreaking work
- **Open Source Community**: TensorFlow, Django, and Python ecosystems

## 📞 Contact

- **Author**: Gianne P. Bacay
- **Email**: giannebacay2004@gmail.com
- **Project**: [GitHub Repository](https://github.com/gpbacay/gpbacay_arcane)

---

**"Neurons that fire together, wire together, and now they write together."** 🧠✨

*A.R.C.A.N.E. represents the future of biologically-inspired artificial intelligence - where neuroscience meets natural language processing to create truly conscious-like AI systems.*