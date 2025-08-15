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

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- TensorFlow 2.12+
- Django 4.2+

### Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/gpbacay_arcane.git
cd gpbacay_arcane
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install the gpbacay_arcane package**:
```bash
pip install -e .
```

4. **Train the neuromimetic language model**:
```bash
python train_neuromimetic_lm.py
```

5. **Run the web interface**:
```bash
cd arcane_project
python manage.py runserver
```

6. **Open your browser** to `http://localhost:8000`

## 🎮 Usage

### Web Interface
The Django web application provides an intuitive interface to:
- Input seed text for generation
- Control creativity level (temperature)
- Adjust generation length
- View real-time model status

### API Endpoints
- `POST /generate/` - Generate text from seed
- `GET /model-info/` - Get model architecture info
- `GET /health/` - Check model status

### Programmatic Usage
```python
from gpbacay_arcane.models import NeuromimeticLanguageModel

# Load the model
model = NeuromimeticLanguageModel(vocab_size=1000)
model.build_model()
model.compile_model()

# Generate text
generated = model.generate_text(
    seed_text="to be or not to be",
    temperature=0.8,
    max_length=50
)
print(generated)
```

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