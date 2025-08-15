# A.R.C.A.N.E. Project Structure

This document outlines the cleaned and optimized project structure for the neuromimetic language foundation model.

## 📁 Root Directory

```
gpbacay_arcane/
├── README.md                    # Main project documentation
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── runtime.txt                  # Python version for deployment
├── Procfile                     # Heroku deployment configuration
├── deploy.md                    # Deployment guide
├── env.example                  # Environment variables template
├── PROJECT_STRUCTURE.md         # This file
├── setup.py                     # Package installation script
└── train_neuromimetic_lm.py     # Main training script
```

## 🧠 Core Package (`gpbacay_arcane/`)

```
gpbacay_arcane/
├── __init__.py                  # Package initialization
├── layers.py                    # Custom neuromimetic layers
├── models.py                    # Neuromimetic language model classes
├── callbacks.py                 # Training callbacks
└── cli_commands.py              # Command-line interface
```

### Key Components

**`layers.py`**: Contains all custom neuromimetic layers:
- `DenseGSER`: Dense Gated Spiking Elastic Reservoir
- `BioplasticDenseLayer`: Hebbian learning and homeostatic plasticity
- `GSER`: Original Gated Spiking Elastic Reservoir  
- `LatentTemporalCoherence`: Temporal processing layer
- `RelationalConceptModeling`: Concept extraction layer

**`models.py`**: Main model architecture:
- `NeuromimeticLanguageModel`: Primary language foundation model
- `load_neuromimetic_model()`: Model loading utility

## 🌐 Web Application (`arcane_project/`)

```
arcane_project/
├── manage.py                    # Django management script
├── db.sqlite3                   # Development database
├── arcane_project/             # Django project settings
│   ├── __init__.py
│   ├── settings.py             # Production-ready configuration
│   ├── urls.py                 # URL routing
│   ├── wsgi.py                 # WSGI application
│   └── asgi.py                 # ASGI application (async)
├── recognizer/                 # Main Django app
│   ├── __init__.py
│   ├── apps.py                 # App configuration
│   ├── admin.py                # Django admin
│   ├── models.py               # Django models (currently empty)
│   ├── views.py                # API endpoints and views
│   ├── urls.py                 # App URL patterns
│   ├── tests.py                # Unit tests
│   ├── static/recognizer/      # Static assets
│   │   ├── style.css          # Custom CSS
│   │   └── script.js          # JavaScript interface
│   └── templates/recognizer/   # HTML templates
│       └── index.html         # Main interface
├── static/                     # Project-wide static files
│   └── assets/                 # Logos and icons
│       ├── arcane_logo.png
│       ├── arcane_logo.ico
│       └── _arcane_logo.png
└── staticfiles/               # Collected static files (production)
```

### Web Features

**Views (`recognizer/views.py`)**:
- `index()`: Main interface renderer
- `generate_text()`: Text generation API endpoint
- `model_info()`: Model architecture information
- `health_check()`: System status monitoring
- `reload_model()`: Model reloading capability

**Frontend (`templates/recognizer/index.html`)**:
- Responsive web interface
- Real-time model status indicator
- Interactive text generation controls
- Temperature and length sliders
- Sample text suggestions
- Model architecture visualization

## 📦 Models Directory (`Models/`)

```
Models/
├── neuromimetic_lm.keras              # Trained model (created after training)
└── neuromimetic_lm_tokenizer.pickle   # Tokenizer (created after training)
```

**Note**: Model files are created after running `train_neuromimetic_lm.py`

## 🗂️ Distribution (`dist/`, `build/`, etc.)

```
dist/                           # Distribution packages
build/                          # Build artifacts  
gpbacay_arcane.egg-info/        # Package metadata
venv/                           # Virtual environment (local development)
```

**Note**: These directories are for package distribution and local development.

## 🚀 Key Features by Directory

### Root Level
- **Training**: `train_neuromimetic_lm.py` - Complete training pipeline
- **Deployment**: `Procfile`, `requirements.txt`, `runtime.txt` - Production deployment
- **Documentation**: `README.md`, `deploy.md` - Comprehensive guides

### Core Package (`gpbacay_arcane/`)
- **Innovation**: Custom neuromimetic neural network layers
- **Research**: Biologically-inspired AI architectures
- **Modularity**: Reusable components for other projects

### Web Application (`arcane_project/`)
- **User Interface**: Beautiful, responsive web interface
- **API**: RESTful endpoints for text generation
- **Production**: Scalable Django application with security

## 🎯 Usage Workflow

1. **Training**: `python train_neuromimetic_lm.py`
2. **Development**: `cd arcane_project && python manage.py runserver`
3. **Production**: Follow `deploy.md` for cloud deployment

## 🧪 For Researchers

Key files for understanding the neuromimetic architecture:
- `gpbacay_arcane/layers.py` - Core biological neural mechanisms
- `gpbacay_arcane/models.py` - Language model architecture
- `train_neuromimetic_lm.py` - Training methodology

## 🌐 For Developers

Key files for web development:
- `arcane_project/recognizer/views.py` - Backend API
- `arcane_project/recognizer/templates/recognizer/index.html` - Frontend
- `arcane_project/arcane_project/settings.py` - Configuration

## 📊 Production Deployment

Essential files for deployment:
- `requirements.txt` - Dependencies
- `Procfile` - Heroku/Railway configuration  
- `runtime.txt` - Python version specification
- `deploy.md` - Platform-specific instructions

---

This structure represents a clean, production-ready neuromimetic language foundation model with both research and commercial applications. Every component serves a specific purpose in the ecosystem of biologically-inspired AI.
