from .cli_commands import about

# Convenience re-exports for layers
from .layers import (
    GSER,
    DenseGSER, 
    BioplasticDenseLayer,
    HebbianHomeostaticNeuroplasticity,
    LatentTemporalCoherence,
    RelationalConceptModeling,
    RelationalGraphAttentionReasoning,
    MultiheadLinearSelfAttentionKernalization,
    PositionalEncodingLayer,
)

# Convenience re-exports for models  
from .models import (
    NeuromimeticLanguageModel,
    load_neuromimetic_model,
)

# Ollama integration (optional)
try:
    from .ollama_integration import (
        OllamaARCANEHybrid,
        create_custom_lm_with_ollama
    )
except ImportError:
    # Ollama integration not available (missing dependencies)
    pass

# Legacy model aliases (deprecated but maintained for compatibility)
DSTSMGSER = NeuromimeticLanguageModel
GSERModel = NeuromimeticLanguageModel
CoherentThoughtModel = NeuromimeticLanguageModel

__version__ = "2.1.0"
__author__ = "Gianne P. Bacay"
__description__ = "Neuromimetic Language Foundation Model with Biologically-Inspired Neural Mechanisms"
