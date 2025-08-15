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

# Legacy models (deprecated but maintained for compatibility)
from .models import (
    DSTSMGSER,
    GSERModel, 
    CoherentThoughtModel,
)

__version__ = "2.0.0"
__author__ = "Gianne P. Bacay"
__description__ = "Neuromimetic Language Foundation Model with Biologically-Inspired Neural Mechanisms"
