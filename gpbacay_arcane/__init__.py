"""
gpbacay_arcane - A.R.C.A.N.E. Neuromimetic Language Foundation Model

Augmented Reconstruction of Consciousness through Artificial Neural Evolution

A Python library for neuromimetic neural network mechanisms featuring:
- Hierarchical Neural Resonance
- Spiking Neural Networks (ResonantGSER)
- Hebbian Learning (BioplasticDenseLayer)
- Homeostatic Plasticity
- Reservoir Computing
"""

from .cli_commands import about

# Convenience re-exports for layers
from .layers import (
    # Core reservoir layers
    GSER,
    DenseGSER,
    ResonantGSER,
    # Bioplastic layers
    BioplasticDenseLayer,
    HebbianHomeostaticNeuroplasticity,
    # Attention and concept layers
    RelationalConceptModeling,
    RelationalGraphAttentionReasoning,
    MultiheadLinearSelfAttentionKernalization,
    # Temporal and positional layers
    LatentTemporalCoherence,
    PositionalEncodingLayer,
    # Utility layers
    ExpandDimensionLayer,
    SpatioTemporalSummaryMixingLayer,
    SpatioTemporalSummarization,
)

# Convenience re-exports for models  
from .models import (
    NeuromimeticLanguageModel,
    HierarchicalResonanceFoundationModel,
    load_neuromimetic_model,
)

# Convenience re-exports for callbacks
from .callbacks import (
    NeuralResonanceCallback,
    DynamicSelfModelingReservoirCallback,
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

__version__ = "3.0.0"
__author__ = "Gianne P. Bacay"
__description__ = "Neuromimetic Language Foundation Model with Biologically-Inspired Neural Mechanisms"
__all__ = [
    # Layers
    "GSER",
    "DenseGSER",
    "ResonantGSER",
    "BioplasticDenseLayer",
    "HebbianHomeostaticNeuroplasticity",
    "RelationalConceptModeling",
    "RelationalGraphAttentionReasoning",
    "MultiheadLinearSelfAttentionKernalization",
    "LatentTemporalCoherence",
    "PositionalEncodingLayer",
    "ExpandDimensionLayer",
    "SpatioTemporalSummaryMixingLayer",
    "SpatioTemporalSummarization",
    # Models
    "NeuromimeticLanguageModel",
    "HierarchicalResonanceFoundationModel",
    "load_neuromimetic_model",
    # Callbacks
    "NeuralResonanceCallback",
    "DynamicSelfModelingReservoirCallback",
    # Legacy aliases
    "DSTSMGSER",
    "GSERModel",
    "CoherentThoughtModel",
]
