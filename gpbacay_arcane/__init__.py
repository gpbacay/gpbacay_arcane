"""
gpbacay_arcane - ARCANE Neuromimetic Semantic Foundation Model

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
    DenseGSER,
    ResonantGSER,
    PredictiveResonantLayer,
    BioplasticDenseLayer,
    HebbianHomeostaticNeuroplasticity,
    RelationalConceptModeling,
    RelationalGraphAttentionReasoning,
    RelationalConceptGraphReasoning,
    LatentTemporalCoherence,
    PositionalEncodingLayer,
    ExpandDimensionLayer,
    SpatioTemporalSummarization,
    ArcaneDecoderBlock,
)

# Convenience re-exports for activations
from .activations import (
    straight_through_spike,
    resonant_spike,
    homeostatic_gelu,
    adaptive_softplus,
    NeuromimeticActivation,
)

# Convenience re-exports for mechanisms
from .mechanisms import (
    GSER,
    ResonantGSERCell,
    PredictiveResonantCell,
    MultiheadLinearSelfAttentionKernalization,
    CausalLinearSelfAttention,
    CausalSoftmaxSelfAttention,
    RMSNorm,
    build_rope_cache,
    apply_rope,
    ResonantSequenceMixer,
    SpatioTemporalSummaryMixingLayer,
    AttentionResidual,
    BlockAttentionResidual,
)


# Convenience re-exports for models  
from .models import (
    NeuromimeticSemanticModel,
    load_neuromimetic_model,
)

# Convenience re-exports for foundational models
from .foundational_models import (
    HierarchicalResonanceFoundationModel,
)

from .language_model import (
    ArcaneSLMConfig,
    ArcaneSmallLanguageModel,
)

from .tokenization import BytePairTokenizer

# Distillation support (Qwen vocab adapter needs `transformers`, imported lazily).
from .distillation import (
    ArcaneDistiller,
    WarmupCosine,
    distillation_loss,
    topk_kd_loss,
    read_distill_dataset,
    write_shard,
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
DSTSMGSER = NeuromimeticSemanticModel
GSERModel = NeuromimeticSemanticModel
CoherentThoughtModel = NeuromimeticSemanticModel

__version__ = "3.0.0"
__author__ = "Gianne P. Bacay"
__description__ = "Neuromimetic Semantic Foundation Model with Biologically-Inspired Neural Mechanisms"
__all__ = [
    # Layers
    "GSER",
    "DenseGSER",
    "ResonantGSER",
    "PredictiveResonantLayer",
    "BioplasticDenseLayer",
    "HebbianHomeostaticNeuroplasticity",
    "RelationalConceptModeling",
    "RelationalGraphAttentionReasoning",
    "RelationalConceptGraphReasoning",
    "MultiheadLinearSelfAttentionKernalization",
    "AttentionResidual",
    "BlockAttentionResidual",
    "LatentTemporalCoherence",
    "PositionalEncodingLayer",
    "ExpandDimensionLayer",
    "SpatioTemporalSummaryMixingLayer",
    "SpatioTemporalSummarization",
    "CausalLinearSelfAttention",
    "CausalSoftmaxSelfAttention",
    "RMSNorm",
    "build_rope_cache",
    "apply_rope",
    "ResonantSequenceMixer",
    "ArcaneDecoderBlock",
    # Activations
    "straight_through_spike",
    "resonant_spike",
    "homeostatic_gelu",
    "adaptive_softplus",
    "NeuromimeticActivation",
    # Models
    "NeuromimeticSemanticModel",
    "HierarchicalResonanceFoundationModel",
    "ArcaneSmallLanguageModel",
    # Distillation
    "ArcaneDistiller",
    "WarmupCosine",
    "distillation_loss",
    "topk_kd_loss",
    "read_distill_dataset",
    "write_shard",
    "ArcaneSLMConfig",
    "BytePairTokenizer",
    "load_neuromimetic_model",
    # Callbacks
    "NeuralResonanceCallback",
    "DynamicSelfModelingReservoirCallback",
    # Legacy aliases
    "DSTSMGSER",
    "GSERModel",
    "CoherentThoughtModel",
]
