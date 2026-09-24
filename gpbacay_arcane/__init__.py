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

import warnings

# Keras 3 probes `np.object`, which NumPy 2.4 emits as a FutureWarning on import.
warnings.filterwarnings(
    "ignore",
    message=r"In the future `np\.object` will be defined",
    category=FutureWarning,
)

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
    ResonantChannelMixer,
    Arc1DecoderBlock,
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
    ConceptEngram,
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

from .arc1 import (
    Arc1Config,
    Arc1Model,
    nested_ladder_indices,
)

from .tools import (
    ToolParam,
    ToolSpec,
    tool,
    Arc1Agent,
    parse_agent_json,
    format_tools_prompt,
    coerce_value,
)

from .arc1_codec import Arc1Codec

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

# Ollama integration is optional and pulls torch / sentence-transformers.
# Import it lazily so TensorFlow-only entry points (the SLM chat API) do not
# load a NumPy-1.x torch wheel and print `_ARRAY_API not found`.

# Legacy model aliases (deprecated but maintained for compatibility)
DSTSMGSER = NeuromimeticSemanticModel
GSERModel = NeuromimeticSemanticModel
CoherentThoughtModel = NeuromimeticSemanticModel

__version__ = "3.0.0"
__author__ = "Gianne P. Bacay"
__description__ = "Neuromimetic Semantic Foundation Model with Biologically-Inspired Neural Mechanisms"

_OPTIONAL_EXPORTS = {
    "OllamaARCANEHybrid": (".ollama_integration", "OllamaARCANEHybrid"),
    "create_custom_lm_with_ollama": (".ollama_integration", "create_custom_lm_with_ollama"),
}


def __getattr__(name):
    target = _OPTIONAL_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value


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
    "ResonantChannelMixer",
    "Arc1DecoderBlock",
    "ConceptEngram",
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
    "Arc1Model",
    "Arc1Config",
    "nested_ladder_indices",
    "Arc1Agent",
    "ToolParam",
    "ToolSpec",
    "tool",
    "parse_agent_json",
    "format_tools_prompt",
    "coerce_value",
    "Arc1Codec",
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
