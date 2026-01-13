"""
gpbacay_arcane CLI commands

Command-line interface for the ARCANE neuromimetic semantic model library.
"""

import argparse


def about():
    """Display information about the library."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              ARCANE                                    ║
║    Augmented Reconstruction of Consciousness through Artificial Neural       ║
║                              Evolution                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  A neuromimetic semantic foundation model library featuring:                 ║
║                                                                              ║
║  • Hierarchical Neural Resonance - Bi-directional state alignment            ║
║  • Spiking Neural Networks - ResonantGSER with reservoir computing           ║
║  • Hebbian Learning - BioplasticDenseLayer with synaptic plasticity          ║
║  • Homeostatic Plasticity - Activity-dependent neural regulation             ║
║  • Temporal Integration - LSTM and spiking dynamics                          ║
║                                                                              ║
║  Author: Gianne P. Bacay                                                     ║
║  GitHub: https://github.com/gpbacay/gpbacay_arcane                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


def list_models():
    """List available models in the library."""
    print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Available Models                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. HierarchicalResonanceFoundationModel  ⭐ NEW                              │
│     Deep neuromimetic architecture with bi-directional resonance.            │
│     Features: Multi-level ResonantGSER hierarchy, cross-level skip           │
│     connections, temporal coherence, attention fusion, BCM plasticity.       │
│     Use with NeuralResonanceCallback for "System 2" reasoning.               │
│                                                                              │
│  2. NeuromimeticSemanticModel                                                │
│     Standard semantic model with neural resonance and Hebbian learning.      │
│                                                                              │
│  Legacy Aliases (for backward compatibility):                                │
│  • DSTSMGSER - Dynamic Spatio-Temporal Self-Modeling Gated Spiking Elastic   │
│                Reservoir                                                     │
│  • GSERModel - Gated Spiking Elastic Reservoir Model                         │
│  • CoherentThoughtModel - Coherent Thought Model                             │
│                                                                              │
│  Optional (requires ollama package):                                         │
│  • OllamaARCANEHybrid - Hybrid model combining Ollama with ARCANE layers     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
    """)


def list_layers():
    """List available layers in the library."""
    print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Available Layers                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Reservoir Layers:                                                           │
│   1. GSER                    - Gated Spiking Elastic Reservoir               │
│   2. DenseGSER               - Dense layer with spiking dynamics             │
│   3. ResonantGSER            - Hierarchical resonant layer with feedback     │
│                                                                              │
│  Bioplastic Layers:                                                          │
│   4. BioplasticDenseLayer    - Hebbian learning + homeostatic plasticity     │
│   5. HebbianHomeostaticNeuroplasticity - Simplified Hebbian learning         │
│                                                                              │
│  Attention & Concept Layers:                                                 │
│   6. RelationalConceptModeling           - Multi-head attention for concepts │
│   7. RelationalGraphAttentionReasoning   - Graph attention for reasoning     │
│   8. RelationalConceptGraphReasoning     - Unified relational reasoning      │
│   8. MultiheadLinearSelfAttentionKernalization - Linear attention            │
│                                                                              │
│  Temporal & Positional Layers:                                               │
│   9. LatentTemporalCoherence  - Temporal coherence distillation              │
│  10. PositionalEncodingLayer  - Sinusoidal positional encoding               │
│                                                                              │
│  Utility Layers:                                                             │
│  11. ExpandDimensionLayer     - Dimension expansion utility                  │
│  12. SpatioTemporalSummaryMixingLayer   - Spatio-temporal processing         │
│  13. SpatioTemporalSummarization        - Sequence summarization             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
    """)


def list_callbacks():
    """List available callbacks in the library."""
    print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Available Callbacks                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. NeuralResonanceCallback                                                  │
│     Orchestrates neural resonance cycles for prospective alignment           │
│                                                                              │
│  2. DynamicSelfModelingReservoirCallback                                     │
│     Manages neurogenesis and synaptic pruning based on performance           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
    """)


def version():
    """Display the current version of the library."""
    try:
        import pkg_resources
        ver = pkg_resources.get_distribution("gpbacay-arcane").version
        print(f"gpbacay-arcane version: {ver}")
    except pkg_resources.DistributionNotFound:
        # Fallback to reading from __init__.py
        try:
            from gpbacay_arcane import __version__
            print(f"gpbacay-arcane version: {__version__} (development)")
        except ImportError:
            print("gpbacay-arcane version: unknown (not installed)")


def cli():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ARCANE - Neuromimetic Semantic Foundation Model CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gpbacay-arcane about          Show library information
  gpbacay-arcane list-models    List available models
  gpbacay-arcane list-layers    List available layers
  gpbacay-arcane list-callbacks List available callbacks
  gpbacay-arcane version        Show version
        """
    )
    parser.add_argument(
        "command",
        choices=["about", "list-models", "list-layers", "list-callbacks", "version"],
        help="Command to execute"
    )

    args = parser.parse_args()

    commands = {
        "about": about,
        "list-models": list_models,
        "list-layers": list_layers,
        "list-callbacks": list_callbacks,
        "version": version,
    }

    commands[args.command]()


if __name__ == "__main__":
    cli()
