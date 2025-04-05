import argparse

def about():
    print("""
        gpbacay_arcane: A Python library for neuromorphic neural network mechanisms.
        Features include dynamic reservoirs, spiking neurons, Hebbian learning, and more.
    """)

def list_models():
    print("""
        Available Models:
        1. DSTSMGSER - Dynamic Spatio-Temporal Self-Modeling Gated Spiking Elastic Reservoir
        2. GSERModel - Simplified Gated Spiking Elastic Reservoir Model
    """)

def list_layers():
    print("""
        Available Layers:
        1. ExpandDimensionLayer
        2. GSER (Gated Spiking Elastic Reservoir)
        3. DenseGSER
        4. RelationalConceptModeling
        5. RelationalGraphAttentionReasoning
        6. HebbianHomeostaticNeuroplasticity
        7. SpatioTemporalSummaryMixingLayer
        8. SpatioTemporalSummarization
        9. MultiheadLinearSelfAttentionKernalization
        10. PositionalEncodingLayer
    """)

def cli():
    parser = argparse.ArgumentParser(description="gpbacay_arcane CLI")
    parser.add_argument(
        "command",
        choices=["about", "list_models", "list_layers"],
        help="- about: Show information about the library\n- list_models: List available models\n- list_layers: List available layers"
    )

    args = parser.parse_args()

    if args.command == "about":
        about()
    elif args.command == "list_models":
        list_models()
    elif args.command == "list_layers":
        list_layers()

if __name__ == "__main__":
    cli()

