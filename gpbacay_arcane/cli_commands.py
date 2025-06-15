import argparse
import pkg_resources

# python cli_commands.py about
def about():
    print("""
        gpbacay_arcane: The A.R.C.A.N.E. Project, which means 
        Augmented Reconstruction of Consciousness through Artificial Neural Evolution, 
        Is a Python library for neuromorphic neural network mechanisms.
        Features include dynamic reservoirs, spiking neurons, Hebbian learning, and more.
    """)

# python cli_commands.py list_models
def list_models():
    print("""
        Available Models:
        1. DSTSMGSER - Dynamic Spatio-Temporal Self-Modeling Gated Spiking Elastic Reservoir
        2. GSERModel - Simplified Gated Spiking Elastic Reservoir Model
    """)

# python cli_commands.py list_layers
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

# python cli_commands.py version
def version():
    try:
        version = pkg_resources.get_distribution("gpbacay-arcane").version
        print(f"gpbacay_arcane version: {version}")
    except pkg_resources.DistributionNotFound:
        print("gpbacay_arcane is not installed as a package.")

# python cli_commands.py --help
def cli():
    parser = argparse.ArgumentParser(description="gpbacay_arcane CLI")
    parser.add_argument(
        "command",
        choices=["about", "list_models", "list_layers", "version"],
        help="- about: Show information about the library\n- list_models: List available models\n- list_layers: List available layers\n- version: Show the current version of the library"
    )

    args = parser.parse_args()

    if args.command == "about":
        about()
    elif args.command == "list_models":
        list_models()
    elif args.command == "list_layers":
        list_layers()
    elif args.command == "version":
        version()

if __name__ == "__main__":
    cli()

