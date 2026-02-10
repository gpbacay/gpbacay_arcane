"""
Example: Inference-Time Resonance in ARCANE

This example demonstrates how to use inference-time resonance for deliberative
reasoning at inference time, enabling "System 2" thinking without training.
"""

import numpy as np
import tensorflow as tf
from gpbacay_arcane import HierarchicalResonanceFoundationModel, NeuromimeticSemanticModel

def example_hierarchical_resonance():
    """Example using HierarchicalResonanceFoundationModel with inference-time resonance."""
    print("=" * 60)
    print("Example: Inference-Time Resonance with HierarchicalResonanceFoundationModel")
    print("=" * 60)
    
    # Create model
    vocab_size = 1000
    model = HierarchicalResonanceFoundationModel(
        vocab_size=vocab_size,
        seq_len=32,
        embed_dim=64,
        hidden_dim=128,
        num_resonance_levels=4,
        resonance_factor=0.15
    )
    
    # Build and compile
    model.build_model()
    model.compile_model(learning_rate=5e-4)
    
    print("\n✓ Model built successfully")
    print(f"  - Resonance levels: {model.num_resonance_levels}")
    print(f"  - Resonant layers: {len(model.resonant_layer_objects)}")
    
    # Create dummy input
    dummy_input = np.random.randint(0, vocab_size, size=(1, 32))
    
    # Standard inference (no resonance)
    print("\n--- Standard Inference (No Resonance) ---")
    predictions_standard = model.model.predict(dummy_input, verbose=0)
    print(f"  Prediction shape: {predictions_standard.shape}")
    print(f"  Top 5 predicted tokens: {np.argsort(predictions_standard[0])[-5:][::-1]}")
    
    # Inference with resonance cycles
    print("\n--- Inference-Time Resonance ---")
    print("  Running 5 resonance cycles for hierarchical state alignment...")
    predictions_resonant = model.predict_with_resonance(
        dummy_input, 
        resonance_cycles=5, 
        verbose=1
    )
    print(f"  Prediction shape: {predictions_resonant.shape}")
    print(f"  Top 5 predicted tokens: {np.argsort(predictions_resonant[0])[-5:][::-1]}")
    
    # Manual resonance cycle control
    print("\n--- Manual Resonance Cycle Control ---")
    divergences = model.run_resonance_cycle(num_cycles=10)
    print(f"  Resonance divergences: {[f'{d:.6f}' for d in divergences[:5]]}...")
    print(f"  Final divergence: {divergences[-1]:.6f}")
    
    # Compare predictions
    print("\n--- Comparison ---")
    top_token_standard = np.argmax(predictions_standard[0])
    top_token_resonant = np.argmax(predictions_resonant[0])
    print(f"  Standard inference top token: {top_token_standard}")
    print(f"  Resonant inference top token: {top_token_resonant}")
    print(f"  Prediction changed: {top_token_standard != top_token_resonant}")
    
    print("\n✓ Inference-time resonance example completed!")


def example_neuromimetic_resonance():
    """Example using NeuromimeticSemanticModel with inference-time resonance."""
    print("\n" + "=" * 60)
    print("Example: Inference-Time Resonance with NeuromimeticSemanticModel")
    print("=" * 60)
    
    # Create model
    vocab_size = 1000
    model = NeuromimeticSemanticModel(
        vocab_size=vocab_size,
        seq_len=16,
        embed_dim=32,
        hidden_dim=64
    )
    
    # Build and compile
    model.build_model()
    model.compile_model()
    
    print("\n✓ Model built successfully")
    print(f"  - Resonant layers: {len(model.resonant_layer_objects)}")
    
    # Create dummy input
    dummy_input = np.random.randint(0, vocab_size, size=(1, 16))
    
    # Standard inference
    print("\n--- Standard Inference (No Resonance) ---")
    predictions_standard = model.model.predict(dummy_input, verbose=0)
    print(f"  Prediction shape: {predictions_standard.shape}")
    
    # Inference with resonance
    print("\n--- Inference-Time Resonance ---")
    predictions_resonant = model.predict_with_resonance(
        dummy_input,
        resonance_cycles=5,
        verbose=1
    )
    print(f"  Prediction shape: {predictions_resonant.shape}")
    
    # Manual resonance cycles
    print("\n--- Manual Resonance Cycles ---")
    divergences = model.run_resonance_cycle(num_cycles=5)
    print(f"  Resonance divergences: {[f'{d:.6f}' for d in divergences]}")
    
    print("\n✓ Neuromimetic inference-time resonance example completed!")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    try:
        example_hierarchical_resonance()
        example_neuromimetic_resonance()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("  1. Inference-time resonance enables deliberative reasoning")
        print("  2. Use predict_with_resonance() for automatic resonance")
        print("  3. Use run_resonance_cycle() for manual control")
        print("  4. Resonance cycles align hierarchical states before prediction")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
