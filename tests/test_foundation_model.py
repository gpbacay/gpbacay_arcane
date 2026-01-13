#!/usr/bin/env python3
"""
Test script for the Arcane Foundational Small Semantic Model.
This script tests the model without requiring a tokenizer.
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from gpbacay_arcane.ollama_integration import OllamaARCANEHybrid
    print("ARCANE modules loaded successfully")
except ImportError as e:
    print(f"Error importing ARCANE modules: {e}")
    sys.exit(1)

def test_foundation_model():
    """Test the Arcane Foundational Small Semantic Model."""
    print("Creating Arcane Foundational Small Semantic Model...")
    
    try:
        # Create the hybrid model
        foundational_model = OllamaARCANEHybrid(
            ollama_model="llama3.2:1b",
            vocab_size=5000,
            embed_dim=256,
            seq_len=32,
            model_name="arcane_foundational_slm"
        )
        
        # Load the saved model (without tokenizer)
        model_path = "Models/arcane_foundational_slm_saved"
        if os.path.exists(model_path):
            foundational_model.load_model(model_path)
            print(f"Loaded Arcane Foundational Model from: {model_path}")
        else:
            print("Pre-trained model not found, using initialization...")
            # Build the model architecture
            foundational_model.build_neuromimetic_architecture()
        
        # Test the model with a simple prompt
        print("\nTesting model with prompt: 'Hello, how are you?'")
        response = foundational_model.generate_text_with_improved_sampling(
            seed_text="Hello, how are you?",
            max_length=100,
            temperature=0.8
        )
        
        print(f"ARCANE Response: {response}")
        
        # Test with the prompt that was causing issues
        print("\nTesting model with prompt: 'hi'")
        response = foundational_model.generate_text_with_improved_sampling(
            seed_text="hi",
            max_length=100,
            temperature=0.8
        )
        
        print(f"ARCANE Response: {response}")
        
        # Test with another prompt
        print("\nTesting model with prompt: 'What is artificial intelligence?'")
        response = foundational_model.generate_text_with_improved_sampling(
            seed_text="What is artificial intelligence?",
            max_length=100,
            temperature=0.8
        )
        
        print(f"ARCANE Response: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_foundation_model()