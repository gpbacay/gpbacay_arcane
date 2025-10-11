#!/usr/bin/env python3
"""
Simple test script for the Arcane Foundational Small Language Model.
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from gpbacay_arcane.ollama_integration import OllamaARCANEHybrid
    print("✅ A.R.C.A.N.E. modules loaded successfully")
except ImportError as e:
    print(f"❌ Error importing A.R.C.A.N.E. modules: {e}")
    sys.exit(1)

def test_model():
    """Test the Arcane Foundational Small Language Model."""
    print("🔄 Creating Arcane Foundational Small Language Model...")
    
    try:
        # Create the hybrid model
        foundational_model = OllamaARCANEHybrid(
            ollama_model="llama3.2:1b",
            model_name="arcane_foundational_slm"
        )
        
        # Load the saved model
        model_path = "Models/arcane_foundational_slm_saved"
        if os.path.exists(model_path):
            foundational_model.load_model(model_path)
            print(f"✅ Loaded Arcane Foundational Model from: {model_path}")
        else:
            print("⚠️  Pre-trained model not found, using initialization...")
        
        # Test the model with a simple prompt
        print("\n🔍 Testing model with prompt: 'Hello, how are you?'")
        response = foundational_model.generate_text_with_improved_sampling(
            seed_text="Hello, how are you?",
            max_length=50,
            temperature=0.8
        )
        
        print(f"🤖 A.R.C.A.N.E. Response: {response}")
        
        # Test with the prompt that was causing issues
        print("\n🔍 Testing model with prompt: 'hi'")
        response = foundational_model.generate_text_with_improved_sampling(
            seed_text="hi",
            max_length=50,
            temperature=0.8
        )
        
        print(f"🤖 A.R.C.A.N.E. Response: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()