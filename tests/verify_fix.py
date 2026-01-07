#!/usr/bin/env python3
"""
Verification script to confirm the fix for the A.R.C.A.N.E. model.
This script demonstrates that the model now produces coherent responses instead of numbers.
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

def verify_fix():
    """Verify that the fix is working correctly."""
    print("="*60)
    print("🔍 VERIFYING A.R.C.A.N.E. MODEL FIX")
    print("="*60)
    
    try:
        # Create the Arcane Foundational Small Semantic Model
        foundational_model = OllamaARCANEHybrid(
            ollama_model="llama3.2:1b",
            model_name="arcane_foundational_slm"
        )
        
        # Load the saved model
        model_path = "Models/arcane_foundational_slm_saved"
        if os.path.exists(model_path):
            foundational_model.load_model(model_path)
            print(f"Loaded Arcane Foundational Model from: {model_path}")
        else:
            print("Pre-trained model not found, using initialization...")
            # Build the model architecture
            foundational_model.build_neuromimetic_architecture()
        
        print("\n" + "="*60)
        print("TESTING MODEL RESPONSES")
        print("="*60)
        
        # Test cases that were previously producing numbers
        test_cases = [
            "hi",
            "Hello, how are you?",
            "Who are you?",
            "What is artificial intelligence?",
            "Tell me a joke"
        ]
        
        all_responses_coherent = True
        
        for i, prompt in enumerate(test_cases, 1):
            print(f"\nTest {i}: '{prompt}'")
            response = foundational_model.generate_text_with_improved_sampling(
                seed_text=prompt,
                max_length=100,
                temperature=0.8
            )
            
            print(f"A.R.C.A.N.E.: {response}")
            
            # Check if response contains numbers (which would indicate the old problem)
            # We'll check if the response is mostly numbers
            words = response.split()
            numeric_words = [word for word in words if word.isdigit()]
            
            if len(numeric_words) > len(words) * 0.5:
                print("ISSUE: Response contains mostly numbers (old problem)")
                all_responses_coherent = False
            else:
                print("Response appears coherent")
        
        print("\n" + "="*60)
        if all_responses_coherent:
            print("SUCCESS: All responses are coherent!")
            print("The fix is working correctly.")
            print("The Arcane Foundational Small Semantic Model now enhances Ollama responses")
            print("   instead of generating numbers.")
        else:
            print("FAILURE: Some responses still contain numbers.")
            print("🔧 The fix may need additional work.")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_fix()