#!/usr/bin/env python3
"""
Test script to verify concise responses from the Arcane Foundational Small Semantic Model.
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

def test_concise_responses():
    """Test that the model produces concise and direct responses."""
    print("="*60)
    print("TESTING CONCISE RESPONSES")
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
        print("TESTING CONCISE RESPONSES")
        print("="*60)
        
        # Test cases that should produce concise responses
        test_cases = [
            ("1+1=?", "Should be a direct numerical answer"),
            ("What is the capital of Philippines?", "Should directly name the capital"),
            ("Who is the president of USA?", "Should directly name the president"),
            ("How many continents are there?", "Should directly state the number"),
            ("What is 2+2?", "Should be a direct numerical answer"),
            ("Tell me a joke", "Can be longer but should be focused")
        ]
        
        for i, (prompt, expectation) in enumerate(test_cases, 1):
            print(f"\nTest {i}: '{prompt}'")
            print(f"Expectation: {expectation}")
            
            response = foundational_model.generate_text_with_improved_sampling(
                seed_text=prompt,
                max_length=100,
                temperature=0.7  # Slightly lower temperature for more focused responses
            )
            
            print(f"ARCANE: {response}")
            
            # Basic checks for conciseness
            word_count = len(response.split())
            if word_count > 30:
                print(f"Response is quite long ({word_count} words)")
            else:
                print(f"Response is reasonably concise ({word_count} words)")
        
        print("\n" + "="*60)
        print("Concise response testing completed!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_concise_responses()