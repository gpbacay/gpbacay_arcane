#!/usr/bin/env python3
"""
Arcane Foundational Small Semantic Model
Enhances Ollama model responses using A.R.C.A.N.E. neuromimetic principles.

Author: Gianne P. Bacay
Project: A.R.C.A.N.E. (Augmented Reconstruction of Consciousness through Artificial Neural Evolution)
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from gpbacay_arcane.ollama_integration import OllamaARCANEHybrid
    from gpbacay_arcane.layers import DenseGSER, BioplasticDenseLayer, GSER
    print("A.R.C.A.N.E. modules loaded successfully")
except ImportError as e:
    print(f"Error importing A.R.C.A.N.E. modules: {e}")
    sys.exit(1)


def create_arcane_foundational_model():
    """
    Create the Arcane Foundational Small Semantic Model.
    This model enhances Ollama responses rather than generating text from scratch.
    """
    print("Creating Arcane Foundational Small Semantic Model...")
    print("=" * 60)
    
    # Create the hybrid model
    foundational_model = OllamaARCANEHybrid(
        ollama_model="llama3.2:1b",
        vocab_size=5000,
        embed_dim=256,
        seq_len=32,
        model_name="arcane_foundational_slm"
    )
    
    # Build the neuromimetic architecture (for potential future use)
    print("🧬 Building neuromimetic enhancement layers...")
    model = foundational_model.build_neuromimetic_architecture()
    
    # Save the model
    print("💾 Saving Arcane Foundational Model...")
    save_path = foundational_model.save_model()
    
    print(f"Arcane Foundational Small Semantic Model created successfully!")
    print(f"Model Location: {save_path}")
    print(f"Features: Ollama Response Enhancement + Neuromimetic Principles")
    print(f"Base Model: Ollama llama3.2:1b")
    
    return foundational_model


def chat_with_arcane_foundational_model():
    """
    Chat interface for the Arcane Foundational Small Semantic Model.
    """
    print("\n" + "="*60)
    print("Arcane Foundational Small Semantic Model Chat Interface")
    print("="*60)
    
    # Load the model
    try:
        foundational_model = OllamaARCANEHybrid(
            ollama_model="llama3.2:1b",
            model_name="arcane_foundational_slm"
        )
        
        model_path = "Models/arcane_foundational_slm_saved"
        if os.path.exists(model_path):
            foundational_model.load_model(model_path)
            print(f"✅ Loaded Arcane Foundational Model from: {model_path}")
        else:
            print("⚠️  Pre-trained model not found, using initialization...")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print("\n" + "="*60)
    print("💬 Chat with your Arcane Foundational Model!")
    print("   Type 'quit' or 'exit' to end the conversation")
    print("   Type 'clear' to clear the conversation history")
    print("="*60)
    
    # Chat loop
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for quit commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nGoodbye! Thanks for chatting with A.R.C.A.N.E.")
                break
            
            # Check for clear command
            if user_input.lower() in ['clear', 'reset']:
                conversation_history = []
                print("🗑️  Conversation history cleared.")
                continue
            
            if not user_input:
                continue
            
            # Add to conversation history
            conversation_history.append(f"You: {user_input}")
            
            print("🧠 A.R.C.A.N.E. is thinking...")
            
            # Generate enhanced response using the foundational model
            response = foundational_model.generate_text_with_improved_sampling(
                seed_text=user_input,
                max_length=100,
                temperature=0.8
            )
            
            # Add to conversation history
            conversation_history.append(f"A.R.C.A.N.E.: {response}")
            
            # Display response
            print(f"🤖 A.R.C.A.N.E.: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for chatting with A.R.C.A.N.E.")
            break
        except Exception as e:
            print(f"Error during chat: {e}")
            print("A.R.C.A.N.E.: I'm having trouble responding right now. Please try again.")


def main():
    """Main function to run the Arcane Foundational Model."""
    if len(sys.argv) > 1 and sys.argv[1] == "--create":
        create_arcane_foundational_model()
    else:
        chat_with_arcane_foundational_model()


if __name__ == "__main__":
    main()