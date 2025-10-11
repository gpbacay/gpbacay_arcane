#!/usr/bin/env python3
"""
A.R.C.A.N.E. Transfer Learning Script
Initialize and train the A.R.C.A.N.E. model with knowledge transferred from Ollama.

Author: Gianne P. Bacay
Project: A.R.C.A.N.E. (Augmented Reconstruction of Consciousness through Artificial Neural Evolution)
"""

import os
import sys
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from gpbacay_arcane.ollama_integration import OllamaARCANEHybrid
    OLLAMA_AVAILABLE = True
    print("✅ Ollama integration available")
except ImportError:
    print("❌ Ollama integration not available. Install with: pip install ollama")
    OLLAMA_AVAILABLE = False
    sys.exit(1)


def create_transfer_learned_model():
    """Create a foundation model with transfer learning from Ollama."""
    print("🧠 A.R.C.A.N.E. Transfer Learning Model Creator")
    print("=" * 60)
    
    if not OLLAMA_AVAILABLE:
        print("❌ Ollama integration required for transfer learning.")
        return
    
    # Check if llama3.2:1b is available
    print("🦙 Checking Ollama model availability...")
    try:
        import ollama
        models_response = ollama.list()
        
        # Handle different response formats
        if isinstance(models_response, dict) and 'models' in models_response:
            available_models = [model['name'] for model in models_response['models']]
        elif isinstance(models_response, list):
            available_models = [model['name'] for model in models_response]
        else:
            # Fallback: try to use the model directly
            print("⚠️  Could not parse model list, attempting direct usage")
            available_models = ['llama3.2:1b']  # Assume it's available
        
        if 'llama3.2:1b' not in available_models:
            print("⚠️  llama3.2:1b not found. Pulling model...")
            try:
                ollama.pull('llama3.2:1b')
                print("✅ Successfully pulled llama3.2:1b")
            except Exception as e:
                print(f"❌ Failed to pull llama3.2:1b: {e}")
                print("Please run: ollama pull llama3.2:1b")
                return
        else:
            print("✅ llama3.2:1b is available")
    except Exception as e:
        print(f"⚠️  Could not check Ollama status: {e}")
        print("🔄 Continuing anyway, assuming Ollama and llama3.2:1b are available...")
    
    # Create the foundation model with transfer learning
    print("\n🏗️ Creating A.R.C.A.N.E. Transfer Learning Model...")
    transfer_model = OllamaARCANEHybrid(
        ollama_model="llama3.2:1b",
        vocab_size=5000,  # Larger vocabulary for better quality
        embed_dim=512,    # Enhanced embedding dimension
        seq_len=64,       # Longer sequence length for better context
        model_name="arcane_transfer_learned_lm"
    )
    
    # Generate comprehensive training data with varied prompts
    print("📚 Generating high-quality training data...")
    training_prompts = [
        # Conversational prompts
        "Hello, how are you today?",
        "What's on your mind?",
        "Tell me about yourself.",
        "How can I help you?",
        "What are your interests?",
        
        # General knowledge
        "What is artificial intelligence?",
        "Explain machine learning in simple terms.",
        "How do neural networks work?",
        "What is deep learning?",
        "Describe the history of AI.",
        
        # Science & Technology
        "What is quantum computing?",
        "Explain the theory of relativity.",
        "How does DNA work?",
        "What causes climate change?",
        "Describe how vaccines work.",
        
        # Philosophy & Reasoning
        "What is consciousness?",
        "What makes something ethical?",
        "How do we distinguish facts from opinions?",
        "What is the meaning of life?",
        "Explain the concept of free will.",
        
        # Practical Knowledge
        "How do I solve problems effectively?",
        "What makes a good leader?",
        "How can I improve my communication skills?",
        "What are the principles of good decision making?",
        "How do I learn new skills efficiently?",
        
        # Creative prompts
        "Tell me a story about a robot learning to feel emotions.",
        "Describe a future where AI and humans work together.",
        "What would happen if we could upload our consciousness?",
        "Imagine a world without the internet.",
        "Describe the perfect AI assistant.",
    ]
    
    # Generate training data with multiple responses per prompt
    training_data = transfer_model.generate_training_data_with_ollama(
        prompts=training_prompts,
        responses_per_prompt=5,  # More responses for diversity
        temperature_range=[0.3, 0.7, 1.0, 1.3]  # Varied creativity levels
    )
    
    # Add some conversational examples
    conversational_prompts = [
        ("Human: Hi there!\nAssistant:", "Hello! It's nice to meet you. How are you doing today?"),
        ("Human: How's the weather?\nAssistant:", "I don't have access to real-time weather data, but I'd be happy to discuss weather patterns or help you find that information elsewhere."),
        ("Human: What's your favorite color?\nAssistant:", "As an AI, I don't experience colors the way humans do, but I find the concept of color fascinating. Blue is often associated with calmness and depth."),
        ("Human: Tell me a joke\nAssistant:", "Why don't scientists trust atoms? Because they make up everything!"),
        ("Human: What should I do today?\nAssistant:", "That depends on your interests and goals. You could learn something new, connect with friends, work on a project, or simply take time to relax and recharge."),
    ]
    
    # Add conversational examples to training data
    for prompt, response in conversational_prompts:
        training_data.append(f"{prompt} {response}")
    
    print(f"✅ Generated {len(training_data)} high-quality training examples")
    
    # Build neuromimetic architecture
    print("🧬 Building enhanced neuromimetic architecture...")
    model = transfer_model.build_neuromimetic_architecture()
    
    print("\n📊 Transfer Learning Model Architecture:")
    model.summary()
    
    # Train the model with knowledge transfer
    print("\n🎯 Training A.R.C.A.N.E. with Knowledge Transfer...")
    print("This may take several minutes depending on your hardware...")
    
    history = transfer_model.train_with_knowledge_transfer(
        training_texts=training_data,
        epochs=30,  # More epochs for better transfer
        batch_size=32,  # Smaller batch size for better gradient updates
        validation_split=0.2,
        enable_self_modeling=False
    )
    
    # Save the transfer learned model to /Models folder
    print("\n💾 Saving A.R.C.A.N.E. Transfer Learning Model...")
    save_path = transfer_model.save_model()  # Automatically saves to /Models
    
    print(f"✅ Transfer learning model saved to: {save_path}")
    
    # Get model statistics
    stats = transfer_model.get_model_stats()
    print("\n📈 Transfer Learning Model Statistics:")
    print(f"  Model Name: {stats['model_name']}")
    print(f"  Base Model: {stats['ollama_base_model']}")
    print(f"  Total Parameters: {stats['total_parameters']:,}")
    print(f"  Vocabulary Size: {stats['actual_vocab_size']:,}")
    print(f"  Sequence Length: {stats['sequence_length']}")
    print(f"  Embedding Dimension: {stats['embedding_dim']}")
    print(f"  Neuromimetic Layers: {stats['neuromimetic_layers']}")
    
    # Test the transfer learned model
    print("\n🎨 Testing Transfer Learning Model Generation...")
    test_prompts = [
        "Hello, how are you?",
        "What is artificial intelligence?",
        "Tell me a story about a robot.",
    ]
    
    for prompt in test_prompts:
        generated = transfer_model.generate_text_with_improved_sampling(
            seed_text=prompt,
            max_length=60,
            temperature=0.8
        )
        print(f"Input: '{prompt}'")
        print(f"Output: {generated}")
        print("-" * 50)
    
    print("\n🎉 A.R.C.A.N.E. Transfer Learning Model Created Successfully!")
    print(f"📂 Model Location: {save_path}")
    print(f"🧠 Type: Neuromimetic Transfer Learning Model")
    print(f"⚡ Features: Spiking Neural Dynamics + Hebbian Learning + Knowledge Transfer")
    print(f"🤖 Base: Ollama llama3.2:1b + A.R.C.A.N.E. Neural Components")


def main():
    """Main function to create transfer learning model."""
    create_transfer_learned_model()


if __name__ == "__main__":
    main()