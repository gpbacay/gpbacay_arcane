#!/usr/bin/env python3
"""
A.R.C.A.N.E. Foundation Small Language Model Creator

This script creates a single neuromimetic foundation small language model by combining:
- Ollama's llama3.2:1b pre-trained knowledge
- A.R.C.A.N.E.'s biological neural mechanisms (spiking dynamics, Hebbian learning)

The model is saved to /Models folder as required.

Author: Gianne P. Bacay
Project: A.R.C.A.N.E. (Augmented Reconstruction of Consciousness through Artificial Neural Evolution)
"""

import os
import sys
import numpy as np
from typing import List

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from gpbacay_arcane.ollama_integration import OllamaARCANEHybrid
    OLLAMA_AVAILABLE = True
except ImportError:
    print("❌ Ollama integration not available. Install with: pip install ollama")
    OLLAMA_AVAILABLE = False
    sys.exit(1)


def create_foundation_training_data() -> List[str]:
    """Generate diverse training prompts for a foundation model."""
    foundation_prompts = [
        # General Knowledge
        "What is artificial intelligence and how does it work?",
        "Explain the concept of machine learning in simple terms.",
        "How do neural networks process information?",
        "What are the main types of artificial intelligence?",
        "Describe how computers understand human language.",
        
        # Science & Technology
        "What is quantum computing and why is it important?",
        "Explain the theory of relativity in basic terms.",
        "How does DNA store genetic information?",
        "What causes climate change and its effects?",
        "Describe how vaccines work to protect health.",
        
        # Mathematics & Logic
        "What is the significance of prime numbers?",
        "Explain probability and statistics concepts.",
        "How do algorithms solve complex problems?",
        "What is calculus and its real-world applications?",
        "Describe geometric principles in nature.",
        
        # Philosophy & Reasoning
        "What makes consciousness unique to living beings?",
        "How do we distinguish between knowledge and belief?",
        "What is the nature of creativity and innovation?",
        "Explain the concept of free will and determinism.",
        "How does language shape human thought?",
        
        # Practical Knowledge
        "How do we solve environmental sustainability challenges?",
        "What makes effective communication between people?",
        "How do economic systems function in society?",
        "What are the principles of good leadership?",
        "How does education foster human development?",
        
        # Technical Understanding
        "How do computer networks enable global communication?",
        "What are the principles of software engineering?",
        "How do databases organize and store information?",
        "What makes cybersecurity important in digital age?",
        "How do programming languages work together?"
    ]
    
    return foundation_prompts


def create_foundation_model_demo():
    """Create a foundation model for demonstration (works without Ollama)."""
    print("\n🏢 Creating Demo Foundation Model (without Ollama dependency)...")
    
    from gpbacay_arcane import NeuromimeticLanguageModel
    from tensorflow.keras.preprocessing.text import Tokenizer
    import numpy as np
    
    # Sample training texts (simulating Ollama-generated content)
    demo_texts = [
        "artificial intelligence machine learning neural networks deep learning",
        "consciousness cognition brain neuroscience biology evolution adaptation",
        "quantum computing physics mathematics algorithms optimization",
        "language processing communication understanding meaning context",
        "creativity innovation problem solving reasoning logic",
        "technology science research development progress future",
        "knowledge information data analysis patterns recognition",
        "learning memory attention perception decision making"
    ]
    
    # Create the model
    model = NeuromimeticLanguageModel(
        vocab_size=1000,
        seq_len=16,
        embed_dim=64,
        hidden_dim=128
    )
    
    # Build and compile
    arch = model.build_model()
    model.compile_model()
    
    # Create tokenizer
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(demo_texts)
    
    # Prepare training data
    sequences = []
    for text in demo_texts:
        seq = tokenizer.texts_to_sequences([text])[0]
        for i in range(len(seq) - 16):
            sequences.append(seq[i:i+17])
    
    if sequences:
        X = np.array([seq[:-1] for seq in sequences])
        y = np.array([seq[-1] for seq in sequences])
        
        print(f"Training with {len(X)} sequences...")
        arch.fit(X, y, epochs=10, batch_size=16, verbose=1)
    
    # Save to Models folder
    import os
    os.makedirs("Models", exist_ok=True)
    arch.save("Models/arcane_foundation_demo.h5")
    
    import pickle
    with open("Models/arcane_foundation_demo_tokenizer.pkl", 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print("✅ Demo Foundation Model saved to Models/")
    return arch, tokenizer


def main():
    """Create the A.R.C.A.N.E. Foundation Small Language Model."""
    print("🧠 A.R.C.A.N.E. Foundation Small Language Model Creator")
    print("=" * 60)
    
    # Check if user wants demo or full Ollama integration
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        create_foundation_model_demo()
        return
    
    if not OLLAMA_AVAILABLE:
        print("❌ Ollama integration required for full functionality.")
        print("💡 Run with --demo flag for demonstration without Ollama:")
        print("   python create_foundation_model.py --demo")
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
                print("🔄 Continuing anyway, assuming model is available...")
        else:
            print("✅ llama3.2:1b is available")
    except Exception as e:
        print(f"⚠️  Could not check Ollama status: {e}")
        print("🔄 Continuing anyway, assuming Ollama and llama3.2:1b are available...")
    
    # Create the foundation model
    print("\n🏗️ Creating A.R.C.A.N.E. Foundation Model...")
    foundation_model = OllamaARCANEHybrid(
        ollama_model="llama3.2:1b",
        vocab_size=4000,  # Larger vocabulary for foundation model
        embed_dim=384,    # Enhanced embedding dimension
        seq_len=48,       # Longer sequence length for better context
        model_name="arcane_foundation_small_lm"
    )
    
    # Generate comprehensive training data
    print("📚 Generating foundation training data...")
    training_prompts = create_foundation_training_data()
    training_data = foundation_model.generate_training_data_with_ollama(
        prompts=training_prompts,
        responses_per_prompt=3,  # Multiple responses for diversity
        temperature_range=[0.4, 0.8, 1.2]  # Varied creativity levels
    )
    
    # Build neuromimetic architecture
    print("🧬 Building neuromimetic foundation architecture...")
    model = foundation_model.build_neuromimetic_architecture()
    
    print("\n📊 Foundation Model Architecture:")
    model.summary()
    
    # Train the foundation model
    print("\n🎯 Training A.R.C.A.N.E. Foundation Model with Knowledge Transfer...")
    print("This may take several minutes depending on your hardware...")
    
    history = foundation_model.train_with_knowledge_transfer(
        training_texts=training_data,
        epochs=25,  # Increased epochs for better knowledge transfer
        batch_size=48,  # Optimized batch size
        validation_split=0.15,
        enable_self_modeling=False  # Simplified for foundation model
    )
    
    # Save the foundation model to /Models folder
    print("\n💾 Saving A.R.C.A.N.E. Foundation Model...")
    save_path = foundation_model.save_model()  # Automatically saves to /Models
    
    print(f"✅ Foundation model saved to: {save_path}")
    
    # Get model statistics
    stats = foundation_model.get_model_stats()
    print("\n📈 Foundation Model Statistics:")
    print(f"  Model Name: {stats['model_name']}")
    print(f"  Base Model: {stats['ollama_base_model']}")
    print(f"  Total Parameters: {stats['total_parameters']:,}")
    print(f"  Vocabulary Size: {stats['actual_vocab_size']:,}")
    print(f"  Sequence Length: {stats['sequence_length']}")
    print(f"  Embedding Dimension: {stats['embedding_dim']}")
    print(f"  Neuromimetic Layers: {stats['neuromimetic_layers']}")
    
    # Test the foundation model
    print("\n🎨 Testing Foundation Model Generation...")
    test_prompts = [
        "Artificial intelligence is",
        "The future of technology",
        "Human consciousness involves"
    ]
    
    for prompt in test_prompts:
        generated = foundation_model.generate_text(
            seed_text=prompt,
            max_length=40,
            temperature=0.8
        )
        print(f"Input: '{prompt}'")
        print(f"Output: {generated}")
        print("-" * 50)
    
    print("\n🎉 A.R.C.A.N.E. Foundation Small Language Model Created Successfully!")
    print(f"📂 Model Location: {save_path}")
    print(f"🧠 Type: Neuromimetic Foundation Model")
    print(f"⚡ Features: Spiking Neural Dynamics + Hebbian Learning")
    print(f"🤖 Base: Ollama llama3.2:1b + A.R.C.A.N.E. Neural Components")


if __name__ == "__main__":
    main()