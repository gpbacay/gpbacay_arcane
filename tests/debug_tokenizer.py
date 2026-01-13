#!/usr/bin/env python3
"""
Debug script to test tokenizer functionality with the latest model.
"""

import os
import sys
import pickle
import numpy as np
import tensorflow as tf

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from gpbacay_arcane.ollama_integration import OllamaARCANEHybrid
    from gpbacay_arcane.layers import DenseGSER, BioplasticDenseLayer, GSER
    print("✅ ARCANE modules loaded successfully")
except ImportError as e:
    print(f"❌ Error importing ARCANE modules: {e}")
    sys.exit(1)

def debug_tokenizer():
    """Debug the tokenizer functionality."""
    print("🔄 Loading Ollama-ARCANE Hybrid model...")
    
    try:
        # Create hybrid model instance
        hybrid_model = OllamaARCANEHybrid(
            ollama_model="llama3.2:1b",
            model_name="arcane_transfer_learned_lm"
        )
        
        # Load the saved model
        model_dir = "Models/arcane_transfer_learned_lm_saved"
        hybrid_model.load_model(model_dir)
        
        print(f"✅ Hybrid model loaded from: {model_dir}")
        
        # Get model statistics
        print("\n📊 Model Statistics:")
        stats = hybrid_model.get_model_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test the tokenizer
        if hybrid_model.tokenizer is not None:
            print("\n🔍 Testing tokenizer functionality...")
            
            # Test text
            test_text = "Hello, how are you?"
            print(f"Original text: {test_text}")
            
            # Convert to tokens
            tokens = hybrid_model.tokenizer.texts_to_sequences([test_text])[0]
            print(f"Token IDs: {tokens}")
            
            # Create reverse mapping
            reverse_tokenizer = {v: k for k, v in hybrid_model.tokenizer.word_index.items()}
            print(f"Reverse tokenizer sample (first 10 items): {dict(list(reverse_tokenizer.items())[:10])}")
            print(f"Vocabulary size: {len(reverse_tokenizer)}")
            
            # Convert back to words
            words = [reverse_tokenizer.get(token, "<UNK>") for token in tokens]
            print(f"Reconstructed words: {words}")
            print(f"Reconstructed text: {' '.join(words)}")
            
            # Test generation with a simple input
            print("\n🔍 Testing text generation...")
            seed_text = "Hello"
            print(f"Seed text: {seed_text}")
            
            # Let's manually test the generation process
            print("\n🔍 Manual generation test...")
            seed_tokens = hybrid_model.tokenizer.texts_to_sequences([seed_text.lower()])[0]
            print(f"Seed tokens: {seed_tokens}")
            
            # Prepare sequence
            seq_len = hybrid_model.seq_len
            if len(seed_tokens) < seq_len:
                seed_tokens = [0] * (seq_len - len(seed_tokens)) + seed_tokens
            else:
                seed_tokens = seed_tokens[-seq_len:]
            print(f"Padded seed tokens: {seed_tokens}")
            
            # Predict next token
            pred = hybrid_model.model.predict(np.array([seed_tokens]), verbose=0)[0]
            print(f"Prediction shape: {pred.shape}")
            print(f"Prediction sample (first 10): {pred[:10]}")
            print(f"Prediction sum: {np.sum(pred)}")
            
            # Check the distribution
            sorted_indices = np.argsort(pred)[::-1]
            print(f"Top 5 predicted token IDs: {sorted_indices[:5]}")
            print(f"Top 5 prediction probabilities: {pred[sorted_indices[:5]]}")
            
            # Get the most likely token
            next_token = np.argmax(pred)
            print(f"Most likely next token ID: {next_token}")
            
            # Check if this token exists in the reverse tokenizer
            word = reverse_tokenizer.get(next_token, "<UNK>")
            print(f"Word for token {next_token}: '{word}'")
            
            # Test with a few more tokens
            print("\n🔍 Testing multiple token generation...")
            current_seq = seed_tokens.copy()
            generated_tokens = []
            
            for i in range(5):  # Generate 5 tokens
                pred = hybrid_model.model.predict(np.array([current_seq]), verbose=0)[0]
                next_token = np.argmax(pred)  # Use greedy decoding for simplicity
                generated_tokens.append(next_token)
                current_seq = current_seq[1:] + [next_token]
                print(f"Step {i+1}: Token {next_token}, Word '{reverse_tokenizer.get(next_token, '<UNK>')}'")
            
            print(f"Generated tokens: {generated_tokens}")
            generated_words = [reverse_tokenizer.get(token, "<UNK>") for token in generated_tokens]
            print(f"Generated words: {generated_words}")
            
            # Now test the actual method
            print("\n🔍 Testing actual generation method...")
            response = hybrid_model.generate_text_with_improved_sampling(
                seed_text=seed_text,
                max_length=20,
                temperature=0.8
            )
            print(f"Generated response: {response}")
        else:
            print("❌ Tokenizer not loaded")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_tokenizer()