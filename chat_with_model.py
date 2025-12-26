#!/usr/bin/env python3
"""
A.R.C.A.N.E. Model Chat Interface
Chat with your pretrained neuromimetic language model.

Author: Gianne P. Bacay
Project: A.R.C.A.N.E. (Augmented Reconstruction of Consciousness through Artificial Neural Evolution)
"""

import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import random
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from gpbacay_arcane.ollama_integration import OllamaARCANEHybrid
    from gpbacay_arcane.layers import DenseGSER, BioplasticDenseLayer, GSER
    print("A.R.C.A.N.E. modules loaded successfully")
except ImportError as e:
    print(f"Error importing A.R.C.A.N.E. modules: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install ollama sentence-transformers")
    sys.exit(1)


def improve_response_quality(response_text):
    """
    Improve the quality of a response through post-processing.
    
    Args:
        response_text: Raw response text from the model
        
    Returns:
        Improved response text
    """
    # Clean up the response
    response = response_text.strip()
    
    # Remove extra whitespace
    response = re.sub(r'\s+', ' ', response)
    
    # If response is too short or seems like gibberish, provide a better fallback
    if len(response.split()) < 3 or is_gibberish(response):
        return generate_better_fallback_response()
    
    # Ensure proper capitalization
    if response and response[0].isalpha():
        response = response[0].upper() + response[1:]
    
    # Ensure it ends with proper punctuation
    if response and response[-1].isalnum():
        response += "."
    
    # Remove repetitive words
    response = remove_repetitive_words(response)
    
    # For concise responses, limit length but ensure completeness
    words = response.split()
    if len(words) > 50:  # Limit to 50 words for conciseness
        # Try to find a natural stopping point
        truncated = " ".join(words[:50])
        # Add punctuation if missing
        if not truncated.endswith(('.', '!', '?')):
            truncated += "."
        return truncated
    
    return response


def is_gibberish(text):
    """
    Check if text appears to be gibberish.
    
    Args:
        text: Text to check
        
    Returns:
        True if text appears to be gibberish
    """
    words = text.split()
    
    # If very few words, might be gibberish
    if len(words) < 3:
        return True
    
    # Check for too many single characters
    single_char_words = [word for word in words if len(word) == 1 and word.isalpha()]
    if len(single_char_words) > len(words) * 0.3:
        return True
    
    # Check for lack of vowels (common in gibberish)
    vowels = "aeiou"
    vowel_count = sum(1 for char in text.lower() if char in vowels)
    if vowel_count < len(text) * 0.1 and len(text) > 10:
        return True
    
    return False


def remove_repetitive_words(text):
    """
    Remove repetitive words from text.
    
    Args:
        text: Text to process
        
    Returns:
        Text with repetitive words removed
    """
    words = text.split()
    if len(words) <= 1:
        return text
    
    # Remove consecutive duplicate words
    deduped_words = [words[0]]
    for i in range(1, len(words)):
        if words[i] != words[i-1]:
            deduped_words.append(words[i])
    
    return " ".join(deduped_words)


def generate_better_fallback_response():
    """
    Generate a better fallback response when the model produces gibberish.
    
    Returns:
        Better fallback response
    """
    fallback_responses = [
        "I'm an AI assistant based on the A.R.C.A.N.E. neuromimetic model. I can help answer questions and have conversations.",
        "I'm a bio-inspired language model that combines neural network principles with biological learning mechanisms.",
        "Hello! I'm A.R.C.A.N.E., a neuromimetic AI that uses spiking neural dynamics and Hebbian learning.",
        "I'm designed to understand and generate human-like text using advanced neural processing techniques.",
        "As an AI, I can discuss various topics including science, technology, philosophy, and more.",
        "I'm part of the A.R.C.A.N.E. project, which bridges neuroscience and artificial intelligence.",
        "I'm here to help with information, creative tasks, problem-solving, and conversation.",
        "My architecture includes spiking neural networks and biological learning principles for more natural processing."
    ]
    
    return random.choice(fallback_responses)


def load_pretrained_model(model_path="Models/arcane_transfer_learned_lm_best.h5", 
                         tokenizer_path="Models/arcane_transfer_learned_lm_saved/tokenizer.pkl"):
    """
    Load a pretrained A.R.C.A.N.E. model and tokenizer.
    
    Args:
        model_path: Path to the saved model
        tokenizer_path: Path to the saved tokenizer
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print("Loading pretrained A.R.C.A.N.E. model...")
    
    try:
        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        print(f"Tokenizer loaded from: {tokenizer_path}")
        
        # Load model with custom objects
        custom_objects = {
            'DenseGSER': DenseGSER,
            'BioplasticDenseLayer': BioplasticDenseLayer,
            'GSER': GSER
        }
        
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects
        )
        print(f"Model loaded from: {model_path}")
        print(f"Model parameters: {model.count_params():,}")
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def load_ollama_hybrid_model(model_dir="Models/arcane_transfer_learned_lm_saved"):
    """
    Load a pretrained OllamaARCANEHybrid model.
    
    Args:
        model_dir: Directory containing the saved model
        
    Returns:
        OllamaARCANEHybrid instance
    """
    print("Loading Ollama-A.R.C.A.N.E. Hybrid model...")
    
    try:
        # Create hybrid model instance
        hybrid_model = OllamaARCANEHybrid(
            ollama_model="llama3.2:1b",
            model_name="arcane_transfer_learned_lm"
        )
        
        # Load the saved model
        hybrid_model.load_model(model_dir)
        
        print(f"Hybrid model loaded from: {model_dir}")
        return hybrid_model
    except Exception as e:
        print(f"Error loading hybrid model: {e}")
        return None


def generate_text_with_model(model, tokenizer, seed_text, max_length=50, temperature=0.8):
    """
    Generate text using the pretrained model with improved sampling.
    
    Args:
        model: Trained TensorFlow model
        tokenizer: Fitted tokenizer
        seed_text: Starting text for generation
        max_length: Maximum length of generated text
        temperature: Sampling temperature (higher = more random)
        
    Returns:
        Generated text
    """
    try:
        # Tokenize input text
        tokenized_input = tokenizer.texts_to_sequences([seed_text])[0]
        
        # Get model parameters
        seq_len = model.input_shape[1]  # Sequence length expected by model
        
        # Generate text
        generated_tokens = list(tokenized_input)
        
        # Create reverse tokenizer mapping
        reverse_tokenizer = {v: k for k, v in tokenizer.word_index.items()}
        
        for _ in range(max_length):
            # Prepare input sequence (pad or truncate to model's expected length)
            if len(generated_tokens) > seq_len:
                input_sequence = generated_tokens[-seq_len:]
            else:
                input_sequence = generated_tokens + [0] * (seq_len - len(generated_tokens))
            
            # Convert to numpy array and reshape for model input
            input_array = np.array([input_sequence])
            
            # Get model prediction
            predictions = model.predict(input_array, verbose=0)[0]
            
            # Apply improved temperature scaling and sampling
            if temperature < 0.5:
                # Very conservative: greedy decoding
                next_token = np.argmax(predictions)
            elif temperature < 0.8:
                # Conservative: top-k sampling
                k = 20
                top_indices = np.argsort(predictions)[-k:]
                top_probs = predictions[top_indices]
                top_probs = top_probs / np.sum(top_probs)
                next_token = np.random.choice(top_indices, p=top_probs)
            else:
                # Balanced/creative: temperature sampling with nucleus
                predictions = np.log(predictions + 1e-8) / temperature
                predictions = np.exp(predictions)
                predictions = predictions / np.sum(predictions)
                
                # Nucleus sampling (top-p)
                sorted_indices = np.argsort(predictions)[::-1]
                cumsum_probs = np.cumsum(predictions[sorted_indices])
                cutoff_idx = np.where(cumsum_probs > 0.9)[0]
                if len(cutoff_idx) > 0:
                    cutoff_idx = cutoff_idx[0] + 1
                else:
                    cutoff_idx = 30
                
                nucleus_indices = sorted_indices[:cutoff_idx]
                nucleus_probs = predictions[nucleus_indices]
                nucleus_probs = nucleus_probs / np.sum(nucleus_probs)
                
                next_token = np.random.choice(nucleus_indices, p=nucleus_probs)
            
            # Add to generated tokens
            generated_tokens.append(next_token)
            
            # Stop if we generate an end token (assuming 0 is end/padding)
            if next_token == 0:
                break
            
            # Stop at natural sentence endings (but ensure minimum length)
            word = reverse_tokenizer.get(next_token, "")
            if word in [".", "!", "?"] and len(generated_tokens) > 8:
                break
        
        # Convert tokens back to text
        generated_words = []
        for token in generated_tokens[len(tokenized_input):]:  # Only new tokens
            word = reverse_tokenizer.get(token, "")
            if word and word != "<UNK>" and word.strip():
                generated_words.append(word)
        
        generated_text = " ".join(generated_words)
        
        # Clean up common artifacts
        generated_text = generated_text.replace("  ", " ")  # Remove double spaces
        generated_text = generated_text.strip()
        
        # Ensure the result starts with a capital letter if it's not empty
        if generated_text and generated_text[0].isalpha():
            generated_text = generated_text[0].upper() + generated_text[1:]
        
        return generated_text if generated_text else "I'm processing your request..."
    except Exception as e:
        print(f"Error during text generation: {e}")
        return "I'm having trouble responding right now. Please try again."


def chat_with_model():
    """
    Interactive chat interface with the pretrained model.
    """
    print("\n" + "="*60)
    print("A.R.C.A.N.E. Neuromimetic Language Model Chat Interface")
    print("="*60)
    
    # Try to load models in order of preference
    model = None
    tokenizer = None
    hybrid_model = None
    
    # Try loading the Arcane Foundational Small Language Model first
    foundational_model_path = "Models/arcane_foundational_slm_saved"
    if os.path.exists(foundational_model_path):
        print(f"\nAttempting to load Arcane Foundational Small Language Model...")
        try:
            hybrid_model = load_ollama_hybrid_model(foundational_model_path)
            if hybrid_model is not None:
                print(f"Successfully loaded Arcane Foundational Small Language Model")
        except Exception as e:
            print(f"Failed to load Arcane Foundational Small Language Model: {e}")
            hybrid_model = None
    
    # If not found, try loading the transfer learned hybrid model
    if hybrid_model is None:
        hybrid_model_path = "Models/arcane_transfer_learned_lm_saved"
        if os.path.exists(hybrid_model_path):
            print(f"\nAttempting to load Transfer Learning Model...")
            try:
                hybrid_model = load_ollama_hybrid_model(hybrid_model_path)
                if hybrid_model is not None:
                    print(f"Successfully loaded Transfer Learning Model")
            except Exception as e:
                print(f"Failed to load Transfer Learning Model: {e}")
                hybrid_model = None
    
    # If no hybrid model, try loading standard models
    if hybrid_model is None:
        model_paths = [
            ("Transfer Learned Model", "Models/arcane_transfer_learned_lm_best.h5", "Models/arcane_transfer_learned_lm_saved/tokenizer.pkl"),
            ("Transfer Learned Model (Directory)", "Models/arcane_transfer_learned_lm_saved/model.h5", 
             "Models/arcane_transfer_learned_lm_saved/tokenizer.pkl")
        ]
        
        # Try loading models in order
        for model_name, model_path, tokenizer_path in model_paths:
            if os.path.exists(model_path) and os.path.exists(tokenizer_path):
                print(f"\nAttempting to load {model_name}...")
                model, tokenizer = load_pretrained_model(model_path, tokenizer_path)
                if model is not None and tokenizer is not None:
                    print(f"Successfully loaded {model_name}")
                    break
            else:
                print(f"{model_name} not found at expected location")
    
    if hybrid_model is None and (model is None or tokenizer is None):
        print("No pretrained model found. Please train a model first.")
        print("Run 'python transfer_learn_arcane.py' to create a transfer learning model")
        return
    
    print("\n" + "="*60)
    print("Chat with your A.R.C.A.N.E. model!")
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
                print("Conversation history cleared.")
                continue
            
            if not user_input:
                continue
            
            # Add to conversation history
            conversation_history.append(f"You: {user_input}")
            
            # Prepare seed text (combine recent conversation for context)
            recent_history = conversation_history[-3:]  # Last 3 exchanges
            seed_text = " ".join([line.split(": ", 1)[1] for line in recent_history if ": " in line])
            
            # If seed text is empty, use just the user input
            if not seed_text:
                seed_text = user_input
            
            print("A.R.C.A.N.E. is thinking...")
            
            # Generate response using the appropriate method
            if hybrid_model is not None:
                # Use the hybrid model's improved generation method
                response = hybrid_model.generate_text_with_improved_sampling(
                    seed_text=seed_text,
                    max_length=80,
                    temperature=0.8
                )
            else:
                # Use the traditional method
                response = generate_text_with_model(
                    model=model,
                    tokenizer=tokenizer,
                    seed_text=seed_text,
                    max_length=60,
                    temperature=0.7
                )
            
            # For the Arcane Foundational Small Language Model, the response is already enhanced
            # Just do basic cleanup for conciseness
            improved_response = response.strip()
            # Remove extra whitespace
            improved_response = re.sub(r'\s+', ' ', improved_response)
            # Ensure proper capitalization
            if improved_response and improved_response[0].isalpha():
                improved_response = improved_response[0].upper() + improved_response[1:]
            # Ensure it ends with proper punctuation
            if improved_response and improved_response[-1].isalnum():
                improved_response += "."
            
            # Add to conversation history
            conversation_history.append(f"A.R.C.A.N.E.: {improved_response}")
            
            # Display response
            print(f"A.R.C.A.N.E.: {improved_response}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! Thanks for chatting with A.R.C.A.N.E.")
            break
        except Exception as e:
            print(f"Error during chat: {e}")
            fallback_response = generate_better_fallback_response()
            print(f"A.R.C.A.N.E.: {fallback_response}")


def main():
    """
    Main function to run the chat interface.
    """
    try:
        chat_with_model()
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()