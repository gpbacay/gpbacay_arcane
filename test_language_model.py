import tensorflow as tf
import numpy as np
import pickle
import os
from gpbacay_arcane.layers import (
    GSER, 
    DenseGSER, 
    LatentTemporalCoherence, 
    RelationalConceptModeling, 
    RelationalGraphAttentionReasoning, 
    HebbianHomeostaticNeuroplasticity, 
    BioplasticDenseLayer,
    MultiheadLinearSelfAttentionKernalization
)

# Define custom objects for model loading
custom_objects = {
    'GSER': GSER,
    'DenseGSER': DenseGSER,
    'LatentTemporalCoherence': LatentTemporalCoherence,
    'RelationalConceptModeling': RelationalConceptModeling,
    'RelationalGraphAttentionReasoning': RelationalGraphAttentionReasoning,
    'HebbianHomeostaticNeuroplasticity': HebbianHomeostaticNeuroplasticity,
    'BioplasticDenseLayer': BioplasticDenseLayer,
    'MultiheadLinearSelfAttentionKernalization': MultiheadLinearSelfAttentionKernalization
}

def load_model_and_tokenizer():
    """Load the trained language model and tokenizer."""
    try:
        # Load model
        model_path = 'Models/neuromimetic_language_model.keras'
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("Language model loaded successfully.")
        
        # Load tokenizer
        tokenizer_path = 'Models/tokenizer.pickle'
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        print("Tokenizer loaded successfully.")
        
        return model, tokenizer
        
    except FileNotFoundError as e:
        print(f"Error: Required files not found. Please train the model first using train_language_model.py")
        print(f"Missing: {e}")
        return None, None
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None

def generate_text_with_model(model, tokenizer, seed_text, max_length=100, temperature=1.0):
    """Generate text using the loaded model."""
    max_sequence_length = model.input_shape[1]
    
    # Convert seed text to tokens
    seed_tokens = tokenizer.texts_to_sequences([seed_text])[0]
    
    # Handle empty or out-of-vocabulary seeds
    if not seed_tokens:
        seed_tokens = [1]  # Use a default token
    
    # Pad or truncate to model's expected length
    if len(seed_tokens) > max_sequence_length:
        seed_tokens = seed_tokens[-max_sequence_length:]
    else:
        seed_tokens = [0] * (max_sequence_length - len(seed_tokens)) + seed_tokens
    
    generated_tokens = seed_tokens.copy()
    
    print(f"Generating text with seed: '{seed_text}'")
    print("Generated text:")
    print(seed_text, end="")
    
    for i in range(max_length):
        # Prepare input
        input_sequence = np.array([generated_tokens[-max_sequence_length:]])
        
        # Predict next token
        predictions = model.predict(input_sequence, verbose=0)[0]
        
        # Apply temperature sampling
        predictions = predictions / temperature
        predictions = tf.nn.softmax(predictions).numpy()
        
        # Sample next token (avoid padding token 0)
        next_token = np.random.choice(len(predictions), p=predictions)
        
        # Convert token to word
        word = ""
        for w, idx in tokenizer.word_index.items():
            if idx == next_token:
                word = w
                break
        
        if word and word != "<UNK>":
            print(f" {word}", end="")
        
        # Add to sequence
        generated_tokens.append(next_token)
        
        # Stop if we hit padding token or unknown
        if next_token == 0 or word == "<UNK>":
            break
    
    print("\n" + "="*80)

def interactive_generation():
    """Interactive text generation interface."""
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        return
    
    print("\nNeuromimetic Language Model - Interactive Text Generation")
    print("="*60)
    print("Enter seed text to generate from, or 'quit' to exit")
    print("You can also adjust temperature (0.1-2.0) by typing 'temp X.X'")
    print("="*60)
    
    temperature = 1.0
    
    while True:
        user_input = input(f"\n[temp={temperature}] Enter seed text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input.lower().startswith('temp '):
            try:
                new_temp = float(user_input.split()[1])
                if 0.1 <= new_temp <= 2.0:
                    temperature = new_temp
                    print(f"Temperature set to {temperature}")
                else:
                    print("Temperature must be between 0.1 and 2.0")
            except (ValueError, IndexError):
                print("Invalid temperature format. Use 'temp X.X' (e.g., 'temp 0.8')")
            continue
        
        if not user_input:
            user_input = "To be or not to be"
            print(f"Using default seed: '{user_input}'")
        
        try:
            generate_text_with_model(
                model, tokenizer, user_input, 
                max_length=80, temperature=temperature
            )
        except Exception as e:
            print(f"Error generating text: {e}")

def run_sample_generations():
    """Run some predefined sample generations."""
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        return
    
    print("\nRunning sample text generations...")
    print("="*60)
    
    sample_seeds = [
        "ROMEO:",
        "To be or not to be",
        "What light through yonder",
        "Fair is foul and foul",
        "Once upon a time",
        "The king said",
        "Love is",
        "In fair Verona"
    ]
    
    temperatures = [0.5, 0.8, 1.0, 1.2]
    
    for seed in sample_seeds[:4]:  # Limit to first 4 for brevity
        for temp in [0.8, 1.2]:  # Test two temperatures
            print(f"\nSeed: '{seed}' (temperature={temp})")
            print("-" * 40)
            try:
                generate_text_with_model(
                    model, tokenizer, seed,
                    max_length=60, temperature=temp
                )
            except Exception as e:
                print(f"Error: {e}")

def evaluate_model():
    """Evaluate the model on some metrics."""
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        return
    
    print("\nModel Information:")
    print("="*40)
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print(f"Vocabulary size: {len(tokenizer.word_index) + 1}")
    print(f"Total parameters: {model.count_params():,}")
    
    # Show model architecture
    print(f"\nModel Architecture Summary:")
    model.summary()

if __name__ == "__main__":
    print("Neuromimetic Language Model - Testing Interface")
    print("="*50)
    
    while True:
        print("\nChoose an option:")
        print("1. Interactive text generation")
        print("2. Run sample generations")
        print("3. Show model information")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            interactive_generation()
        elif choice == '2':
            run_sample_generations()
        elif choice == '3':
            evaluate_model()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-4.")


# Neuromimetic Language Model Testing
# python test_language_model.py
