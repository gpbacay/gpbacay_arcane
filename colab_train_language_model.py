# Neuromimetic Language Model Training for Google Colab
# This script is optimized for Colab with GPU support

# Install required packages (run this cell first in Colab)
"""
!pip install requests matplotlib
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import requests
import os
import matplotlib.pyplot as plt
import pickle

# Check if GPU is available
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
print("TensorFlow Version: ", tf.__version__)

# Upload your gpbacay_arcane folder to Colab or mount Google Drive
# For Google Drive: uncomment the next lines
"""
from google.colab import drive
drive.mount('/content/drive')
# Then copy your gpbacay_arcane folder to /content/
"""

# If uploading directly, uncomment:
"""
from google.colab import files
uploaded = files.upload()  # Upload your gpbacay_arcane.zip file
!unzip gpbacay_arcane.zip
"""

# Add the path to your custom modules
import sys
sys.path.append('/content')  # Adjust path as needed

# Import your custom modules
from gpbacay_arcane.models import NeuromimeticLanguageModel
from gpbacay_arcane.callbacks import DynamicSelfModelingReservoirCallback
from gpbacay_arcane.layers import (
    GSER, DenseGSER, LatentTemporalCoherence, RelationalConceptModeling, 
    RelationalGraphAttentionReasoning, HebbianHomeostaticNeuroplasticity, 
    BioplasticDenseLayer, MultiheadLinearSelfAttentionKernalization
)

def download_shakespeare():
    """Download Shakespeare text dataset."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    if not os.path.exists("shakespeare.txt"):
        print("Downloading Shakespeare dataset...")
        response = requests.get(url)
        with open("shakespeare.txt", "w", encoding="utf-8") as f:
            f.write(response.text)
        print("Dataset downloaded successfully!")
    
    with open("shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    return text

def prepare_text_data(text, max_sequence_length=32, vocab_size=2000):
    """Prepare text data for language modeling."""
    
    # Create tokenizer
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
    tokenizer.fit_on_texts([text])
    
    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences([text])[0]
    
    # Create input-output pairs
    X, y = [], []
    
    for i in range(len(sequences) - max_sequence_length):
        X.append(sequences[i:i + max_sequence_length])
        y.append(sequences[i + max_sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} training sequences")
    print(f"Vocabulary size: {len(tokenizer.word_index) + 1}")
    print(f"Sequence length: {max_sequence_length}")
    
    return X, y, tokenizer

def train_language_model():
    """Train the neuromimetic language model."""
    
    # Download and prepare data
    text = download_shakespeare()
    
    # Hyperparameters (optimized for Colab GPU)
    max_sequence_length = 24
    vocab_size = 2000  # Further reduced for faster training
    embedding_dim = 64
    d_model = 96      # Slightly reduced
    num_heads = 4
    reservoir_dim = 192  # Reduced
    spectral_radius = 1.2
    leak_rate = 0.1
    spike_threshold = 0.3
    max_dynamic_reservoir_dim = 384
    d_coherence = 48  # Reduced
    num_thought_steps = 4  # Reduced for faster training
    
    # Prepare data
    X, y, tokenizer = prepare_text_data(text, max_sequence_length, vocab_size)
    
    # Split data
    split_idx = int(0.9 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Get actual vocab size from tokenizer
    actual_vocab_size = len(tokenizer.word_index) + 1
    
    # Initialize model
    model_builder = NeuromimeticLanguageModel(
        vocab_size=actual_vocab_size,
        max_sequence_length=max_sequence_length,
        embedding_dim=embedding_dim,
        d_model=d_model,
        num_heads=num_heads,
        reservoir_dim=reservoir_dim,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_dynamic_reservoir_dim=max_dynamic_reservoir_dim,
        d_coherence=d_coherence,
        num_thought_steps=num_thought_steps,
        dropout_rate=0.15
    )
    
    model_builder.build_model()
    model_builder.compile_model(learning_rate=2e-3)  # Higher learning rate for faster training
    
    # Print model summary
    model_builder.model.summary()
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=4,  # Reduced patience for faster training
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        'neuromimetic_language_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    dynamic_reservoir_callback = DynamicSelfModelingReservoirCallback(
        reservoir_layer=model_builder.reservoir_layer,
        performance_metric='val_accuracy',
        target_metric=0.35,  # Lower target
        growth_rate=3,
        prune_rate=0.02
    )
    
    # Train the model
    print("Starting training with GPU acceleration...")
    try:
        history = model_builder.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=12,  # Reduced epochs for faster training
            batch_size=64,  # Larger batch size for GPU
            callbacks=[early_stopping, reduce_lr, model_checkpoint, dynamic_reservoir_callback],
            verbose=1
        )
    except Exception as e:
        print(f"Training failed with error: {e}")
        return
    
    # Evaluate the model
    val_loss, val_accuracy, val_perplexity = model_builder.model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Perplexity: {val_perplexity:.4f}")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['perplexity'], label='Training Perplexity')
    plt.plot(history.history['val_perplexity'], label='Validation Perplexity')
    plt.title('Model Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Save tokenizer
    with open('tokenizer.pickle', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print(f"\nModel and tokenizer saved!")
    
    # Download files to local machine (Colab specific)
    from google.colab import files
    files.download('neuromimetic_language_model.keras')
    files.download('tokenizer.pickle')
    
    # Generate some sample text
    print("\nGenerating sample text...")
    sample_texts = [
        "ROMEO:",
        "To be or not",
        "What light",
        "Fair is foul"
    ]
    
    for seed in sample_texts:
        try:
            generated = model_builder.generate_text(
                seed_text=seed,
                tokenizer=tokenizer,
                max_length=40,
                temperature=0.8
            )
            print(f"\nSeed: '{seed}'")
            print(f"Generated: '{generated}'")
        except Exception as e:
            print(f"Generation failed for '{seed}': {e}")

if __name__ == "__main__":
    train_language_model()

# Instructions for Google Colab:
# 1. Upload this file to Colab
# 2. Upload your gpbacay_arcane folder (zip it first)
# 3. Enable GPU: Runtime -> Change runtime type -> GPU
# 4. Run the cells in order
# 5. The trained model will be automatically downloaded
