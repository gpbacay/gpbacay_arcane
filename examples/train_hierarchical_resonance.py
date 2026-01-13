#!/usr/bin/env python3
"""
Train the Hierarchical Neural Resonance Foundation Model

This example demonstrates how to train the HierarchicalResonanceFoundationModel
with the NeuralResonanceCallback for deliberative "System 2" reasoning.

Features demonstrated:
- Multi-level ResonantGSER hierarchy with bi-directional feedback
- Neural Resonance cycles (Thinking Phase) during training
- Prospective Configuration for stable deep learning
- Multi-pathway feature fusion with attention
- Bioplastic Hebbian learning layers

Author: Gianne P. Bacay
Project: ARCANE (Augmented Reconstruction of Consciousness through Artificial Neural Evolution)
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpbacay_arcane import (
    HierarchicalResonanceFoundationModel,
    NeuralResonanceCallback,
    DynamicSelfModelingReservoirCallback,
)


def load_text_data(file_path, max_chars=50000):
    """Load and preprocess text data."""
    print(f"📖 Loading text data from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()[:max_chars]
    
    # Clean text
    text = text.lower()
    text = ' '.join(text.split())  # Normalize whitespace
    
    print(f"   Loaded {len(text):,} characters")
    return text


def prepare_sequences(text, tokenizer, seq_len=32):
    """Prepare training sequences from text."""
    print(f"🔢 Preparing sequences (seq_len={seq_len})...")
    
    # Tokenize
    sequences = tokenizer.texts_to_sequences([text])[0]
    
    # Create input-output pairs
    X, y = [], []
    for i in range(len(sequences) - seq_len):
        X.append(sequences[i:i + seq_len])
        y.append(sequences[i + seq_len])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"   Created {len(X):,} training sequences")
    return X, y


def train_hierarchical_resonance_model(
    text_path="data/shakespeare_small.txt",
    vocab_size=3000,
    seq_len=32,
    embed_dim=64,
    hidden_dim=128,
    num_resonance_levels=4,
    resonance_cycles=10,
    batch_size=32,
    epochs=20,
    save_dir="Models"
):
    """
    Train the Hierarchical Neural Resonance Foundation Model.
    
    Args:
        text_path: Path to training text file
        vocab_size: Maximum vocabulary size
        seq_len: Sequence length for training
        embed_dim: Embedding dimension
        hidden_dim: Hidden layer dimension
        num_resonance_levels: Number of resonance hierarchy levels (2-6)
        resonance_cycles: Number of resonance cycles per batch (5-15)
        batch_size: Training batch size
        epochs: Number of training epochs
        save_dir: Directory to save trained model
    """
    print("\n" + "=" * 70)
    print("  🧠 HIERARCHICAL NEURAL RESONANCE FOUNDATION MODEL TRAINING")
    print("=" * 70)
    
    # === Load Data ===
    text = load_text_data(text_path)
    
    # === Create Tokenizer ===
    print(f"📝 Creating tokenizer (vocab_size={vocab_size})...")
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
    tokenizer.fit_on_texts([text])
    actual_vocab_size = min(len(tokenizer.word_index) + 1, vocab_size)
    print(f"   Actual vocabulary size: {actual_vocab_size:,}")
    
    # === Prepare Sequences ===
    X, y = prepare_sequences(text, tokenizer, seq_len)
    
    # Split data
    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Validation samples: {len(X_val):,}")
    
    # === Create Model ===
    print(f"\n🏗️  Building Hierarchical Resonance Foundation Model...")
    print(f"   • Resonance levels: {num_resonance_levels}")
    print(f"   • Hidden dimension: {hidden_dim}")
    print(f"   • Embedding dimension: {embed_dim}")
    
    model = HierarchicalResonanceFoundationModel(
        vocab_size=actual_vocab_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_resonance_levels=num_resonance_levels,
        resonance_factor=0.15,
        use_temporal_coherence=True,
        use_attention_fusion=True,
        dropout_rate=0.1
    )
    
    # Build and compile
    keras_model = model.build_model()
    model.compile_model(learning_rate=3e-4)
    
    # Print model summary
    model.summary()
    
    # === Setup Callbacks ===
    print(f"\n⚙️  Setting up training callbacks...")
    
    # 1. Neural Resonance Callback - The "Thinking Phase"
    # This orchestrates bi-directional feedback between layers
    resonance_callback = NeuralResonanceCallback(
        resonance_cycles=resonance_cycles,
        learning_rate=0.01
    )
    print(f"   ✓ NeuralResonanceCallback (cycles={resonance_cycles})")
    
    # 2. Model Checkpoint
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "hierarchical_resonance_best.keras")
    checkpoint_callback = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    print(f"   ✓ ModelCheckpoint ({checkpoint_path})")
    
    # 3. Early Stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    print(f"   ✓ EarlyStopping (patience=5)")
    
    # 4. Learning Rate Reduction
    lr_reduction = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    print(f"   ✓ ReduceLROnPlateau")
    
    callbacks = [
        resonance_callback,
        checkpoint_callback,
        early_stopping,
        lr_reduction
    ]
    
    # === Train Model ===
    print(f"\n🚀 Starting training...")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Epochs: {epochs}")
    print(f"   • Resonance cycles per batch: {resonance_cycles}")
    print("\n" + "-" * 70)
    
    history = keras_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # === Save Final Model ===
    print(f"\n💾 Saving final model...")
    final_path = os.path.join(save_dir, "hierarchical_resonance_final.keras")
    keras_model.save(final_path)
    print(f"   Saved to: {final_path}")
    
    # Save tokenizer
    import pickle
    tokenizer_path = os.path.join(save_dir, "hierarchical_resonance_tokenizer.pkl")
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"   Tokenizer saved to: {tokenizer_path}")
    
    # === Test Generation ===
    print(f"\n✨ Testing text generation...")
    test_prompts = [
        "the king",
        "love is",
        "to be or",
    ]
    
    for prompt in test_prompts:
        generated = model.generate_text(
            seed_text=prompt,
            tokenizer=tokenizer,
            max_length=30,
            temperature=0.8
        )
        print(f"\n   Prompt: '{prompt}'")
        print(f"   Generated: {generated}")
    
    # === Training Summary ===
    print("\n" + "=" * 70)
    print("  📊 TRAINING SUMMARY")
    print("=" * 70)
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"\n   Final Training Accuracy:   {final_train_acc:.4f}")
    print(f"   Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"   Final Training Loss:       {final_train_loss:.4f}")
    print(f"   Final Validation Loss:     {final_val_loss:.4f}")
    
    print(f"\n   Model saved to: {save_dir}/")
    print(f"   Total epochs trained: {len(history.history['loss'])}")
    
    print("\n" + "=" * 70)
    print("  ✅ TRAINING COMPLETE")
    print("=" * 70 + "\n")
    
    return model, tokenizer, history


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train the Hierarchical Neural Resonance Foundation Model"
    )
    parser.add_argument(
        "--data", "-d",
        default="data/shakespeare_small.txt",
        help="Path to training text file"
    )
    parser.add_argument(
        "--vocab-size", "-v",
        type=int, default=3000,
        help="Maximum vocabulary size"
    )
    parser.add_argument(
        "--seq-len", "-s",
        type=int, default=32,
        help="Sequence length"
    )
    parser.add_argument(
        "--hidden-dim", "-H",
        type=int, default=128,
        help="Hidden layer dimension"
    )
    parser.add_argument(
        "--levels", "-l",
        type=int, default=4,
        help="Number of resonance hierarchy levels (2-6)"
    )
    parser.add_argument(
        "--cycles", "-c",
        type=int, default=10,
        help="Resonance cycles per batch (5-15)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int, default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int, default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--output", "-o",
        default="Models",
        help="Directory to save trained model"
    )
    
    args = parser.parse_args()
    
    train_hierarchical_resonance_model(
        text_path=args.data,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_resonance_levels=args.levels,
        resonance_cycles=args.cycles,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_dir=args.output
    )


if __name__ == "__main__":
    main()

