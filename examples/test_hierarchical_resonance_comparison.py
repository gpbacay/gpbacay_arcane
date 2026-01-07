#!/usr/bin/env python3
"""
Test and Compare: HierarchicalResonanceFoundationModel vs Traditional Models

This script comprehensively tests the new HierarchicalResonanceFoundationModel
and compares it against traditional LSTM-based architectures.

Comparison includes:
- Training accuracy and loss curves
- Validation performance
- Text generation quality
- Training time analysis
- Memory efficiency

Author: Gianne P. Bacay
Project: A.R.C.A.N.E. (Augmented Reconstruction of Consciousness through Artificial Neural Evolution)
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, GlobalAveragePooling1D, Dropout
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpbacay_arcane import (
    HierarchicalResonanceFoundationModel,
    NeuromimeticSemanticModel,
    NeuralResonanceCallback,
)


# Set for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def load_text_data(file_path="data/shakespeare_small.txt", max_chars=15000):
    """Load and preprocess text data."""
    print(f"\n Loading text data...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()[:max_chars]
    
    text = text.lower()
    text = ' '.join(text.split())
    
    print(f"   Characters: {len(text):,}")
    return text


def prepare_sequences(text, tokenizer, seq_len=16):
    """Prepare training sequences."""
    sequences = tokenizer.texts_to_sequences([text])[0]
    
    X, y = [], []
    for i in range(len(sequences) - seq_len):
        X.append(sequences[i:i + seq_len])
        y.append(sequences[i + seq_len])
    
    return np.array(X), np.array(y)


def build_traditional_lstm(vocab_size, seq_len=16, embed_dim=32, hidden_dim=64):
    """Build a traditional deep LSTM model for comparison."""
    inputs = Input(shape=(seq_len,), name='input')
    
    # Embedding
    x = Embedding(vocab_size, embed_dim, name='embedding')(inputs)
    
    # Deep LSTM stack (similar depth to resonance hierarchy)
    x = LSTM(hidden_dim, return_sequences=True, dropout=0.1, name='lstm_1')(x)
    x = LSTM(hidden_dim, return_sequences=True, dropout=0.1, name='lstm_2')(x)
    x = LSTM(hidden_dim, return_sequences=True, dropout=0.1, name='lstm_3')(x)
    x = LSTM(hidden_dim, dropout=0.1, name='lstm_4')(x)
    
    # Dense layers
    x = Dense(hidden_dim * 2, activation='gelu', name='dense_1')(x)
    x = Dropout(0.1)(x)
    x = Dense(hidden_dim, activation='gelu', name='dense_2')(x)
    
    outputs = Dense(vocab_size, activation='softmax', name='output')(x)
    
    model = Model(inputs, outputs, name='Traditional_Deep_LSTM')
    return model


def generate_text_traditional(model, seed_text, tokenizer, seq_len=16, max_length=30, temperature=0.8):
    """Generate text using traditional LSTM model."""
    reverse_tokenizer = {v: k for k, v in tokenizer.word_index.items()}
    
    seed_tokens = tokenizer.texts_to_sequences([seed_text.lower()])[0]
    if not seed_tokens:
        seed_tokens = [1]
    
    if len(seed_tokens) < seq_len:
        seed_tokens = [0] * (seq_len - len(seed_tokens)) + seed_tokens
    else:
        seed_tokens = seed_tokens[-seq_len:]
    
    current_seq = seed_tokens.copy()
    generated_words = []
    
    for _ in range(max_length):
        pred = model.predict(np.array([current_seq]), verbose=0)[0]
        
        # Temperature sampling
        pred = np.log(pred + 1e-10) / temperature
        pred = np.exp(pred) / np.sum(np.exp(pred))
        
        next_token = np.random.choice(len(pred), p=pred)
        word = reverse_tokenizer.get(next_token, "")
        
        if word and word != "<UNK>" and word.strip():
            generated_words.append(word)
        
        current_seq = current_seq[1:] + [next_token]
        
        if word in [".", "!", "?"] and len(generated_words) > 5:
            break
    
    return " ".join(generated_words)


def run_comparison_test():
    """Run comprehensive comparison between models."""
    
    print("\n" + "=" * 80)
    print("   HIERARCHICAL RESONANCE vs TRADITIONAL MODEL COMPARISON")
    print("=" * 80)
    
    # ========== Configuration ==========
    VOCAB_SIZE = 2000
    SEQ_LEN = 16
    EMBED_DIM = 32
    HIDDEN_DIM = 64
    BATCH_SIZE = 32
    EPOCHS = 10
    RESONANCE_CYCLES = 5
    RESONANCE_LEVELS = 3
    
    print(f"\n Configuration:")
    print(f"   Vocab Size: {VOCAB_SIZE}")
    print(f"   Sequence Length: {SEQ_LEN}")
    print(f"   Hidden Dimension: {HIDDEN_DIM}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Resonance Levels: {RESONANCE_LEVELS}")
    print(f"   Resonance Cycles: {RESONANCE_CYCLES}")
    
    # ========== Load Data ==========
    text = load_text_data()
    
    print(f"\n Creating tokenizer...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<UNK>")
    tokenizer.fit_on_texts([text])
    actual_vocab = min(len(tokenizer.word_index) + 1, VOCAB_SIZE)
    print(f"   Vocabulary: {actual_vocab:,} words")
    
    X, y = prepare_sequences(text, tokenizer, SEQ_LEN)
    
    # Split data
    split_idx = int(len(X) * 0.85)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Validation samples: {len(X_val):,}")
    
    results = {}
    
    # ========== Model 1: Traditional Deep LSTM ==========
    print("\n" + "-" * 80)
    print(" MODEL 1: Traditional Deep LSTM")
    print("-" * 80)
    
    trad_model = build_traditional_lstm(
        vocab_size=actual_vocab,
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM
    )
    
    trad_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"   Parameters: {trad_model.count_params():,}")
    
    print("\n   Training...")
    start_time = time.time()
    trad_history = trad_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1
    )
    trad_time = time.time() - start_time
    
    results['Traditional LSTM'] = {
        'train_acc': trad_history.history['accuracy'][-1],
        'val_acc': trad_history.history['val_accuracy'][-1],
        'train_loss': trad_history.history['loss'][-1],
        'val_loss': trad_history.history['val_loss'][-1],
        'time': trad_time,
        'params': trad_model.count_params(),
        'history': trad_history.history
    }
    
    # ========== Model 2: NeuromimeticSemanticModel (Standard) ==========
    print("\n" + "-" * 80)
    print(" MODEL 2: NeuromimeticSemanticModel (Standard A.R.C.A.N.E.)")
    print("-" * 80)
    
    neuro_model = NeuromimeticSemanticModel(
        vocab_size=actual_vocab,
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM
    )
    neuro_keras = neuro_model.build_model()
    neuro_model.compile_model(learning_rate=1e-3)
    
    print(f"   Parameters: {neuro_keras.count_params():,}")
    
    # Use resonance callback
    resonance_cb_neuro = NeuralResonanceCallback(resonance_cycles=3)
    
    print("\n   Training with Neural Resonance...")
    start_time = time.time()
    neuro_history = neuro_keras.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[resonance_cb_neuro],
        verbose=1
    )
    neuro_time = time.time() - start_time
    
    results['Neuromimetic (Standard)'] = {
        'train_acc': neuro_history.history['accuracy'][-1],
        'val_acc': neuro_history.history['val_accuracy'][-1],
        'train_loss': neuro_history.history['loss'][-1],
        'val_loss': neuro_history.history['val_loss'][-1],
        'time': neuro_time,
        'params': neuro_keras.count_params(),
        'history': neuro_history.history
    }
    
    # ========== Model 3: HierarchicalResonanceFoundationModel ==========
    print("\n" + "-" * 80)
    print(" MODEL 3: HierarchicalResonanceFoundationModel (Advanced)")
    print("-" * 80)
    
    hier_model = HierarchicalResonanceFoundationModel(
        vocab_size=actual_vocab,
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_resonance_levels=RESONANCE_LEVELS,
        resonance_factor=0.15,
        use_temporal_coherence=True,
        use_attention_fusion=True,
        dropout_rate=0.1
    )
    hier_keras = hier_model.build_model()
    hier_model.compile_model(learning_rate=5e-4)
    
    print(f"   Parameters: {hier_keras.count_params():,}")
    print(f"   Resonance Levels: {RESONANCE_LEVELS}")
    print(f"   Temporal Coherence: Enabled")
    print(f"   Attention Fusion: Enabled")
    
    # Use resonance callback with more cycles for the deep hierarchy
    resonance_cb_hier = NeuralResonanceCallback(resonance_cycles=RESONANCE_CYCLES)
    
    print(f"\n   Training with Neural Resonance ({RESONANCE_CYCLES} cycles)...")
    start_time = time.time()
    hier_history = hier_keras.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[resonance_cb_hier],
        verbose=1
    )
    hier_time = time.time() - start_time
    
    results['Hierarchical Resonance'] = {
        'train_acc': hier_history.history['accuracy'][-1],
        'val_acc': hier_history.history['val_accuracy'][-1],
        'train_loss': hier_history.history['loss'][-1],
        'val_loss': hier_history.history['val_loss'][-1],
        'time': hier_time,
        'params': hier_keras.count_params(),
        'history': hier_history.history
    }
    
    # ========== Results Comparison ==========
    print("\n" + "=" * 80)
    print("   FINAL RESULTS COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Model':<30} | {'Val Acc':>10} | {'Val Loss':>10} | {'Time (s)':>10} | {'Params':>12}")
    print("-" * 80)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<30} | {metrics['val_acc']:>10.4f} | {metrics['val_loss']:>10.4f} | {metrics['time']:>10.1f} | {metrics['params']:>12,}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['val_acc'])
    print("-" * 80)
    print(f"\n Best Model by Validation Accuracy: {best_model[0]}")
    print(f"   Validation Accuracy: {best_model[1]['val_acc']:.4f}")
    
    # Calculate improvements
    trad_val_acc = results['Traditional LSTM']['val_acc']
    hier_val_acc = results['Hierarchical Resonance']['val_acc']
    
    if hier_val_acc > trad_val_acc:
        improvement = ((hier_val_acc - trad_val_acc) / trad_val_acc) * 100
        print(f"\n   Hierarchical Resonance improved by {improvement:.2f}% over Traditional LSTM")
    else:
        print(f"\n   Traditional LSTM performed better in this configuration")
    
    # ========== Text Generation Comparison ==========
    print("\n" + "=" * 80)
    print("   SEMANTIC GENERATION COMPARISON")
    print("=" * 80)
    
    test_prompts = ["the king", "we shall", "to be"]
    
    for prompt in test_prompts:
        print(f"\n Prompt: '{prompt}'")
        print("-" * 60)
        
        # Traditional LSTM
        trad_gen = generate_text_traditional(
            trad_model, prompt, tokenizer, 
            seq_len=SEQ_LEN, max_length=25, temperature=0.8
        )
        print(f"   Traditional LSTM:        {trad_gen}")
        
        # Neuromimetic
        neuro_gen = neuro_model.generate_text(
            prompt, tokenizer, max_length=25, temperature=0.8
        )
        print(f"   Neuromimetic (Standard): {neuro_gen}")
        
        # Hierarchical Resonance
        hier_gen = hier_model.generate_text(
            prompt, tokenizer, max_length=25, temperature=0.8
        )
        print(f"   Hierarchical Resonance:  {hier_gen}")
    
    # ========== Training Dynamics Analysis ==========
    print("\n" + "=" * 80)
    print("   TRAINING DYNAMICS ANALYSIS")
    print("=" * 80)
    
    for model_name, metrics in results.items():
        history = metrics['history']
        
        # Calculate stability (variance of validation loss)
        val_loss_var = np.var(history['val_loss'])
        
        # Calculate convergence speed (epochs to reach 90% of final accuracy)
        final_acc = history['val_accuracy'][-1]
        target = final_acc * 0.9
        convergence_epoch = next(
            (i for i, acc in enumerate(history['val_accuracy']) if acc >= target), 
            len(history['val_accuracy'])
        ) + 1
        
        print(f"\n   {model_name}:")
        print(f"      Convergence (90% of final): Epoch {convergence_epoch}")
        print(f"      Validation Loss Variance:   {val_loss_var:.6f}")
        print(f"      Final Train/Val Gap:        {abs(history['accuracy'][-1] - history['val_accuracy'][-1]):.4f}")
    
    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("   SUMMARY")
    print("=" * 80)
    
    print("""
   Key Observations:
   
   1. Hierarchical Resonance Foundation Model:
      - Multi-level ResonantGSER hierarchy enables deliberative processing
      - Neural Resonance cycles act as "Thinking Phase" before weight updates
      - Cross-level skip connections enable multi-scale feature extraction
      - Temporal coherence distillation captures sequential dynamics
      - Higher parameter count due to attention fusion and bioplastic layers
   
   2. Standard Neuromimetic Model:
      - Simpler 2-level resonance hierarchy
      - Good balance of complexity and performance
      - Faster training than full hierarchical model
   
   3. Traditional Deep LSTM:
      - Pure feed-forward (no resonance/harmonization)
      - Faster per-epoch training
      - May struggle with very deep configurations
   
   The Hierarchical Resonance model implements "System 2" reasoning through
   bi-directional feedback and prospective configuration, which can lead to
   better generalization on complex sequential patterns.
    """)
    
    print("=" * 80)
    print("   TEST COMPLETE")
    print("=" * 80 + "\n")
    
    return results


if __name__ == "__main__":
    # Ensure CPU for consistent comparison
    # tf.config.set_visible_devices([], 'GPU')
    
    run_comparison_test()

