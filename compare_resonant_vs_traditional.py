import os
import numpy as np
import tensorflow as tf
import requests
import re
import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, GlobalAveragePooling1D, Concatenate, LayerNormalization, Dropout
from tensorflow.keras import Model
from gpbacay_arcane.layers import ResonantGSER, BioplasticDenseLayer
from gpbacay_arcane.callbacks import NeuralResonanceCallback

# Set stability policy
tf.keras.mixed_precision.set_global_policy('float32')

def download_data():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not os.path.exists("shakespeare_small.txt"):
        print("Downloading small dataset...")
        response = requests.get(url)
        # Use only a small portion for fast comparison
        text = response.text[:50000]
        with open("shakespeare_small.txt", "w", encoding="utf-8") as f:
            f.write(text)
    else:
        with open("shakespeare_small.txt", "r", encoding="utf-8") as f:
            text = f.read()
    return text

def prepare_data(text, seq_len=16, vocab_size=1000):
    text = text.lower()
    text = re.sub(r'[^\w\s\.\,\!\?\:\;]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>", lower=True)
    tokenizer.fit_on_texts([text])
    sequences = tokenizer.texts_to_sequences([text])[0]
    
    X, y = [], []
    for i in range(0, len(sequences) - seq_len, 3):
        X.append(sequences[i:i + seq_len])
        y.append(sequences[i + seq_len])
    
    return np.array(X), np.array(y), len(tokenizer.word_index) + 1

def build_traditional_model(vocab_size, seq_len, embed_dim, hidden_dim):
    inputs = Input(batch_shape=(32, seq_len))
    x = Embedding(vocab_size, embed_dim)(inputs)
    x = LSTM(hidden_dim, return_sequences=True)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(hidden_dim, activation='relu')(x)
    outputs = Dense(vocab_size, activation='softmax')(x)
    return Model(inputs, outputs, name="Traditional_LSTM")

def build_resonant_model(vocab_size, seq_len, embed_dim, hidden_dim):
    inputs = Input(batch_shape=(32, seq_len))
    x = Embedding(vocab_size, embed_dim)(inputs)
    
    # Resonant Layer 1
    x = ResonantGSER(
        units=hidden_dim,
        spectral_radius=0.9,
        leak_rate=0.1,
        spike_threshold=0.35,
        activation='gelu',
        name='resonant_1'
    )(x)
    
    x = LayerNormalization()(x)
    
    # Resonant Layer 2
    x = ResonantGSER(
        units=hidden_dim,
        spectral_radius=0.8,
        leak_rate=0.12,
        spike_threshold=0.3,
        activation='gelu',
        name='resonant_2'
    )(x)
    
    x = LSTM(hidden_dim, return_sequences=True)(x)
    x = GlobalAveragePooling1D()(x)
    
    # Hebbian/Bioplastic Layer
    x = BioplasticDenseLayer(
        units=hidden_dim,
        learning_rate=1e-3,
        activation='gelu',
        name='bioplastic'
    )(x)
    
    outputs = Dense(vocab_size, activation='softmax')(x)
    return Model(inputs, outputs, name="Resonant_ARCANE")

def run_comparison():
    print("Starting A.R.C.A.N.E. Resonant vs Traditional Model Comparison")
    print("=" * 70)
    
    # 1. Data Setup
    text = download_data()
    X, y, vocab_size = prepare_data(text)
    
    # Use a subset for a more significant comparison
    subset = 15000
    X, y = X[:subset], y[:subset]
    
    # Ensure dataset size is multiple of batch size (32)
    X = X[:(len(X) // 32) * 32]
    y = y[:(len(y) // 32) * 32]
    
    split = int(0.8 * len(X))
    # Ensure split is also multiple of 32
    split = (split // 32) * 32
    
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    params = {
        "vocab_size": vocab_size,
        "seq_len": 16,
        "embed_dim": 32,
        "hidden_dim": 64
    }
    
    # 2. Traditional Model (Updated to be deeper for fairness)
    print("\nTraining Traditional LSTM Model (Deepened)...")
    trad_model = build_traditional_model(**params)
    # Adding an extra layer to Traditional for fairness in depth
    inputs_trad = Input(batch_shape=(32, 16))
    x_trad = Embedding(vocab_size, 32)(inputs_trad)
    x_trad = LSTM(64, return_sequences=True)(x_trad)
    x_trad = LSTM(64, return_sequences=True)(x_trad) # Extra layer
    x_trad = GlobalAveragePooling1D()(x_trad)
    x_trad = Dense(64, activation='relu')(x_trad)
    outputs_trad = Dense(vocab_size, activation='softmax')(x_trad)
    trad_model = Model(inputs_trad, outputs_trad, name="Deep_Traditional_LSTM")
    
    trad_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    start_time = time.time()
    trad_history = trad_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        verbose=1
    )
    trad_time = time.time() - start_time
    
    # 3. Resonant Model
    print("\nTraining Resonant A.R.C.A.N.E. Model...")
    res_model = build_resonant_model(**params)
    res_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Resonance Callback with more cycles for deep alignment
    resonance_cb = NeuralResonanceCallback(resonance_cycles=10)
    
    start_time = time.time()
    res_history = res_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        callbacks=[resonance_cb],
        verbose=1
    )
    res_time = time.time() - start_time
    
    # 4. Results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS (15 Epochs)")
    print("=" * 70)
    print(f"{'Metric':<20} | {'Traditional':<15} | {'Resonant':<15}")
    print("-" * 70)
    print(f"{'Final Train Acc':<20} | {trad_history.history['accuracy'][-1]:.4f}          | {res_history.history['accuracy'][-1]:.4f}")
    print(f"{'Final Val Acc':<20} | {trad_history.history['val_accuracy'][-1]:.4f}          | {res_history.history['val_accuracy'][-1]:.4f}")
    print(f"{'Final Val Loss':<20} | {trad_history.history['val_loss'][-1]:.4f}          | {res_history.history['val_loss'][-1]:.4f}")
    print(f"{'Training Time':<20} | {trad_time:.2f}s           | {res_time:.2f}s")
    print("-" * 70)
    
    if res_history.history['val_accuracy'][-1] > trad_history.history['val_accuracy'][-1]:
        print("Resonant Model outperformed Traditional Model in accuracy!")
    else:
        print("Traditional Model performed better/equal in this short run.")

if __name__ == "__main__":
    run_comparison()

