import os
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.layers import Input, Dense, LSTM, GlobalAveragePooling1D, LayerNormalization, Dropout, Reshape
from tensorflow.keras import Model
from gpbacay_arcane.layers import ResonantGSER, BioplasticDenseLayer
from gpbacay_arcane.callbacks import NeuralResonanceCallback

# Set stability policy
tf.keras.mixed_precision.set_global_policy('float32')

def prepare_mnist_data():
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # We will treat the 28x28 image as a sequence of 28 rows, each with 28 features
    # This aligns with the temporal/sequential focus of the A.R.C.A.N.E. architecture
    
    # Ensure dataset size is multiple of batch size (32) for stateful/resonant layers if needed
    train_limit = (len(x_train) // 32) * 32
    test_limit = (len(x_test) // 32) * 32
    
    x_train, y_train = x_train[:train_limit], y_train[:train_limit]
    x_test, y_test = x_test[:test_limit], y_test[:test_limit]
    
    return x_train, y_train, x_test, y_test

def build_traditional_model(seq_len, feature_dim, hidden_dim):
    inputs = Input(batch_shape=(32, seq_len, feature_dim))
    x = LSTM(hidden_dim, return_sequences=True)(inputs)
    x = LSTM(hidden_dim, return_sequences=True)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(hidden_dim, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    return Model(inputs, outputs, name="Deep_Traditional_LSTM")

def build_resonant_model(seq_len, feature_dim, hidden_dim):
    inputs = Input(batch_shape=(32, seq_len, feature_dim))
    
    # First ResonantGSER layer - Processing spatial-temporal row dynamics
    x = ResonantGSER(
        units=hidden_dim,
        spectral_radius=0.9,
        leak_rate=0.1,
        spike_threshold=0.35,
        activation='gelu',
        name='resonant_1'
    )(inputs)
    
    x = LayerNormalization()(x)
    
    # Second ResonantGSER layer - Higher-level feature integration
    x = ResonantGSER(
        units=hidden_dim,
        spectral_radius=0.8,
        leak_rate=0.12,
        spike_threshold=0.3,
        activation='gelu',
        name='resonant_2'
    )(x)
    
    # Temporal integration via LSTM
    x = LSTM(hidden_dim, return_sequences=True)(x)
    x = GlobalAveragePooling1D()(x)
    
    # Hebbian/Bioplastic Layer for final feature refinement
    x = BioplasticDenseLayer(
        units=hidden_dim,
        learning_rate=1e-3,
        activation='gelu',
        name='bioplastic'
    )(x)
    
    outputs = Dense(10, activation='softmax')(x)
    return Model(inputs, outputs, name="Resonant_ARCANE")

def run_comparison():
    print("Starting A.R.C.A.N.E. Resonant vs Traditional Model Comparison on MNIST")
    print("=" * 80)
    
    # 1. Data Setup
    x_train, y_train, x_test, y_test = prepare_mnist_data()
    
    # Use a subset for faster demonstration if needed, but MNIST is fast enough
    # subset = 20000
    # x_train, y_train = x_train[:subset], y_train[:subset]
    
    params = {
        "seq_len": 28,
        "feature_dim": 28,
        "hidden_dim": 64
    }
    
    # 2. Traditional Model
    print("\nTraining Deep Traditional LSTM Model...")
    trad_model = build_traditional_model(**params)
    trad_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    start_time = time.time()
    trad_history = trad_model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=10,
        batch_size=32,
        verbose=1
    )
    trad_time = time.time() - start_time
    
    # 3. Resonant Model
    print("\nTraining Resonant A.R.C.A.N.E. Model...")
    res_model = build_resonant_model(**params)
    res_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Resonance Callback with cycles for prospective alignment
    resonance_cb = NeuralResonanceCallback(resonance_cycles=5) # 5 cycles is usually enough for MNIST
    
    start_time = time.time()
    res_history = res_model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=10,
        batch_size=32,
        callbacks=[resonance_cb],
        verbose=1
    )
    res_time = time.time() - start_time
    
    # 4. Results Comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON RESULTS (MNIST - 10 Epochs)")
    print("=" * 80)
    print(f"{'Metric':<20} | {'Traditional LSTM':<20} | {'Resonant ARCANE':<20}")
    print("-" * 80)
    print(f"{'Final Train Acc':<20} | {trad_history.history['accuracy'][-1]:.4f}               | {res_history.history['accuracy'][-1]:.4f}")
    print(f"{'Final Test Acc':<20} | {trad_history.history['val_accuracy'][-1]:.4f}               | {res_history.history['val_accuracy'][-1]:.4f}")
    print(f"{'Final Test Loss':<20} | {trad_history.history['val_loss'][-1]:.4f}               | {res_history.history['val_loss'][-1]:.4f}")
    print(f"{'Total Training Time':<20} | {trad_time:.2f}s                | {res_time:.2f}s")
    print("-" * 80)
    
    res_acc = res_history.history['val_accuracy'][-1]
    trad_acc = trad_history.history['val_accuracy'][-1]
    
    if res_acc > trad_acc:
        diff = (res_acc - trad_acc) * 100
        print(f"Resonant Model outperformed Traditional Model by {diff:.2f}% accuracy!")
        print("The hierarchical resonance and prospective alignment provided better feature extraction.")
    else:
        print("Traditional Model performed better/equal in this specific configuration.")
    
    print("=" * 80)

if __name__ == "__main__":
    # Ensure we use CPU for this test to avoid local setup issues if GPU memory is tight
    # with tf.device('/CPU:0'):
    run_comparison()

if __name__ == "__main__":
    run_comparison()

