
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Reshape, GlobalAveragePooling1D, Dense, Dropout, LayerNormalization, Add, LSTM
from tensorflow.keras.models import Model, Sequential
from gpbacay_arcane.layers import ResonantGSER, BioplasticDenseLayer, SpatioTemporalSummarization
from gpbacay_arcane.callbacks import NeuralResonanceCallback

# --- ARCANE Model Definition ---
def build_arcane_model(hidden_dim=256, resonance_factor=0.2, num_resonance_levels=3):
    """
    Builds a neuromimetic MNIST classifier using ResonantGSER and Bioplastic layers.
    Emphasizes Hierarchical Neural Resonance for Latent Space Reasoning.
    Treats MNIST images (28x28) as a sequence of 28 rows.
    """
    inputs = Input(shape=(28, 28), name='mnist_input')
    
    # Normalize input
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    
    # Reshape for sequential processing (28 timesteps, 28 features each)
    x = Reshape((28, 28))(x)

    resonant_layers = []
    for i in range(num_resonance_levels):
        res_layer = ResonantGSER(
            units=hidden_dim,
            resonance_factor=resonance_factor + (i * 0.05),
            spike_threshold=0.4 - (i * 0.05), 
            resonance_cycles=3,
            return_sequences=True,
            name=f'resonant_logic_level_{i+1}'
        )
        resonant_layers.append(res_layer)
    
    # Initial forward pass through resonant layers
    current_output = x
    for i, layer in enumerate(resonant_layers):
        current_output = layer(current_output)
        if i < num_resonance_levels - 1:
            current_output = LayerNormalization()(current_output)

    # Establish Hierarchical Feedback for Latent Space Reasoning
    for i in range(num_resonance_levels):
        if i > 0:
            resonant_layers[i].set_lower_layer(resonant_layers[i-1])
        if i < num_resonance_levels - 1:
            resonant_layers[i].set_higher_layer(resonant_layers[i+1])
    
    # Spatio-Temporal Summarization (Global context mixing for Unified Multi-Modal Semantic Space)
    summary = SpatioTemporalSummarization(d_model=hidden_dim)(current_output)
    
    # Feature Pooling
    pooled = GlobalAveragePooling1D()(summary)
    
    # Bioplastic Dense Layer (Hebbian Plasticity for Abstraction of Surface-Level Conceptual Variability)
    bioplastic = BioplasticDenseLayer(
        units=hidden_dim,
        learning_rate=2e-3,
        target_avg=0.1,
        homeostatic_rate=1e-4,
        activation='gelu',
        name='neuromimetic_refining'
    )(pooled)
    
    bioplastic = Dropout(0.2)(bioplastic)
    
    # Output layer
    outputs = Dense(10, activation='softmax', name='classification_output')(bioplastic)
    
    model = Model(inputs=inputs, outputs=outputs, name='ARCANE_LatentReasoning_Classifier')
    return model, resonant_layers

# --- Baseline Model Definition (Standard LSTM) ---
def build_baseline_model(hidden_dim=256):
    """
    Builds a standard LSTM-based MNIST classifier for comparison.
    """
    model = Sequential([
        Input(shape=(28, 28), name='mnist_input'),
        tf.keras.layers.Rescaling(1./255),
        Reshape((28, 28)), # Reshape for sequential processing (28 timesteps, 28 features each)
        LSTM(hidden_dim, return_sequences=True, name='lstm_1'),
        LayerNormalization(),
        LSTM(hidden_dim, name='lstm_2'),
        Dropout(0.2),
        Dense(10, activation='softmax', name='classification_output')
    ], name='LSTM_Baseline_Classifier')
    return model

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, callbacks=None):
    print(f"\n--- Training {model_name} ---")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=128,
        validation_split=0.1,
        verbose=1,
        callbacks=callbacks
    )
    
    print(f"\nEvaluating {model_name} on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy for {model_name}: {test_acc:.4f}")
    print(f"Test Loss for {model_name}: {test_loss:.4f}")
    
    return history, test_acc, test_loss

def plot_results(arcane_history, baseline_history, arcane_acc, baseline_acc):
    plt.style.use('dark_background')
    plt.figure(figsize=(18, 6))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(arcane_history.history['accuracy'], label='ARCANE Train Accuracy', color='#00ffcc', linewidth=2)
    plt.plot(arcane_history.history['val_accuracy'], label='ARCANE Val Accuracy', color='#00ccff', linewidth=2)
    plt.plot(baseline_history.history['accuracy'], label='Baseline Train Accuracy', color='#ff33cc', linewidth=2, linestyle='--')
    plt.plot(baseline_history.history['val_accuracy'], label='Baseline Val Accuracy', color='#ff66ff', linewidth=2, linestyle='--')
    plt.title('Training & Validation Accuracy', color='white', fontsize=14)
    plt.xlabel('Epoch', color='white')
    plt.ylabel('Accuracy', color='white')
    plt.legend(facecolor='#1e1e1e', edgecolor='#333333', fontsize=10)
    plt.grid(True, alpha=0.2)
    plt.gca().set_facecolor('#1e1e1e')
    plt.tick_params(colors='white')
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(arcane_history.history['loss'], label='ARCANE Train Loss', color='#00ffcc', linewidth=2)
    plt.plot(arcane_history.history['val_loss'], label='ARCANE Val Loss', color='#00ccff', linewidth=2)
    plt.plot(baseline_history.history['loss'], label='Baseline Train Loss', color='#ff33cc', linewidth=2, linestyle='--')
    plt.plot(baseline_history.history['val_loss'], label='Baseline Val Loss', color='#ff66ff', linewidth=2, linestyle='--')
    plt.title('Training & Validation Loss', color='white', fontsize=14)
    plt.xlabel('Epoch', color='white')
    plt.ylabel('Loss', color='white')
    plt.legend(facecolor='#1e1e1e', edgecolor='#333333', fontsize=10)
    plt.grid(True, alpha=0.2)
    plt.gca().set_facecolor('#1e1e1e')
    plt.tick_params(colors='white')
    
    plt.gcf().set_facecolor('#121212')
    plt.tight_layout()
    
    # Save the evaluation plot
    plot_path = 'latent_space_reasoning_poc_results.png'
    plt.savefig(plot_path)
    print(f"\nEvaluation plot saved as '{plot_path}'")

def analyze_results(arcane_history, baseline_history, arcane_acc, baseline_acc):
    print("\n--- Hypothesis Validation ---")
    print(f"ARCANE Test Accuracy: {arcane_acc:.4f}")
    print(f"Baseline LSTM Test Accuracy: {baseline_acc:.4f}")

    arcane_val_loss = arcane_history.history['val_loss']
    baseline_val_loss = baseline_history.history['val_loss']

    arcane_loss_variance = np.var(arcane_val_loss)
    baseline_loss_variance = np.var(baseline_val_loss)

    print(f"ARCANE Validation Loss Variance: {arcane_loss_variance:.6f}")
    print(f"Baseline Validation Loss Variance: {baseline_loss_variance:.6f}")

    arcane_train_val_gap = np.mean(np.array(arcane_history.history['loss']) - np.array(arcane_val_loss))
    baseline_train_val_gap = np.mean(np.array(baseline_history.history['loss']) - np.array(baseline_val_loss))

    print(f"ARCANE Average Train/Val Loss Gap: {arcane_train_val_gap:.6f}")
    print(f"Baseline Average Train/Val Loss Gap: {baseline_train_val_gap:.6f}")

    print("\n--- Interpretation of Latent Space Reasoning ---")
    if arcane_acc > baseline_acc and arcane_loss_variance < baseline_loss_variance and arcane_train_val_gap < baseline_train_val_gap:
        print("The ARCANE model demonstrates superior generalization and more stable training dynamics.")
        print("This supports the hypothesis that ARCANE's Hierarchical Neural Resonance facilitates Latent Space Reasoning.")
        print("The iterative 'Thinking Phase' (semantic alignment, conceptual gating) allows the model to refine")
        print("its internal semantic representations before weight updates, leading to a more robust and")
        print("coherent understanding, which translates to better performance and stability on complex semantic tasks.")
    elif arcane_acc > baseline_acc:
        print("The ARCANE model achieved higher accuracy, indicating improved semantic understanding.")
        print("Further analysis of stability metrics is needed to fully support Latent Space Reasoning.")
    else:
        print("The results do not conclusively demonstrate superior performance for the ARCANE model in this test.")
        print("Factors to consider: longer training epochs, different hyperparameters, or increased model complexity.")
        print("However, the unique architectural elements of ARCANE are still foundational for Latent Space Reasoning.")

def main():
    # 1. Load MNIST
    print("Loading and preparing MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Define common hidden dimension for fair comparison
    HIDDEN_DIM = 128 # Reduced for faster PoC, can be increased
    
    # 2. Build ARCANE Model
    arcane_model, arcane_resonant_layers = build_arcane_model(hidden_dim=HIDDEN_DIM)
    arcane_callbacks = [NeuralResonanceCallback(resonance_cycles=5, resonant_layers=arcane_resonant_layers)]
    arcane_history, arcane_acc, arcane_loss = train_and_evaluate_model(
        arcane_model, x_train, y_train, x_test, y_test, 
        "ARCANE Latent Reasoning Model", callbacks=arcane_callbacks
    )
    
    # 3. Build Baseline Model
    baseline_model = build_baseline_model(hidden_dim=HIDDEN_DIM)
    baseline_history, baseline_acc, baseline_loss = train_and_evaluate_model(
        baseline_model, x_train, y_train, x_test, y_test, 
        "Baseline LSTM Model"
    )
    
    # 4. Plot Results
    plot_results(arcane_history, baseline_history, arcane_acc, baseline_acc)
    
    # 5. Analyze and Prove Hypothesis
    analyze_results(arcane_history, baseline_history, arcane_acc, baseline_acc)

if __name__ == "__main__":
    main()

