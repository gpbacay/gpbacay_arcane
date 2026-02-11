import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Reshape, GlobalAveragePooling1D, Dense, Dropout, LayerNormalization, Add
from tensorflow.keras.models import Model

# Ensure we import the local workspace version of gpbacay_arcane, not the pip-installed one
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from gpbacay_arcane.layers import (
    ResonantGSER,
    PredictiveResonantLayer,
    BioplasticDenseLayer,
    SpatioTemporalSummarization,
)

def build_mnist_arcane_model(
    hidden_dim=128,
    resonance_factor=0.2,
    resonance_cycles=1,
    persistent_predictive=False,
    bioplastic_inference_plasticity=False,
):
    """
    Builds a neuromimetic MNIST classifier using ResonantGSER and Bioplastic layers.
    Treats MNIST images (28x28) as a sequence of 28 rows.
    """
    inputs = Input(shape=(28, 28), name='mnist_input')
    
    # Normalize input
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    
    # Predictive Resonant Levels
    # We use two PredictiveResonantLayer blocks stacked for deeper reasoning.
    pred_layer_1 = PredictiveResonantLayer(
        units=hidden_dim,
        resonance_cycles=resonance_cycles,
        resonance_step_size=0.2,
        spike_threshold=0.4,
        return_sequences=True,
        persist_alignment=persistent_predictive,
        name="predictive_resonant_primary",
    )

    pred_layer_2 = PredictiveResonantLayer(
        units=hidden_dim,
        resonance_cycles=resonance_cycles,
        resonance_step_size=0.25,
        spike_threshold=0.35,
        return_sequences=True,
        persist_alignment=persistent_predictive,
        name="predictive_resonant_secondary",
    )

    # Initial forward pass through predictive resonant hierarchy
    s1 = pred_layer_1(x)
    s1_norm = LayerNormalization(name="pr_layer_norm_1")(s1)

    s2 = pred_layer_2(s1_norm)
    s2_norm = LayerNormalization(name="pr_layer_norm_2")(s2)
    
    # Spatio-Temporal Summarization (Global context mixing)
    summary = SpatioTemporalSummarization(d_model=hidden_dim)(s2_norm)
    
    # Feature Pooling
    pooled = GlobalAveragePooling1D()(summary)
    
    # Bioplastic Dense Layer (Hebbian Plasticity for classification refinement)
    # This helps the model learn categories through neural activity correlation
    bioplastic = BioplasticDenseLayer(
        units=hidden_dim,
        learning_rate=2e-3,
        target_avg=0.1,
        homeostatic_rate=1e-4,
        activation='gelu',
        enable_inference_plasticity=bioplastic_inference_plasticity,
        name='neuromimetic_refining'
    )(pooled)
    
    bioplastic = Dropout(0.2)(bioplastic)
    
    # Output layer
    outputs = Dense(10, activation='softmax', name='classification_output')(bioplastic)
    
    model = Model(inputs=inputs, outputs=outputs, name='ARCANE_MNIST_Classifier')
    return model

def train_and_evaluate():
    # 1. Load MNIST
    print("Loading and preparing MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # 2. Build Model
    # Use a lighter ARCANE configuration by default so MNIST trains in minutes,
    # even on CPU-only machines. Build with the same architecture as test_infer
    # (persistent_predictive + bioplastic_inference_plasticity) so saved weights
    # are compatible and inference can use stateful resonance and inference learning.
    model = build_mnist_arcane_model(
        hidden_dim=128,
        resonance_factor=0.2,
        resonance_cycles=1,
        persistent_predictive=True,
        bioplastic_inference_plasticity=True,
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 3. Train
    print("\nStarting training with ResonantGSER architecture...")
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=256,
        validation_split=0.1,
        verbose=1
    )
    
    # 4. Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # 5. Visualization
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='#00ffcc', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='#ff00ff', linewidth=2)
    plt.title('Training & Validation Accuracy', color='white')
    plt.xlabel('Epoch', color='white')
    plt.ylabel('Accuracy', color='white')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().set_facecolor('#1e1e1e')
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='#00ffcc', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', color='#ff00ff', linewidth=2)
    plt.title('Training & Validation Loss', color='white')
    plt.xlabel('Epoch', color='white')
    plt.ylabel('Loss', color='white')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().set_facecolor('#1e1e1e')
    
    plt.gcf().set_facecolor('#121212')
    plt.tick_params(colors='white')
    plt.tight_layout()
    
    # Save the evaluation plot
    plt.savefig('mnist_arcane_results.png')
    print("\nEvaluation plot saved as 'mnist_arcane_results.png'")
    
    # Save the model
    model_save_path = 'mnist_arcane_model.keras'
    model.save(model_save_path)
    model.save_weights('mnist_arcane_model.weights.h5')
    print(f"Model saved as '{model_save_path}' and weights as 'mnist_arcane_model.weights.h5'")
    
    if test_acc >= 0.99:
        print("\nSUCCESS: Reached the 99% accuracy goal!")
    else:
        print(f"\nGoal: 99%. Achieved: {test_acc*100:.2f}%. Model may need more epochs or higher hidden dimension.")

if __name__ == "__main__":
    train_and_evaluate()

# python examples/mnist_arcane.py
# python examples/test_infer.py