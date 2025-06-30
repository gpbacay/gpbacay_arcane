import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

from gpbacay_arcane.models import CoherentThoughtModel
from gpbacay_arcane.callbacks import DynamicSelfModelingReservoirCallback

def main():
    # Load and preprocess MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Hyperparameters
    input_shape = (28, 28, 1)
    reservoir_dim = 512
    spectral_radius = 2.0
    leak_rate = 0.2
    spike_threshold = 0.5
    max_dynamic_reservoir_dim = 1024 # Reduced for this model to manage complexity
    output_dim = 10
    num_thought_steps = 15 # The number of internal "thought" steps
    d_coherence = 256 # The dimensionality of the coherence vector

    # Initialize the model
    model_instance = CoherentThoughtModel(
        input_shape=input_shape,
        reservoir_dim=reservoir_dim,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_dynamic_reservoir_dim=max_dynamic_reservoir_dim,
        output_dim=output_dim,
        num_thought_steps=num_thought_steps,
        d_coherence=d_coherence
    )
    model_instance.build_model()
    model_instance.compile_model()

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, mode='max')
    dynamic_reservoir_callback = DynamicSelfModelingReservoirCallback(
        reservoir_layer=model_instance.reservoir_layer,
        performance_metric='val_accuracy',
        target_metric=0.98,
        stagnation_epochs=7 # Prune neurons if no improvement for 7 epochs
    )

    # Train the model
    history = model_instance.model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=10, # Increased epochs for this more complex model
        batch_size=64,
        callbacks=[early_stopping, reduce_lr, dynamic_reservoir_callback]
    )

    # Evaluate the model
    loss, acc = model_instance.model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest Accuracy: {acc:.4f}, Loss: {loss:.4f}")

    # Plot Training History
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Create Models directory if it doesn't exist
    os.makedirs('Models', exist_ok=True)

    # Save the model
    model_path = os.path.join('Models', 'coherent_thought_model.keras')
    model_instance.model.save(model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()

# Dynamic Spatio-Temporal Self-Modeling Gated Spiking Elastic Reservoir (DST-SM-GSER)
# python main.py

