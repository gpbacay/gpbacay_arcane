import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

from gpbacay_arcane.models import HighAccuracyDigitModel, NeuromimeticLanguageModel
from gpbacay_arcane.callbacks import DynamicSelfModelingReservoirCallback

def train_model():
    # Load and preprocess MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Prepare flattened inputs for self-modeling target
    x_train_flat = x_train.reshape((x_train.shape[0], -1))
    x_test_flat = x_test.reshape((x_test.shape[0], -1))

    # Hyperparameters (tuned for >=99% MNIST accuracy)
    input_shape = (28, 28, 1)
    output_dim = 10

    # Custom reservoir/GSER parameters
    reservoir_dim = 1024
    spectral_radius = 1.5
    leak_rate = 0.2
    spike_threshold = 0.5
    max_dynamic_reservoir_dim = 2048

    # Custom embedding/attention/coherence parameters
    d_model = 128
    num_heads = 8
    num_thought_steps = 8
    d_coherence = 256

    # Initialize the model
    model_builder = HighAccuracyDigitModel(
        input_shape=input_shape,
        output_dim=output_dim,
        reservoir_dim=reservoir_dim,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_dynamic_reservoir_dim=max_dynamic_reservoir_dim,
        d_model=d_model,
        num_heads=num_heads,
        num_thought_steps=num_thought_steps,
        d_coherence=d_coherence,
        use_weighted_summary=True,
    )
    model_builder.build_model()
    model_builder.compile_model()

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_clf_out_accuracy', patience=8, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_clf_out_accuracy', factor=0.5, patience=3, mode='max', min_lr=1e-5)
    dynamic_reservoir_callback = DynamicSelfModelingReservoirCallback(
        reservoir_layer=model_builder.reservoir_layer,
        performance_metric='val_clf_out_accuracy',
        target_metric=0.99
    )

    # Train the model
    history = model_builder.model.fit(
        x_train, {'clf_out': y_train, 'sm_out': x_train_flat},
        validation_data=(x_test, {'clf_out': y_test, 'sm_out': x_test_flat}),
        epochs=20,
        batch_size=128,
        callbacks=[early_stopping, reduce_lr, dynamic_reservoir_callback]
    )

    # Evaluate the model
    eval_dict = model_builder.model.evaluate(
        x_test, {'clf_out': y_test, 'sm_out': x_test_flat}, verbose=2, return_dict=True
    )
    print(f"\nTest Accuracy: {eval_dict.get('clf_out_accuracy', 0):.4f}, Loss: {eval_dict.get('loss', 0):.4f}")

    # Plot Training History
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['clf_out_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_clf_out_accuracy'], label='Validation Accuracy')
    plt.title('Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['sm_out_mse'], label='Train MSE')
    plt.plot(history.history['val_sm_out_mse'], label='Validation MSE')
    plt.title('Self-Modeling MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Create Models directory if it doesn't exist
    os.makedirs('Models', exist_ok=True)

    # Save the model
    model_path = os.path.join('Models', 'high_accuracy_digit_model.keras')
    model_builder.model.save(model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    train_model()





# Dynamic Spatio-Temporal Self-Modeling Gated Spiking Elastic Reservoir (DST-SM-GSER)
# with Dynamic Self Modeling Reservoir Callback
# python train_model.py

