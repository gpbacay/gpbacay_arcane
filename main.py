import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

from gpbacay_arcane.models import DSTSMGSER

def main():
    # Load and preprocess MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Prepare flattened inputs for self-modeling target
    x_train_flat = x_train.reshape((x_train.shape[0], -1))
    x_test_flat = x_test.reshape((x_test.shape[0], -1))

    # Hyperparameters
    input_shape = (28, 28, 1)
    reservoir_dim = 512
    spectral_radius = 2.0
    leak_rate = 0.2
    spike_threshold = 0.5
    max_dynamic_reservoir_dim = 4096
    output_dim = 10

    # Initialize the model
    dstsmgser = DSTSMGSER(
        input_shape=input_shape,
        reservoir_dim=reservoir_dim,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_dynamic_reservoir_dim=max_dynamic_reservoir_dim,
        output_dim=output_dim,
        use_weighted_summary=True
    )
    dstsmgser.build_model()
    dstsmgser.compile_model()

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_clf_out_accuracy', patience=10, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_clf_out_accuracy', factor=0.1, patience=5, mode='max')

    # Train the model
    history = dstsmgser.model.fit(
        x_train, {'clf_out': y_train, 'sm_out': x_train_flat},
        validation_data=(x_test, {'clf_out': y_test, 'sm_out': x_test_flat}),
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr]
    )

    # Evaluate the model
    evaluation_results = dstsmgser.model.evaluate(x_test, {'clf_out': y_test, 'sm_out': x_test_flat}, verbose=2)
    acc, loss = evaluation_results[3], evaluation_results[0]
    print(f"\nTest Accuracy: {acc:.4f}, Loss: {loss:.4f}")

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
    model_path = os.path.join('Models', 'dstsmgser_model.keras')
    dstsmgser.model.save(model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()





# Dynamic Spatio-Temporal Self-Modeling Gated Spiking Elastic Reservoir (DST-SM-GSER)
# python main.py

