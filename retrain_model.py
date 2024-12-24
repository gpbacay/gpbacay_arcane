import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os
import matplotlib.pyplot as plt

from gpbacay_arcane.callbacks import DynamicSelfModelingReservoirCallback


def retrain_model():
    # Load the saved model
    model_path = os.path.join('Models', 'dstsmgser_retrained_model.keras')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found at {model_path}")

    model = tf.keras.models.load_model(model_path)

    # Load and preprocess MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Prepare flattened inputs for self-modeling target
    x_train_flat = x_train.reshape((x_train.shape[0], -1))
    x_test_flat = x_test.reshape((x_test.shape[0], -1))

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_clf_out_accuracy', patience=10, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_clf_out_accuracy', factor=0.1, patience=5, mode='max')

    # Retrain the model
    history = model.fit(
        x_train, {'clf_out': y_train, 'sm_out': x_train_flat},
        validation_data=(x_test, {'clf_out': y_test, 'sm_out': x_test_flat}),
        epochs=10,  # Adjust as needed
        batch_size=64,
        callbacks=[early_stopping, reduce_lr]
    )

    # Evaluate the retrained model
    evaluation_results = model.evaluate(x_test, {'clf_out': y_test, 'sm_out': x_test_flat}, verbose=2)
    acc, loss = evaluation_results[3], evaluation_results[0]
    print(f"\nRetrained Model Test Accuracy: {acc:.4f}, Loss: {loss:.4f}")

    # Plot Training History
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['clf_out_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_clf_out_accuracy'], label='Validation Accuracy')
    plt.title('Classification Accuracy (Retraining)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['sm_out_mse'], label='Train MSE')
    plt.plot(history.history['val_sm_out_mse'], label='Validation MSE')
    plt.title('Self-Modeling MSE (Retraining)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save the retrained model
    retrained_model_path = os.path.join('Models', 'dstsmgser_retrained_model.keras')
    model.save(retrained_model_path)
    print(f"\nRetrained model saved to {retrained_model_path}")


if __name__ == "__main__":
    retrain_model()

# python retrain_model.py