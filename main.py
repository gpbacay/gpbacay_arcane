import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RNN, Input, Dropout, Flatten, BatchNormalization, LayerNormalization, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

from gpbacay_arcane import MultiheadLinearSelfAttentionKernalizationLayer
from gpbacay_arcane import HebbianHomeostaticLayer
from gpbacay_arcane import GatedSpikingElasticReservoirLayer
from gpbacay_arcane import DenseReservoirLayer
from gpbacay_arcane import ExpandDimensionLayer
from gpbacay_arcane import DynamicSelfModelingReservoirCallback


class DSTSMGSER:
    def __init__(self, input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, max_dynamic_reservoir_dim, output_dim, use_weighted_summary=False):
        self.input_shape = input_shape
        self.reservoir_dim = reservoir_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_dynamic_reservoir_dim = max_dynamic_reservoir_dim
        self.output_dim = output_dim
        self.use_weighted_summary = use_weighted_summary
        self.model = None
        self.reservoir_layer = None

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        # Preprocessing
        x = BatchNormalization()(inputs)
        x = Flatten()(x)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)

        # Attention Layer
        linear_attention_layer = MultiheadLinearSelfAttentionKernalizationLayer(
            d_model=128, num_heads=8, use_weighted_summary=self.use_weighted_summary)
        x = ExpandDimensionLayer()(x)
        x = linear_attention_layer(x)

        # Reservoir layer
        self.reservoir_layer = GatedSpikingElasticReservoirLayer(
            initial_reservoir_size=self.reservoir_dim,
            input_dim=x.shape[-1],
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold,
            max_dynamic_reservoir_dim=self.max_dynamic_reservoir_dim
        )
        lnn_layer = RNN(self.reservoir_layer, return_sequences=True)
        lnn_output = lnn_layer(x)

        # Hebbian homeostatic layer
        hebbian_homeostatic_layer = HebbianHomeostaticLayer(units=self.reservoir_dim, name='hebbian_homeostatic_layer')
        x = hebbian_homeostatic_layer(lnn_output)

        # Classification output
        clf_out = DenseReservoirLayer(
            units=self.output_dim,
            input_dim=x.shape[-1],
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold,
            max_dynamic_units=self.max_dynamic_reservoir_dim,
            activation='softmax',
            name='clf_out'
        )(Flatten()(x))

        # Self-modeling output
        sm_out = DenseReservoirLayer(
            units=np.prod(self.input_shape),
            input_dim=x.shape[-1],
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold,
            max_dynamic_units=self.max_dynamic_reservoir_dim,
            activation='sigmoid',
            name='sm_out'
        )(Flatten()(x))

        # Compile the model
        self.model = tf.keras.Model(inputs=inputs, outputs=[clf_out, sm_out])

    def compile_model(self):
        self.model.compile(
            optimizer='adam',
            loss={
                'clf_out': 'categorical_crossentropy',
                'sm_out': 'mse'
            },
            loss_weights={
                'clf_out': 1.0,
                'sm_out': 0.5
            },
            metrics={
                'clf_out': 'accuracy',
                'sm_out': 'mse'
            }
        )


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
    reservoir_dim = 1024
    spectral_radius = 2.0
    leak_rate = 0.2
    spike_threshold = 0.5
    max_dynamic_reservoir_dim = 8192
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
    dynamic_reservoir_callback = DynamicSelfModelingReservoirCallback(
        reservoir_layer=dstsmgser.reservoir_layer,
        performance_metric='val_clf_out_accuracy',
        target_metric=0.70
    )

    # Train the model
    history = dstsmgser.model.fit(
        x_train, {'clf_out': y_train, 'sm_out': x_train_flat},
        validation_data=(x_test, {'clf_out': y_test, 'sm_out': x_test_flat}),
        epochs=10,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr, dynamic_reservoir_callback]
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
# with Multihead Linear Self Attention Aggregation Mechanism and Dynamic Self Modeling Reservoir Callback
# python main.py
# Test Accuracy: 0.9722
