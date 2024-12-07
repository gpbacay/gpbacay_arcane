import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RNN, Dense, Input, Dropout, Flatten, Reshape, BatchNormalization, LayerNormalization
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class HebbianHomeostaticLayer(tf.keras.Layer):
    def __init__(self, units, learning_rate=0.00001, target_avg=0.1, homeostatic_rate=0.00001, **kwargs):
        super(HebbianHomeostaticLayer, self).__init__(**kwargs)
        self.units = units
        self.learning_rate = learning_rate
        self.target_avg = target_avg
        self.homeostatic_rate = homeostatic_rate
    
    def build(self, input_shape):
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.001)
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                    initializer=initializer,
                                    trainable=True)
    
    def call(self, inputs):
        inputs = tf.squeeze(inputs, axis=1)
        # Normalize inputs
        inputs = tf.nn.l2_normalize(inputs, axis=-1)
        
        # Forward pass with normalized weights
        normalized_kernel = tf.nn.l2_normalize(self.kernel, axis=0)
        outputs = tf.matmul(inputs, normalized_kernel)
        
        # Scale outputs
        outputs = tf.clip_by_value(outputs, -1.0, 1.0)
        
        # Hebbian update with strong regularization
        if self.learning_rate > 0:
            delta_weights = self.learning_rate * tf.matmul(tf.transpose(inputs), outputs)
            delta_weights = tf.clip_by_value(delta_weights, -0.01, 0.01)
            self.kernel.assign(tf.clip_by_norm(self.kernel + delta_weights, 1.0))
        
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'learning_rate': self.learning_rate,
            'target_avg': self.target_avg,
            'homeostatic_rate': self.homeostatic_rate,
        })
        return config

class SpatioTemporalSummaryMixing(tf.keras.Layer):
    def __init__(self, d_model, dropout_rate=0.1, **kwargs):
        super(SpatioTemporalSummaryMixing, self).__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.local_dense1 = Dense(4 * self.d_model, activation='gelu')
        self.local_dense2 = Dense(self.d_model)
        self.local_dropout = Dropout(self.dropout_rate)
        self.summary_dense1 = Dense(4 * self.d_model, activation='gelu')
        self.summary_dense2 = Dense(self.d_model)
        self.summary_dropout = Dropout(self.dropout_rate)
        self.combiner_dense1 = Dense(4 * self.d_model, activation='gelu')
        self.combiner_dense2 = Dense(self.d_model)
        self.combiner_dropout = Dropout(self.dropout_rate)
        self.layer_norm = LayerNormalization(epsilon=1e-6)
        self.dynamic_dense = Dense(self.d_model)
        super(SpatioTemporalSummaryMixing, self).build(input_shape)

    def call(self, inputs, training=False):
        local_output = self.local_dense1(inputs)
        local_output = self.local_dense2(local_output)
        local_output = self.local_dropout(local_output, training=training)
        
        summary = self.summary_dense1(inputs)
        summary = self.summary_dense2(summary)
        summary = self.summary_dropout(summary, training=training)
        mean_summary = tf.reduce_mean(summary, axis=1, keepdims=True)
        mean_summary = tf.tile(mean_summary, [1, tf.shape(inputs)[1], 1])
        
        combined = tf.concat([local_output, mean_summary], axis=-1)
        output = self.combiner_dense1(combined)
        output = self.combiner_dense2(output)
        output = self.combiner_dropout(output, training=training)
        
        inputs = self.dynamic_dense(inputs)  # Project inputs to match dimensions
        return self.layer_norm(inputs + output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate,
        })
        return config

class GatedSpikingElasticLNNStep(tf.keras.layers.Layer):
    def __init__(self, initial_reservoir_size, input_dim, spectral_radius, leak_rate, spike_threshold, max_dynamic_reservoir_dim, **kwargs):
        super().__init__(**kwargs)
        self.initial_reservoir_size = initial_reservoir_size
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_dynamic_reservoir_dim = max_dynamic_reservoir_dim
        
        self.state_size = [self.max_dynamic_reservoir_dim]
        self.output_size = self.max_dynamic_reservoir_dim
        
        self.initialize_weights()

    def initialize_weights(self):
        self.spatiotemporal_reservoir_weights = self.add_weight(
            shape=(self.initial_reservoir_size, self.initial_reservoir_size),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=False,
            name='spatiotemporal_reservoir_weights'
        )
        self.spatiotemporal_input_weights = self.add_weight(
            shape=(self.initial_reservoir_size, self.input_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=False,
            name='spatiotemporal_input_weights'
        )
        self.spiking_gate_weights = self.add_weight(
            shape=(3 * self.initial_reservoir_size, self.input_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=False,
            name='spiking_gate_weights'
        )

    def call(self, inputs, states):
        prev_state = states[0][:, :tf.shape(self.spatiotemporal_reservoir_weights)[0]]

        input_part = tf.matmul(inputs, self.spatiotemporal_input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.spatiotemporal_reservoir_weights)
        gate_part = tf.matmul(inputs, self.spiking_gate_weights, transpose_b=True)

        i_gate, f_gate, o_gate = tf.split(tf.sigmoid(gate_part), 3, axis=-1)

        state = (1 - self.leak_rate) * (f_gate * prev_state) + self.leak_rate * tf.tanh(i_gate * (input_part + reservoir_part))
        state = o_gate * state

        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)

        padded_state = tf.concat([state, tf.zeros([tf.shape(state)[0], self.max_dynamic_reservoir_dim - tf.shape(state)[-1]])], axis=1)

        return padded_state, [padded_state]

    def add_neurons(self, growth_rate):
        current_size = tf.shape(self.spatiotemporal_reservoir_weights)[0]
        new_neurons = min(growth_rate, self.max_dynamic_reservoir_dim - current_size)
        if new_neurons <= 0:
            return

        new_reservoir_weights = tf.random.normal((new_neurons, current_size + new_neurons))
        full_new_weights = tf.concat([
            tf.concat([self.spatiotemporal_reservoir_weights, tf.zeros((current_size, new_neurons))], axis=1),
            new_reservoir_weights
        ], axis=0)
        
        spectral_radius = tf.math.real(tf.reduce_max(tf.abs(tf.linalg.eigvals(full_new_weights))))
        scaling_factor = self.spectral_radius / spectral_radius
        new_reservoir_weights *= scaling_factor

        self.spatiotemporal_reservoir_weights = tf.Variable(full_new_weights, trainable=False)
        
        new_input_weights = tf.random.normal((new_neurons, self.input_dim)) * 0.1
        self.spatiotemporal_input_weights = tf.concat([self.spatiotemporal_input_weights, new_input_weights], axis=0)
        
        new_gate_weights = tf.random.normal((3 * new_neurons, self.input_dim)) * 0.1
        self.spiking_gate_weights = tf.concat([self.spiking_gate_weights, new_gate_weights], axis=0)

    def prune_connections(self, prune_rate):
        weights = self.spatiotemporal_reservoir_weights.numpy()
        threshold = np.percentile(np.abs(weights), prune_rate * 100)
        mask = np.abs(weights) > threshold
        self.spatiotemporal_reservoir_weights.assign(weights * mask)

    def get_config(self):
        config = super().get_config()
        config.update({
            'initial_reservoir_size': self.initial_reservoir_size,
            'input_dim': self.input_dim,
            'spectral_radius': self.spectral_radius,
            'leak_rate': self.leak_rate,
            'spike_threshold': self.spike_threshold,
            'max_dynamic_reservoir_dim': self.max_dynamic_reservoir_dim
        })
        return config

def create_dstsmgselnn_model(input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, max_dynamic_reservoir_dim, output_dim):
    inputs = Input(shape=input_shape)

    x = BatchNormalization()(inputs)
    x = Flatten()(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)

    # Summary mixing layer
    summary_mixing_layer = SpatioTemporalSummaryMixing(d_model=128)
    x = Reshape((1, x.shape[-1]))(x)
    x = summary_mixing_layer(x)

    # Reservoir layer
    reservoir_layer = GatedSpikingElasticLNNStep(
        initial_reservoir_size=reservoir_dim,
        input_dim=x.shape[-1],
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_dynamic_reservoir_dim=max_dynamic_reservoir_dim
    )
    lnn_layer = RNN(reservoir_layer, return_sequences=True)
    lnn_output = lnn_layer(x)

    # Hebbian homeostatic layer
    hebbian_homeostatic_layer = HebbianHomeostaticLayer(units=reservoir_dim, name='hebbian_homeostatic_layer')
    hebbian_output = hebbian_homeostatic_layer(lnn_output)

    x = Flatten()(hebbian_output)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)

    # Self-modeling Mechanism: Auxiliary task (predict flattened input)
    self_modeling_output = Dense(np.prod(input_shape), activation='sigmoid', name='self_modeling_output')(x)
    
    # Classification output
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    classification_output = Dense(output_dim, activation='softmax', name='classification_output', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

    model = tf.keras.Model(inputs=inputs, outputs=[classification_output, self_modeling_output])

    return model, reservoir_layer

class SelfModelingCallback(Callback):
    def __init__(self, reservoir_layer, performance_metric='accuracy', target_metric=0.95, 
                add_neurons_threshold=0.01, prune_connections_threshold=0.1, growth_phase_length=10, pruning_phase_length=5):
        super().__init__()
        self.reservoir_layer = reservoir_layer
        self.performance_metric = performance_metric
        self.target_metric = target_metric
        self.initial_add_neurons_threshold = add_neurons_threshold
        self.initial_prune_connections_threshold = prune_connections_threshold
        self.growth_phase_length = growth_phase_length
        self.pruning_phase_length = pruning_phase_length
        self.current_phase = 'growth'
        self.phase_counter = 0
        self.performance_history = []

    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.performance_metric, 0)
        
        self.performance_history.append(current_metric)
        
        self.add_neurons_threshold = self.initial_add_neurons_threshold * (1 - current_metric)
        self.prune_connections_threshold = self.initial_prune_connections_threshold * current_metric

        self.phase_counter += 1
        if self.current_phase == 'growth' and self.phase_counter >= self.growth_phase_length:
            self.current_phase = 'pruning'
            self.phase_counter = 0
        elif self.current_phase == 'pruning' and self.phase_counter >= self.pruning_phase_length:
            self.current_phase = 'growth'
            self.phase_counter = 0

        if len(self.performance_history) > 5:
            improvement_rate = (current_metric - self.performance_history[-5]) / 5
            
            if improvement_rate > 0.01:
                self.reservoir_layer.add_neurons(growth_rate=10)
            elif improvement_rate < 0.001:
                self.reservoir_layer.prune_connections(prune_rate=0.1)

        if current_metric >= self.target_metric:
            print(f" - Performance metric {self.performance_metric} reached target {self.target_metric}. Current phase: {self.current_phase}")
            if self.current_phase == 'growth':
                self.reservoir_layer.add_neurons(growth_rate=10)
            elif self.current_phase == 'pruning':
                self.reservoir_layer.prune_connections(prune_rate=0.1)

def main():
    # Load and preprocess MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Flatten images for self-modeling task
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

    # Create model
    model, reservoir_layer = create_dstsmgselnn_model(
        input_shape=input_shape,
        reservoir_dim=reservoir_dim,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_dynamic_reservoir_dim=max_dynamic_reservoir_dim,
        output_dim=output_dim
    )

    # Compile the model
    model.compile(
        optimizer='adam', 
        loss={'classification_output': 'categorical_crossentropy', 'self_modeling_output': 'mse'},
        metrics={'classification_output': 'accuracy', 'self_modeling_output': 'mse'}
    )

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_classification_output_accuracy', patience=10, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_classification_output_accuracy', factor=0.1, patience=5, mode='max')
    self_modeling_callback = SelfModelingCallback(
        reservoir_layer=reservoir_layer,
        performance_metric='val_classification_output_accuracy',
        target_metric=0.95,
        add_neurons_threshold=0.005,
        prune_connections_threshold=0.1,
        growth_phase_length=10,
        pruning_phase_length=5
    )

    # Train the model
    history = model.fit(
        x_train, 
        {'classification_output': y_train, 'self_modeling_output': x_train_flat}, 
        validation_data=(x_test, {'classification_output': y_test, 'self_modeling_output': x_test_flat}),
        epochs=10,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr, self_modeling_callback]
    )
    
    # Evaluate the model
    evaluation_results = model.evaluate(x_test, {'classification_output': y_test, 'self_modeling_output': x_test_flat}, verbose=2)
    classification_acc = evaluation_results[1]
    print(f"Test accuracy: {classification_acc:.4f}")

    # Plot Training History
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['classification_output_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_classification_output_accuracy'], label='Validation Accuracy')
    plt.title('Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['self_modeling_output_mse'], label='Train MSE')
    plt.plot(history.history['val_self_modeling_output_mse'], label='Validation MSE')
    plt.title('Self-Modeling MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

# Dynamic Spatio-Temporal Self-Modeling Gated Spiking Elastic Liquid Neural Network (DST-SM-GSELNN)
# python dstsmgselnn_mnist.py
# Test Accuracy: 0.9788
