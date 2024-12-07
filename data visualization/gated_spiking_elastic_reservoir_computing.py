import tensorflow as tf
import numpy as np


class GatedSpikingElasticReservoirLayer(tf.keras.layers.Layer):
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
        # Get the current weights of the reservoir
        weights = self.reservoir_weights.numpy()  # Get the weights as numpy array for manipulation
        
        # Calculate a threshold based on the prune_rate (e.g., top prune_rate % of smallest weights)
        threshold = np.percentile(np.abs(weights), prune_rate * 100)  # Threshold value
        
        # Create a mask to zero out small weights
        mask = np.abs(weights) > threshold  # Only keep weights above the threshold
        
        # Apply the mask to the reservoir weights
        self.reservoir_weights.assign(weights * mask)

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