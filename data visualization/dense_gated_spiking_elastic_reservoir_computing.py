import tensorflow as tf

class DenseReservoirLayer(tf.keras.layers.Layer):
    def __init__(self, units, input_dim=None, spectral_radius=None, leak_rate=None, spike_threshold=None, 
                 max_dynamic_units=None, activation='relu', kernel_initializer='glorot_uniform', 
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, name=None, **kwargs):
        super(DenseReservoirLayer, self).__init__(name=name, **kwargs)
        
        self.units = units
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_dynamic_units = max_dynamic_units
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

    def build(self, input_shape):
        # Initialize input weights with kernel_initializer
        self.input_weights = self.add_weight(
            shape=(self.input_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name='input_weights'
        )
        
        # Initialize reservoir weights (non-trainable)
        self.reservoir_weights = self.add_weight(
            shape=(self.input_dim, self.units),
            initializer=self.kernel_initializer,
            trainable=False,
            name='reservoir_weights'
        )
        
        # Initialize gate weights with kernel_initializer
        self.gate_weights = self.add_weight(
            shape=(3 * self.units, self.input_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name='gate_weights'
        )
        
        # Initialize biases for gates
        self.gate_bias = self.add_weight(
            shape=(3 * self.units,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            trainable=True,
            name='gate_bias'
        )
        
        # Internal dense layer for output adjustment with kernel and bias initializers
        self.dense = tf.keras.layers.Dense(
            self.units,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="output_dense"
        )

    def call(self, inputs):
        # Compute input, reservoir, and gate components
        input_part = tf.matmul(inputs, self.input_weights)
        reservoir_part = tf.matmul(inputs, self.reservoir_weights)
        gate_part = tf.matmul(inputs, self.gate_weights, transpose_b=True) + self.gate_bias

        # Split gates
        i_gate, f_gate, o_gate = tf.split(tf.sigmoid(gate_part), 3, axis=-1)

        # Compute state update
        state = (1 - self.leak_rate) * (f_gate * reservoir_part) + self.leak_rate * tf.tanh(i_gate * (input_part + reservoir_part))
        state = o_gate * state

        # Apply spike threshold
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)

        # Apply Dense layer for final transformation
        state = self.dense(state)

        return state

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'input_dim': self.input_dim,
            'spectral_radius': self.spectral_radius,
            'leak_rate': self.leak_rate,
            'spike_threshold': self.spike_threshold,
            'max_dynamic_units': self.max_dynamic_units,
            'activation': tf.keras.activations.serialize(self.activation),
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
        })
        return config
