import tensorflow as tf

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