import tensorflow as tf
import numpy as np
from .mechanisms import (
    GSER, 
    ResonantGSERCell, 
    MultiheadLinearSelfAttentionKernalization, 
    SpatioTemporalSummaryMixingLayer
)

class ExpandDimensionLayer(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super(ExpandDimensionLayer, self).__init__(**kwargs)
        self.axis = axis
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)
    def get_config(self):
        config = super().get_config()
        config.update({'axis': self.axis})
        return config

class DenseGSER(tf.keras.layers.Layer):
    def __init__(self, units, input_dim=None, spectral_radius=0.9, leak_rate=0.1, spike_threshold=0.5, 
                 max_dynamic_units=None, activation='gelu', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform', name='kernel')
        self.bias = self.add_weight(shape=(self.units,), initializer='zeros', name='bias')
    def call(self, inputs):
        return tf.nn.gelu(tf.matmul(inputs, self.kernel) + self.bias)

class ResonantGSER(tf.keras.layers.RNN):
    def __init__(self, units, resonance_factor=0.1, spike_threshold=0.5, 
                 return_sequences=False, return_state=False, **kwargs):
        self._cell = ResonantGSERCell(units, resonance_factor=resonance_factor, spike_threshold=spike_threshold)
        super(ResonantGSER, self).__init__(self._cell, return_sequences=return_sequences, return_state=return_state, **kwargs)
        self.units = units
        self.resonance_factor = resonance_factor
    def project_feedback(self, representation=None):
        return tf.matmul(representation if representation is not None else self._cell.last_h, self._cell.feedback_weights)
    def harmonize_states(self, projection):
        self._cell.resonance_alignment.assign(projection)

class RelationalConceptModeling(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
    def call(self, inputs):
        return self.mha(inputs, inputs)

class RelationalGraphAttentionReasoning(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.dense = tf.keras.layers.Dense(num_classes)
    def call(self, inputs):
        x = self.mha(inputs, inputs)
        return self.dense(tf.reduce_mean(x, axis=1))

class BioplasticDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, learning_rate=1e-3, anti_hebbian_rate=0.1, target_avg=0.12, 
                 homeostatic_rate=5e-5, bcm_tau=800.0, activation='gelu', normalization='l2', 
                 dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.learning_rate = learning_rate
        self.anti_hebbian_rate = anti_hebbian_rate
        self.target_avg = target_avg
        self.homeostatic_rate = homeostatic_rate
        self.bcm_tau = bcm_tau
        self.activation = tf.keras.activations.get(activation)
        self.normalization_type = normalization
        self.dropout_rate = dropout_rate
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        self.trace = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=False,
            name='plasticity_trace'
        )

    def call(self, inputs, training=False):
        x = tf.matmul(inputs, self.kernel) + self.bias
        x = self.activation(x)
        if training:
            x = self.dropout(x, training=training)
        return x

class HebbianHomeostaticNeuroplasticity(tf.keras.layers.Layer):
    def __init__(self, units, learning_rate=1e-3, target_activity=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.learning_rate = learning_rate
        self.target_activity = target_activity

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel) + self.bias

class SpatioTemporalSummarization(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.mixing = SpatioTemporalSummaryMixingLayer(d_model)
    def call(self, inputs):
        return self.mixing(inputs)

class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, max_position, d_model, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return inputs

class LatentTemporalCoherence(tf.keras.layers.Layer):
    def __init__(self, d_coherence, **kwargs):
        super().__init__(**kwargs)
        self.d_coherence = d_coherence
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.d_coherence), initializer='glorot_uniform', name='kernel')
    def call(self, inputs):
        return tf.matmul(tf.reduce_mean(inputs, axis=1), self.kernel)