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
    """
    A neuromimetic dense layer with Gated Spiking Elastic Reservoir (GSER) properties,
    designed for Direct Semantic Optimization and Abstraction of Surface-Level Conceptual Variability.
    It incorporates a conceptual gating mechanism to dynamically filter and emphasize
    semantically relevant features in the latent space.
    """
    def __init__(self, units, input_dim=None, spectral_radius=0.9, leak_rate=0.1, spike_threshold=0.5, 
                 max_dynamic_units=None, activation='gelu', use_conceptual_gate=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.spectral_radius = spectral_radius # Retained for potential future GSER-specific logic
        self.leak_rate = leak_rate             # Retained for potential future GSER-specific logic
        self.spike_threshold = spike_threshold # Retained for potential future GSER-specific logic
        self.activation = tf.keras.activations.get(activation)
        self.use_conceptual_gate = use_conceptual_gate

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            name='kernel'
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='bias'
        )
        
        if self.use_conceptual_gate:
            self.conceptual_gate_kernel = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='glorot_uniform',
                name='conceptual_gate_kernel'
            )
            self.conceptual_gate_bias = self.add_weight(
                shape=(self.units,),
                initializer='zeros',
                name='conceptual_gate_bias'
            )
        self.built = True

    def call(self, inputs):
        # Standard dense transformation
        x = tf.matmul(inputs, self.kernel) + self.bias
        x = self.activation(x)

        if self.use_conceptual_gate:
            # Compute conceptual gate
            gate_activations = tf.matmul(inputs, self.conceptual_gate_kernel) + self.conceptual_gate_bias
            conceptual_gate = tf.sigmoid(gate_activations) # Sigmoid to produce gating values between 0 and 1
            x = x * conceptual_gate # Apply conceptual gate
            
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'spectral_radius': self.spectral_radius,
            'leak_rate': self.leak_rate,
            'spike_threshold': self.spike_threshold,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_conceptual_gate': self.use_conceptual_gate,
        })
        return config

class ResonantGSER(tf.keras.layers.RNN):
    """
    A wrapper layer for ResonantGSERCell, implementing hierarchical resonance for
    Latent Space Reasoning, Unified Multi-Modal Semantic Space integration, and
    Direct Semantic Optimization. It facilitates iterative state alignment and
    feedback propagation within a multi-layered semantic hierarchy.
    """
    def __init__(self, units, resonance_factor=0.1, spike_threshold=0.5, 
                 resonance_cycles=3, convergence_epsilon=1e-4,
                 return_sequences=False, return_state=False, **kwargs):
        
        if hasattr(units, 'state_size'):
            cell = units
            self.units = getattr(cell, 'units', None)
        else:
            self.units = units
            cell = ResonantGSERCell(
                units, 
                resonance_factor=resonance_factor, 
                spike_threshold=spike_threshold,
                resonance_cycles=resonance_cycles,
                convergence_epsilon=convergence_epsilon
            )
        
        super(ResonantGSER, self).__init__(
            cell, 
            return_sequences=return_sequences, 
            return_state=return_state, 
            **kwargs
        )
        self.resonance_factor = resonance_factor
        self.resonance_cycles = resonance_cycles
        # Use names instead of direct object references to avoid recursion errors
        self.higher_layer_name = None
        self.lower_layer_name = None

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "resonance_factor": self.resonance_factor,
            "resonance_cycles": self.resonance_cycles
        })
        return config
    
    def set_higher_layer(self, layer):
        self.higher_layer_name = layer.name if layer else None
    
    def set_lower_layer(self, layer):
        self.lower_layer_name = layer.name if layer else None
    
    def project_feedback(self, representation=None):
        """
        Top-Down Projection: P_{i→i-1} = f_proj(S_i; W_i)
        
        Projects the current layer's representation down to the lower layer.
        If no representation is provided, uses the cell's last hidden state.
        """
        if representation is None:
            # Use the cell's tracked state
            representation = tf.expand_dims(self._cell.last_h, 0)
        
        # Use the cell's projection function
        projection = self._cell.project_feedback(representation)
        return projection
    
    def harmonize_states(self, projection):
        """
        Bottom-Up Harmonization: Receive top-down projection and set alignment target.
        
        This sets the resonance_alignment which will be used in the next forward pass
        to guide the iterative harmonization loop.
        """
        # Squeeze to match the alignment shape if needed
        if len(projection.shape) > 1:
            projection = tf.squeeze(projection, axis=0)
        
        self._cell.resonance_alignment.assign(projection)
    
    def get_divergence(self):
        """Get the current global divergence metric from the cell."""
        return self._cell.global_divergence.numpy()
    
    def propagate_feedback_to_lower(self):
        """
        Propagate this layer's state as a top-down projection to the lower layer.
        This implements the "Project (Top-Down)" step from Algorithm 1.
        """
        if self._lower_layer is not None:
            projection = self.project_feedback()
            self._lower_layer.harmonize_states(projection)
    
    def receive_feedback_from_higher(self):
        """
        Receive and apply top-down projection from the higher layer.
        This implements the "Harmonize (Bottom-Up)" step from Algorithm 1.
        """
        if self._higher_layer is not None:
            projection = self._higher_layer.project_feedback()
            self.harmonize_states(projection)

class RelationalConceptModeling(tf.keras.layers.Layer):
    """
    A layer designed to model and abstract relational concepts within a Unified Multi-Modal Semantic Space.
    It uses multi-head attention to identify and extract salient conceptual relationships from input features,
    contributing to Latent Space Reasoning by focusing on interconnected semantic entities.
    """
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
    def call(self, inputs):
        return self.mha(inputs, inputs)

class RelationalGraphAttentionReasoning(tf.keras.layers.Layer):
    """
    A layer for performing Latent Space Reasoning by applying graph-like attention over relational semantic embeddings.
    It extracts and processes intricate relationships between conceptual entities, contributing to a deeper
    semantic understanding.
    """
    def __init__(self, d_model, num_heads, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.dense = tf.keras.layers.Dense(num_classes)
    def call(self, inputs):
        x = self.mha(inputs, inputs)
        return self.dense(tf.reduce_mean(x, axis=1))

class BioplasticDenseLayer(tf.keras.layers.Layer):
    """
    A bioplastic dense layer incorporating Hebbian learning and homeostatic plasticity for
    Direct Semantic Optimization and Abstraction of Surface-Level Conceptual Variability.
    This layer adapts its synaptic weights based on neural activity, forming robust and
    adaptive semantic representations in the latent space.
    """
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

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "learning_rate": self.learning_rate,
            "anti_hebbian_rate": self.anti_hebbian_rate,
            "target_avg": self.target_avg,
            "homeostatic_rate": self.homeostatic_rate,
            "bcm_tau": self.bcm_tau,
            "activation": tf.keras.activations.serialize(self.activation),
            "normalization": self.normalization_type,
            "dropout_rate": self.dropout_rate
        })
        return config

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
    """
    A layer implementing Hebbian learning with homeostatic plasticity for robust and adaptive
    semantic feature learning. It promotes the formation of stable and meaningful connections
    in the latent space by regulating neural activity and synaptic strength.
    """
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
    """
    A layer for unifying multi-modal spatio-temporal features into coherent Semantic Summaries.
    It supports a Unified Multi-Modal Semantic Space by abstracting away surface-level conceptual variability,
    producing compact representations suitable for Latent Space Reasoning.
    """
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.mixing = SpatioTemporalSummaryMixingLayer(d_model)
    def call(self, inputs):
        return self.mixing(inputs)
    def get_config(self):
        config = super().get_config()
        # Since self.mixing is created in __init__ with d_model, 
        # we should probably pass it back if we want to be perfect, 
        # but let's just make sure it serializes.
        config.update({"d_model": self.mixing.d_model})
        return config

class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, max_position, d_model, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return inputs

class LatentTemporalCoherence(tf.keras.layers.Layer):
    """
    A layer designed to distill a compact 'semantic coherence vector' from temporal inputs,
    facilitating Latent Space Reasoning and Abstraction of Surface-Level Conceptual Variability
    in sequential data. It captures the essential semantic flow over time.
    """
    def __init__(self, d_coherence, **kwargs):
        super().__init__(**kwargs)
        self.d_coherence = d_coherence
    def build(self, input_shape):
        # Kernel to project pooled temporal features into a semantic coherence vector
        self.coherence_kernel = self.add_weight(shape=(input_shape[-1], self.d_coherence), initializer='glorot_uniform', name='coherence_kernel')
    def call(self, inputs):
        # Average pool across the temporal dimension to get a global temporal context
        pooled_temporal_features = tf.reduce_mean(inputs, axis=1)
        # Project into the semantic coherence space
        semantic_coherence_vector = tf.matmul(pooled_temporal_features, self.coherence_kernel)
        return semantic_coherence_vector