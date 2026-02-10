import tensorflow as tf
import numpy as np
from .mechanisms import (
    GSER,
    ResonantGSERCell,
    PredictiveResonantCell,
    MultiheadLinearSelfAttentionKernalization,
    SpatioTemporalSummaryMixingLayer,
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
        self._higher_layer_ref = None
        self._lower_layer_ref = None

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "resonance_factor": self.resonance_factor,
            "resonance_cycles": self.resonance_cycles
        })
        return config
    
    
    def project_feedback(self, representation=None):
        """
        Top-Down Projection: P_{i→i-1} = f_proj(S_i; W_i)
        
        Projects the current layer's representation down to the lower layer.
        If no representation is provided, uses the cell's last hidden state.
        """
        if representation is None:
            # Use the cell's tracked state
            representation = tf.expand_dims(self.cell.last_h, 0)
        
        # Use the cell's projection function
        projection = self.cell.project_feedback(representation)
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
        
        self.cell.resonance_alignment.assign(projection)
    
    def get_divergence(self):
        """Get the current global divergence metric from the cell."""
        return self.cell.global_divergence.numpy()
    
    def propagate_feedback_to_lower(self):
        """
        Convenience method to propagate feedback to the lower layer.
        Projects this layer's state down to the lower layer and sets its resonance_alignment.
        
        Note: This method requires that set_lower_layer() has been called to establish
        the hierarchical connection, and that the layer has access to the model via _model attribute.
        """
        if self.lower_layer_name is None:
            return
        
        # Find the lower layer by name from the model
        lower_layer = None
        if hasattr(self, '_model') and self._model is not None:
            for layer in self._model.layers:
                if layer.name == self.lower_layer_name:
                    lower_layer = layer
                    break
        
        if lower_layer is None:
            # If model reference not available, try direct reference if stored
            if hasattr(self, '_lower_layer_ref'):
                lower_layer = self._lower_layer_ref
            else:
                return
        
        # Project feedback from this layer to the lower layer
        projection = self.project_feedback()
        lower_layer.harmonize_states(projection)
    
    def set_lower_layer(self, layer):
        """Set the lower layer reference for hierarchical feedback."""
        self.lower_layer_name = layer.name if layer else None
        self._lower_layer_ref = layer  # Store direct reference as backup
    
    def set_higher_layer(self, layer):
        """Set the higher layer reference for hierarchical feedback."""
        self.higher_layer_name = layer.name if layer else None
        self._higher_layer_ref = layer  # Store direct reference as backup


class PredictiveResonantLayer(tf.keras.layers.RNN):
    """
    PredictiveResonantLayer
    ------------------------

    A new neuromimetic recurrent layer that implements *local* predictive resonance.

    Key ideas:
    - Per-example alignment: alignment is part of the recurrent state (h, c, align),
      not a single global vector shared across the batch.
    - Self-contained: does not depend on external callbacks or model references.
    - Predictive coding flavour: the layer maintains a slow-moving alignment state
      that acts as an internal prediction of future activity, and the fast state
      resonates toward that prediction on each step.
    """

    def __init__(
        self,
        units,
        resonance_cycles=3,
        resonance_step_size=0.2,
        spike_threshold=0.5,
        return_sequences=False,
        return_state=False,
        **kwargs,
    ):
        cell = PredictiveResonantCell(
            units=units,
            resonance_cycles=resonance_cycles,
            resonance_step_size=resonance_step_size,
            spike_threshold=spike_threshold,
        )

        super(PredictiveResonantLayer, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            **kwargs,
        )
        self.units = units
        self.resonance_cycles = resonance_cycles
        self.resonance_step_size = resonance_step_size
        self.spike_threshold = spike_threshold

    def get_config(self):
        config = super(PredictiveResonantLayer, self).get_config()
        config.update(
            {
                "units": self.units,
                "resonance_cycles": self.resonance_cycles,
                "resonance_step_size": self.resonance_step_size,
                "spike_threshold": self.spike_threshold,
            }
        )
        return config

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

class RelationalConceptGraphReasoning(tf.keras.layers.Layer):
    """
    A unified mechanism combining relational concept modeling and graph attention reasoning.
    This novel mechanism integrates multi-head attention with configurable semantic processing
    to enable both concept extraction and relational reasoning within a unified framework.

    Features:
    - Multi-head attention for relational semantic processing
    - Configurable output modes: concept features or classification predictions
    - Enhanced semantic processing with residual connections and layer normalization
    - Support for hierarchical reasoning with multiple attention layers
    - Adaptive pooling strategies for different semantic tasks

    This mechanism advances Latent Space Reasoning by providing a flexible architecture
    that can model concepts, reason about relationships, and perform semantic classification
    within the Unified Multi-Modal Semantic Space.
    """
    def __init__(self, d_model, num_heads, output_mode='features', num_classes=None,
                 num_reasoning_layers=1, use_residual=True, use_layer_norm=True,
                 pooling_strategy='mean', semantic_dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.output_mode = output_mode  # 'features', 'classification', or 'both'
        self.num_classes = num_classes
        self.num_reasoning_layers = num_reasoning_layers
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.pooling_strategy = pooling_strategy  # 'mean', 'max', 'attention', 'none'
        self.semantic_dropout = semantic_dropout

        # Core attention mechanism
        self.attention_layers = []
        for i in range(num_reasoning_layers):
            self.attention_layers.append(
                tf.keras.layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=d_model,
                    name=f'attention_layer_{i}'
                )
            )

        # Layer normalization for stable training
        if self.use_layer_norm:
            self.layer_norms = [
                tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f'layer_norm_{i}')
                for i in range(num_reasoning_layers)
            ]

        # Semantic enhancement layers
        self.semantic_enhancer = tf.keras.layers.Dense(
            d_model,
            activation='gelu',
            name='semantic_enhancer'
        )

        # Dropout for regularization
        self.dropout = tf.keras.layers.Dropout(semantic_dropout)

        # Output processing based on mode
        if output_mode in ['classification', 'both']:
            if num_classes is None:
                raise ValueError("num_classes must be specified for classification mode")
            self.classifier = tf.keras.layers.Dense(num_classes, name='classifier')

        # Adaptive pooling if needed
        if pooling_strategy == 'attention':
            self.attention_pool = tf.keras.layers.Dense(1, activation='tanh', name='attention_pool')

    def call(self, inputs, training=False):
        """
        Forward pass with hierarchical relational reasoning.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, d_model)
            training: Whether in training mode

        Returns:
            Depending on output_mode:
            - 'features': Enhanced attention features (batch_size, seq_len, d_model)
            - 'classification': Classification logits (batch_size, num_classes)
            - 'both': Tuple of (features, logits)
        """
        x = inputs

        # Hierarchical attention processing
        for i, attention_layer in enumerate(self.attention_layers):
            # Self-attention with residual connection
            attn_output = attention_layer(x, x)

            if self.use_residual:
                x = x + attn_output  # Residual connection
            else:
                x = attn_output

            # Layer normalization for stability
            if self.use_layer_norm:
                x = self.layer_norms[i](x)

        # Semantic enhancement
        x = self.semantic_enhancer(x)
        x = self.dropout(x, training=training)

        # Handle different output modes
        if self.output_mode == 'features':
            return x

        elif self.output_mode == 'classification':
            # Pool the sequence for classification
            pooled = self._pool_sequence(x)
            return self.classifier(pooled)

        elif self.output_mode == 'both':
            # Return both features and classification
            pooled = self._pool_sequence(x)
            return x, self.classifier(pooled)

    def _pool_sequence(self, x):
        """Adaptive pooling strategies for sequence aggregation."""
        if self.pooling_strategy == 'mean':
            return tf.reduce_mean(x, axis=1)
        elif self.pooling_strategy == 'max':
            return tf.reduce_max(x, axis=1)
        elif self.pooling_strategy == 'attention':
            # Learnable attention-based pooling
            attn_weights = self.attention_pool(x)  # (batch, seq_len, 1)
            attn_weights = tf.nn.softmax(attn_weights, axis=1)
            return tf.reduce_sum(x * attn_weights, axis=1)
        elif self.pooling_strategy == 'none':
            return x  # Keep sequence dimension
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'output_mode': self.output_mode,
            'num_classes': self.num_classes,
            'num_reasoning_layers': self.num_reasoning_layers,
            'use_residual': self.use_residual,
            'use_layer_norm': self.use_layer_norm,
            'pooling_strategy': self.pooling_strategy,
            'semantic_dropout': self.semantic_dropout,
        })
        return config

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