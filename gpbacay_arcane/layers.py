import tensorflow as tf
import numpy as np
from .mechanisms import (
    ResonantGSERCell,
    PredictiveResonantCell,
    CausalLinearSelfAttention,
    CausalSoftmaxSelfAttention,
    RMSNorm,
    ResonantSequenceMixer,
    SpatioTemporalSummaryMixingLayer,
    ConceptEngram,
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
                 max_dynamic_units=None, activation='gelu', use_conceptual_gate=True,
                 gate_normalize=False, **kwargs):
        super().__init__(**kwargs)
        self.gate_normalize = gate_normalize
        self.units = units
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.activation = tf.keras.activations.get(activation)
        self.use_conceptual_gate = use_conceptual_gate
        self.max_dynamic_units = max_dynamic_units
        self.input_dim = input_dim

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
        if self.gate_normalize:
            self.gate_threshold = self.add_weight(
                shape=(self.units,),
                initializer=tf.keras.initializers.Constant(self.spike_threshold),
                trainable=True,
                name='gate_threshold'
            )
        self.built = True

    def call(self, inputs):
        x = tf.matmul(inputs, self.kernel) + self.bias
        x = self.activation(x)
        # Leak rate acts as the inverse slope of a soft spike threshold so both
        # GSER parameters affect a non-recurrent dense map.
        sharpness = 1.0 / tf.maximum(tf.cast(self.leak_rate, x.dtype), 1e-3)
        if self.gate_normalize:
            # An absolute threshold on unnormalised post-activation values has no
            # idea what scale x is on; at d_model=768 it attenuated this branch
            # ~6x at init. Normalising first makes the threshold scale-relative,
            # and learnable per channel so the layer can set its own sparsity.
            scale = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + 1e-6)
            gate = tf.nn.sigmoid(sharpness * (x / scale - tf.cast(self.gate_threshold, x.dtype)))
        else:
            gate = tf.nn.sigmoid(sharpness * (x - tf.cast(self.spike_threshold, x.dtype)))
        x = x * gate

        if self.use_conceptual_gate:
            gate_activations = tf.matmul(inputs, self.conceptual_gate_kernel) + self.conceptual_gate_bias
            conceptual_gate = tf.sigmoid(gate_activations)
            x = x * conceptual_gate

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
            'gate_normalize': self.gate_normalize,
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
        self.higher_layer_name = None
        self.lower_layer_name = None
        # Bypass Keras tracking so cross-layer links can be set after build.
        object.__setattr__(self, "_hierarchy", {"higher": None, "lower": None})

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "resonance_factor": self.resonance_factor,
            "resonance_cycles": self.resonance_cycles,
            "spike_threshold": getattr(self.cell, "spike_threshold", 0.5),
            "convergence_epsilon": getattr(self.cell, "convergence_epsilon", 1e-4),
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
            projection = tf.reduce_mean(projection, axis=0)
        self.cell.resonance_alignment.assign(projection)
        self.cell.alignment_set.assign(1.0)
    
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
            if self._hierarchy.get("lower") is not None:
                lower_layer = self._hierarchy["lower"]
            else:
                return
        
        # Project feedback from this layer to the lower layer
        projection = self.project_feedback()
        lower_layer.harmonize_states(projection)
    
    def set_lower_layer(self, layer):
        """Set the lower layer reference for hierarchical feedback."""
        self.lower_layer_name = layer.name if layer else None
        self._hierarchy["lower"] = layer

    def set_higher_layer(self, layer):
        """Set the higher layer reference for hierarchical feedback."""
        self.higher_layer_name = layer.name if layer else None
        self._hierarchy["higher"] = layer


class PredictiveResonantLayer(tf.keras.layers.RNN):
    """
    RNN layer that implements local predictive resonance, inspired by **Predictive Coding** theories in neuroscience.
    
    This layer models the behavior of cortical columns that maintain an internal prediction of future inputs.
    Alignment is stored per example in the recurrent state (h, c, align), functioning like a local
    "expectation memory." A slow-moving alignment vector acts as this internal prediction; each step 
    the hidden state is iteratively harmonized toward it to minimize prediction error.
    
    Set persist_alignment=True to carry these semantic expectations across separate forward passes,
    mimicking long-term potentiation (LTP) in biological synapses.
    """

    def __init__(
        self,
        units,
        resonance_cycles=3,
        resonance_step_size=0.2,
        spike_threshold=0.5,
        return_sequences=False,
        return_state=False,
        persist_alignment=False,
        **kwargs,
    ):
        cell = PredictiveResonantCell(
            units=units,
            resonance_cycles=resonance_cycles,
            resonance_step_size=resonance_step_size,
            spike_threshold=spike_threshold,
            persist_alignment=persist_alignment,
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
        self.persist_alignment = persist_alignment

    def get_config(self):
        config = super(PredictiveResonantLayer, self).get_config()
        config.update(
            {
                "units": self.units,
                "resonance_cycles": self.resonance_cycles,
                "resonance_step_size": self.resonance_step_size,
                "spike_threshold": self.spike_threshold,
                "persist_alignment": self.persist_alignment,
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
                 dropout_rate=0.1, enable_inference_plasticity=False, **kwargs):
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
        # When True, we apply a lightweight Hebbian-style plasticity update
        # to a non-trainable "plastic" weight component during inference.
        # This gives you inference-time learning without interfering with
        # gradient-based training.
        self.enable_inference_plasticity = enable_inference_plasticity

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
            "dropout_rate": self.dropout_rate,
            "enable_inference_plasticity": self.enable_inference_plasticity,
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
        # Non-trainable plastic component that can be updated online during
        # inference without interfering with optimizer updates.
        self.plastic_kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='zeros',
            trainable=False,
            name='plastic_kernel'
        )
        self.trace = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=False,
            name='plasticity_trace'
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        # Effective weights are the sum of the trained kernel and the
        # non-trainable plastic component.
        effective_kernel = self.kernel + self.plastic_kernel
        x = tf.matmul(inputs, effective_kernel) + self.bias
        x = self.activation(x)

        if training:
            # Standard dropout path during supervised training.
            x = self.dropout(x, training=training)
        elif self.enable_inference_plasticity:
            # Lightweight Hebbian-style plasticity during inference:
            # ΔW ∝ pre^T * post (with simple anti-Hebbian and homeostatic
            # components to avoid runaway growth).
            # This runs only when not in training mode, so it won't
            # interfere with backprop.
            pre = tf.reshape(inputs, [-1, tf.shape(inputs)[-1]])
            post = tf.reshape(x, [-1, tf.shape(x)[-1]])

            if self.normalization_type == "l2":
                pre = tf.nn.l2_normalize(pre, axis=-1)
                post = tf.nn.l2_normalize(post, axis=-1)

            batch_size = tf.cast(tf.shape(pre)[0], tf.float32)
            batch_size_safe = tf.maximum(batch_size, 1.0)

            mean_post = tf.reduce_mean(post, axis=0)
            tau = tf.maximum(tf.cast(self.bcm_tau, tf.float32), 1.0)
            new_trace = (1.0 - 1.0 / tau) * self.trace + (1.0 / tau) * mean_post
            self.trace.assign(new_trace)

            bcm = tf.einsum("bi,bj->ij", pre, post * (post - self.trace)) / batch_size_safe
            dw = self.learning_rate * (bcm - self.anti_hebbian_rate * tf.abs(bcm))

            activity = tf.reduce_mean(tf.abs(post))
            homeostatic_scale = self.homeostatic_rate * (self.target_avg - activity)

            self.plastic_kernel.assign_add(dw)
            self.plastic_kernel.assign_add(homeostatic_scale * self.plastic_kernel)
            self.plastic_kernel.assign(tf.clip_by_norm(self.plastic_kernel, 5.0))

        return x

class HebbianHomeostaticNeuroplasticity(tf.keras.layers.Layer):
    """
    Dense map with a non-trainable plastic component updated by Hebbian
    learning and synaptic scaling. The gradient-trained kernel is left
    alone so optimizer updates and local plasticity do not overwrite each other.
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
        self.plastic_kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='zeros',
            trainable=False,
            name='plastic_kernel'
        )
        self.gain = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False,
            name='homeostatic_gain'
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = tf.matmul(inputs, self.kernel + self.plastic_kernel) + self.bias
        x = x * self.gain

        if training:
            pre = tf.reshape(inputs, [-1, tf.shape(inputs)[-1]])
            post = tf.reshape(x, [-1, tf.shape(x)[-1]])
            batch = tf.maximum(tf.cast(tf.shape(pre)[0], tf.float32), 1.0)
            hebb = tf.einsum("bi,bj->ij", pre, post) / batch
            self.plastic_kernel.assign_add(self.learning_rate * hebb)
            self.plastic_kernel.assign(tf.clip_by_norm(self.plastic_kernel, 5.0))
            activity = tf.reduce_mean(tf.abs(post))
            new_gain = self.gain + self.learning_rate * (self.target_activity - activity)
            self.gain.assign(tf.clip_by_value(new_gain, 0.1, 10.0))

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "learning_rate": self.learning_rate,
            "target_activity": self.target_activity,
        })
        return config

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
    """Sinusoidal positional encoding added to the last dimension of a sequence."""

    def __init__(self, max_position, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_position = int(max_position)
        self.d_model = int(d_model)

    def build(self, input_shape):
        position = np.arange(self.max_position)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, self.d_model, 2) * -(np.log(10000.0) / max(self.d_model, 1))
        )
        pe = np.zeros((self.max_position, self.d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        cosine_dim = pe[:, 1::2].shape[1]
        pe[:, 1::2] = np.cos(position * div_term[:cosine_dim])
        self.pe = self.add_weight(
            name="positional_encoding",
            shape=(1, self.max_position, self.d_model),
            initializer=tf.constant_initializer(pe[np.newaxis, ...]),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pe[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_position": self.max_position, "d_model": self.d_model})
        return config

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


class ArcaneDecoderBlock(tf.keras.layers.Layer):
    """One causal LM block: attention, DenseGSER expand, bioplastic project, resonance.

    ``attention_type`` selects linear (default) or full softmax attention, so a
    decoder stack can interleave a minority of softmax layers -- linear attention
    compresses the prefix into a fixed state and cannot do exact key lookup,
    which is precisely what a softmax teacher is good at.

    Residual mixing of prior block outputs is handled by the parent model via
    ``AttentionResidual`` so this block stays a single-tensor in/out layer.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        ffn_mult=2,
        dropout_rate=0.1,
        resonance_factor=0.15,
        resonance_cycles=3,
        spike_threshold=0.4,
        leak_rate=0.1,
        enable_inference_plasticity=False,
        attention_type="linear",
        use_rope=False,
        rope_base=10000.0,
        max_position=2048,
        chunk_size=64,
        use_decay=False,
        reweight_centered=False,
        gate_normalize=False,
        ffn_out_activation="gelu",
        norm_type="layer",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_mult = ffn_mult
        self.dropout_rate = dropout_rate
        self.resonance_factor = resonance_factor
        self.resonance_cycles = resonance_cycles
        self.spike_threshold = spike_threshold
        self.leak_rate = leak_rate
        self.enable_inference_plasticity = enable_inference_plasticity
        self.attention_type = attention_type
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.max_position = max_position
        self.chunk_size = chunk_size
        self.use_decay = use_decay
        self.reweight_centered = reweight_centered
        self.gate_normalize = gate_normalize
        self.ffn_out_activation = ffn_out_activation
        self.norm_type = norm_type
        d_ff = d_model * ffn_mult
        if attention_type == "softmax":
            self.attn = CausalSoftmaxSelfAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                use_rope=use_rope,
                rope_base=rope_base,
                max_position=max_position,
                name="causal_softmax_attn",
            )
        elif attention_type == "linear":
            self.attn = CausalLinearSelfAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                chunk_size=chunk_size,
                use_decay=use_decay,
                use_rope=use_rope,
                rope_base=rope_base,
                max_position=max_position,
                reweight_centered=reweight_centered,
                name="causal_linear_attn",
            )
        else:
            raise ValueError(f"attention_type must be 'linear' or 'softmax', got {attention_type!r}")
        self.gser = DenseGSER(
            units=d_ff,
            leak_rate=leak_rate,
            spike_threshold=spike_threshold,
            activation="gelu",
            use_conceptual_gate=True,
            gate_normalize=gate_normalize,
            name="dense_gser_expand",
        )
        self.bioplastic = BioplasticDenseLayer(
            units=d_model,
            # A GELU on the FFN *down*-projection floors negatives at -0.17 while
            # leaving positives unbounded, so every FFN write into the residual
            # stream is biased positive (measured: positive mass 1.7x negative).
            # Standard transformer FFNs leave this projection linear.
            activation=ffn_out_activation,
            dropout_rate=dropout_rate,
            enable_inference_plasticity=enable_inference_plasticity,
            name="bioplastic_project",
        )
        self.resonance = ResonantSequenceMixer(
            d_model=d_model,
            resonance_factor=resonance_factor,
            resonance_cycles=resonance_cycles,
            spike_threshold=spike_threshold,
            name="resonant_mixer",
        )
        self.ffn_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.ffn_norm = (
            RMSNorm(eps=1e-6) if norm_type == "rms"
            else tf.keras.layers.LayerNormalization(epsilon=1e-6)
        )

    def call(self, inputs, training=False):
        x = self.attn(inputs, training=training)
        h = self.gser(x)
        h = self.bioplastic(h, training=training)
        h = self.ffn_dropout(h, training=training)
        x = self.ffn_norm(x + h)
        return self.resonance(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "ffn_mult": self.ffn_mult,
                "dropout_rate": self.dropout_rate,
                "resonance_factor": self.resonance_factor,
                "resonance_cycles": self.resonance_cycles,
                "spike_threshold": self.spike_threshold,
                "leak_rate": self.leak_rate,
                "enable_inference_plasticity": self.enable_inference_plasticity,
                "attention_type": self.attention_type,
                "use_rope": self.use_rope,
                "rope_base": self.rope_base,
                "max_position": self.max_position,
                "chunk_size": self.chunk_size,
                "use_decay": self.use_decay,
                "reweight_centered": self.reweight_centered,
                "gate_normalize": self.gate_normalize,
                "ffn_out_activation": self.ffn_out_activation,
                "norm_type": self.norm_type,
            }
        )
        return config


class ResonantChannelMixer(tf.keras.layers.Layer):
    """Cheap channel mixer: low-rank expand + DenseGSER gate + project.

    Replaces a dense ``d * ffn_mult`` FFN with a bottleneck of width
    ``d_model // rank_div`` so most ARC 1 capacity can sit in ConceptEngram
    tables instead of matmuls.
    """

    def __init__(
        self,
        d_model,
        rank_div=4,
        leak_rate=0.1,
        spike_threshold=0.4,
        dropout_rate=0.0,
        gate_normalize=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.rank_div = max(2, int(rank_div))
        self.leak_rate = float(leak_rate)
        self.spike_threshold = float(spike_threshold)
        self.dropout_rate = float(dropout_rate)
        self.gate_normalize = bool(gate_normalize)
        self.inner = max(self.d_model // self.rank_div, 16)

    def build(self, input_shape):
        self.down = tf.keras.layers.Dense(self.inner, use_bias=True, name="mix_down")
        self.up = tf.keras.layers.Dense(self.d_model, use_bias=True, name="mix_up")
        self.gser = DenseGSER(
            units=self.d_model,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold,
            activation="gelu",
            use_conceptual_gate=True,
            gate_normalize=self.gate_normalize,
            name="mix_gser",
        )
        self.drop = tf.keras.layers.Dropout(self.dropout_rate)
        self.down.build(input_shape)
        mid = list(input_shape)
        mid[-1] = self.inner
        self.up.build(mid)
        self.gser.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=False):
        h = self.up(self.down(inputs))
        h = self.gser(h)
        return self.drop(h, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "rank_div": self.rank_div,
                "leak_rate": self.leak_rate,
                "spike_threshold": self.spike_threshold,
                "dropout_rate": self.dropout_rate,
                "gate_normalize": self.gate_normalize,
            }
        )
        return config


class Arc1DecoderBlock(tf.keras.layers.Layer):
    """ARC 1 block: causal attn → ResonantChannelMixer → ConceptEngram → resonance."""

    def __init__(
        self,
        d_model,
        num_heads,
        dropout_rate=0.1,
        resonance_factor=0.15,
        resonance_cycles=3,
        spike_threshold=0.4,
        leak_rate=0.1,
        attention_type="linear",
        use_rope=True,
        rope_base=10000.0,
        max_position=2048,
        chunk_size=64,
        use_decay=True,
        reweight_centered=True,
        gate_normalize=True,
        norm_type="rms",
        mixer_rank_div=4,
        engram_table_size=4096,
        engram_rows=8,
        ngram_sizes=(2, 3),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.resonance_factor = resonance_factor
        self.resonance_cycles = resonance_cycles
        self.spike_threshold = spike_threshold
        self.leak_rate = leak_rate
        self.attention_type = attention_type
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.max_position = max_position
        self.chunk_size = chunk_size
        self.use_decay = use_decay
        self.reweight_centered = reweight_centered
        self.gate_normalize = gate_normalize
        self.norm_type = norm_type
        self.mixer_rank_div = mixer_rank_div
        self.engram_table_size = engram_table_size
        self.engram_rows = engram_rows
        self.ngram_sizes = tuple(ngram_sizes)

        if attention_type == "softmax":
            self.attn = CausalSoftmaxSelfAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                use_rope=use_rope,
                rope_base=rope_base,
                max_position=max_position,
                name="causal_softmax_attn",
            )
        elif attention_type == "linear":
            self.attn = CausalLinearSelfAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                chunk_size=chunk_size,
                use_decay=use_decay,
                use_rope=use_rope,
                rope_base=rope_base,
                max_position=max_position,
                reweight_centered=reweight_centered,
                name="causal_linear_attn",
            )
        else:
            raise ValueError(f"attention_type must be 'linear' or 'softmax', got {attention_type!r}")

        self.mixer = ResonantChannelMixer(
            d_model=d_model,
            rank_div=mixer_rank_div,
            leak_rate=leak_rate,
            spike_threshold=spike_threshold,
            dropout_rate=dropout_rate,
            gate_normalize=gate_normalize,
            name="channel_mixer",
        )
        self.engram = ConceptEngram(
            d_model=d_model,
            table_size=engram_table_size,
            ngram_sizes=self.ngram_sizes,
            rows_per_token=engram_rows,
            leak_rate=leak_rate,
            spike_threshold=spike_threshold,
            name="concept_engram",
        )
        self.resonance = ResonantSequenceMixer(
            d_model=d_model,
            resonance_factor=resonance_factor,
            resonance_cycles=resonance_cycles,
            spike_threshold=spike_threshold,
            name="resonant_mixer",
        )
        self.mix_norm = (
            RMSNorm(eps=1e-6) if norm_type == "rms"
            else tf.keras.layers.LayerNormalization(epsilon=1e-6)
        )
        self.engram_norm = (
            RMSNorm(eps=1e-6) if norm_type == "rms"
            else tf.keras.layers.LayerNormalization(epsilon=1e-6)
        )

    def call(self, inputs, token_ids=None, training=False):
        x = self.attn(inputs, training=training)
        h = self.mixer(x, training=training)
        x = self.mix_norm(x + h)
        x = self.engram_norm(self.engram(x, token_ids=token_ids, training=training))
        return self.resonance(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "resonance_factor": self.resonance_factor,
                "resonance_cycles": self.resonance_cycles,
                "spike_threshold": self.spike_threshold,
                "leak_rate": self.leak_rate,
                "attention_type": self.attention_type,
                "use_rope": self.use_rope,
                "rope_base": self.rope_base,
                "max_position": self.max_position,
                "chunk_size": self.chunk_size,
                "use_decay": self.use_decay,
                "reweight_centered": self.reweight_centered,
                "gate_normalize": self.gate_normalize,
                "norm_type": self.norm_type,
                "mixer_rank_div": self.mixer_rank_div,
                "engram_table_size": self.engram_table_size,
                "engram_rows": self.engram_rows,
                "ngram_sizes": self.ngram_sizes,
            }
        )
        return config
