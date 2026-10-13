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
    FieldAttention,
    FieldResonance,
    GatedShortConv,
    GroupedQueryFieldAttention,
    SelectiveCausalResonance,
)
from .activations import graded_spike

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
    A wrapper for ``ResonantGSERCell`` that performs bounded hierarchical state
    alignment and feedback propagation before a direct System 1 prediction.
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
        If no representation is provided, uses the current forward pass's
        batch prototype.
        """
        if representation is None:
            # Use the cell's tracked state
            representation = tf.expand_dims(self.cell.last_h, 0)
        
        # Use the cell's projection function
        projection = self.cell.project_feedback(representation)
        return projection

    def set_runtime_alignment(self, alignment):
        """Install an ephemeral ``(batch, units)`` feedback target."""
        self.cell.set_runtime_alignment(alignment)

    def clear_runtime_alignment(self):
        """Clear a previously installed per-example feedback target."""
        self.cell.clear_runtime_alignment()
    
    def harmonize_states(self, projection):
        """
        Bottom-Up harmonization of the current state toward a top-down projection.
        
        This is the RSAA state update ``S <- S - alpha * (S - P)``. The
        harmonized state is stored as the target used by the next forward pass;
        assigning ``P`` directly would skip the defined state-update equation.
        """
        if len(projection.shape) > 1:
            projection = tf.reduce_mean(projection, axis=0)
        projection = tf.cast(projection, self.cell.last_h.dtype)
        current = self.cell.last_h
        alpha = tf.clip_by_value(
            tf.cast(self.resonance_factor, current.dtype)
            + tf.cast(self.cell.semantic_divergence_weight, current.dtype),
            0.0,
            0.99,
        )
        harmonized = current - alpha * (current - projection)
        self.cell.resonance_alignment.assign(harmonized)
        self.cell.alignment_set.assign(1.0)
        self.cell.global_divergence.assign(
            tf.reduce_mean(tf.square(harmonized - projection))
        )
    
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
    Multi-head attention over relational features for direct classification or
    downstream System 1 processing.
    """
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
    def call(self, inputs):
        return self.mha(inputs, inputs)

class RelationalGraphAttentionReasoning(tf.keras.layers.Layer):
    """
    Graph-like attention over relational embeddings followed by a pooled direct
    classification head.
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
    A configurable stack of relational self-attention blocks for feature
    extraction and direct classification.

    Features:
    - Multi-head attention for relational semantic processing
    - Configurable output modes: concept features or classification predictions
    - Enhanced semantic processing with residual connections and layer normalization
    - Stacked relational processing with multiple attention layers
    - Adaptive pooling strategies for different semantic tasks

    The layer produces features, a direct classification, or both. The historical
    class name is retained for API compatibility.
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

class HierarchicalHomeostaticController(tf.keras.layers.Layer):
    """Coordinate activity targets across a list of hierarchical tensors.

    The controller computes one activity value per example and level, derives
    an optional task-adaptive target from the hierarchy's current operating
    point, and applies bounded per-level gains.  It also returns hierarchy
    imbalance and target signals for downstream plastic layers.
    """

    def __init__(
        self,
        target_activity=0.12,
        regulation_rate=0.1,
        level_target_slope=0.0,
        adaptive_target_rate=0.0,
        target_bounds=(0.01, 1.0),
        gain_bounds=(0.5, 2.0),
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not 0.0 <= adaptive_target_rate <= 1.0:
            raise ValueError("adaptive_target_rate must be between 0 and 1")
        if target_bounds[0] <= 0 or target_bounds[0] >= target_bounds[1]:
            raise ValueError("target_bounds must be positive and increasing")
        if gain_bounds[0] <= 0 or gain_bounds[0] >= gain_bounds[1]:
            raise ValueError("gain_bounds must be positive and increasing")
        self.target_activity = float(target_activity)
        self.regulation_rate = float(regulation_rate)
        self.level_target_slope = float(level_target_slope)
        self.adaptive_target_rate = float(adaptive_target_rate)
        self.target_bounds = tuple(float(value) for value in target_bounds)
        self.gain_bounds = tuple(float(value) for value in gain_bounds)

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) < 2:
            raise ValueError("HierarchicalHomeostaticController requires at least two levels")
        activities = []
        for tensor in inputs:
            axes = tf.range(1, tf.rank(tensor))
            activities.append(tf.reduce_mean(tf.abs(tensor), axis=axes))
        activities = tf.stack(activities, axis=1)
        hierarchy_activity = tf.reduce_mean(activities, axis=1, keepdims=True)
        base_target = tf.cast(self.target_activity, activities.dtype)
        adaptive_target = (
            (1.0 - self.adaptive_target_rate) * base_target
            + self.adaptive_target_rate * hierarchy_activity
        )
        adaptive_target = tf.clip_by_value(
            adaptive_target, self.target_bounds[0], self.target_bounds[1]
        )

        level_count = len(inputs)
        positions = tf.linspace(-1.0, 1.0, level_count)
        target_profile = 1.0 + self.level_target_slope * positions
        targets = adaptive_target * tf.reshape(target_profile, [1, level_count])
        imbalance = tf.reduce_mean(tf.abs(activities - targets), axis=1, keepdims=True)

        outputs = []
        for index, tensor in enumerate(inputs):
            gain = 1.0 + self.regulation_rate * (targets[:, index] - activities[:, index])
            gain = tf.clip_by_value(gain, self.gain_bounds[0], self.gain_bounds[1])
            gain_shape = tf.concat(
                [[tf.shape(tensor)[0]], tf.ones(tf.rank(tensor) - 1, tf.int32)],
                axis=0,
            )
            outputs.append(tensor * tf.reshape(tf.cast(gain, tensor.dtype), gain_shape))
        return [*outputs, imbalance, adaptive_target]

    def get_config(self):
        config = super().get_config()
        config.update({
            "target_activity": self.target_activity,
            "regulation_rate": self.regulation_rate,
            "level_target_slope": self.level_target_slope,
            "adaptive_target_rate": self.adaptive_target_rate,
            "target_bounds": self.target_bounds,
            "gain_bounds": self.gain_bounds,
        })
        return config


class BioplasticDenseLayer(tf.keras.layers.Layer):
    """
    A bioplastic dense layer incorporating Hebbian learning and homeostatic plasticity for
    Direct Semantic Optimization and Abstraction of Surface-Level Conceptual Variability.
    This layer adapts its synaptic weights based on neural activity, forming robust and
    adaptive semantic representations in the latent space.
    """
    def __init__(
        self,
        units,
        learning_rate=1e-3,
        anti_hebbian_rate=0.1,
        target_avg=0.12,
        homeostatic_rate=5e-5,
        bcm_tau=800.0,
        activation='gelu',
        normalization='l2',
        dropout_rate=0.1,
        enable_inference_plasticity=False,
        homeostasis_mode="global",
        local_homeostasis_weight=0.5,
        homeostatic_window_size=0,
        slow_bcm_tau=None,
        slow_trace_weight=0.5,
        adaptive_target_rate=0.0,
        target_bounds=(0.01, 1.0),
        resonance_coupling=0.0,
        hierarchy_level=0,
        num_hierarchy_levels=1,
        hierarchy_target_slope=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if homeostasis_mode not in {"global", "local", "hybrid"}:
            raise ValueError("homeostasis_mode must be 'global', 'local', or 'hybrid'")
        if not 0.0 <= local_homeostasis_weight <= 1.0:
            raise ValueError("local_homeostasis_weight must be between 0 and 1")
        if homeostatic_window_size < 0:
            raise ValueError("homeostatic_window_size must be non-negative")
        if slow_bcm_tau is not None and slow_bcm_tau < 1.0:
            raise ValueError("slow_bcm_tau must be at least 1")
        if not 0.0 <= slow_trace_weight <= 1.0:
            raise ValueError("slow_trace_weight must be between 0 and 1")
        if not 0.0 <= adaptive_target_rate <= 1.0:
            raise ValueError("adaptive_target_rate must be between 0 and 1")
        if target_bounds[0] <= 0 or target_bounds[0] >= target_bounds[1]:
            raise ValueError("target_bounds must be positive and increasing")
        if num_hierarchy_levels < 1 or not 0 <= hierarchy_level < num_hierarchy_levels:
            raise ValueError("hierarchy_level must index num_hierarchy_levels")
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
        self.homeostasis_mode = homeostasis_mode
        self.local_homeostasis_weight = float(local_homeostasis_weight)
        self.homeostatic_window_size = int(homeostatic_window_size)
        self.slow_bcm_tau = None if slow_bcm_tau is None else float(slow_bcm_tau)
        self.slow_trace_weight = float(slow_trace_weight)
        self.adaptive_target_rate = float(adaptive_target_rate)
        self.target_bounds = tuple(float(value) for value in target_bounds)
        self.resonance_coupling = float(resonance_coupling)
        self.hierarchy_level = int(hierarchy_level)
        self.num_hierarchy_levels = int(num_hierarchy_levels)
        self.hierarchy_target_slope = float(hierarchy_target_slope)

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
            "homeostasis_mode": self.homeostasis_mode,
            "local_homeostasis_weight": self.local_homeostasis_weight,
            "homeostatic_window_size": self.homeostatic_window_size,
            "slow_bcm_tau": self.slow_bcm_tau,
            "slow_trace_weight": self.slow_trace_weight,
            "adaptive_target_rate": self.adaptive_target_rate,
            "target_bounds": self.target_bounds,
            "resonance_coupling": self.resonance_coupling,
            "hierarchy_level": self.hierarchy_level,
            "num_hierarchy_levels": self.num_hierarchy_levels,
            "hierarchy_target_slope": self.hierarchy_target_slope,
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
        self.slow_trace = None
        if self.slow_bcm_tau is not None:
            self.slow_trace = self.add_weight(
                shape=(self.units,), initializer='zeros', trainable=False,
                name='slow_plasticity_trace'
            )
        self.activity_window = None
        self.window_index = None
        self.window_count = None
        if self.homeostatic_window_size:
            self.activity_window = self.add_weight(
                shape=(self.homeostatic_window_size, self.units),
                initializer='zeros', trainable=False, name='activity_window'
            )
            self.window_index = self.add_weight(
                shape=(), initializer='zeros', trainable=False, dtype='int32',
                name='activity_window_index'
            )
            self.window_count = self.add_weight(
                shape=(), initializer='zeros', trainable=False, dtype='int32',
                name='activity_window_count'
            )
        self.adaptive_target = None
        if self.adaptive_target_rate > 0.0:
            self.adaptive_target = self.add_weight(
                shape=(self.units,),
                initializer=tf.keras.initializers.Constant(self.target_avg),
                trainable=False,
                name='adaptive_activity_target'
            )
        super().build(input_shape)

    def _attention_row_weights(self, attention_weights, inputs, dtype):
        if attention_weights is None:
            row_count = tf.reduce_prod(tf.shape(inputs)[:-1])
            return tf.ones(tf.stack([row_count, 1]), dtype=dtype)
        weights = tf.cast(attention_weights, dtype)
        if weights.shape.rank == inputs.shape.rank:
            weights = tf.reduce_mean(weights, axis=-1)
        if weights.shape.rank == 1 and inputs.shape.rank and inputs.shape.rank > 2:
            weights = tf.reshape(
                weights,
                tf.concat([[tf.shape(weights)[0]], tf.ones(inputs.shape.rank - 2, tf.int32)], 0),
            )
        weights = tf.broadcast_to(weights, tf.shape(inputs)[:-1])
        return tf.reshape(tf.maximum(weights, 0.0), [-1, 1])

    def _requested_target(self, target_signal, dtype):
        base = tf.fill([self.units], tf.cast(self.target_avg, dtype))
        if self.adaptive_target is None or target_signal is None:
            target = base if self.adaptive_target is None else tf.cast(self.adaptive_target, dtype)
        else:
            signal = tf.cast(target_signal, dtype)
            if signal.shape.rank and signal.shape[-1] == self.units:
                axes = tf.range(tf.rank(signal) - 1)
                requested = tf.reduce_mean(signal, axis=axes)
            else:
                requested = tf.fill([self.units], tf.reduce_mean(signal))
            requested = tf.clip_by_value(
                requested, self.target_bounds[0], self.target_bounds[1]
            )
            updated = (
                (1.0 - self.adaptive_target_rate) * self.adaptive_target
                + self.adaptive_target_rate * requested
            )
            self.adaptive_target.assign(updated)
            target = tf.cast(updated, dtype)
        if self.num_hierarchy_levels > 1:
            position = 2.0 * self.hierarchy_level / (self.num_hierarchy_levels - 1) - 1.0
            target = target * (1.0 + self.hierarchy_target_slope * position)
        return tf.clip_by_value(target, self.target_bounds[0], self.target_bounds[1])

    def call(
        self,
        inputs,
        training=False,
        attention_weights=None,
        resonance_signal=None,
        target_signal=None,
    ):
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

            row_weights = self._attention_row_weights(attention_weights, inputs, post.dtype)
            weight_sum = tf.maximum(tf.reduce_sum(row_weights), tf.cast(1e-7, post.dtype))

            # Canonical BCM uses a non-negative sliding threshold based on
            # recent squared postsynaptic activity.
            mean_post_squared = tf.reduce_sum(
                row_weights * tf.square(post), axis=0
            ) / weight_sum
            tau = tf.maximum(tf.cast(self.bcm_tau, tf.float32), 1.0)
            new_trace = (
                (1.0 - 1.0 / tau) * self.trace
                + (1.0 / tau) * mean_post_squared
            )
            self.trace.assign(new_trace)

            threshold = new_trace
            if self.activity_window is not None:
                index = tf.math.floormod(
                    self.window_index, self.homeostatic_window_size
                )
                updated_window = tf.tensor_scatter_nd_update(
                    self.activity_window,
                    tf.reshape(index, [1, 1]),
                    tf.expand_dims(mean_post_squared, axis=0),
                )
                self.activity_window.assign(updated_window)
                self.window_index.assign_add(1)
                self.window_count.assign(
                    tf.minimum(self.window_count + 1, self.homeostatic_window_size)
                )
                count = tf.maximum(tf.cast(self.window_count, post.dtype), 1.0)
                threshold = tf.reduce_sum(self.activity_window, axis=0) / count
            if self.slow_trace is not None:
                slow_tau = tf.cast(self.slow_bcm_tau, post.dtype)
                updated_slow = (
                    (1.0 - 1.0 / slow_tau) * self.slow_trace
                    + (1.0 / slow_tau) * mean_post_squared
                )
                self.slow_trace.assign(updated_slow)
                threshold = (
                    (1.0 - self.slow_trace_weight) * threshold
                    + self.slow_trace_weight * updated_slow
                )

            bcm = tf.einsum(
                "bi,bj->ij", pre, row_weights * post * (post - threshold)
            ) / weight_sum
            # Decay the fast component instead of subtracting abs(bcm), which
            # incorrectly made every negative correlation more negative.
            dw = self.learning_rate * (
                bcm - self.anti_hebbian_rate * self.plastic_kernel
            )

            local_activity = tf.reduce_sum(
                row_weights * tf.abs(post), axis=0
            ) / weight_sum
            global_activity = tf.reduce_mean(local_activity)
            target = self._requested_target(target_signal, post.dtype)
            local_error = target - local_activity
            global_error = tf.reduce_mean(target) - global_activity
            if self.homeostasis_mode == "local":
                regulation_error = local_error
            elif self.homeostasis_mode == "hybrid":
                regulation_error = (
                    self.local_homeostasis_weight * local_error
                    + (1.0 - self.local_homeostasis_weight) * global_error
                )
            else:
                regulation_error = tf.fill([self.units], global_error)
            coupling = tf.cast(1.0, post.dtype)
            if resonance_signal is not None:
                divergence = tf.reduce_mean(tf.abs(tf.cast(resonance_signal, post.dtype)))
                coupling = tf.clip_by_value(
                    1.0 + self.resonance_coupling * divergence, 0.1, 10.0
                )
            homeostatic_scale = self.homeostatic_rate * coupling * regulation_error

            self.plastic_kernel.assign_add(dw)
            self.plastic_kernel.assign_add(
                self.plastic_kernel * tf.expand_dims(homeostatic_scale, axis=0)
            )
            self.plastic_kernel.assign(tf.clip_by_norm(self.plastic_kernel, 5.0))

            # The adaptation is part of this inference, not only the next one.
            x = tf.matmul(inputs, self.kernel + self.plastic_kernel) + self.bias
            x = self.activation(x)

        return x

    def reset_plasticity(self):
        """Forget inference-time fast weights and their BCM activity trace."""
        if not self.built:
            return
        self.plastic_kernel.assign(tf.zeros_like(self.plastic_kernel))
        self.trace.assign(tf.zeros_like(self.trace))
        if self.slow_trace is not None:
            self.slow_trace.assign(tf.zeros_like(self.slow_trace))
        if self.activity_window is not None:
            self.activity_window.assign(tf.zeros_like(self.activity_window))
            self.window_index.assign(0)
            self.window_count.assign(0)
        if self.adaptive_target is not None:
            self.adaptive_target.assign(
                tf.fill([self.units], tf.cast(self.target_avg, self.adaptive_target.dtype))
            )

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
    producing compact representations for direct System 1 decisions.
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
    producing a compact temporal context for direct decisions from sequential data.
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


class ResonantSwiGLU(tf.keras.layers.Layer):
    """SwiGLU channel expansion with an ARCANE graded-spike residual write.

    The main residual stream remains continuous.  Only the nonlinear branch is
    quantised, avoiding the information bottleneck of repeatedly spiking the
    complete hidden state while retaining a learned sparse/bursting write.
    """

    def __init__(self, d_model, hidden_dim, spike_threshold=0.4,
                 dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.spike_threshold = float(spike_threshold)
        self.dropout_rate = float(dropout_rate)
        self.in_proj = tf.keras.layers.Dense(2 * self.hidden_dim, use_bias=False, name="in_proj")
        self.out_proj = tf.keras.layers.Dense(self.d_model, use_bias=False, name="out_proj")
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def build(self, input_shape):
        self.in_proj.build(input_shape)
        hidden_shape = list(input_shape)
        hidden_shape[-1] = self.hidden_dim
        self.out_proj.build(hidden_shape)
        self.threshold = self.add_weight(
            name="spike_threshold", shape=(self.hidden_dim,),
            initializer=tf.keras.initializers.Constant(self.spike_threshold), trainable=True)
        self.leak_logit = self.add_weight(
            name="spike_leak", shape=(self.hidden_dim,),
            initializer=tf.keras.initializers.Constant(-3.0), trainable=True)
        super().build(input_shape)

    def call(self, inputs, training=False):
        gate, value = tf.split(self.in_proj(inputs), 2, axis=-1)
        hidden = tf.nn.silu(gate) * value
        hidden = graded_spike(hidden, tf.abs(self.threshold), tf.sigmoid(self.leak_logit))
        return self.dropout(self.out_proj(hidden), training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "hidden_dim": self.hidden_dim,
            "spike_threshold": self.spike_threshold,
            "dropout_rate": self.dropout_rate,
        })
        return config


class Arc1HybridBlock(tf.keras.layers.Layer):
    """ARC 1 LM v2 block: LIV local/GQA global + resonant SwiGLU + memory.

    ``sequence_type`` is either ``"conv"`` for inexpensive input-varying local
    mixing or ``"attention"`` for exact global lookup.  Both variants retain
    ARCANE's engram memory and resonance as separate residual branches.
    """

    def __init__(self, d_model, num_heads, num_kv_heads=2, sequence_type="conv",
                 ffn_hidden_dim=None, conv_kernel_size=3, dropout_rate=0.0,
                 spike_threshold=0.4, resonance_decays=(0.5, 0.9, 0.99),
                 use_rope=True, rope_base=10000.0, max_position=2048,
                 use_engram=False, engram_table_size=4096, engram_rows=4,
                 ngram_sizes=(2, 3), residual_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.sequence_type = str(sequence_type)
        self.ffn_hidden_dim = int(ffn_hidden_dim or (2 * d_model))
        self.conv_kernel_size = int(conv_kernel_size)
        self.dropout_rate = float(dropout_rate)
        self.spike_threshold = float(spike_threshold)
        self.resonance_decays = tuple(float(x) for x in resonance_decays)
        self.use_rope = bool(use_rope)
        self.rope_base = float(rope_base)
        self.max_position = int(max_position)
        self.use_engram = bool(use_engram)
        self.engram_table_size = int(engram_table_size)
        self.engram_rows = int(engram_rows)
        self.ngram_sizes = tuple(int(x) for x in ngram_sizes)
        self.residual_scale = float(residual_scale)
        if self.sequence_type == "conv":
            self.sequence = GatedShortConv(
                self.d_model, self.conv_kernel_size, self.dropout_rate, name="gated_short_conv")
        elif self.sequence_type == "attention":
            self.sequence = GroupedQueryFieldAttention(
                self.d_model, self.num_heads, self.num_kv_heads, self.dropout_rate,
                use_rope=self.use_rope, rope_base=self.rope_base,
                max_position=self.max_position, name="grouped_query_attention")
        else:
            raise ValueError("sequence_type must be 'conv' or 'attention'")
        self.sequence_norm = RMSNorm(name="sequence_norm")
        self.ffn_norm = RMSNorm(name="ffn_norm")
        self.ffn = ResonantSwiGLU(
            self.d_model, self.ffn_hidden_dim, self.spike_threshold,
            self.dropout_rate, name="resonant_swiglu")
        self.resonance_norm = RMSNorm(name="resonance_norm")
        self.resonance = SelectiveCausalResonance(
            self.d_model, self.resonance_decays, self.spike_threshold,
            self.dropout_rate, name="selective_resonance")
        self.engram = ConceptEngram(
            d_model=self.d_model, table_size=self.engram_table_size,
            ngram_sizes=self.ngram_sizes, rows_per_token=self.engram_rows,
            spike_threshold=self.spike_threshold, name="concept_engram",
        ) if self.use_engram else None

    def call(self, inputs, token_ids=None, token_mask=None, training=False):
        scale = tf.cast(self.residual_scale, inputs.dtype)
        x = inputs + scale * self.sequence(
            self.sequence_norm(inputs), token_mask=token_mask, training=training)
        x = x + scale * self.ffn(self.ffn_norm(x), training=training)
        if self.engram is not None:
            x = self.engram(x, token_ids=token_ids, training=training)
        x = x + scale * self.resonance(
            self.resonance_norm(x), token_mask=token_mask, training=training)
        if token_mask is not None:
            x *= tf.cast(tf.expand_dims(token_mask, -1), x.dtype)
        return x

    def call_with_cache(self, inputs, token_ids=None, token_mask=None, training=False):
        """Full prompt pass returning the minimal state needed for decoding."""
        scale = tf.cast(self.residual_scale, inputs.dtype)
        seq_out, seq_cache = self.sequence.call_with_cache(
            self.sequence_norm(inputs), token_mask=token_mask, training=training)
        x = inputs + scale * seq_out
        x = x + scale * self.ffn(self.ffn_norm(x), training=training)
        if self.engram is not None:
            x = self.engram(x, token_ids=token_ids, training=training)
        res_out, res_cache = self.resonance.call_with_cache(
            self.resonance_norm(x), token_mask=token_mask, training=training)
        x = x + scale * res_out
        if token_mask is not None:
            x *= tf.cast(tf.expand_dims(token_mask, -1), x.dtype)
        return x, {"sequence": seq_cache, "resonance": res_cache}

    def step(self, inputs, state, position, recent_token_ids, training=False):
        """One-token cached decode. ``inputs`` has shape ``(B, 1, D)``."""
        scale = tf.cast(self.residual_scale, inputs.dtype)
        normed = self.sequence_norm(inputs)
        if self.sequence_type == "attention":
            seq_out, seq_cache = self.sequence.step(
                normed, state["sequence"], position=position, training=training)
        else:
            seq_out, seq_cache = self.sequence.step(normed, state["sequence"], training=training)
        x = inputs + scale * seq_out
        x = x + scale * self.ffn(self.ffn_norm(x), training=training)
        if self.engram is not None:
            x = self.engram.step(x, recent_token_ids, training=training)
        res_out, res_cache = self.resonance.step(
            self.resonance_norm(x), state["resonance"], training=training)
        x = x + scale * res_out
        return x, {"sequence": seq_cache, "resonance": res_cache}

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "sequence_type": self.sequence_type,
            "ffn_hidden_dim": self.ffn_hidden_dim,
            "conv_kernel_size": self.conv_kernel_size,
            "dropout_rate": self.dropout_rate,
            "spike_threshold": self.spike_threshold,
            "resonance_decays": self.resonance_decays,
            "use_rope": self.use_rope,
            "rope_base": self.rope_base,
            "max_position": self.max_position,
            "use_engram": self.use_engram,
            "engram_table_size": self.engram_table_size,
            "engram_rows": self.engram_rows,
            "ngram_sizes": self.ngram_sizes,
            "residual_scale": self.residual_scale,
        })
        return config


class Arc1PerceptionBlock(tf.keras.layers.Layer):
    """ARC 1 perception block over a padded, bidirectional token field.

    ``x += FieldAttention(norm x)`` → ``x += ResonantChannelMixer(norm x)`` →
    optional ``ConceptEngram`` lexical memory → ``FieldResonance``. Every
    stage is pad-masked, so a request reads the same with or without padding.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        dropout_rate=0.0,
        resonance_factor=0.15,
        resonance_cycles=3,
        spike_threshold=0.4,
        leak_rate=0.1,
        use_rope=True,
        max_position=1024,
        gate_normalize=True,
        mixer_rank_div=2,
        use_engram=False,
        engram_table_size=4096,
        engram_rows=4,
        ngram_sizes=(2, 3),
        causal=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # ConceptEngram n-grams already look backward only, so causal just
        # switches attention and resonance to prefix-only.
        self.causal = bool(causal)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout_rate)
        self.resonance_factor = float(resonance_factor)
        self.resonance_cycles = int(resonance_cycles)
        self.spike_threshold = float(spike_threshold)
        self.leak_rate = float(leak_rate)
        self.use_rope = bool(use_rope)
        self.max_position = int(max_position)
        self.gate_normalize = bool(gate_normalize)
        self.mixer_rank_div = int(mixer_rank_div)
        self.use_engram = bool(use_engram)
        self.engram_table_size = int(engram_table_size)
        self.engram_rows = int(engram_rows)
        self.ngram_sizes = tuple(int(n) for n in ngram_sizes)

        self.attn_norm = RMSNorm(eps=1e-6, name="attn_norm")
        self.attn = FieldAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            use_rope=self.use_rope,
            max_position=self.max_position,
            causal=self.causal,
            name="field_attention",
        )
        self.mix_norm = RMSNorm(eps=1e-6, name="mix_norm")
        self.mixer = ResonantChannelMixer(
            d_model=self.d_model,
            rank_div=self.mixer_rank_div,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold,
            dropout_rate=self.dropout_rate,
            gate_normalize=self.gate_normalize,
            name="channel_mixer",
        )
        self.engram = (
            ConceptEngram(
                d_model=self.d_model,
                table_size=self.engram_table_size,
                ngram_sizes=self.ngram_sizes,
                rows_per_token=self.engram_rows,
                leak_rate=self.leak_rate,
                spike_threshold=self.spike_threshold,
                name="concept_engram",
            )
            if self.use_engram
            else None
        )
        self.resonance = FieldResonance(
            d_model=self.d_model,
            resonance_factor=self.resonance_factor,
            resonance_cycles=self.resonance_cycles,
            spike_threshold=self.spike_threshold,
            causal=self.causal,
            name="field_resonance",
        )
        self.out_norm = RMSNorm(eps=1e-6, name="out_norm")

    def build(self, input_shape):
        for layer in (self.attn_norm, self.attn, self.mix_norm, self.mixer, self.resonance, self.out_norm):
            layer.build(input_shape)
        if self.engram is not None:
            self.engram.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, token_ids=None, token_mask=None, training=False):
        x = inputs + self.attn(self.attn_norm(inputs), token_mask=token_mask, training=training)
        x = x + self.mixer(self.mix_norm(x), training=training)
        if self.engram is not None:
            x = self.engram(x, token_ids=token_ids, training=training)
        if self.causal:
            # Pre-norm residual for the LM: re-normalising the stream after every
            # block (below) left a 6-layer LM stuck at the unigram plateau.
            return x + self.resonance(self.out_norm(x), token_mask=token_mask, training=training)
        x = x + self.resonance(x, token_mask=token_mask, training=training)
        return self.out_norm(x)

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
                "use_rope": self.use_rope,
                "max_position": self.max_position,
                "gate_normalize": self.gate_normalize,
                "mixer_rank_div": self.mixer_rank_div,
                "use_engram": self.use_engram,
                "engram_table_size": self.engram_table_size,
                "engram_rows": self.engram_rows,
                "ngram_sizes": self.ngram_sizes,
                "causal": self.causal,
            }
        )
        return config
