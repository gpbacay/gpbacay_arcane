import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer

class GSER(Layer):
    """
    The Gated Spiking Elastic Reservoir (GSER) Mechanism (RNN Cell) for semantic processing.
    Combines dynamic reservoir sizing, spiking neurons, and adaptive gating, with an integrated conceptual
    gating mechanism to dynamically adjust the influence of input and recurrent connections based on semantic relevance.
    This contributes to Direct Semantic Optimization and Abstraction of Surface-Level Conceptual Variability
    by focusing on the most salient semantic features in the latent space, supporting Latent Space Reasoning
    within a Unified Multi-Modal Semantic Space.
    """
    def __init__(self, input_dim, initial_reservoir_size, max_dynamic_reservoir_dim, spectral_radius, leak_rate, spike_threshold, neurogenesis_rate=0.05, pruning_rate=0.1, use_semantic_gate=True, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.initial_reservoir_size = initial_reservoir_size
        self.max_dynamic_reservoir_dim = max_dynamic_reservoir_dim
        self.spectral_radius = spectral_radius
        self.initial_leak_rate = leak_rate
        self.initial_spike_threshold = spike_threshold
        self.neurogenesis_rate = neurogenesis_rate
        self.pruning_rate = pruning_rate
        self.use_semantic_gate = use_semantic_gate
        self.current_reservoir_size = None
        self.state_size = [self.max_dynamic_reservoir_dim]
        self.output_size = self.max_dynamic_reservoir_dim
        
    def build(self, input_shape):
        super().build(input_shape)
        self.current_reservoir_size = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_reservoir_size),
            trainable=False,
            dtype=tf.int32,
            name='current_reservoir_size'
        )
        self.initialize_weights()
        
        if self.use_semantic_gate:
            self.semantic_gate_kernel = self.add_weight(
                shape=(self.input_dim + self.max_dynamic_reservoir_dim, self.max_dynamic_reservoir_dim),
                initializer='glorot_uniform',
                trainable=True,
                name='semantic_gate_kernel'
            )
            self.semantic_gate_bias = self.add_weight(
                shape=(self.max_dynamic_reservoir_dim,),
                initializer='zeros',
                trainable=True,
                name='semantic_gate_bias'
            )

    def initialize_weights(self):
        self.spatiotemporal_reservoir_weights = self.add_weight(
            shape=(self.max_dynamic_reservoir_dim, self.max_dynamic_reservoir_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=False,
            name='spatiotemporal_reservoir_weights'
        )
        self.spatiotemporal_input_weights = self.add_weight(
            shape=(self.max_dynamic_reservoir_dim, self.input_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=False,
            name='spatiotemporal_input_weights'
        )
        self.spiking_gate_weights = self.add_weight(
            shape=(3 * self.max_dynamic_reservoir_dim, self.input_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=False,
            name='spiking_gate_weights'
        )
        self.leak_rate_param = self.add_weight(
            shape=(self.max_dynamic_reservoir_dim,),
            initializer=tf.keras.initializers.Constant(np.log(self.initial_leak_rate / (1 - self.initial_leak_rate))),
            trainable=True,
            name='leak_rate_param'
        )
        self.spike_threshold_param = self.add_weight(
            shape=(self.max_dynamic_reservoir_dim,),
            initializer=tf.keras.initializers.Constant(np.log(np.exp(self.initial_spike_threshold) - 1)),
            trainable=True,
            name='spike_threshold_param'
        )

    def add_neurons(self, new_neurons_count):
        new_size = tf.minimum(self.current_reservoir_size + new_neurons_count, self.max_dynamic_reservoir_dim)
        self.current_reservoir_size.assign(new_size)

    def prune_connections(self, pruning_threshold=0.1):
        active_size = self.current_reservoir_size
        active_weights = self.spatiotemporal_reservoir_weights[:active_size, :active_size]
        mask = tf.abs(active_weights) < pruning_threshold
        pruned_weights = tf.where(mask, tf.zeros_like(active_weights), active_weights)
        self.spatiotemporal_reservoir_weights.assign(tf.tensor_scatter_nd_update(
            self.spatiotemporal_reservoir_weights,
            tf.where(tf.ones((active_size, active_size), dtype=tf.bool)),
            tf.reshape(pruned_weights, [-1])
        ))

    def prune_neurons(self, num_to_prune):
        active_size = self.current_reservoir_size
        if active_size <= num_to_prune:
            return
        activity = tf.reduce_sum(tf.abs(self.spatiotemporal_reservoir_weights[:active_size, :active_size]), axis=1)
        _, indices_to_prune = tf.nn.top_k(-activity, k=num_to_prune)
        for idx_to_prune in tf.sort(indices_to_prune, direction='DESCENDING'):
            last_active_idx = self.current_reservoir_size - 1
            if idx_to_prune >= last_active_idx:
                self.current_reservoir_size.assign_sub(1)
                continue
            p = tf.stack([idx_to_prune, last_active_idx])
            q = tf.stack([last_active_idx, idx_to_prune])
            temp_weights = tf.tensor_scatter_nd_update(self.spatiotemporal_reservoir_weights, tf.expand_dims(p, axis=1), tf.gather(self.spatiotemporal_reservoir_weights, q))
            temp_weights = tf.transpose(temp_weights)
            temp_weights = tf.tensor_scatter_nd_update(temp_weights, tf.expand_dims(p, axis=1), tf.gather(temp_weights, q))
            self.spatiotemporal_reservoir_weights.assign(tf.transpose(temp_weights))
            self.spatiotemporal_input_weights.assign(tf.tensor_scatter_nd_update(self.spatiotemporal_input_weights, tf.expand_dims(p, axis=1), tf.gather(self.spatiotemporal_input_weights, q)))
            self.leak_rate_param.assign(tf.tensor_scatter_nd_update(self.leak_rate_param, tf.expand_dims(p, axis=1), tf.gather(self.leak_rate_param, q)))
            self.spike_threshold_param.assign(tf.tensor_scatter_nd_update(self.spike_threshold_param, tf.expand_dims(p, axis=1), tf.gather(self.spike_threshold_param, q)))
            self.current_reservoir_size.assign_sub(1)

    def call(self, inputs, states):
        inputs = tf.cast(inputs, tf.float32)
        if isinstance(states, (list, tuple)):
            prev_state_full = states[0] if len(states) > 0 else tf.zeros((tf.shape(inputs)[0], self.max_dynamic_reservoir_dim))
        else:
            prev_state_full = states
        active_size = self.current_reservoir_size
        prev_state = prev_state_full[:, :active_size]
        active_input_weights = self.spatiotemporal_input_weights[:active_size, :]
        active_reservoir_weights = self.spatiotemporal_reservoir_weights[:active_size, :active_size]
        active_gate_weights = self.spiking_gate_weights[:3 * active_size, :]
        leak_rate = tf.sigmoid(self.leak_rate_param[:active_size])
        spike_threshold = tf.nn.softplus(self.spike_threshold_param[:active_size])
        input_part = tf.matmul(inputs, active_input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, active_reservoir_weights)
        gate_part = tf.matmul(inputs, active_gate_weights, transpose_b=True)
        i_gate, f_gate, o_gate = tf.split(tf.sigmoid(gate_part), 3, axis=-1)
        state = (1 - leak_rate) * (f_gate * prev_state) + leak_rate * tf.tanh(i_gate * (input_part + reservoir_part))
        state = o_gate * state

        if self.use_semantic_gate:
            # Concatenate inputs and current state for the semantic gate
            combined_features = tf.concat([inputs, state], axis=-1)
            semantic_gate_activations = tf.matmul(combined_features, self.semantic_gate_kernel) + self.semantic_gate_bias
            semantic_gate = tf.sigmoid(semantic_gate_activations[:, :active_size]) # Apply gate to active part
            state = state * semantic_gate # Modulate state based on semantic relevance

        spikes = tf.cast(tf.greater(state, spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - spike_threshold, state)
        padded_state = tf.pad(state, [[0, 0], [0, self.max_dynamic_reservoir_dim - active_size]])
        padded_state.set_shape([None, self.max_dynamic_reservoir_dim])
        return padded_state, [padded_state]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if batch_size is None and inputs is not None:
            batch_size = tf.shape(inputs)[0]
        dtype = dtype or tf.float32
        return [tf.zeros((batch_size, self.max_dynamic_reservoir_dim), dtype=dtype)]

    def get_config(self):
        config = super().get_config()
        config.update({
            'initial_reservoir_size': self.initial_reservoir_size,
            'input_dim': self.input_dim,
            'spectral_radius': self.spectral_radius,
            'leak_rate': self.initial_leak_rate,
            'spike_threshold': self.initial_spike_threshold,
            'max_dynamic_reservoir_dim': self.max_dynamic_reservoir_dim,
            'neurogenesis_rate': self.neurogenesis_rate,
            'pruning_rate': self.pruning_rate,
            'use_semantic_gate': self.use_semantic_gate
        })
        return config


class ResonantGSERCell(Layer):
    """
    Cell for the Resonant Gated Spiking Elastic Reservoir (ResonantGSER).
    This cell is a core component for Latent Space Reasoning, integrating concepts of
    Unified Multi-Modal Semantic Space and Direct Semantic Optimization through its
    deliberative resonance mechanism. It combines robust gated updates with biomimetic
    spiking and iterative harmonization to abstract away surface-level conceptual variability.
    """
    def __init__(self, units, resonance_factor=0.1, spike_threshold=0.5, 
                 resonance_cycles=3, convergence_epsilon=1e-4, semantic_divergence_weight=0.1, **kwargs):
        super(ResonantGSERCell, self).__init__(**kwargs)
        self.units = units
        self.resonance_factor = resonance_factor
        self.spike_threshold = spike_threshold
        self.resonance_cycles = resonance_cycles  # N in the RSAA paper
        self.convergence_epsilon = convergence_epsilon  # ε for early stopping
        self.semantic_divergence_weight = semantic_divergence_weight # Weight for semantic divergence in harmonization
        self.state_size = [units, units]  # [h, c]
        self.output_size = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        
    def build(self, input_shape):
        # Handle symbolic or undefined feature dimension in Functional API
        input_dim = input_shape[-1]
        if input_dim is None:
            input_dim = self.units 
        
        # Build the LSTM cell
        self.lstm_cell.build((None, int(input_dim)))
        
        # Resonance modulation parameters (trainable)
        self.resonance_gate = self.add_weight(
            name='resonance_gate', shape=(self.units,),
            initializer=tf.keras.initializers.Constant(1.0), trainable=True
        )
        self.resonance_bias = self.add_weight(
            name='resonance_bias', shape=(self.units,),
            initializer=tf.keras.initializers.Zeros(), trainable=True
        )
        
        # Top-down projection alignment (non-trainable, updated during resonance)
        self.resonance_alignment = self.add_weight(
            name='resonance_alignment', shape=(self.units,),
            initializer='zeros', trainable=False
        )
        
        # Feedback projection weights (trainable) - for top-down expectations
        self.feedback_weights = self.add_weight(
            name='feedback_weights', shape=(self.units, int(input_dim)),
            initializer='glorot_uniform', trainable=True
        )
        
        # Projection head for feedback (trainable) - maps h_t to projection space
        self.projection_kernel = self.add_weight(
            name='projection_kernel', shape=(self.units, self.units),
            initializer='glorot_uniform', trainable=True
        )
        self.projection_bias = self.add_weight(
            name='projection_bias', shape=(self.units,),
            initializer='zeros', trainable=True
        )
        
        # Track last hidden state for external access
        self.last_h = self.add_weight(
            name='last_h', shape=(self.units,),
            initializer='zeros', trainable=False
        )
        
        # Track global divergence for monitoring
        self.global_divergence = self.add_weight(
            name='global_divergence', shape=(),
            initializer='zeros', trainable=False
        )
        
        self.built = True
    
    def project_feedback(self, state):
        """
        Top-Down Projection: P_{i→i-1} = f_proj(S_i; W_i)
        Projects current state to expectation for lower layer.
        """
        # Apply learned projection transformation
        projection = tf.matmul(state, self.projection_kernel) + self.projection_bias
        return projection
    
    def compute_divergence(self, current_state, projection):
        """
        Prediction Divergence: Δ_{i-1} = S_{i-1} - P_{i→i-1}
        Computes the signed difference between current state and top-down expectation.
        """
        divergence = current_state - projection
        return divergence
    
    def harmonize_state(self, current_state, divergence, gamma):
        """
        State Harmonization: Updates state to reduce semantic divergence from top-down projection.
        Incorporates semantic divergence weighting for Direct Semantic Optimization.
        """
        # Dynamically adjust harmonization based on semantic divergence weight
        harmonized = current_state - (gamma + self.semantic_divergence_weight) * divergence
        return harmonized
    
    def resonance_loop(self, h_initial, projection_from_above=None):
        """
        Implements the core Resonance Loop for Latent Space Reasoning.
        """
        h_current = h_initial
        gamma = self.resonance_factor
        
        if projection_from_above is not None:
            # Iterative synchronization
            for _ in range(self.resonance_cycles):
                delta = self.compute_divergence(h_current, projection_from_above)
                # Divergence check (avoiding early break for graph compatibility)
                h_current = self.harmonize_state(h_current, delta, gamma)
            
            # Final divergence update after loop
            final_delta = self.compute_divergence(h_current, projection_from_above)
            divergence_magnitude = tf.reduce_mean(tf.square(final_delta))
            self.global_divergence.assign(divergence_magnitude)
        
        return h_current
        
    def call(self, inputs, states, **kwargs):
        """
        Forward pass integrating Direct Semantic Optimization and Latent Space Reasoning.
        """
        # Handle states whether they are list or tuple
        if isinstance(states, (list, tuple)):
            h_prev = states[0]
            c_prev = states[1]
        else:
            h_prev = states
            c_prev = None # Should not happen with LSTMCell
            
        training = kwargs.get('training')
        
        # === Step 1: Forward Initialization ===
        # Standard LSTM forward pass
        h_lstm, new_states = self.lstm_cell(inputs, states, training=training)
        h_new = new_states[0]
        c_new = new_states[1]
        
        # === Step 2: Resonance Loop ===
        projection_from_above = self.resonance_alignment
        h_resonated = self.resonance_loop(h_new, projection_from_above)
        
        # === Step 3: Apply Resonance Modulation ===
        res_mod = tf.sigmoid(self.resonance_gate) * self.resonance_factor
        h_modulated = h_resonated * (1.0 + res_mod) + self.resonance_bias
        
        # === Step 4: Spiking Mechanism ===
        spikes = tf.cast(tf.greater(h_modulated, self.spike_threshold), dtype=tf.float32)
        h_final = tf.where(spikes > 0, h_modulated - self.spike_threshold, h_modulated)
        
        # === Step 5: Track State ===
        # Use tf.cond to safely assign in graph mode if needed, 
        # but assign is generally ok on non-trainable variables.
        self.last_h.assign(tf.reduce_mean(h_final, axis=0))
        
        return h_final, [h_final, c_new]
    
    def get_projection(self):
        """
        External interface to get top-down projection for lower layers.
        Returns the projection based on the last processed state.
        """
        # Use the tracked last_h to compute projection
        batch_size = 1  # For stateless projection
        last_h_expanded = tf.expand_dims(self.last_h, 0)
        return self.project_feedback(last_h_expanded)
        
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if batch_size is None and inputs is not None:
            batch_size = tf.shape(inputs)[0]
        dtype = dtype or tf.float32
        return [tf.zeros((batch_size, self.units), dtype=dtype),
                tf.zeros((batch_size, self.units), dtype=dtype)]

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "resonance_factor": self.resonance_factor,
            "spike_threshold": self.spike_threshold,
            "resonance_cycles": self.resonance_cycles,
            "convergence_epsilon": self.convergence_epsilon,
            "semantic_divergence_weight": self.semantic_divergence_weight
        })
        return config


class PredictiveResonantCell(Layer):
    """
    A neuromimetic recurrent cell that implements *local* predictive resonance, modeled after homeostatic loops in biological neural circuits.
    
    This mechanism implements the core principle of **Predictive Coding**: that neural state should minimize the divergence between sensory input and internal expectation.
    
    Compared to `ResonantGSERCell`, this cell:
    - Keeps the core recurrent dynamics explicit (mimicking somatic integration)
    - Maintains a per-sample alignment state acting as a **synaptic memory trace**
    - Uses a lightweight predictive head (like a dendritic prediction) to update the alignment state
      toward a slow-moving internal expectation of future activity

    This creates a fully autonomous, self-contained resonance loop similar to the behavior of
    isolated cortical columns.
    without relying on external callbacks or model references.
    """

    def __init__(
        self,
        units,
        resonance_cycles=3,
        resonance_step_size=0.2,
        spike_threshold=0.5,
        persist_alignment=False,
        **kwargs,
    ):
        super(PredictiveResonantCell, self).__init__(**kwargs)
        self.units = units
        self.resonance_cycles = resonance_cycles
        self.resonance_step_size = resonance_step_size
        self.spike_threshold = spike_threshold
        # If True, we keep a slow-moving alignment memory across separate
        # forward passes (and even across separate inputs within the same
        # process), giving you "stateful resonance across calls".
        self.persist_alignment = persist_alignment

        # state: [h, c, alignment]
        self.state_size = [units, units, units]
        self.output_size = units

        self.lstm_cell = tf.keras.layers.LSTMCell(units)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if input_dim is None:
            input_dim = self.units

        # Build base LSTM cell
        self.lstm_cell.build((None, int(input_dim)))

        # Predictive head: maps current state to a "target" alignment state
        self.prediction_kernel = self.add_weight(
            name="prediction_kernel",
            shape=(self.units, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.prediction_bias = self.add_weight(
            name="prediction_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )

        # Resonance modulation and spiking parameters
        self.resonance_gate = self.add_weight(
            name="resonance_gate",
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True,
        )
        self.resonance_bias = self.add_weight(
            name="resonance_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )

        # Track divergence for monitoring (mean squared error between state and alignment)
        self.global_divergence = self.add_weight(
            name="global_divergence",
            shape=(),
            initializer="zeros",
            trainable=False,
        )

        # Optional persistent alignment memory (batch-agnostic).
        # When enabled, this provides a default alignment state for the next
        # sequence, even if it is a different input or comes from a different
        # call site, enabling stateful resonance behaviour.
        if self.persist_alignment:
            self.alignment_memory = self.add_weight(
                name="alignment_memory",
                shape=(self.units,),
                initializer="zeros",
                trainable=False,
            )

        super(PredictiveResonantCell, self).build(input_shape)

    def _resonance_loop(self, h_initial, alignment):
        """
        Per-example resonance loop: iteratively move h toward alignment.
        """
        if alignment is None:
            return h_initial

        h_current = h_initial
        step = self.resonance_step_size

        for _ in range(self.resonance_cycles):
            delta = h_current - alignment
            h_current = h_current - step * delta

        # Track divergence for analysis
        final_delta = h_current - alignment
        divergence = tf.reduce_mean(tf.square(final_delta))
        self.global_divergence.assign(divergence)

        return h_current

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """
        Initial state for the predictive resonant cell.

        When `persist_alignment` is enabled, the alignment part of the state
        is initialized from a batch-agnostic `alignment_memory` variable so
        that resonance can carry over across independent calls.
        """
        if batch_size is None and inputs is not None:
            batch_size = tf.shape(inputs)[0]
        dtype = dtype or tf.float32

        h0 = tf.zeros((batch_size, self.units), dtype=dtype)
        c0 = tf.zeros((batch_size, self.units), dtype=dtype)

        if getattr(self, "persist_alignment", False) and hasattr(self, "alignment_memory"):
            align0 = tf.tile(tf.expand_dims(tf.cast(self.alignment_memory, dtype), 0), [batch_size, 1])
        else:
            align0 = tf.zeros((batch_size, self.units), dtype=dtype)

        return [h0, c0, align0]

    def call(self, inputs, states, **kwargs):
        """
        Forward pass:
        1. Standard LSTM update
        2. Resonance loop toward current alignment state
        3. Spiking + modulation
        4. Predictive update of alignment state (slow-moving target)
        """
        if isinstance(states, (list, tuple)):
            h_prev, c_prev, align_prev = states
        else:
            # Fallback, should not normally happen
            h_prev = states
            c_prev = tf.zeros_like(h_prev)
            align_prev = tf.zeros_like(h_prev)

        training = kwargs.get("training", None)

        # 1. Base LSTM dynamics
        h_lstm, [h_new, c_new] = self.lstm_cell(inputs, [h_prev, c_prev], training=training)

        # 2. Local resonance toward alignment state (per-example)
        h_resonant = self._resonance_loop(h_new, align_prev)

        # 3. Resonance modulation + spiking
        res_mod = tf.sigmoid(self.resonance_gate)
        h_modulated = h_resonant * (1.0 + res_mod) + self.resonance_bias

        spikes = tf.cast(tf.greater(h_modulated, self.spike_threshold), tf.float32)
        h_final = tf.where(spikes > 0.0, h_modulated - self.spike_threshold, h_modulated)

        # 4. Predictive update of alignment: slow-moving target toward projected future state
        predicted = tf.matmul(h_final, self.prediction_kernel) + self.prediction_bias
        alpha = 0.1  # slow update rate for alignment state
        align_new = (1.0 - alpha) * align_prev + alpha * predicted

        # Optionally store a batch-agnostic summary of the new alignment so
        # that subsequent calls can start from a more informed resonance target.
        if getattr(self, "persist_alignment", False) and hasattr(self, "alignment_memory"):
            mean_align = tf.reduce_mean(align_new, axis=0)
            self.alignment_memory.assign(mean_align)

        return h_final, [h_final, c_new, align_new]

    def get_config(self):
        config = super(PredictiveResonantCell, self).get_config()
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


class MultiheadLinearSelfAttentionKernalization(Layer):
    """
    A Multi-head linear self-attention mechanism with kernel approximation, designed for efficient
    Latent Space Reasoning and establishing coherent relationships within a Unified Multi-Modal Semantic Space.
    It achieves linear complexity (O(n)) for long sequences, and incorporates semantic re-weighting
    to enhance Direct Semantic Optimization by prioritizing semantically important features.
    """
    def __init__(self, d_model, num_heads, dropout_rate=0.1, use_weighted_summary=False, 
                 use_semantic_reweighting=True, eps=1e-6, **kwargs):
        super(MultiheadLinearSelfAttentionKernalization, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_weighted_summary = use_weighted_summary
        self.use_semantic_reweighting = use_semantic_reweighting
        self.eps = eps
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=eps)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def build(self, input_shape):
        d_model = self.d_model
        self.query_weight = self.add_weight(name='query_weight', shape=(d_model, d_model), initializer='glorot_uniform', trainable=True)
        self.query_bias = self.add_weight(name='query_bias', shape=(d_model,), initializer='zeros', trainable=True)
        self.key_weight = self.add_weight(name='key_weight', shape=(d_model, d_model), initializer='glorot_uniform', trainable=True)
        self.key_bias = self.add_weight(name='key_bias', shape=(d_model,), initializer='zeros', trainable=True)
        self.value_weight = self.add_weight(name='value_weight', shape=(d_model, d_model), initializer='glorot_uniform', trainable=True)
        self.value_bias = self.add_weight(name='value_bias', shape=(d_model,), initializer='zeros', trainable=True)
        self.output_weight = self.add_weight(name='output_weight', shape=(d_model, d_model), initializer='glorot_uniform', trainable=True)
        self.output_bias = self.add_weight(name='output_bias', shape=(d_model,), initializer='zeros', trainable=True)
        if self.use_weighted_summary:
            self.summary_weight = self.add_weight(name='summary_weight', shape=(d_model, 1), initializer='glorot_uniform', trainable=True)
            self.summary_bias = self.add_weight(name='summary_bias', shape=(1,), initializer='zeros', trainable=True)
        
        if self.use_semantic_reweighting:
            self.semantic_reweight_kernel = self.add_weight(
                name='semantic_reweight_kernel', shape=(d_model, 1),
                initializer='glorot_uniform', trainable=True
            )
            self.semantic_reweight_bias = self.add_weight(
                name='semantic_reweight_bias', shape=(1,),
                initializer='zeros', trainable=True
            )

        self.layer_norm.build(input_shape)
        super(MultiheadLinearSelfAttentionKernalization, self).build(input_shape)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        queries = tf.matmul(inputs, self.query_weight) + self.query_bias
        keys = tf.matmul(inputs, self.key_weight) + self.key_bias
        values = tf.matmul(inputs, self.value_weight) + self.value_bias
        queries = self.split_heads(queries, batch_size)
        keys = self.split_heads(keys, batch_size)
        values = self.split_heads(values, batch_size)
        queries = tf.nn.elu(queries) + 1.0
        keys = tf.nn.elu(keys) + 1.0
        key_norm = tf.sqrt(tf.reduce_sum(tf.square(keys), axis=-1, keepdims=True) + self.eps)
        keys = keys / key_norm
        scores = tf.einsum("bhqd,bhkd->bhqk", queries, keys)
        attention_output = tf.einsum("bhqk,bhvd->bhqd", scores, values)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, seq_len, self.d_model))
        if self.use_weighted_summary:
            weights = tf.nn.sigmoid(tf.matmul(attention_output, self.summary_weight) + self.summary_bias)
            attention_output = attention_output * weights
        output = tf.matmul(attention_output, self.output_weight) + self.output_bias
        if self.use_semantic_reweighting:
            reweight_factors = tf.sigmoid(tf.matmul(output, self.semantic_reweight_kernel) + self.semantic_reweight_bias)
            output = output * reweight_factors
        output = self.dropout(output, training=training)
        return self.layer_norm(inputs + output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model, "num_heads": self.num_heads, "dropout_rate": self.dropout_rate,
            "use_weighted_summary": self.use_weighted_summary, "use_semantic_reweighting": self.use_semantic_reweighting, "eps": self.eps,
        })
        return config


class SpatioTemporalSummaryMixingLayer(Layer):
    """
    A mechanism that enhances spatio-temporal data by mixing local and global context to generate
    Non-Autoregressive Semantic Predictions for Efficiency and construct a Unified Multi-Modal Semantic Space.
    It uses gated linear units (GLU) for local gating and GELU for high-level semantic summaries,
    actively abstracting away surface-level conceptual variability for Latent Space Reasoning.
    """
    def __init__(self, d_model, dropout_rate=0.1, use_weighted_summary=False, **kwargs):
        super(SpatioTemporalSummaryMixingLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.use_weighted_summary = use_weighted_summary

    def build(self, input_shape):
        from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
        self.local_dense1 = Dense(4 * self.d_model)
        self.local_dense2 = Dense(self.d_model)
        self.local_dropout = Dropout(self.dropout_rate)
        self.summary_dense1 = Dense(4 * self.d_model, activation='gelu')
        self.summary_dense2 = Dense(self.d_model)
        self.summary_dropout = Dropout(self.dropout_rate)
        if self.use_weighted_summary:
            self.summary_weights = Dense(1, activation='softmax')
        self.combiner_dense1 = Dense(4 * self.d_model, activation='gelu')
        self.combiner_dense2 = Dense(self.d_model)
        self.combiner_dropout = Dropout(self.dropout_rate)
        self.dynamic_dense = Dense(self.d_model)
        self.layer_norm = LayerNormalization(epsilon=1e-6)
        super(SpatioTemporalSummaryMixingLayer, self).build(input_shape)

    def call(self, inputs, training=False):
        local_output = self.local_dense1(inputs)
        local_output, gate = tf.split(local_output, 2, axis=-1)
        local_output = local_output * tf.sigmoid(gate)
        local_output = self.local_dense2(local_output)
        local_output = self.local_dropout(local_output, training=training)
        summary = self.summary_dense1(inputs)
        summary = self.summary_dense2(summary)
        summary = self.summary_dropout(summary, training=training)
        if self.use_weighted_summary:
            weights = self.summary_weights(summary)
            weighted_summary = tf.reduce_sum(summary * weights, axis=1, keepdims=True)
        else:
            weighted_summary = tf.reduce_mean(summary, axis=1, keepdims=True)
        weighted_summary = tf.tile(weighted_summary, [1, tf.shape(inputs)[1], 1])
        combined = tf.concat([local_output, weighted_summary], axis=-1)
        output = self.combiner_dense1(combined)
        output = self.combiner_dense2(output)
        output = self.combiner_dropout(output, training=training)
        inputs_trans = self.dynamic_dense(inputs)
        return self.layer_norm(inputs_trans + output)

    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model, 'dropout_rate': self.dropout_rate, 'use_weighted_summary': self.use_weighted_summary})
        return config
