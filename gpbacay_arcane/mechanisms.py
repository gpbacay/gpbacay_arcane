from __future__ import annotations
from typing import Optional
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization

from .activations import straight_through_spike

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
        self.current_reservoir_size: Optional[tf.Variable] = None
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
        leak = float(np.clip(self.initial_leak_rate, 1e-4, 1.0 - 1e-4))
        leak_logit = np.log(leak / (1.0 - leak))
        threshold = float(max(self.initial_spike_threshold, 1e-4))
        threshold_pre = np.log(np.expm1(threshold))
        spectral_radius = float(self.spectral_radius)

        def spectral_reservoir_initializer(shape, dtype=None):
            w = np.random.randn(*shape).astype(np.float32) * 0.1
            try:
                rho = float(np.max(np.abs(np.linalg.eigvals(w))))
                if rho > 1e-8:
                    w *= spectral_radius / rho
            except np.linalg.LinAlgError:
                pass
            return tf.cast(w, dtype or tf.float32)

        self.spatiotemporal_reservoir_weights = self.add_weight(
            shape=(self.max_dynamic_reservoir_dim, self.max_dynamic_reservoir_dim),
            initializer=spectral_reservoir_initializer,
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
            initializer=tf.keras.initializers.Constant(leak_logit),
            trainable=True,
            name='leak_rate_param'
        )
        self.spike_threshold_param = self.add_weight(
            shape=(self.max_dynamic_reservoir_dim,),
            initializer=tf.keras.initializers.Constant(threshold_pre),
            trainable=True,
            name='spike_threshold_param'
        )

    def add_neurons(self, new_neurons_count):
        new_size = tf.minimum(self.current_reservoir_size + new_neurons_count, self.max_dynamic_reservoir_dim)
        self.current_reservoir_size.assign(new_size)

    def prune_connections(self, pruning_threshold=0.1):
        active_size = tf.cast(tf.convert_to_tensor(self.current_reservoir_size), tf.int32)
        active_weights = tf.slice(self.spatiotemporal_reservoir_weights, [0, 0], [active_size, active_size])
        mask = tf.abs(active_weights) < pruning_threshold
        pruned_weights = tf.where(mask, tf.zeros_like(active_weights), active_weights)
        paddings = tf.stack([
            tf.stack([tf.constant(0, dtype=tf.int32), tf.cast(self.max_dynamic_reservoir_dim, tf.int32) - active_size]),
            tf.stack([tf.constant(0, dtype=tf.int32), tf.cast(self.max_dynamic_reservoir_dim, tf.int32) - active_size]),
        ])
        padded = tf.pad(pruned_weights, paddings)
        # Keep inactive block of the original matrix intact.
        inactive_mask = tf.pad(
            tf.ones(tf.stack([active_size, active_size]), dtype=tf.float32),
            paddings,
        )
        updated = padded + self.spatiotemporal_reservoir_weights * (1.0 - inactive_mask)
        self.spatiotemporal_reservoir_weights.assign(updated)

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
        active_size = tf.cast(tf.convert_to_tensor(self.current_reservoir_size), tf.int32)
        prev_state = tf.slice(prev_state_full, [0, 0], [-1, active_size])
        active_input_weights = tf.slice(self.spatiotemporal_input_weights, [0, 0], [active_size, -1])
        active_reservoir_weights = tf.slice(self.spatiotemporal_reservoir_weights, [0, 0], [active_size, active_size])
        active_gate_weights = tf.slice(self.spiking_gate_weights, [0, 0], [3 * active_size, -1])
        leak_rate = tf.sigmoid(tf.slice(self.leak_rate_param, [0], [active_size]))
        spike_threshold = tf.nn.softplus(tf.slice(self.spike_threshold_param, [0], [active_size]))
        input_part = tf.matmul(inputs, active_input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, active_reservoir_weights)
        gate_part = tf.matmul(inputs, active_gate_weights, transpose_b=True)
        i_gate, f_gate, o_gate = tf.split(tf.sigmoid(gate_part), 3, axis=-1)
        state = (1 - leak_rate) * (f_gate * prev_state) + leak_rate * tf.tanh(i_gate * (input_part + reservoir_part))
        state = o_gate * state

        if self.use_semantic_gate:
            pad_width = tf.cast(self.max_dynamic_reservoir_dim - active_size, tf.int32)
            paddings = tf.stack([
                tf.constant([0, 0], dtype=tf.int32),
                tf.stack([tf.constant(0, dtype=tf.int32), pad_width]),
            ])
            padded_for_gate = tf.pad(state, paddings)
            combined_features = tf.concat([inputs, padded_for_gate], axis=-1)
            semantic_gate_activations = tf.matmul(combined_features, self.semantic_gate_kernel) + self.semantic_gate_bias
            semantic_gate = tf.sigmoid(semantic_gate_activations[:, :tf.cast(active_size, tf.int32)])
            state = state * semantic_gate

        spikes = straight_through_spike(state, spike_threshold)
        state = state - spikes * spike_threshold
        pad_width = tf.cast(self.max_dynamic_reservoir_dim - active_size, tf.int32)
        paddings = tf.stack([
            tf.constant([0, 0], dtype=tf.int32),
            tf.stack([tf.constant(0, dtype=tf.int32), pad_width]),
        ])
        padded_state = tf.pad(state, paddings)
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
        
        # Top-down projection: hidden state -> lower-layer expectation space.
        # feedback_weights maps to the cell's input dim; projection_kernel
        # stays in hidden space so sibling layers of equal width can align.
        self.feedback_weights = self.add_weight(
            name='feedback_weights', shape=(self.units, int(input_dim)),
            initializer='glorot_uniform', trainable=True
        )
        self.projection_kernel = self.add_weight(
            name='projection_kernel', shape=(self.units, self.units),
            initializer='glorot_uniform', trainable=True
        )
        self.projection_bias = self.add_weight(
            name='projection_bias', shape=(self.units,),
            initializer='zeros', trainable=True
        )
        self.alignment_set = self.add_weight(
            name='alignment_set', shape=(),
            initializer='zeros', trainable=False
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
    
    def project_feedback(self, state, to_input_space=False):
        """
        Top-Down Projection: P_{i→i-1} = f_proj(S_i; W_i)

        By default projects within hidden space so equal-width layers can
        align. Set `to_input_space=True` to reconstruct the cell's input
        using `feedback_weights`.
        """
        if to_input_space:
            return tf.matmul(state, self.feedback_weights)
        return tf.matmul(state, self.projection_kernel) + self.projection_bias
    
    def compute_divergence(self, current_state, projection):
        """
        Prediction Divergence: Δ_{i-1} = S_{i-1} - P_{i→i-1}
        Computes the signed difference between current state and top-down expectation.
        """
        divergence = current_state - projection
        return divergence
    
    def harmonize_state(self, current_state, divergence, gamma):
        """
        State Harmonization: h <- h - α (h - p), with α clipped to (0, 1).
        """
        alpha = tf.clip_by_value(
            tf.cast(gamma, current_state.dtype) + self.semantic_divergence_weight,
            0.0,
            0.99,
        )
        return current_state - alpha * divergence

    def resonance_loop(self, h_initial, projection_from_above=None):
        """
        Closed-form equivalent of N harmonization steps:

            h_N = (1-α)^N h + (1-(1-α)^N) p

        Skips alignment when no top-down target has been set, so the first
        forward pass does not pull representations toward zero.
        """
        if projection_from_above is None:
            return h_initial

        align = tf.reshape(tf.cast(projection_from_above, h_initial.dtype), [1, self.units])
        alignment_energy = tf.reduce_sum(tf.square(align)) + self.alignment_set

        def _apply():
            alpha = tf.clip_by_value(
                tf.cast(self.resonance_factor, h_initial.dtype) + self.semantic_divergence_weight,
                0.0,
                0.99,
            )
            n = tf.cast(self.resonance_cycles, h_initial.dtype)
            decay = tf.pow(1.0 - alpha, n)
            h_resonated = decay * h_initial + (1.0 - decay) * align
            final_delta = h_resonated - align
            self.global_divergence.assign(tf.reduce_mean(tf.square(final_delta)))
            return h_resonated

        return tf.cond(alignment_energy > 0.0, _apply, lambda: h_initial)
        
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
        
        # === Step 4: Spiking Mechanism (STE so spike decisions train) ===
        spikes = straight_through_spike(h_modulated, self.spike_threshold)
        h_final = h_modulated - spikes * self.spike_threshold

        # === Step 5: Track a slow prototype of the hidden state ===
        batch_mean = tf.reduce_mean(h_final, axis=0)
        self.last_h.assign(0.9 * self.last_h + 0.1 * batch_mean)
        
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
        Closed-form local resonance: h_N = (1-η)^N h + (1-(1-η)^N) alignment.
        """
        if alignment is None:
            return h_initial

        step = tf.clip_by_value(tf.cast(self.resonance_step_size, h_initial.dtype), 0.0, 0.99)
        n = tf.cast(self.resonance_cycles, h_initial.dtype)
        decay = tf.pow(1.0 - step, n)
        h_current = decay * h_initial + (1.0 - decay) * alignment
        self.global_divergence.assign(tf.reduce_mean(tf.square(h_current - alignment)))
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

        spikes = straight_through_spike(h_modulated, self.spike_threshold)
        h_final = h_modulated - spikes * self.spike_threshold

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
        config = super().get_config()
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
        # Feature map φ(x) = elu(x)+1 as in Katharopoulos et al. linear attention.
        # Compute φ(Q)(φ(K)^T V) / φ(Q)(φ(K)^T 1) in O(n d^2), not QK^T in O(n^2).
        queries = tf.nn.elu(queries) + 1.0
        keys = tf.nn.elu(keys) + 1.0
        kv = tf.einsum("bhsd,bhse->bhde", keys, values)
        normalizer = tf.reduce_sum(keys, axis=2)
        numerator = tf.einsum("bhsd,bhde->bhse", queries, kv)
        denominator = tf.einsum("bhsd,bhd->bhs", queries, normalizer)
        attention_output = numerator / (tf.expand_dims(denominator, -1) + self.eps)
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


def build_rope_cache(max_position, head_dim, base=10000.0):
    """Precompute (cos, sin) rotary tables of shape ``(max_position, head_dim // 2)``."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (np.arange(0, half, dtype=np.float64) / float(half)))
    angles = np.outer(np.arange(max_position, dtype=np.float64), inv_freq)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def apply_rope(x, cos, sin):
    """Rotate ``(batch, heads, seq, head_dim)`` using the split-half convention.

    ``(x1, x2) -> (x1 cos - x2 sin, x2 cos + x1 sin)``, matching the reference
    Llama/Qwen RoPE so a distilled student sees the same positional geometry as
    the teacher.
    """
    seq_len = tf.shape(x)[2]
    # Materialise as tensors first: the caches are stored as numpy, and inside a
    # tf.function ``seq_len`` is a Tensor, which numpy cannot slice with.
    cos_t = tf.convert_to_tensor(cos, dtype=x.dtype)
    sin_t = tf.convert_to_tensor(sin, dtype=x.dtype)
    # expand_dims rather than [None, None]: new-axis slicing has no TFLite builtin.
    c = tf.expand_dims(tf.expand_dims(cos_t[:seq_len], 0), 0)
    s = tf.expand_dims(tf.expand_dims(sin_t[:seq_len], 0), 0)
    x1, x2 = tf.split(x, 2, axis=-1)
    return tf.concat([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


class RMSNorm(Layer):
    """Root-mean-square layer norm (no mean subtraction, no bias), as used by Qwen."""

    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma", shape=(input_shape[-1],), initializer="ones", trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        variance = tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True)
        return inputs * tf.math.rsqrt(variance + self.eps) * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


class CausalLinearSelfAttention(Layer):
    """Chunkwise causal linear attention with optional per-head decay and RoPE.

    Feature map phi(x) = elu(x) + 1 (Katharopoulos et al.). Mathematically this is
    the same causal prefix sum over the outer products phi(k_i) v_i as a naive
    ``cumsum``, but the (d_head, d_head) state is materialised once per *chunk*
    rather than once per *token*. That takes activation memory from
    O(n . d_head^2) down to O(d_head^2 + n . chunk) -- for the 100M preset at
    seq_len 256 the old form held 2.4 GB of prefix tensors at batch 2, which is
    what made distillation-scale batches impossible.

    ``use_decay`` adds a learned per-head forget factor gamma in (0, 1) so the KV
    state stops weighting the entire prefix uniformly.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        dropout_rate=0.1,
        use_semantic_reweighting=True,
        eps=1e-6,
        chunk_size=64,
        use_decay=False,
        use_rope=False,
        rope_base=10000.0,
        max_position=2048,
        reweight_centered=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_semantic_reweighting = use_semantic_reweighting
        self.eps = eps
        self.chunk_size = int(chunk_size)
        self.use_decay = use_decay
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.max_position = int(max_position)
        self.reweight_centered = reweight_centered
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.depth = d_model // num_heads
        if self.use_rope and self.depth % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=eps)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def build(self, input_shape):
        d_model = self.d_model
        self.query_weight = self.add_weight(
            name="query_weight", shape=(d_model, d_model), initializer="glorot_uniform", trainable=True
        )
        self.query_bias = self.add_weight(
            name="query_bias", shape=(d_model,), initializer="zeros", trainable=True
        )
        self.key_weight = self.add_weight(
            name="key_weight", shape=(d_model, d_model), initializer="glorot_uniform", trainable=True
        )
        self.key_bias = self.add_weight(
            name="key_bias", shape=(d_model,), initializer="zeros", trainable=True
        )
        self.value_weight = self.add_weight(
            name="value_weight", shape=(d_model, d_model), initializer="glorot_uniform", trainable=True
        )
        self.value_bias = self.add_weight(
            name="value_bias", shape=(d_model,), initializer="zeros", trainable=True
        )
        self.output_weight = self.add_weight(
            name="output_weight", shape=(d_model, d_model), initializer="glorot_uniform", trainable=True
        )
        self.output_bias = self.add_weight(
            name="output_bias", shape=(d_model,), initializer="zeros", trainable=True
        )
        if self.use_semantic_reweighting:
            self.semantic_reweight_kernel = self.add_weight(
                name="semantic_reweight_kernel",
                shape=(d_model, 1),
                initializer="glorot_uniform",
                trainable=True,
            )
            self.semantic_reweight_bias = self.add_weight(
                name="semantic_reweight_bias", shape=(1,), initializer="zeros", trainable=True
            )
        if self.use_decay:
            # sigmoid(4.0) ~ 0.982: close to no forgetting at init, free to sharpen.
            self.decay_logit = self.add_weight(
                name="decay_logit",
                shape=(self.num_heads,),
                initializer=tf.keras.initializers.Constant(4.0),
                trainable=True,
            )
        if self.use_rope:
            # Stored as numpy: a tf.constant created in build() belongs to the
            # build scratch graph and is out of scope when call() is traced.
            self.rope_cos, self.rope_sin = build_rope_cache(
                self.max_position, self.depth, self.rope_base
            )
        # Static exponent tables for the intra-chunk decay mask.
        c = self.chunk_size
        idx = np.arange(c)
        self._tri = (idx[:, None] >= idx[None, :]).astype(np.float32)
        self._mask_exp = (idx[:, None] - idx[None, :]).astype(np.float32)
        self._q_exp = (idx + 1).astype(np.float32)
        self._tail_exp = (c - 1 - idx).astype(np.float32)
        self.layer_norm.build(input_shape)
        super().build(input_shape)

    def _split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _decay_tables(self, dtype):
        """Per-head (query decay, intra-chunk mask, tail weights, whole-chunk decay)."""
        c = self.chunk_size
        tri = tf.constant(self._tri, dtype=dtype)
        if not self.use_decay:
            ones_q = tf.ones((self.num_heads, c), dtype=dtype)
            mask = tf.tile(tri[None], [self.num_heads, 1, 1])
            return ones_q, mask, ones_q, tf.ones((self.num_heads,), dtype=dtype)
        gamma = tf.cast(tf.sigmoid(self.decay_logit), dtype)[:, None]
        decay_q = tf.pow(gamma, tf.constant(self._q_exp, dtype=dtype)[None, :])
        mask = tf.pow(gamma[:, :, None], tf.constant(self._mask_exp, dtype=dtype)[None]) * tri[None]
        tail = tf.pow(gamma, tf.constant(self._tail_exp, dtype=dtype)[None, :])
        chunk_decay = tf.pow(tf.squeeze(gamma, -1), tf.cast(c, dtype))
        return decay_q, mask, tail, chunk_decay

    def _single_chunk_attention(self, q, k, v, seq_len: int):
        """Causal linear attention when the whole sequence fits in one chunk.

        Identical to one ``tf.scan`` iteration with a zero carry, without the
        scan overhead that dominates CPU step time at this scale.
        """
        c = self.chunk_size
        dtype = q.dtype
        pad = (c - seq_len) % c
        q = tf.pad(q, [[0, 0], [0, 0], [0, pad], [0, 0]])
        k = tf.pad(k, [[0, 0], [0, 0], [0, pad], [0, 0]])
        v = tf.pad(v, [[0, 0], [0, 0], [0, pad], [0, 0]])
        _, mask, _, _ = self._decay_tables(dtype)
        scores = tf.einsum("bhcd,bhkd->bhck", q, k) * mask[None]
        out = tf.einsum("bhck,bhke->bhce", scores, v)
        den = tf.reduce_sum(scores, axis=-1, keepdims=True)
        return (out / (den + self.eps))[:, :, :seq_len, :]

    def _chunked_attention(self, q, k, v, seq_len):
        """Scan chunk-by-chunk, carrying the (d_head, d_head) KV state forward."""
        c = self.chunk_size
        dtype = q.dtype
        pad = (-seq_len) % c
        q = tf.pad(q, [[0, 0], [0, 0], [0, pad], [0, 0]])
        k = tf.pad(k, [[0, 0], [0, 0], [0, pad], [0, 0]])
        v = tf.pad(v, [[0, 0], [0, 0], [0, pad], [0, 0]])
        batch = tf.shape(q)[0]
        heads, depth = self.num_heads, self.depth
        n_chunks = (seq_len + pad) // c

        def to_chunks(t):
            t = tf.reshape(t, (batch, heads, n_chunks, c, depth))
            return tf.transpose(t, perm=[2, 0, 1, 3, 4])

        qs, ks, vs = to_chunks(q), to_chunks(k), to_chunks(v)
        decay_q, mask, tail, chunk_decay = self._decay_tables(dtype)
        decay_q_n = decay_q[None, :, :, None]
        decay_q_d = decay_q[None, :, :]
        mask_b = mask[None]
        tail_b = tail[None, :, :, None]
        chunk_kv = chunk_decay[None, :, None, None]
        chunk_z = chunk_decay[None, :, None]

        def step(carry, elems):
            _, kv_state, z_state = carry
            q_c, k_c, v_c = elems
            inter_num = tf.einsum("bhcd,bhde->bhce", q_c, kv_state) * decay_q_n
            inter_den = tf.einsum("bhcd,bhd->bhc", q_c, z_state) * decay_q_d
            scores = tf.einsum("bhcd,bhkd->bhck", q_c, k_c) * mask_b
            intra_num = tf.einsum("bhck,bhke->bhce", scores, v_c)
            intra_den = tf.reduce_sum(scores, axis=-1)
            out = (inter_num + intra_num) / (tf.expand_dims(inter_den + intra_den, -1) + self.eps)
            k_w = k_c * tail_b
            kv_next = kv_state * chunk_kv + tf.einsum("bhkd,bhke->bhde", k_w, v_c)
            z_next = z_state * chunk_z + tf.reduce_sum(k_w, axis=2)
            return out, kv_next, z_next

        init = (
            tf.zeros((batch, heads, c, depth), dtype=dtype),
            tf.zeros((batch, heads, depth, depth), dtype=dtype),
            tf.zeros((batch, heads, depth), dtype=dtype),
        )
        outs, _, _ = tf.scan(step, (qs, ks, vs), initializer=init)
        outs = tf.transpose(outs, perm=[1, 2, 0, 3, 4])
        outs = tf.reshape(outs, (batch, heads, n_chunks * c, depth))
        return outs[:, :, :seq_len, :]

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        queries = tf.matmul(inputs, self.query_weight) + self.query_bias
        keys = tf.matmul(inputs, self.key_weight) + self.key_bias
        values = tf.matmul(inputs, self.value_weight) + self.value_bias
        queries = self._split_heads(queries, batch_size)
        keys = self._split_heads(keys, batch_size)
        values = self._split_heads(values, batch_size)
        if self.use_rope:
            # Rotate before the feature map so phi(.) stays non-negative and the
            # linear-attention denominator cannot change sign.
            queries = apply_rope(queries, self.rope_cos, self.rope_sin)
            keys = apply_rope(keys, self.rope_cos, self.rope_sin)
        queries = tf.nn.elu(queries) + 1.0
        keys = tf.nn.elu(keys) + 1.0
        static_len = inputs.shape[1]
        if static_len is not None and self.chunk_size >= int(static_len):
            attention_output = self._single_chunk_attention(
                queries, keys, values, int(static_len)
            )
        else:
            attention_output = self._chunked_attention(queries, keys, values, seq_len)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, seq_len, self.d_model))
        output = tf.matmul(attention_output, self.output_weight) + self.output_bias
        if self.use_semantic_reweighting:
            reweight_factors = tf.sigmoid(
                tf.matmul(output, self.semantic_reweight_kernel) + self.semantic_reweight_bias
            )
            # Centred form spans (0, 2) with unit gain at init; the original
            # (0, 1) form can only attenuate and halves the branch at init.
            if self.reweight_centered:
                reweight_factors = 2.0 * reweight_factors
            output = output * reweight_factors
        output = self.dropout(output, training=training)
        return self.layer_norm(inputs + output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "use_semantic_reweighting": self.use_semantic_reweighting,
                "eps": self.eps,
                "chunk_size": self.chunk_size,
                "use_decay": self.use_decay,
                "use_rope": self.use_rope,
                "rope_base": self.rope_base,
                "max_position": self.max_position,
                "reweight_centered": self.reweight_centered,
            }
        )
        return config


class CausalSoftmaxSelfAttention(Layer):
    """Standard causal softmax attention with RoPE, for hybrid ARCANE decoders.

    Linear attention compresses the whole prefix into a fixed d_head x d_head
    state and so cannot do exact key lookup; a softmax teacher can. Interleaving
    a minority of these layers restores recall without paying quadratic cost at
    every depth.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        dropout_rate=0.1,
        eps=1e-6,
        use_rope=True,
        rope_base=10000.0,
        max_position=2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.eps = eps
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.max_position = int(max_position)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.depth = d_model // num_heads
        if self.use_rope and self.depth % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=eps)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def build(self, input_shape):
        d_model = self.d_model
        for name in ("query", "key", "value", "output"):
            setattr(
                self,
                f"{name}_weight",
                self.add_weight(
                    name=f"{name}_weight",
                    shape=(d_model, d_model),
                    initializer="glorot_uniform",
                    trainable=True,
                ),
            )
            setattr(
                self,
                f"{name}_bias",
                self.add_weight(
                    name=f"{name}_bias", shape=(d_model,), initializer="zeros", trainable=True
                ),
            )
        if self.use_rope:
            # Stored as numpy: a tf.constant created in build() belongs to the
            # build scratch graph and is out of scope when call() is traced.
            self.rope_cos, self.rope_sin = build_rope_cache(
                self.max_position, self.depth, self.rope_base
            )
        self.layer_norm.build(input_shape)
        super().build(input_shape)

    def _split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        q = self._split_heads(tf.matmul(inputs, self.query_weight) + self.query_bias, batch_size)
        k = self._split_heads(tf.matmul(inputs, self.key_weight) + self.key_bias, batch_size)
        v = self._split_heads(tf.matmul(inputs, self.value_weight) + self.value_bias, batch_size)
        if self.use_rope:
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.depth, inputs.dtype))
        # tf.where rather than ``scores += (1 - causal) * -1e9``: on CPU, oneDNN
        # fuses that broadcast add into the batch matmul and runs ~7x slower.
        causal = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.bool), -1, 0)
        scores = tf.where(causal[None, None, :, :], scores, tf.cast(-1e9, inputs.dtype))
        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights, training=training)
        context = tf.matmul(weights, v)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, seq_len, self.d_model))
        output = tf.matmul(context, self.output_weight) + self.output_bias
        output = self.dropout(output, training=training)
        return self.layer_norm(inputs + output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "eps": self.eps,
                "use_rope": self.use_rope,
                "rope_base": self.rope_base,
                "max_position": self.max_position,
            }
        )
        return config

class ResonantSequenceMixer(Layer):
    """Token-parallel closed-form resonance from ResonantGSERCell, made causal.

    Harmonizes each token toward the causal running mean of the sequence
    (a prefix prototype), then applies STE subtractive reset. No RNN unroll,
    so it can sit inside a 100M-scale decoder block.
    """

    def __init__(
        self,
        d_model,
        resonance_factor=0.15,
        resonance_cycles=3,
        spike_threshold=0.4,
        semantic_divergence_weight=0.1,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.resonance_factor = resonance_factor
        self.resonance_cycles = resonance_cycles
        self.spike_threshold = spike_threshold
        self.semantic_divergence_weight = semantic_divergence_weight
        self.eps = eps
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=eps)

    def build(self, input_shape):
        self.projection_kernel = self.add_weight(
            name="projection_kernel",
            shape=(self.d_model, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.resonance_gate = self.add_weight(
            name="resonance_gate",
            shape=(self.d_model,),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True,
        )
        self.resonance_bias = self.add_weight(
            name="resonance_bias",
            shape=(self.d_model,),
            initializer="zeros",
            trainable=True,
        )
        self.layer_norm.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=False):
        seq_len = tf.shape(inputs)[1]
        dtype = inputs.dtype
        steps = tf.cast(tf.range(1, seq_len + 1), dtype)[None, :, None]
        prefix_mean = tf.cumsum(inputs, axis=1) / steps
        prototype = tf.matmul(prefix_mean, self.projection_kernel)
        alpha = tf.clip_by_value(
            tf.cast(self.resonance_factor, dtype) + tf.cast(self.semantic_divergence_weight, dtype),
            0.0,
            0.99,
        )
        decay = tf.pow(1.0 - alpha, tf.cast(self.resonance_cycles, dtype))
        h_resonated = decay * inputs + (1.0 - decay) * prototype
        res_mod = tf.sigmoid(self.resonance_gate) * tf.cast(self.resonance_factor, dtype)
        h_modulated = h_resonated * (1.0 + res_mod) + self.resonance_bias
        spikes = straight_through_spike(h_modulated, self.spike_threshold)
        h_final = h_modulated - spikes * tf.cast(self.spike_threshold, dtype)
        return self.layer_norm(inputs + h_final)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "resonance_factor": self.resonance_factor,
                "resonance_cycles": self.resonance_cycles,
                "spike_threshold": self.spike_threshold,
                "semantic_divergence_weight": self.semantic_divergence_weight,
                "eps": self.eps,
            }
        )
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
        self.local_dense1 = Dense(4 * self.d_model)
        self.local_dense2 = Dense(self.d_model)
        self.local_dropout = Dropout(self.dropout_rate)
        self.summary_dense1 = Dense(4 * self.d_model, activation='gelu')
        self.summary_dense2 = Dense(self.d_model)
        self.summary_dropout = Dropout(self.dropout_rate)
        if self.use_weighted_summary:
            self.summary_weights = Dense(1)
        self.combiner_dense1 = Dense(4 * self.d_model, activation='gelu')
        self.combiner_dense2 = Dense(self.d_model)
        self.combiner_dropout = Dropout(self.dropout_rate)
        self.dynamic_dense = Dense(self.d_model)
        self.layer_norm = LayerNormalization(epsilon=1e-6)
        super().build(input_shape)

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
            scores = self.summary_weights(summary)
            weights = tf.nn.softmax(scores, axis=1)
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


class AttentionResidual(Layer):
    """
    Attention Residuals (AttnRes) for Arcane.
    
    Replaces standard fixed-weight additive residual connections with learned,
    input-dependent softmax attention over the full depth of preceding layer outputs.
    
    Based on the 'Attention Residuals' paper (Kimi Team, 2026):
       h_l = Σ α_{i→l} · v_i,   where α_{i→l} = softmax(w_l · RMSNorm(v_i))
    
    Key advantages over standard residuals:
    - Selective depth aggregation: each layer can selectively retrieve earlier
      layer outputs rather than blindly accumulating all prior info.
    - Bounded output magnitudes: prevents the PreNorm dilution problem where
      hidden-state magnitudes grow as O(L) with depth.
    - Uniform gradient flow: softmax competition prevents gradient over-concentration
      in the earliest layers.
    - Enables learned skip connections across depth (e.g. early embedding retrieval).
    
    Usage:
        history = [embed_out]
        for block in blocks:
            x = AttentionResidual(d_model)(history)  # selective read
            x = block(x)
            history.append(x)
    """
    def __init__(self, d_model, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.eps = eps

    def build(self, _):
        # Pseudo-query vector w_l: zero-init → uniform weighting at start (safe training onset)
        self.wl = self.add_weight(
            shape=(self.d_model,),
            initializer='zeros',
            trainable=True,
            name='pseudo_query'
        )
        # RMSNorm scaling parameter for keys
        self.gamma = self.add_weight(
            shape=(self.d_model,),
            initializer='ones',
            trainable=True,
            name='rmsnorm_gamma'
        )
        super().build(_)

    def call(self, history):
        """
        Args:
            history: list of tensors, each shape (batch_size, ..., d_model).
                     Represents v_0, v_1, ..., v_{l-1} (embedding + all prior layer outputs).
        Returns:
            Tensor of shape (batch_size, ..., d_model): selectively aggregated representation.
        """
        V = tf.stack(history, axis=-2)   # (..., L, d_model)
        rms = tf.sqrt(tf.reduce_mean(V ** 2, axis=-1, keepdims=True) + self.eps)
        K = (V / rms) * self.gamma
        scores = tf.einsum('d,...ld->...l', self.wl, K)
        alpha = tf.nn.softmax(scores, axis=-1)
        out = tf.einsum('...l,...ld->...d', alpha, V)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model, 'eps': self.eps})
        return config


class BlockAttentionResidual(Layer):
    """
    Block Attention Residuals (Block AttnRes) for Arcane.
    
    A memory-efficient variant of AttentionResidual. Instead of attending over
    all L individual layer outputs (O(Ld) memory), layers are partitioned into N
    blocks. Within each block, outputs are summed into a single block representation 
    b_n. Across blocks, softmax attention is applied over the N block-level summaries
    plus the current partial intra-block accumulation.
    
    Memory: O(Nd) instead of O(Ld) — practical at scale with N ≈ 8.
    
    Training behaviour:
    - Intra-block: standard additive residuals (local)
    - Inter-block: learned softmax attention over block summaries (global)
    
    Usage:
        completed_blocks = [embed_block_rep]  # b_0 = embedding block
        partial = embed_out
        for i, sublayer in enumerate(sublayers):
            x = BlockAttentionResidual(d_model)([completed_blocks, partial])
            x = sublayer(x)
            partial = partial + x
            if end_of_block:
                completed_blocks.append(partial)
                partial = zero
    """
    def __init__(self, d_model, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.eps = eps

    def build(self, _):
        self.wl = self.add_weight(
            shape=(self.d_model,),
            initializer='zeros',
            trainable=True,
            name='pseudo_query'
        )
        self.gamma = self.add_weight(
            shape=(self.d_model,),
            initializer='ones',
            trainable=True,
            name='rmsnorm_gamma'
        )
        super().build(_)

    def call(self, inputs):
        """
        Args:
            inputs: tuple/list of (completed_blocks, partial_block)
                - completed_blocks: list of tensors, each (batch, d_model) — one per finished block
                - partial_block: tensor (batch, d_model) — intra-block accumulation so far
        Returns:
            Tensor (batch, d_model): aggregated from all block sources.
        """
        completed_blocks, partial_block = inputs
        all_sources = list(completed_blocks) + [partial_block]

        V = tf.stack(all_sources, axis=-2)        # (..., N+1, d_model)
        rms = tf.sqrt(tf.reduce_mean(V ** 2, axis=-1, keepdims=True) + self.eps)
        K = (V / rms) * self.gamma
        scores = tf.einsum('d,...nd->...n', self.wl, K)
        alpha = tf.nn.softmax(scores, axis=-1)
        out = tf.einsum('...n,...nd->...d', alpha, V)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model, 'eps': self.eps})
        return config


class ConceptEngram(Layer):
    """Hashed n-gram lookup memory written into the residual stream.

    Most capacity lives in gatherable table rows rather than dense matmuls:
    each token hashes local n-grams, gathers table rows, and gates the write
    with a cheap DenseGSER-style sigmoid. Arithmetic cost is O(n_rows * d),
    independent of table size beyond the gather.
    """

    def __init__(
        self,
        d_model,
        table_size=4096,
        ngram_sizes=(2, 3),
        rows_per_token=8,
        hash_base=1000003,
        leak_rate=0.1,
        spike_threshold=0.4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.table_size = int(table_size)
        self.ngram_sizes = tuple(int(n) for n in ngram_sizes)
        self.rows_per_token = int(rows_per_token)
        self.hash_base = int(hash_base)
        self.leak_rate = float(leak_rate)
        self.spike_threshold = float(spike_threshold)

    def build(self, input_shape):
        self.table = self.add_weight(
            name="engram_table",
            shape=(self.table_size, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.gate_kernel = self.add_weight(
            name="gate_kernel",
            shape=(self.d_model, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.gate_bias = self.add_weight(
            name="gate_bias",
            shape=(self.d_model,),
            initializer="zeros",
            trainable=True,
        )
        self.value_scale = self.add_weight(
            name="value_scale",
            shape=(self.d_model,),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
        )
        super().build(input_shape)

    _PRIME = 2147483647  # 2^31 - 1; keeps every intermediate product inside int64

    def _ngram_keys(self, token_ids, n):
        """Rolling polynomial hash of n-grams, reduced mod a Mersenne prime.

        Reducing inside the loop keeps ``acc * base`` below 2^62 for any n.
        """
        ids = tf.cast(token_ids, tf.int64)
        batch = tf.shape(ids)[0]
        seq = tf.shape(ids)[1]
        prime = tf.constant(self._PRIME, dtype=tf.int64)
        base = tf.constant(self.hash_base % self._PRIME, dtype=tf.int64)
        pad = max(n - 1, 0)
        padded = tf.pad(ids, [[0, 0], [pad, 0]])
        acc = tf.fill((batch, seq), tf.constant(n, dtype=tf.int64))
        for offset in range(n):
            sl = padded[:, offset : offset + seq]
            acc = (acc * base + sl) % prime
        return acc

    def _lookup_keys(self, token_ids):
        """``rows_per_token`` independent universal hashes over the n-gram keys.

        Row r uses n-gram family ``r % len(ngram_sizes)`` and its own (a_r, b_r),
        so two n-grams that collide in one row almost never collide in the others.
        """
        families = [self._ngram_keys(token_ids, n) for n in self.ngram_sizes]
        prime = tf.constant(self._PRIME, dtype=tf.int64)
        table = tf.constant(self.table_size, dtype=tf.int64)
        rows = []
        for r in range(self.rows_per_token):
            a = tf.constant(2 * (40503 * (r + 1) % 65521) + 1, dtype=tf.int64)
            b = tf.constant(2654435761 * (r + 7) % self._PRIME, dtype=tf.int64)
            src = families[r % len(families)]
            rows.append(((src * a + b) % prime) % table)
        return tf.stack(rows, axis=-1)

    def call(self, inputs, token_ids=None, training=False):
        """``inputs`` is (B, T, D). ``token_ids`` optional (B, T)."""
        x = inputs
        dtype = x.dtype
        if token_ids is None:
            pseudo = tf.cast(
                tf.reduce_sum(tf.cast(x * 97.0, tf.float32), axis=-1),
                tf.int64,
            )
            token_ids = tf.math.abs(pseudo)
        keys = self._lookup_keys(token_ids)  # (B, T, R)
        gathered = tf.gather(self.table, keys)  # (B, T, R, D)
        memory = tf.reduce_mean(tf.cast(gathered, dtype), axis=2)
        memory = memory * tf.cast(self.value_scale, dtype)
        gate_pre = tf.matmul(x, tf.cast(self.gate_kernel, dtype)) + tf.cast(self.gate_bias, dtype)
        sharpness = 1.0 / tf.maximum(tf.cast(self.leak_rate, dtype), 1e-3)
        gate = tf.nn.sigmoid(sharpness * (gate_pre - tf.cast(self.spike_threshold, dtype)))
        return x + gate * memory

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "table_size": self.table_size,
                "ngram_sizes": self.ngram_sizes,
                "rows_per_token": self.rows_per_token,
                "hash_base": self.hash_base,
                "leak_rate": self.leak_rate,
                "spike_threshold": self.spike_threshold,
            }
        )
        return config



# ---------------------------------------------------------------------------
# ARC 1 — Resonant Schema Binding mechanisms
# ---------------------------------------------------------------------------
_MASK_NEG = -1e9


class FieldAttention(Layer):
    """Bidirectional, pad-aware multi-head attention with RoPE.

    ARC 1 perceives a short utterance as one standing "field": every token may
    attend to every other real token (no causal mask), and right padding is
    masked out as keys so outputs are identical with or without padding.
    """

    def __init__(self, d_model, num_heads, dropout_rate=0.0, use_rope=True,
                 rope_base=10000.0, max_position=1024, causal=False, **kwargs):
        super().__init__(**kwargs)
        self.causal = bool(causal)  # True: token t sees only tokens <= t (ARC 1 LM)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.depth = self.d_model // self.num_heads
        if use_rope and self.depth % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        self.dropout_rate = float(dropout_rate)
        self.use_rope = bool(use_rope)
        self.rope_base = float(rope_base)
        self.max_position = int(max_position)
        self.qkv = Dense(3 * self.d_model, name="qkv")
        self.out = Dense(self.d_model, name="out")
        self.dropout = Dropout(self.dropout_rate)

    def build(self, input_shape):
        self.qkv.build(input_shape)
        self.out.build(input_shape)
        if self.use_rope:
            self.rope_cos, self.rope_sin = build_rope_cache(self.max_position, self.depth, self.rope_base)
        super().build(input_shape)

    def call(self, inputs, token_mask=None, training=False):
        batch = tf.shape(inputs)[0]
        seq = tf.shape(inputs)[1]
        qkv = tf.reshape(self.qkv(inputs), (batch, seq, 3, self.num_heads, self.depth))
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])  # (3, B, H, T, dh)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_rope:
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.depth, inputs.dtype))
        if token_mask is not None:
            keys = tf.reshape(tf.cast(token_mask, tf.bool), (batch, 1, 1, seq))
            scores = tf.where(keys, scores, tf.cast(_MASK_NEG, inputs.dtype))
        if self.causal:
            past = tf.linalg.band_part(tf.ones((seq, seq), dtype=tf.bool), -1, 0)
            scores = tf.where(past, scores, tf.cast(_MASK_NEG, inputs.dtype))
        weights = self.dropout(tf.nn.softmax(scores, axis=-1), training=training)
        context = tf.transpose(tf.matmul(weights, v), perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch, seq, self.d_model))
        return self.dropout(self.out(context), training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model, "num_heads": self.num_heads, "dropout_rate": self.dropout_rate,
            "use_rope": self.use_rope, "rope_base": self.rope_base, "max_position": self.max_position,
            "causal": self.causal,
        })
        return config


class FieldResonance(Layer):
    """Global (bidirectional) resonance over a padded token field.

    Every real token is harmonized toward the field prototype (the projected,
    masked mean of all real tokens) for ``resonance_cycles`` closed-form cycles,
    then passes a straight-through spike with subtractive reset. Padding never
    enters the prototype, so the layer is invariant to right padding.
    """

    def __init__(self, d_model, resonance_factor=0.15, resonance_cycles=3, spike_threshold=0.4,
                 causal=False, **kwargs):
        super().__init__(**kwargs)
        self.causal = bool(causal)  # True: prototype is the running mean of tokens <= t
        self.d_model = int(d_model)
        self.resonance_factor = float(resonance_factor)
        self.resonance_cycles = int(resonance_cycles)
        self.spike_threshold = float(spike_threshold)

    def build(self, input_shape):
        self.projection_kernel = self.add_weight(
            name="projection_kernel", shape=(self.d_model, self.d_model),
            initializer="glorot_uniform", trainable=True)
        self.resonance_gate = self.add_weight(
            name="resonance_gate", shape=(self.d_model,),
            initializer=tf.keras.initializers.Constant(1.0), trainable=True)
        self.resonance_bias = self.add_weight(
            name="resonance_bias", shape=(self.d_model,), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, inputs, token_mask=None, training=False):
        dtype = inputs.dtype
        if token_mask is None:
            token_mask = tf.ones(tf.shape(inputs)[:2], dtype=tf.bool)
        m = tf.expand_dims(tf.cast(token_mask, dtype), -1)
        if self.causal:
            prototype = tf.cumsum(inputs * m, axis=1) / tf.maximum(tf.cumsum(m, axis=1), 1.0)
        else:
            prototype = tf.reduce_sum(inputs * m, axis=1, keepdims=True) / tf.maximum(
                tf.reduce_sum(m, axis=1, keepdims=True), 1.0)
        prototype = tf.matmul(prototype, self.projection_kernel)
        alpha = tf.clip_by_value(tf.cast(self.resonance_factor, dtype), 0.0, 0.99)
        decay = tf.pow(1.0 - alpha, tf.cast(self.resonance_cycles, dtype))
        h = decay * inputs + (1.0 - decay) * prototype
        h = h * (1.0 + tf.sigmoid(self.resonance_gate) * alpha) + self.resonance_bias
        spikes = straight_through_spike(h, self.spike_threshold)
        return h - spikes * tf.cast(self.spike_threshold, dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model, "resonance_factor": self.resonance_factor,
            "resonance_cycles": self.resonance_cycles, "spike_threshold": self.spike_threshold,
            "causal": self.causal,
        })
        return config


class ResonantBinding(Layer):
    """Bind schema probes to an utterance field through iterative resonance.

    Each probe ``p`` (a tool, parameter, or enum-option engram) listens to the
    utterance field ``U`` for ``cycles`` rounds with shared weights:

        a_c = softmax_t( (W_q p_{c-1}) . (W_k U_t) )           where it resonates
        r_c = W_l sum_t a_c[t] W_v U_t                         what it hears
        g_c = sigmoid( (W_g [p_{c-1}; r_c] - theta) / leak )   GSER spiking gate
        p_c = p' + Mixer(p'),   p' = p_{c-1} + g_c * r_c

    Keys and values are computed once per utterance and shared by all probes,
    so binding P probes costs O(P * T * d) on top of a single encoder pass.
    Fewer cycles means less compute, and every cycle count is trained.

    Returns the settled probe state, the last heard signal, and the peak
    per-head resonance score (a direct "something here matches me" signal).
    """

    def __init__(self, d_model, num_heads=4, cycles=3, leak_rate=0.1, spike_threshold=0.4,
                 mixer_rank_div=2, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.depth = self.d_model // self.num_heads
        self.cycles = int(cycles)
        self.leak_rate = float(leak_rate)
        self.spike_threshold = float(spike_threshold)
        self.mixer_rank_div = int(mixer_rank_div)
        self.dropout_rate = float(dropout_rate)
        self.inner = max(self.d_model // max(self.mixer_rank_div, 1), 16)
        self.q_norm = RMSNorm(name="probe_norm")
        self.query = Dense(self.d_model, use_bias=False, name="query")
        self.key = Dense(self.d_model, use_bias=False, name="key")
        self.value = Dense(self.d_model, use_bias=False, name="value")
        self.listen = Dense(self.d_model, name="listen")
        self.gate = Dense(self.d_model, name="gate")
        self.mix_down = Dense(self.inner, activation="gelu", name="mix_down")
        self.mix_up = Dense(self.d_model, name="mix_up")
        self.mix_norm = RMSNorm(name="mix_norm")
        self.dropout = Dropout(self.dropout_rate)

    def build(self, input_shape):
        d = self.d_model
        for layer, width in ((self.q_norm, d), (self.query, d), (self.key, d), (self.value, d),
                             (self.listen, d), (self.gate, 2 * d), (self.mix_down, d),
                             (self.mix_up, self.inner), (self.mix_norm, d)):
            if not layer.built:  # field_memory() may already have built key/value
                layer.build((None, width))
        self.gate_threshold = self.add_weight(
            name="gate_threshold", shape=(d,),
            initializer=tf.keras.initializers.Constant(self.spike_threshold), trainable=True)
        super().build(input_shape)

    def field_memory(self, field):
        """Per-utterance keys/values ``(B, H, T, dh)`` shared by every probe."""
        batch = tf.shape(field)[0]
        seq = tf.shape(field)[1]

        def heads(x):
            return tf.transpose(tf.reshape(x, (batch, seq, self.num_heads, self.depth)), [0, 2, 1, 3])

        return heads(self.key(field)), heads(self.value(field))

    def call(self, probes, keys, values, token_mask, probe_example, cycles=None, training=False):
        """``probes`` (P, D); ``keys``/``values`` (B, H, T, dh); ``token_mask`` (B, T);
        ``probe_example`` (P,) int, the utterance each probe binds to."""
        cycles = self.cycles if cycles is None else int(max(1, min(int(cycles), self.cycles)))
        dtype = probes.dtype
        k = tf.gather(keys, probe_example)      # (P, H, T, dh)
        v = tf.gather(values, probe_example)
        mask = tf.expand_dims(tf.gather(tf.cast(token_mask, tf.bool), probe_example), 1)  # (P, 1, T)
        n_probe = tf.shape(probes)[0]
        scale = tf.sqrt(tf.cast(self.depth, dtype))
        sharp = 1.0 / tf.maximum(tf.cast(self.leak_rate, dtype), 1e-3)
        p = probes
        heard = tf.zeros_like(probes)
        peak = tf.zeros((n_probe, self.num_heads), dtype=dtype)
        for _ in range(cycles):
            q = tf.reshape(self.query(self.q_norm(p)), (n_probe, self.num_heads, 1, self.depth))
            scores = tf.squeeze(tf.matmul(q, k, transpose_b=True), axis=2) / scale  # (P, H, T)
            scores = tf.where(mask, scores, tf.cast(_MASK_NEG, dtype))
            peak = tf.reduce_max(scores, axis=-1)
            attn = self.dropout(tf.nn.softmax(scores, axis=-1), training=training)
            ctx = tf.reshape(tf.matmul(tf.expand_dims(attn, 2), v), (n_probe, self.d_model))
            heard = self.listen(ctx)
            g = self.gate(tf.concat([p, heard], axis=-1))
            g_scale = tf.sqrt(tf.reduce_mean(tf.square(g), axis=-1, keepdims=True) + 1e-6)
            g = tf.sigmoid(sharp * (g / g_scale - self.gate_threshold))
            p = p + g * heard
            p = p + self.dropout(self.mix_up(self.mix_down(self.mix_norm(p))), training=training)
        return p, heard, peak

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model, "num_heads": self.num_heads, "cycles": self.cycles,
            "leak_rate": self.leak_rate, "spike_threshold": self.spike_threshold,
            "mixer_rank_div": self.mixer_rank_div, "dropout_rate": self.dropout_rate,
        })
        return config
