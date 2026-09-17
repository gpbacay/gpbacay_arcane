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


class CausalLinearSelfAttention(Layer):
    """Katharopoulos linear attention with a causal prefix (cumsum) instead of a full-sequence KV.

    Same feature map as ``MultiheadLinearSelfAttentionKernalization`` (φ(x) = elu(x)+1),
    but position t only aggregates keys/values from indices 0..t. Complexity is still
    O(n d^2); memory for the prefix outer product is O(n d_head^2).
    """

    def __init__(
        self,
        d_model,
        num_heads,
        dropout_rate=0.1,
        use_semantic_reweighting=True,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_semantic_reweighting = use_semantic_reweighting
        self.eps = eps
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.depth = d_model // num_heads
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
        self.layer_norm.build(input_shape)
        super().build(input_shape)

    def _split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        queries = tf.matmul(inputs, self.query_weight) + self.query_bias
        keys = tf.matmul(inputs, self.key_weight) + self.key_bias
        values = tf.matmul(inputs, self.value_weight) + self.value_bias
        queries = self._split_heads(queries, batch_size)
        keys = self._split_heads(keys, batch_size)
        values = self._split_heads(values, batch_size)
        queries = tf.nn.elu(queries) + 1.0
        keys = tf.nn.elu(keys) + 1.0
        # Causal prefix: KV_t = Σ_{i≤t} φ(k_i)^T v_i, z_t = Σ_{i≤t} φ(k_i)
        kv = tf.expand_dims(keys, -1) * tf.expand_dims(values, -2)
        kv_cs = tf.cumsum(kv, axis=2)
        z_cs = tf.cumsum(keys, axis=2)
        numerator = tf.einsum("bhsd,bhsde->bhse", queries, kv_cs)
        denominator = tf.einsum("bhsd,bhsd->bhs", queries, z_cs)
        attention_output = numerator / (tf.expand_dims(denominator, -1) + self.eps)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, seq_len, self.d_model))
        output = tf.matmul(attention_output, self.output_weight) + self.output_bias
        if self.use_semantic_reweighting:
            reweight_factors = tf.sigmoid(
                tf.matmul(output, self.semantic_reweight_kernel) + self.semantic_reweight_bias
            )
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
