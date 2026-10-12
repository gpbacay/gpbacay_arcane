import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Embedding, LSTM, GlobalAveragePooling1D, 
    Dropout, LayerNormalization, Concatenate, Add, 
    GlobalMaxPooling1D, RNN, Multiply, Reshape
)
from tensorflow.keras import Model
from gpbacay_arcane.layers import (
    ResonantGSER,
    BioplasticDenseLayer,
    DenseGSER,
    LatentTemporalCoherence,
    PositionalEncodingLayer,
    HierarchicalHomeostaticController,
)
from gpbacay_arcane.resonance import (
    attach_resonance_probe,
    run_per_example_resonance,
    train_feedback_step,
)
from gpbacay_arcane.test_time_adaptation import adapt_at_inference

class HierarchicalResonanceFoundationModel:
    """
    Hierarchical Neural Resonance Foundation Model
    
    A deep neuromimetic architecture for experimental hierarchical latent-state
    refinement. Task-level reasoning benefits require empirical validation.
    """
    
    def __init__(
        self, 
        vocab_size, 
        seq_len=32, 
        embed_dim=64, 
        hidden_dim=128,
        num_resonance_levels=4,
        resonance_factor=0.15,
        use_temporal_coherence=True,
        use_attention_fusion=True,
        dropout_rate=0.1,
        enable_inference_plasticity=False,
        use_hierarchical_homeostasis=False,
        homeostasis_config=None,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_resonance_levels = max(2, min(6, num_resonance_levels))
        self.resonance_factor = resonance_factor
        self.use_temporal_coherence = use_temporal_coherence
        self.use_attention_fusion = use_attention_fusion
        self.dropout_rate = dropout_rate
        self.enable_inference_plasticity = bool(enable_inference_plasticity)
        self.use_hierarchical_homeostasis = bool(use_hierarchical_homeostasis)
        self.homeostasis_config = dict(homeostasis_config or {})
        self.model = None
        self.resonant_layers = []
        
    
    def build_model(self):
        self.resonant_layers = []
        with tf.device('/CPU:0'):
            # === Input and Embedding ===
            # Ensure static sequence length and batch dimension for heavy initialization
            inputs = Input(shape=(self.seq_len,), batch_size=None, name='text_input')
            
            # 1. Expand dims to (batch, seq_len, 1) if necessary
            # 2. Embedding + Positional Encoding
            embedded = Embedding(self.vocab_size, self.embed_dim, name='token_embedding')(inputs)
            positioned = PositionalEncodingLayer(max_position=self.seq_len, d_model=self.embed_dim, name='positional_encoding')(embedded)
            
            # Initial projection to hidden dimension
            projected = Dense(self.hidden_dim, activation='gelu', name='input_projection')(positioned)
            
            # Ensure 3D shape for RNN (batch, steps, features) with static dimensions
            projected = Reshape((int(self.seq_len), int(self.hidden_dim)), name='reshape_to_3d')(projected)
            
            level_outputs = []
            skip_connections = []
            resonant_layer_objects = []  # Store actual layer objects for hierarchy setup
            current = projected
            
            for level in range(self.num_resonance_levels):
                # Calculate level-specific parameters
                # Higher levels have higher resonance for deeper deliberation
                resonance_factor = self.resonance_factor + (level * 0.05)
                spike_threshold = 0.4 - (level * 0.05)
                level_units = self.hidden_dim
                
                # Create resonant layer with RSAA parameters
                resonant_layer = ResonantGSER(
                    units=level_units,
                    spike_threshold=spike_threshold,
                    resonance_factor=resonance_factor,
                    resonance_cycles=3,  # N cycles per forward pass
                    convergence_epsilon=1e-4,
                    return_sequences=True,
                    name=f'resonant_level_{level}'
                )
                
                # Apply the layer
                level_out = resonant_layer(current)
                
                # Store references
                resonant_layer_objects.append(resonant_layer)
                self.resonant_layers.append(level_out)
                
                # Normalization and dropout
                normed = LayerNormalization(epsilon=1e-6, name=f'layer_norm_{level}')(level_out)
                dropped = Dropout(self.dropout_rate, name=f'dropout_{level}')(normed)
                
                level_outputs.append(dropped)
                skip_connections.append(level_out)
                
                if level > 0:
                    current = Add(name=f'skip_add_{level}')([dropped, current])
                else:
                    current = dropped
            
            # === Establish Hierarchical Feedback Connections ===
            # Wire up the resonant layers to enable top-down/bottom-up communication
            for i in range(len(resonant_layer_objects)):
                if i > 0:
                    # Set lower layer reference
                    resonant_layer_objects[i].set_lower_layer(resonant_layer_objects[i-1])
                if i < len(resonant_layer_objects) - 1:
                    # Set higher layer reference
                    resonant_layer_objects[i].set_higher_layer(resonant_layer_objects[i+1])
            
            # Store for external access
            self.resonant_layer_objects = resonant_layer_objects

            resonance_signal = None
            target_signal = None
            if self.use_hierarchical_homeostasis:
                controlled = HierarchicalHomeostaticController(
                    target_activity=self.homeostasis_config.get("target_avg", 0.12),
                    regulation_rate=self.homeostasis_config.get("hierarchical_rate", 0.1),
                    level_target_slope=self.homeostasis_config.get("hierarchy_target_slope", 0.0),
                    adaptive_target_rate=self.homeostasis_config.get("hierarchy_adaptive_rate", 0.1),
                    name="hierarchical_homeostasis",
                )([*skip_connections, current])
                skip_connections = controlled[:self.num_resonance_levels]
                current = controlled[self.num_resonance_levels]
                resonance_signal = controlled[-2]
                target_signal = controlled[-1]
            
            lstm_out = LSTM(self.hidden_dim, return_sequences=True, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate / 2, name='temporal_lstm')(current)
            gser_out = DenseGSER(units=self.hidden_dim, spectral_radius=0.9, leak_rate=0.1, spike_threshold=0.3, activation='gelu', name='dense_gser_reservoir')(lstm_out)
            
            pathways = []
            pathways.append(GlobalAveragePooling1D(name='avg_pool_final')(current))
            pathways.append(GlobalMaxPooling1D(name='max_pool_final')(current))
            pathways.append(GlobalAveragePooling1D(name='avg_pool_lstm')(lstm_out))
            pathways.append(GlobalAveragePooling1D(name='avg_pool_gser')(gser_out))
            
            skip_pooled = [GlobalAveragePooling1D(name=f'skip_pool_{i}')(s) for i, s in enumerate(skip_connections)]
            skip_compressed = Dense(self.hidden_dim, activation='gelu', name='skip_compression')(Concatenate(name='skip_fusion')(skip_pooled))
            pathways.append(skip_compressed)
            
            if self.use_temporal_coherence:
                pathways.append(LatentTemporalCoherence(d_coherence=min(64, self.hidden_dim), name='temporal_coherence')(gser_out))
            
            fused = Concatenate(name='pathway_fusion')(pathways)
            attention_weights = None
            if self.use_attention_fusion:
                gate = Dense(fused.shape[-1] or self.hidden_dim * 4, activation='sigmoid', name='pathway_gate')(fused)
                fused = Multiply(name='gated_fusion')([fused, gate])
                attention_weights = gate
            
            bioplastic_kwargs = {
                "normalization": "l2",
                "activation": "gelu",
            }
            if self.enable_inference_plasticity:
                bioplastic_kwargs.update({
                    "homeostasis_mode": "hybrid",
                    "homeostatic_window_size": 16,
                    "slow_bcm_tau": 4000.0,
                    "adaptive_target_rate": 0.05,
                    "resonance_coupling": 0.5,
                })
            bioplastic_kwargs.update(self.homeostasis_config)
            for key in (
                "hierarchical_rate",
                "hierarchy_adaptive_rate",
                "hierarchy_level",
                "num_hierarchy_levels",
            ):
                bioplastic_kwargs.pop(key, None)
            bioplastic_1 = BioplasticDenseLayer(
                units=self.hidden_dim * 2,
                enable_inference_plasticity=self.enable_inference_plasticity,
                hierarchy_level=0,
                num_hierarchy_levels=2,
                name='bioplastic_1',
                **bioplastic_kwargs,
            )
            bioplastic1 = bioplastic_1(
                fused,
                attention_weights=attention_weights,
                resonance_signal=resonance_signal,
                target_signal=target_signal,
            )
            bioplastic_2 = BioplasticDenseLayer(
                units=self.hidden_dim,
                enable_inference_plasticity=self.enable_inference_plasticity,
                hierarchy_level=1,
                num_hierarchy_levels=2,
                name='bioplastic_2',
                **bioplastic_kwargs,
            )
            bioplastic2 = bioplastic_2(
                bioplastic1,
                resonance_signal=resonance_signal,
                target_signal=target_signal,
            )
            
            dense_out = Dense(self.hidden_dim, activation='gelu', name='output_dense')(bioplastic2)
            dense_out = Dropout(self.dropout_rate / 2, name='output_dropout')(dense_out)
            outputs = Dense(self.vocab_size, activation='softmax', name='semantic_output')(dense_out)
            self.model = Model(inputs=inputs, outputs=outputs, name='hierarchical_resonance_foundation_model')
            attach_resonance_probe(self, inputs, outputs, self.resonant_layers)
        return self.model
    
    def reset_resonance_state(self, clear_prototypes=True):
        """Clear transient resonance state so separate requests are independent."""
        if not hasattr(self, 'resonant_layer_objects'):
            raise ValueError("Model must be built before resetting resonance state")
        for layer in self.resonant_layer_objects:
            layer.cell.resonance_alignment.assign(
                tf.zeros_like(layer.cell.resonance_alignment)
            )
            layer.cell.alignment_set.assign(0.0)
            layer.cell.global_divergence.assign(0.0)
            if clear_prototypes:
                layer.cell.last_h.assign(tf.zeros_like(layer.cell.last_h))

    def _project_and_harmonize(self):
        """Stage one simultaneous top-down target for every non-top layer."""
        projections = {
            i - 1: self.resonant_layer_objects[i].project_feedback()
            for i in range(len(self.resonant_layer_objects) - 1, 0, -1)
        }
        for i, layer in enumerate(self.resonant_layer_objects):
            if i in projections:
                layer.harmonize_states(projections[i])

    def _hierarchy_divergence(self):
        """Fresh mismatch between this pass's lower states and top-down projections."""
        total = 0.0
        for i in range(1, len(self.resonant_layer_objects)):
            projection = self.resonant_layer_objects[i].project_feedback()
            projection = tf.reduce_mean(projection, axis=0)
            lower_state = self.resonant_layer_objects[i - 1].cell.last_h
            total += float(tf.reduce_mean(tf.square(lower_state - projection)).numpy())
        return total

    def run_resonance_cycle(
        self, num_cycles=1, use_parallel=True, inputs=None, reset_state=True
    ):
        """
        Execute the full RSAA algorithm across the hierarchy.
        
        With ``inputs``, this implements an actual current-example refinement:
        initialize the hierarchy with a forward pass, project top-down targets,
        re-run the same input so lower states respond to those targets, and only
        then measure convergence. Each cycle therefore changes the state used by
        the following cycle.
        
        Args:
            num_cycles: Number of full hierarchy resonance cycles to run
            use_parallel: Retained for API compatibility. TensorFlow schedules
                independent projection ops; no O(1) parallelism is promised.
            inputs: Current batch to refine. If omitted, one target is staged
                from the most recent forward pass, without claiming convergence.
            reset_state: Clear cross-request prototypes/alignment before the
                initialization pass. Disable only for an intentional stream.
        
        Returns:
            List of divergence values for each cycle
        """
        del use_parallel
        if inputs is None:
            if num_cycles < 1:
                raise ValueError("num_cycles must be at least 1 without current inputs")
            self._project_and_harmonize()
            return [float(sum(layer.get_divergence() for layer in self.resonant_layer_objects))]
        return run_per_example_resonance(
            self, inputs, num_cycles=num_cycles, reset_state=reset_state
        )

    def train_feedback_step(self, inputs, labels=None, **kwargs):
        """Train task weights and feedback projections through reconstruction."""
        return train_feedback_step(self, inputs, labels=labels, **kwargs)

    def adapt_at_inference(self, inputs, **kwargs):
        """Run bounded objective-driven adaptation with episodic reset by default."""
        return adapt_at_inference(self, inputs, **kwargs)
    
    def predict_with_resonance(
        self, inputs, resonance_cycles=5, verbose=0, reset_state=True
    ):
        """
        Perform prediction with inference-time resonance cycles.
        
        The current input first initializes the hierarchy. Every resonance cycle
        then stages top-down targets and performs another forward pass of that
        same input. Weights remain fixed; this is iterative latent-state inference.
        
        Args:
            inputs: Input data (numpy array or tensor)
            resonance_cycles: Number of resonance cycles to run before prediction
            verbose: Verbosity level (0 or 1)
            reset_state: Make this request independent of prior requests. Set
                False only for an intentional continuous stream.
        
        Returns:
            Model predictions after resonance alignment
        """
        if self.model is None:
            raise ValueError("Model must be built before prediction")
        
        if verbose > 0:
            print(f"Running {resonance_cycles} resonance cycles for inference-time alignment...")
        
        divergences = self.run_resonance_cycle(
            num_cycles=resonance_cycles,
            inputs=inputs,
            reset_state=reset_state,
        )
        
        if verbose > 0:
            best = self.last_resonance_best_cycle
            if best:
                print(
                    f"Resonance finished. Best divergence: "
                    f"{divergences[best - 1]:.6f} (cycle {best})"
                )
            else:
                print("Resonance disabled (0 cycles).")
        
        predictions = self._last_resonance_predictions
        return predictions.numpy() if hasattr(predictions, "numpy") else predictions
    
    def compile_model(self, learning_rate=5e-4):
        if self.model is None: raise ValueError("Build model first.")
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return self.model
    
    def get_resonant_layers(self):
        return [l for l in self.model.layers if 'resonant_level' in l.name] if self.model else []

    def get_model_info(self):
        return {
            "name": "Hierarchical Neural Resonance Foundation Model",
            "resonance_levels": self.num_resonance_levels,
            "hidden_dim": self.hidden_dim,
            "homeostasis": {
                "inference_plasticity": self.enable_inference_plasticity,
                "hierarchical": self.use_hierarchical_homeostasis,
                "config": self.homeostasis_config,
            }
        }
