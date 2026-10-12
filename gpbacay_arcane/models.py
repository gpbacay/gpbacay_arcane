"""
gpbacay_arcane.models

This module contains the neuromimetic semantic model architectures for the ARCANE project.
Augmented Reconstruction of Consciousness through Artificial Neural Evolution.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Embedding, LSTM, GlobalAveragePooling1D,
    Dropout, LayerNormalization, Concatenate,
)
from tensorflow.keras import Model

from gpbacay_arcane.layers import (
    ResonantGSER,
    PredictiveResonantLayer,
    BioplasticDenseLayer,
    DenseGSER,
    LatentTemporalCoherence,
    HebbianHomeostaticNeuroplasticity,
    HierarchicalHomeostaticController,
    PositionalEncodingLayer,
)
from gpbacay_arcane.mechanisms import (
    GSER,
    MultiheadLinearSelfAttentionKernalization,
)
from gpbacay_arcane.resonance import (
    attach_resonance_probe,
    run_per_example_resonance,
    train_feedback_step,
)
from gpbacay_arcane.test_time_adaptation import adapt_at_inference


class NeuromimeticSemanticModel:
    """
    Neuromimetic Semantic Foundation Model
    
    A novel model architecture that incorporates biological neural principles for advanced semantic understanding across various data types:
    - Hierarchical neural resonance and prospective alignment
    - Spiking neural dynamics via ResonantGSER layers
    - Hebbian learning via BioplasticDenseLayer
    - Homeostatic plasticity for activity regulation
    - Temporal sequence processing via LSTM
    
    This model represents the first implementation of a neuromimetic semantic foundation model,
    bridging neuroscience and artificial intelligence for comprehensive semantic engineering.
    """
    
    def __init__(
        self,
        vocab_size,
        seq_len=16,
        embed_dim=32,
        hidden_dim=64,
        enable_inference_plasticity=False,
        use_hierarchical_homeostasis=False,
        homeostasis_config=None,
    ):
        """
        Initialize the neuromimetic semantic model.
        
        Args:
            vocab_size (int): Size of the vocabulary (or embedding space for non-textual data)
            seq_len (int): Length of input sequences
            embed_dim (int): Embedding dimension
            hidden_dim (int): Hidden layer dimension
        """
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.enable_inference_plasticity = bool(enable_inference_plasticity)
        self.use_hierarchical_homeostasis = bool(use_hierarchical_homeostasis)
        self.homeostasis_config = dict(homeostasis_config or {})
        self.model = None

    def build_model(self):
        """Build the neuromimetic semantic model architecture."""
        
        # Force CPU device for variables to avoid GPU/CPU conflicts
        with tf.device('/CPU:0'):
            inputs = Input(shape=(self.seq_len,), name='text_input')
            
            # Embedding layer
            embedded = Embedding(
                self.vocab_size, 
                self.embed_dim,
                name='embedding'
            )(inputs)
            
            # First ResonantGSER layer - Primary neural processing with Prospective Alignment
            resonant_layer_1 = ResonantGSER(
                units=self.hidden_dim,
                spike_threshold=0.35,
                resonance_factor=0.1,
                resonance_cycles=3,
                return_sequences=True,
                name='resonant_gser_1'
            )
            gser1 = resonant_layer_1(embedded)
            
            # Layer normalization and dropout for stability
            gser1_norm = LayerNormalization(name='layer_norm_1')(gser1)
            gser1_drop = Dropout(0.15, name='dropout_1')(gser1_norm)
            
            # Second ResonantGSER layer - Secondary neural processing with Hierarchical Feedback
            resonant_layer_2 = ResonantGSER(
                units=self.hidden_dim,
                spike_threshold=0.3,
                resonance_factor=0.12,
                resonance_cycles=3,
                return_sequences=True,
                name='resonant_gser_2'
            )
            gser2 = resonant_layer_2(gser1_drop)

            # Establish Hierarchical Feedback Connections
            resonant_layer_2.set_lower_layer(resonant_layer_1)
            resonant_layer_1.set_higher_layer(resonant_layer_2)
            
            # Store resonant layer objects for inference-time resonance
            self.resonant_layer_objects = [resonant_layer_1, resonant_layer_2]
            self.resonant_layers = [gser1, gser2]

            resonance_signal = None
            target_signal = None
            gser2_effective = gser2
            if self.use_hierarchical_homeostasis:
                controlled = HierarchicalHomeostaticController(
                    target_activity=self.homeostasis_config.get("target_avg", 0.11),
                    regulation_rate=self.homeostasis_config.get("hierarchical_rate", 0.1),
                    level_target_slope=self.homeostasis_config.get("hierarchy_target_slope", 0.0),
                    adaptive_target_rate=self.homeostasis_config.get("hierarchy_adaptive_rate", 0.1),
                    name="hierarchical_homeostasis",
                )([gser1, gser2])
                _, gser2_effective, resonance_signal, target_signal = controlled
            
            # LSTM for sequential temporal processing
            lstm_out = LSTM(
                self.hidden_dim,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.1,
                name='lstm_temporal'
            )(gser2_effective)
            
            # Multiple pooling strategies for feature extraction
            avg_pool = GlobalAveragePooling1D(name='avg_pool')(lstm_out)
            gser2_pool = GlobalAveragePooling1D(name='gser2_pool')(gser2_effective)
            
            # Feature fusion from multiple neural pathways
            combined = Concatenate(name='feature_fusion')([avg_pool, gser2_pool])
            
            # BioplasticDenseLayer - Hebbian learning and homeostatic plasticity
            bioplastic_kwargs = {
                "learning_rate": 1.5e-3,
                "target_avg": 0.11,
                "homeostatic_rate": 8e-5,
                "activation": "gelu",
                "dropout_rate": 0.12,
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
                "hierarchical_rate", "hierarchy_adaptive_rate"
            ):
                bioplastic_kwargs.pop(key, None)
            bioplastic_layer = BioplasticDenseLayer(
                units=self.hidden_dim * 2,  # Match combined features dimension
                enable_inference_plasticity=self.enable_inference_plasticity,
                name='bioplastic_main',
                **bioplastic_kwargs,
            )
            bioplastic = bioplastic_layer(
                combined,
                resonance_signal=resonance_signal,
                target_signal=target_signal,
            )
            
            # Additional dense processing layer
            dense_hidden = Dense(
                self.hidden_dim,
                activation='gelu',
                name='dense_processing'
            )(bioplastic)
            
            dense_dropout = Dropout(0.1, name='dense_dropout')(dense_hidden)
            
            # Output layer for semantic modeling
            outputs = Dense(
                self.vocab_size,
                activation='softmax',
                name='semantic_output'
            )(dense_dropout)
            
            self.model = Model(
                inputs=inputs,
                outputs=outputs,
                name='neuromimetic_semantic_foundation_model'
            )
            attach_resonance_probe(self, inputs, outputs, self.resonant_layers)
        
        return self.model
    
    def compile_model(self, learning_rate=1e-3):
        """Compile the model with appropriate optimizer and loss function."""
        if self.model is None:
            raise ValueError("Model must be built before compiling. Call build_model() first.")
        
        # Stable optimizer with gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.999
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return self.model
    
    def generate_text(self, seed_text, tokenizer, max_length=50, temperature=0.8):
        """
        Generate semantic output (e.g., text) using the trained neuromimetic semantic model.
        
        Args:
            seed_text (str): Initial input to start generation (e.g., text, sequence)
            tokenizer: Keras tokenizer or similar mapping function used during training
            max_length (int): Maximum length of output to generate
            temperature (float): Sampling temperature for creativity control
            
        Returns:
            str: Generated output (e.g., text, sequence representation)
        """
        if self.model is None:
            raise ValueError("Model must be built before text generation.")
        
        # Create reverse mapping
        reverse_tokenizer = {v: k for k, v in tokenizer.word_index.items()}
        
        # Convert seed to tokens
        seed_tokens = tokenizer.texts_to_sequences([seed_text.lower()])[0]
        if not seed_tokens:
            seed_tokens = [1]  # fallback
        
        # Pad to sequence length
        if len(seed_tokens) < self.seq_len:
            seed_tokens = [0] * (self.seq_len - len(seed_tokens)) + seed_tokens
        else:
            seed_tokens = seed_tokens[-self.seq_len:]
        
        # Generate text
        current_seq = seed_tokens.copy()
        generated_words = []
        
        for _ in range(max_length):
            pred = self.model.predict(np.array([current_seq]), verbose=0)[0]
            
            # Temperature sampling
            if temperature < 0.9:
                # Conservative: top-k sampling
                k = 10
                top_indices = np.argsort(pred)[-k:]
                top_probs = pred[top_indices]
                top_probs = top_probs / top_probs.sum()
                next_token = np.random.choice(top_indices, p=top_probs)
            else:
                # Creative: temperature + nucleus sampling
                pred = pred / temperature
                pred = tf.nn.softmax(pred).numpy()
                
                # Nucleus sampling (top-p = 0.9)
                sorted_indices = np.argsort(pred)[::-1]
                cumsum_probs = np.cumsum(pred[sorted_indices])
                cutoff_idx = np.where(cumsum_probs > 0.9)[0]
                if len(cutoff_idx) > 0:
                    cutoff_idx = cutoff_idx[0] + 1
                else:
                    cutoff_idx = 15
                
                nucleus_indices = sorted_indices[:cutoff_idx]
                nucleus_probs = pred[nucleus_indices]
                nucleus_probs = nucleus_probs / nucleus_probs.sum()
                
                next_token = np.random.choice(nucleus_indices, p=nucleus_probs)
            
            # Convert token to word
            word = reverse_tokenizer.get(next_token, "")
            
            if word and word != "<UNK>" and word.strip():
                generated_words.append(word)
            
            # Update sequence
            current_seq = current_seq[1:] + [next_token]
            
            # Natural stopping points
            if word in [".", "!", "?"] and len(generated_words) > 5:
                break
        
        return " ".join(generated_words)
    
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
        Execute resonance cycles across the resonant layers for inference-time alignment.
        
        When ``inputs`` is provided, a forward initialization on the current
        batch is followed by project -> harmonize -> forward refinement cycles.
        
        Args:
            num_cycles: Number of resonance cycles to run
            use_parallel: Retained for backwards compatibility.
            inputs: Current batch to refine. Without it, only one target is
                staged from the most recently cached states.
            reset_state: Clear state inherited from earlier requests.
        
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
        
        Initialize from the current input, then alternate top-down projection
        and a fresh forward refinement of that same input.
        
        Args:
            inputs: Input data (numpy array or tensor)
            resonance_cycles: Number of resonance cycles to run before prediction
            verbose: Verbosity level (0 or 1)
            reset_state: Keep requests independent unless explicitly disabled.
        
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
    
    def get_model_info(self):
        """Get information about the neuromimetic model architecture."""
        return {
            "name": "Neuromimetic Semantic Foundation Model",
            "description": "Bio-inspired model with spiking neural dynamics for semantic understanding",
            "features": [
                "Dual DenseGSER spiking neural layers",
                "BioplasticDenseLayer Hebbian learning",
                "LSTM temporal processing",
                "Homeostatic plasticity regulation",
                "Advanced semantic generation capabilities"
            ],
            "parameters": {
                "vocab_size": self.vocab_size,
                "sequence_length": self.seq_len,
                "embedding_dim": self.embed_dim,
                "hidden_dim": self.hidden_dim
            },
            "homeostasis": {
                "inference_plasticity": self.enable_inference_plasticity,
                "hierarchical": self.use_hierarchical_homeostasis,
                "config": self.homeostasis_config,
            },
        }


# For backward compatibility, keep the legacy class name as an alias
NeuromimeticSemanticFoundationModel = NeuromimeticSemanticModel




def load_neuromimetic_model(model_path, tokenizer_path=None):
    """
    Load a pre-trained neuromimetic semantic model.
    
    Args:
        model_path (str): Path to the saved model file
        tokenizer_path (str): Path to the saved tokenizer file
        
    Returns:
        tuple: (model, tokenizer) if tokenizer_path provided, else just model
    """
    # Custom objects for loading all ARCANE layers
    custom_objects = {
        'DenseGSER': DenseGSER,
        'ResonantGSER': ResonantGSER,
        'PredictiveResonantLayer': PredictiveResonantLayer,
        'BioplasticDenseLayer': BioplasticDenseLayer,
        'GSER': GSER,
        'LatentTemporalCoherence': LatentTemporalCoherence,
        'HebbianHomeostaticNeuroplasticity': HebbianHomeostaticNeuroplasticity,
        'PositionalEncodingLayer': PositionalEncodingLayer,
        'MultiheadLinearSelfAttentionKernalization': MultiheadLinearSelfAttentionKernalization,
    }
    
    # Load model
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    if tokenizer_path:
        import pickle
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    
    return model


# Legacy model aliases for backward compatibility
DSTSMGSER = NeuromimeticSemanticModel
GSERModel = NeuromimeticSemanticModel
CoherentThoughtModel = NeuromimeticSemanticModel
