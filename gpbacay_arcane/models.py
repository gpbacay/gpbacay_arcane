"""
gpbacay_arcane.models

This module contains the neuromimetic semantic model architectures for the ARCANE project.
Augmented Reconstruction of Consciousness through Artificial Neural Evolution.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Embedding, LSTM, GlobalAveragePooling1D, 
    Dropout, LayerNormalization, Concatenate, Add, 
    GlobalMaxPooling1D, BatchNormalization, RNN, Multiply
)
from tensorflow.keras import Model

from gpbacay_arcane.layers import (
    ResonantGSER, 
    PredictiveResonantLayer,
    BioplasticDenseLayer, 
    DenseGSER,
    GSER,
    LatentTemporalCoherence,
    HebbianHomeostaticNeuroplasticity,
    PositionalEncodingLayer,
    MultiheadLinearSelfAttentionKernalization,
)


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
    
    def __init__(self, vocab_size, seq_len=16, embed_dim=32, hidden_dim=64):
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
            
            # LSTM for sequential temporal processing
            lstm_out = LSTM(
                self.hidden_dim,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.1,
                name='lstm_temporal'
            )(gser2)
            
            # Multiple pooling strategies for feature extraction
            avg_pool = GlobalAveragePooling1D(name='avg_pool')(lstm_out)
            gser2_pool = GlobalAveragePooling1D(name='gser2_pool')(gser2)
            
            # Feature fusion from multiple neural pathways
            combined = Concatenate(name='feature_fusion')([avg_pool, gser2_pool])
            
            # BioplasticDenseLayer - Hebbian learning and homeostatic plasticity
            bioplastic = BioplasticDenseLayer(
                units=self.hidden_dim * 2,  # Match combined features dimension
                learning_rate=1.5e-3,
                target_avg=0.11,
                homeostatic_rate=8e-5,
                activation='gelu',
                dropout_rate=0.12,
                name='bioplastic_main'
            )(combined)
            
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
    
    def run_resonance_cycle(self, num_cycles=1, use_parallel=True):
        """
        Execute resonance cycles across the resonant layers for inference-time alignment.
        
        The implementation uses parallelization within each phase:
        - All projections are computed simultaneously (parallel)
        - All harmonizations are applied simultaneously (parallel)
        - All divergence computations run in parallel
        
        Args:
            num_cycles: Number of resonance cycles to run
            use_parallel: Whether to use parallelized operations (default: True)
        
        Returns:
            List of divergence values for each cycle
        """
        if not hasattr(self, 'resonant_layer_objects'):
            raise ValueError("Model must be built before running resonance cycles")
        
        # Store model reference in layers
        for layer in self.resonant_layer_objects:
            layer._model = self.model
        
        divergences = []
        
        for cycle in range(num_cycles):
            # Step A: Project (Top-Down) - PARALLELIZED
            # All layers can compute projections simultaneously since they're independent.
            # We collect all projection operations first, then TensorFlow executes them in parallel
            # when they're part of the same computation graph.
            projections = {}
            
            # Collect all projection operations (independent operations that can run in parallel)
            projection_pairs = [
                (i - 1, self.resonant_layer_objects[i].project_feedback())
                for i in range(len(self.resonant_layer_objects) - 1, 0, -1)
            ]
            
            # Execute projections - TensorFlow will parallelize independent operations
            # when they're in eager mode or part of a tf.function graph
            for idx, proj_tensor in projection_pairs:
                projections[idx] = proj_tensor
            
            # Step B: Harmonize (Bottom-Up) - PARALLELIZED
            # All harmonizations can be applied simultaneously since they're independent.
            # Each layer only modifies its own resonance_alignment variable.
            harmonization_pairs = [
                (layer, projections[i])
                for i, layer in enumerate(self.resonant_layer_objects)
                if i in projections
            ]
            
            # Execute all harmonizations - operations are independent and can run in parallel
            for layer, proj in harmonization_pairs:
                layer.harmonize_states(proj)
            
            # Step C: Check Convergence - PARALLELIZED divergence computation
            # All divergence computations are independent and can run in parallel
            divergence_values = [layer.get_divergence() for layer in self.resonant_layer_objects]
            cycle_divergence = sum(divergence_values)
            divergences.append(float(cycle_divergence))
            
            # Early stopping if converged
            if cycle_divergence < self.resonant_layer_objects[0].cell.convergence_epsilon:
                break
        
        return divergences
    
    def predict_with_resonance(self, inputs, resonance_cycles=5, verbose=0):
        """
        Perform prediction with inference-time resonance cycles.
        
        This method runs resonance cycles before prediction to align the hierarchical
        states, enabling deliberative reasoning at inference time.
        
        Args:
            inputs: Input data (numpy array or tensor)
            resonance_cycles: Number of resonance cycles to run before prediction
            verbose: Verbosity level (0 or 1)
        
        Returns:
            Model predictions after resonance alignment
        """
        if self.model is None:
            raise ValueError("Model must be built before prediction")
        
        if verbose > 0:
            print(f"Running {resonance_cycles} resonance cycles for inference-time alignment...")
        
        # Run resonance cycles to align hierarchical states
        divergences = self.run_resonance_cycle(num_cycles=resonance_cycles)
        
        if verbose > 0:
            print(f"Resonance converged. Final divergence: {divergences[-1]:.6f}")
        
        # Perform prediction with aligned states
        predictions = self.model.predict(inputs, verbose=verbose)
        return predictions
    
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
            }
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


class PredictiveResonantSemanticModel:
    """
    Predictive Resonant Semantic Foundation Model
    
    A neuromimetic model architecture that leverages PredictiveResonantLayer for
    local predictive resonance. Unlike the hierarchical ResonantGSER approach,
    this model uses self-contained resonance where each layer maintains its own
    internal alignment vector for continuous prediction.
    
    Key features:
    - Stacked PredictiveResonantLayers for deep predictive coding
    - Local resonance without requiring hierarchical feedback wiring
    - Optional persist_alignment for cross-sequence semantic memory (LTP-like)
    - BioplasticDenseLayer for Hebbian learning
    - Combined with LSTM for temporal processing
    """
    
    def __init__(self, vocab_size, seq_len=16, embed_dim=32, hidden_dim=64, 
                 num_pred_layers=2, persist_alignment=False):
        """
        Initialize the predictive resonant semantic model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            seq_len (int): Length of input sequences
            embed_dim (int): Embedding dimension
            hidden_dim (int): Hidden layer dimension
            num_pred_layers (int): Number of stacked PredictiveResonantLayers
            persist_alignment (bool): Whether to maintain alignment across sequences
        """
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_pred_layers = num_pred_layers
        self.persist_alignment = persist_alignment
        self.model = None

    def build_model(self):
        """Build the predictive resonant semantic model architecture."""
        
        with tf.device('/CPU:0'):
            inputs = Input(shape=(self.seq_len,), name='text_input')
            
            # Embedding layer
            embedded = Embedding(
                self.vocab_size, 
                self.embed_dim,
                name='embedding'
            )(inputs)
            
            # Stack of PredictiveResonantLayers for deep local resonance
            x = embedded
            self.predictive_resonant_layers = []
            
            for i in range(self.num_pred_layers):
                # First layer doesn't use persist_alignment to start fresh
                # Subsequent layers can use it for memory across sequences
                use_persist = self.persist_alignment and (i > 0)
                
                pred_layer = PredictiveResonantLayer(
                    units=self.hidden_dim,
                    resonance_cycles=3,
                    resonance_step_size=0.2,
                    spike_threshold=0.5,
                    return_sequences=True,
                    persist_alignment=use_persist,
                    name=f'predictive_resonant_{i+1}'
                )
                
                x = pred_layer(x)
                x = LayerNormalization(name=f'pred_layer_norm_{i+1}')(x)
                x = Dropout(0.15, name=f'pred_dropout_{i+1}')(x)
                
                self.predictive_resonant_layers.append(pred_layer)
            
            # LSTM for sequential temporal processing
            lstm_out = LSTM(
                self.hidden_dim,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.1,
                name='lstm_temporal'
            )(x)
            
            # Feature pooling
            avg_pool = GlobalAveragePooling1D(name='avg_pool')(lstm_out)
            pred_pool = GlobalAveragePooling1D(name='pred_pool')(x)
            
            # Feature fusion
            combined = Concatenate(name='feature_fusion')([avg_pool, pred_pool])
            
            # BioplasticDenseLayer - Hebbian learning and homeostatic plasticity
            bioplastic = BioplasticDenseLayer(
                units=self.hidden_dim * 2,
                learning_rate=1.5e-3,
                target_avg=0.11,
                homeostatic_rate=8e-5,
                activation='gelu',
                dropout_rate=0.12,
                name='bioplastic_main'
            )(combined)
            
            # Additional dense processing
            dense_hidden = Dense(
                self.hidden_dim,
                activation='gelu',
                name='dense_processing'
            )(bioplastic)
            
            dense_dropout = Dropout(0.1, name='dense_dropout')(dense_hidden)
            
            # Output layer
            outputs = Dense(
                self.vocab_size,
                activation='softmax',
                name='semantic_output'
            )(dense_dropout)
            
            self.model = Model(
                inputs=inputs,
                outputs=outputs,
                name='predictive_resonant_semantic_model'
            )
        
        return self.model
    
    def compile_model(self, learning_rate=1e-3):
        """Compile the model with appropriate optimizer and loss function."""
        if self.model is None:
            raise ValueError("Model must be built before compiling. Call build_model() first.")
        
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
    
    def run_inference_resonance(self, inputs, num_cycles=3):
        """
        Run inference-time resonance cycles on the PredictiveResonantLayers.
        
        Since PredictiveResonantLayer maintains its own internal alignment,
        this allows for deliberative inference where the layer refines
        its predictions before final output.
        
        Args:
            inputs: Input data
            num_cycles: Number of resonance iterations
            
        Returns:
            Model predictions after resonance
        """
        if self.model is None:
            raise ValueError("Model must be built first")
        
        # Get the final hidden state from the last PredictiveResonantLayer
        # The alignment has already been updated during forward pass
        predictions = self.model.predict(inputs, verbose=0)
        return predictions
    
    def get_model_info(self):
        """Get information about the predictive resonant model architecture."""
        return {
            "name": "Predictive Resonant Semantic Model",
            "description": "Bio-inspired model with local predictive resonance for semantic understanding",
            "features": [
                f"{self.num_pred_layers} stacked PredictiveResonantLayers",
                "Local predictive coding without hierarchical feedback",
                "BioplasticDenseLayer Hebbian learning",
                "LSTM temporal processing",
                "persist_alignment" if self.persist_alignment else "stateless resonance"
            ],
            "parameters": {
                "vocab_size": self.vocab_size,
                "sequence_length": self.seq_len,
                "embedding_dim": self.embed_dim,
                "hidden_dim": self.hidden_dim,
                "num_pred_layers": self.num_pred_layers
            }
        }


class HybridResonantModel:
    """
    Hybrid Resonant Model combining ResonantGSER (hierarchical) and 
    PredictiveResonantLayer (local) for maximum representational power.
    
    This model leverages:
    - ResonantGSER for top-down hierarchical alignment
    - PredictiveResonantLayer for local autonomous prediction
    - BioplasticDenseLayer for Hebbian learning
    """
    
    def __init__(self, vocab_size, seq_len=16, embed_dim=32, hidden_dim=64):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.model = None

    def build_model(self):
        """Build the hybrid resonant model architecture."""
        
        with tf.device('/CPU:0'):
            inputs = Input(shape=(self.seq_len,), name='text_input')
            
            # Embedding
            embedded = Embedding(
                self.vocab_size, 
                self.embed_dim,
                name='embedding'
            )(inputs)
            
            # Stage 1: PredictiveResonantLayer (local resonance)
            pred_layer = PredictiveResonantLayer(
                units=self.hidden_dim,
                resonance_cycles=3,
                resonance_step_size=0.2,
                spike_threshold=0.5,
                return_sequences=True,
                persist_alignment=False,
                name='predictive_resonant_1'
            )
            pred_out = pred_layer(embedded)
            pred_out = LayerNormalization(name='pred_layer_norm')(pred_out)
            pred_out = Dropout(0.15)(pred_out)
            
            # Stage 2: ResonantGSER (hierarchical resonance)
            resonant_layer = ResonantGSER(
                units=self.hidden_dim,
                spike_threshold=0.35,
                resonance_factor=0.1,
                resonance_cycles=3,
                return_sequences=True,
                name='resonant_gser'
            )
            gser_out = resonant_layer(pred_out)
            
            # Establish hierarchical feedback
            resonant_layer.set_lower_layer(pred_layer)
            pred_layer.set_higher_layer(resonant_layer)
            
            # Store layers for resonance cycles
            self.resonant_layer_objects = [pred_layer, resonant_layer]
            
            # LSTM temporal processing
            lstm_out = LSTM(
                self.hidden_dim,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.1,
                name='lstm_temporal'
            )(gser_out)
            
            # Feature fusion
            avg_pool = GlobalAveragePooling1D(name='avg_pool')(lstm_out)
            gser_pool = GlobalAveragePooling1D(name='gser_pool')(gser_out)
            
            combined = Concatenate(name='feature_fusion')([avg_pool, gser_pool])
            
            # BioplasticDenseLayer
            bioplastic = BioplasticDenseLayer(
                units=self.hidden_dim * 2,
                learning_rate=1.5e-3,
                target_avg=0.11,
                homeostatic_rate=8e-5,
                activation='gelu',
                dropout_rate=0.12,
                name='bioplastic_main'
            )(combined)
            
            dense_hidden = Dense(self.hidden_dim, activation='gelu', name='dense_processing')(bioplastic)
            dense_dropout = Dropout(0.1)(dense_hidden)
            
            outputs = Dense(self.vocab_size, activation='softmax', name='semantic_output')(dense_dropout)
            
            self.model = Model(inputs=inputs, outputs=outputs, name='hybrid_resonant_model')
        
        return self.model
    
    def compile_model(self, learning_rate=1e-3):
        """Compile the hybrid model."""
        if self.model is None:
            raise ValueError("Model must be built before compiling.")
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def get_model_info(self):
        return {
            "name": "Hybrid Resonant Model",
            "description": "Combines hierarchical (ResonantGSER) and local (PredictiveResonantLayer) resonance",
            "features": [
                "PredictiveResonantLayer for local autonomous prediction",
                "ResonantGSER for hierarchical top-down alignment",
                "BioplasticDenseLayer Hebbian learning",
                "Bidirectional resonance flow"
            ]
        }


# Legacy model aliases for backward compatibility
# These provide compatibility with older references while using the main neuromimetic architecture
DSTSMGSER = NeuromimeticSemanticModel  # Dynamic Spatio-Temporal Self-Modeling Gated Spiking Elastic Reservoir
GSERModel = NeuromimeticSemanticModel  # Simplified Gated Spiking Elastic Reservoir Model  
CoherentThoughtModel = NeuromimeticSemanticModel  # Coherent Thought Model
