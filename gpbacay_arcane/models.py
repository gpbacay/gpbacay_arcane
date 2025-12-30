"""
gpbacay_arcane.models

This module contains the neuromimetic language model architectures for the A.R.C.A.N.E. project.
Augmented Reconstruction of Consciousness through Artificial Neural Evolution.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Embedding, LSTM, GlobalAveragePooling1D, 
    Dropout, LayerNormalization, Concatenate, Add, 
    GlobalMaxPooling1D, BatchNormalization, RNN
)
from tensorflow.keras import Model

from gpbacay_arcane.layers import (
    ResonantGSER, 
    BioplasticDenseLayer, 
    DenseGSER,
    GSER,
    LatentTemporalCoherence,
    HebbianHomeostaticNeuroplasticity,
    PositionalEncodingLayer,
    MultiheadLinearSelfAttentionKernalization,
)


class NeuromimeticLanguageModel:
    """
    Neuromimetic Language Foundation Model
    
    A novel language model architecture that incorporates biological neural principles:
    - Hierarchical neural resonance and prospective alignment
    - Spiking neural dynamics via ResonantGSER layers
    - Hebbian learning via BioplasticDenseLayer
    - Homeostatic plasticity for activity regulation
    - Temporal sequence processing via LSTM
    
    This model represents the first implementation of a neuromimetic language foundation model,
    bridging neuroscience and natural language processing.
    """
    
    def __init__(self, vocab_size, seq_len=16, embed_dim=32, hidden_dim=64):
        """
        Initialize the neuromimetic language model.
        
        Args:
            vocab_size (int): Size of the vocabulary
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
        """Build the neuromimetic language model architecture."""
        
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
            gser1 = ResonantGSER(
                units=self.hidden_dim,
                spectral_radius=0.9,
                leak_rate=0.1,
                spike_threshold=0.35,
                activation='gelu',
                name='resonant_gser_1'
            )(embedded)
            
            # Layer normalization and dropout for stability
            gser1_norm = LayerNormalization(name='layer_norm_1')(gser1)
            gser1_drop = Dropout(0.15, name='dropout_1')(gser1_norm)
            
            # Second ResonantGSER layer - Secondary neural processing with Hierarchical Feedback
            gser2 = ResonantGSER(
                units=self.hidden_dim,
                spectral_radius=0.8,
                leak_rate=0.12,
                spike_threshold=0.3,
                activation='gelu',
                name='resonant_gser_2'
            )(gser1_drop)
            
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
            
            # Output layer for language modeling
            outputs = Dense(
                self.vocab_size,
                activation='softmax',
                name='language_output'
            )(dense_dropout)
            
            self.model = Model(
                inputs=inputs,
                outputs=outputs,
                name='neuromimetic_language_foundation_model'
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
        Generate text using the trained neuromimetic language model.
        
        Args:
            seed_text (str): Initial text to start generation
            tokenizer: Keras tokenizer used during training
            max_length (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature for creativity control
            
        Returns:
            str: Generated text
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
    
    def get_model_info(self):
        """Get information about the neuromimetic model architecture."""
        return {
            "name": "Neuromimetic Language Foundation Model",
            "description": "Bio-inspired language model with spiking neural dynamics",
            "features": [
                "Dual DenseGSER spiking neural layers",
                "BioplasticDenseLayer Hebbian learning",
                "LSTM temporal processing",
                "Homeostatic plasticity regulation",
                "Advanced text generation capabilities"
            ],
            "parameters": {
                "vocab_size": self.vocab_size,
                "sequence_length": self.seq_len,
                "embedding_dim": self.embed_dim,
                "hidden_dim": self.hidden_dim
            }
        }


# For backward compatibility, keep the legacy class name as an alias
NeuromimeticLanguageFoundationModel = NeuromimeticLanguageModel


class HierarchicalResonanceFoundationModel:
    """
    Hierarchical Neural Resonance Foundation Model
    
    A deep neuromimetic architecture implementing the full Hierarchical Neural Resonance
    mechanism for deliberative "System 2" reasoning. This model features:
    
    Architecture:
    - Multi-level ResonantGSER hierarchy with bi-directional feedback
    - Cross-level skip connections for multi-scale resonance
    - Temporal coherence integration via LatentTemporalCoherence
    - Multi-pathway feature fusion with attention-based aggregation
    - Bioplastic output layers with Hebbian learning
    
    Biological Principles:
    - Prospective Configuration: Neural activities optimized before weight updates
    - Neural Resonance: Iterative "Thinking Phase" for hierarchical alignment
    - Harmonization: Local error correction through top-down feedback
    - Prediction Divergence: Driving signal for resonance-based learning
    
    This model is designed to work with the NeuralResonanceCallback to orchestrate
    the resonance cycles during training.
    
    References:
        - docs/NEURAL_RESONANCE.md for theoretical background
        - Adaptive Resonance Theory (ART) by Grossberg
        - Predictive Coding theories of brain function
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
        dropout_rate=0.1
    ):
        """
        Initialize the Hierarchical Resonance Foundation Model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            seq_len (int): Length of input sequences
            embed_dim (int): Embedding dimension
            hidden_dim (int): Hidden layer dimension for resonance layers
            num_resonance_levels (int): Number of hierarchical resonance levels (2-6)
            resonance_factor (float): Strength of resonance feedback (0.05-0.3)
            use_temporal_coherence (bool): Whether to use LatentTemporalCoherence
            use_attention_fusion (bool): Whether to use attention for pathway fusion
            dropout_rate (float): Dropout rate for regularization
        """
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_resonance_levels = max(2, min(6, num_resonance_levels))
        self.resonance_factor = resonance_factor
        self.use_temporal_coherence = use_temporal_coherence
        self.use_attention_fusion = use_attention_fusion
        self.dropout_rate = dropout_rate
        self.model = None
        self.resonant_layers = []
        
    def _create_resonance_level(self, x, level_idx, units, spectral_radius, leak_rate, spike_threshold):
        """Create a single resonance level with normalization and dropout."""
        resonant = ResonantGSER(
            units=units,
            spectral_radius=spectral_radius,
            leak_rate=leak_rate,
            spike_threshold=spike_threshold,
            activation='gelu',
            resonance_factor=self.resonance_factor,
            name=f'resonant_level_{level_idx}'
        )(x)
        
        # Store reference for callback access
        self.resonant_layers.append(resonant)
        
        # Normalization and dropout for stability
        normed = LayerNormalization(
            epsilon=1e-6, 
            name=f'layer_norm_{level_idx}'
        )(resonant)
        dropped = Dropout(
            self.dropout_rate, 
            name=f'dropout_{level_idx}'
        )(normed)
        
        return dropped, resonant
    
    def build_model(self):
        """
        Build the Hierarchical Resonance Foundation Model architecture.
        
        The architecture creates a deep hierarchy of ResonantGSER layers with:
        - Gradually decreasing spectral radius for stability
        - Varying leak rates for multi-scale temporal dynamics
        - Skip connections for cross-level information flow
        - Multiple pooling pathways for feature extraction
        
        Returns:
            tf.keras.Model: The compiled Keras model
        """
        self.resonant_layers = []
        
        with tf.device('/CPU:0'):
            # === Input and Embedding ===
            inputs = Input(shape=(self.seq_len,), name='text_input')
            
            # Embedding with positional encoding
            embedded = Embedding(
                self.vocab_size, 
                self.embed_dim,
                name='token_embedding'
            )(inputs)
            
            # Positional encoding for sequence awareness
            positioned = PositionalEncodingLayer(
                max_position=self.seq_len,
                d_model=self.embed_dim,
                name='positional_encoding'
            )(embedded)
            
            # Initial projection to hidden dimension
            projected = Dense(
                self.hidden_dim, 
                activation='gelu',
                name='input_projection'
            )(positioned)
            
            # === Hierarchical Resonance Levels ===
            # Create multi-level resonance hierarchy with varying dynamics
            level_outputs = []
            skip_connections = []
            current = projected
            
            for level in range(self.num_resonance_levels):
                # Calculate level-specific parameters
                # Higher levels have lower spectral radius for stability
                # and higher leak rates for faster dynamics
                spectral_radius = 0.95 - (level * 0.08)
                leak_rate = 0.08 + (level * 0.03)
                spike_threshold = 0.4 - (level * 0.05)
                
                # Determine units (can vary by level for hierarchical abstraction)
                level_units = self.hidden_dim
                
                # Create resonance level
                level_out, raw_resonant = self._create_resonance_level(
                    current,
                    level_idx=level,
                    units=level_units,
                    spectral_radius=spectral_radius,
                    leak_rate=leak_rate,
                    spike_threshold=spike_threshold
                )
                
                level_outputs.append(level_out)
                skip_connections.append(raw_resonant)
                
                # Add skip connection from previous levels (cross-level resonance)
                if level > 0:
                    # Residual connection with previous level
                    current = Add(name=f'skip_add_{level}')([level_out, current])
                else:
                    current = level_out
            
            # === Temporal Processing ===
            # LSTM for sequential temporal dependencies
            lstm_out = LSTM(
                self.hidden_dim,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate / 2,
                name='temporal_lstm'
            )(current)
            
            # === Dynamic Reservoir Processing ===
            # RNN with GSER cell for spiking dynamics
            gser_cell = GSER(
                input_dim=self.hidden_dim,
                initial_reservoir_size=self.hidden_dim,
                max_dynamic_reservoir_dim=self.hidden_dim * 2,
                spectral_radius=0.9,
                leak_rate=0.1,
                spike_threshold=0.3,
                name='gser_cell'
            )
            gser_rnn = RNN(
                gser_cell, 
                return_sequences=True,
                name='gser_reservoir'
            )(lstm_out)
            
            # === Temporal Coherence (Optional) ===
            if self.use_temporal_coherence:
                # Distill temporal dynamics into coherence vector
                coherence_dim = min(64, self.hidden_dim)
                temporal_coherence = LatentTemporalCoherence(
                    d_coherence=coherence_dim,
                    name='temporal_coherence'
                )(gser_rnn)
            
            # === Multi-Pathway Feature Extraction ===
            pathways = []
            
            # Pathway 1: Global average pooling of final resonance
            avg_pool_final = GlobalAveragePooling1D(name='avg_pool_final')(current)
            pathways.append(avg_pool_final)
            
            # Pathway 2: Global max pooling for salient features
            max_pool_final = GlobalMaxPooling1D(name='max_pool_final')(current)
            pathways.append(max_pool_final)
            
            # Pathway 3: LSTM temporal features
            avg_pool_lstm = GlobalAveragePooling1D(name='avg_pool_lstm')(lstm_out)
            pathways.append(avg_pool_lstm)
            
            # Pathway 4: GSER reservoir features
            avg_pool_gser = GlobalAveragePooling1D(name='avg_pool_gser')(gser_rnn)
            pathways.append(avg_pool_gser)
            
            # Pathway 5: Multi-level skip connection fusion
            skip_pooled = []
            for i, skip in enumerate(skip_connections):
                skip_pool = GlobalAveragePooling1D(name=f'skip_pool_{i}')(skip)
                skip_pooled.append(skip_pool)
            skip_fused = Concatenate(name='skip_fusion')(skip_pooled)
            skip_compressed = Dense(
                self.hidden_dim, 
                activation='gelu',
                name='skip_compression'
            )(skip_fused)
            pathways.append(skip_compressed)
            
            # Add temporal coherence if enabled
            if self.use_temporal_coherence:
                pathways.append(temporal_coherence)
            
            # === Pathway Fusion ===
            if self.use_attention_fusion and len(pathways) > 2:
                # Stack pathways for attention-based fusion
                pathway_stack = tf.stack(pathways, axis=1)  # (batch, num_pathways, features)
                
                # Self-attention over pathways
                attention_fusion = MultiheadLinearSelfAttentionKernalization(
                    d_model=self.hidden_dim,
                    num_heads=4,
                    dropout_rate=self.dropout_rate,
                    name='pathway_attention'
                )(pathway_stack)
                
                # Aggregate attention output
                fused = GlobalAveragePooling1D(name='attention_aggregate')(attention_fusion)
            else:
                # Simple concatenation fusion
                fused = Concatenate(name='pathway_fusion')(pathways)
            
            # === Bioplastic Processing ===
            # Hebbian learning with homeostatic plasticity
            bioplastic1 = BioplasticDenseLayer(
                units=self.hidden_dim * 2,
                learning_rate=1e-3,
                anti_hebbian_rate=0.1,
                target_avg=0.12,
                homeostatic_rate=5e-5,
                bcm_tau=800.0,
                activation='gelu',
                dropout_rate=self.dropout_rate,
                name='bioplastic_1'
            )(fused)
            
            bioplastic2 = BioplasticDenseLayer(
                units=self.hidden_dim,
                learning_rate=8e-4,
                anti_hebbian_rate=0.08,
                target_avg=0.1,
                homeostatic_rate=3e-5,
                bcm_tau=1000.0,
                activation='gelu',
                dropout_rate=self.dropout_rate / 2,
                name='bioplastic_2'
            )(bioplastic1)
            
            # === Output Layer ===
            # Final dense processing
            dense_out = Dense(
                self.hidden_dim,
                activation='gelu',
                name='output_dense'
            )(bioplastic2)
            
            dense_out = Dropout(self.dropout_rate / 2, name='output_dropout')(dense_out)
            
            # Output projection
            outputs = Dense(
                self.vocab_size,
                activation='softmax',
                name='language_output'
            )(dense_out)
            
            # === Build Model ===
            self.model = Model(
                inputs=inputs,
                outputs=outputs,
                name='hierarchical_resonance_foundation_model'
            )
        
        return self.model
    
    def compile_model(self, learning_rate=5e-4, warmup_steps=1000):
        """
        Compile the model with optimized settings for resonance training.
        
        Args:
            learning_rate (float): Base learning rate
            warmup_steps (int): Number of warmup steps for learning rate
            
        Returns:
            tf.keras.Model: The compiled model
        """
        if self.model is None:
            raise ValueError("Model must be built before compiling. Call build_model() first.")
        
        # Adam optimizer with gradient clipping for stability
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def get_resonant_layers(self):
        """
        Get references to all ResonantGSER layers for the NeuralResonanceCallback.
        
        Returns:
            list: List of ResonantGSER layer references
        """
        if self.model is None:
            return []
        
        resonant_layers = []
        for layer in self.model.layers:
            if 'ResonantGSER' in str(type(layer)) or 'resonant_level' in layer.name:
                resonant_layers.append(layer)
        return resonant_layers
    
    def generate_text(self, seed_text, tokenizer, max_length=100, temperature=0.8, top_p=0.9):
        """
        Generate text using the trained Hierarchical Resonance model.
        
        Uses nucleus sampling with temperature control for high-quality generation.
        
        Args:
            seed_text (str): Initial text to start generation
            tokenizer: Keras tokenizer used during training
            max_length (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature (0.5-1.5)
            top_p (float): Nucleus sampling threshold
            
        Returns:
            str: Generated text
        """
        if self.model is None:
            raise ValueError("Model must be built before text generation.")
        
        # Create reverse mapping
        reverse_tokenizer = {v: k for k, v in tokenizer.word_index.items()}
        
        # Convert seed to tokens
        seed_tokens = tokenizer.texts_to_sequences([seed_text.lower()])[0]
        if not seed_tokens:
            seed_tokens = [1]
        
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
            
            # Apply temperature
            pred = np.log(pred + 1e-10) / temperature
            pred = np.exp(pred) / np.sum(np.exp(pred))
            
            # Nucleus (top-p) sampling
            sorted_indices = np.argsort(pred)[::-1]
            cumsum_probs = np.cumsum(pred[sorted_indices])
            cutoff_idx = np.searchsorted(cumsum_probs, top_p) + 1
            cutoff_idx = max(1, min(cutoff_idx, len(pred)))
            
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
            if word in [".", "!", "?"] and len(generated_words) > 10:
                break
        
        return " ".join(generated_words)
    
    def get_model_info(self):
        """Get detailed information about the model architecture."""
        return {
            "name": "Hierarchical Neural Resonance Foundation Model",
            "description": "Deep neuromimetic architecture with bi-directional resonance and deliberative reasoning",
            "architecture": {
                "resonance_levels": self.num_resonance_levels,
                "resonance_factor": self.resonance_factor,
                "temporal_coherence": self.use_temporal_coherence,
                "attention_fusion": self.use_attention_fusion,
            },
            "features": [
                f"{self.num_resonance_levels}-level ResonantGSER hierarchy",
                "Bi-directional feedback projections",
                "Cross-level skip connections",
                "LSTM + GSER temporal processing",
                "LatentTemporalCoherence distillation" if self.use_temporal_coherence else "Standard temporal pooling",
                "Multi-pathway attention fusion" if self.use_attention_fusion else "Concatenation fusion",
                "Dual BioplasticDenseLayer with BCM plasticity",
                "Hebbian learning with anti-Hebbian decorrelation",
                "Homeostatic activity regulation",
            ],
            "biological_principles": [
                "Prospective Configuration",
                "Neural Resonance (System 2 thinking)",
                "Harmonization via feedback",
                "Prediction Divergence minimization",
                "Spiking neural dynamics",
                "BCM metaplasticity",
            ],
            "parameters": {
                "vocab_size": self.vocab_size,
                "sequence_length": self.seq_len,
                "embedding_dim": self.embed_dim,
                "hidden_dim": self.hidden_dim,
                "dropout_rate": self.dropout_rate,
            },
            "recommended_training": {
                "callback": "NeuralResonanceCallback(resonance_cycles=5-15)",
                "batch_size": "32-64",
                "learning_rate": "1e-4 to 5e-4",
                "epochs": "Depends on dataset size",
            }
        }
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            print("Model not built yet. Call build_model() first.")
            return
        
        print("\n" + "=" * 70)
        print("  HIERARCHICAL NEURAL RESONANCE FOUNDATION MODEL")
        print("=" * 70)
        info = self.get_model_info()
        
        print(f"\n Architecture:")
        for key, value in info["architecture"].items():
            print(f"   • {key}: {value}")
        
        print(f"\n Features:")
        for feature in info["features"]:
            print(f"   ✓ {feature}")
        
        print(f"\n Biological Principles:")
        for principle in info["biological_principles"]:
            print(f"   • {principle}")
        
        print(f"\n Parameters:")
        for key, value in info["parameters"].items():
            print(f"   • {key}: {value}")
        
        print(f"\n Recommended Training:")
        for key, value in info["recommended_training"].items():
            print(f"   • {key}: {value}")
        
        print("\n" + "=" * 70)
        print("\n Keras Model Summary:")
        self.model.summary()


def load_neuromimetic_model(model_path, tokenizer_path=None):
    """
    Load a pre-trained neuromimetic language model.
    
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
# These provide compatibility with older references while using the main neuromimetic architecture
DSTSMGSER = NeuromimeticLanguageModel  # Dynamic Spatio-Temporal Self-Modeling Gated Spiking Elastic Reservoir
GSERModel = NeuromimeticLanguageModel  # Simplified Gated Spiking Elastic Reservoir Model  
CoherentThoughtModel = NeuromimeticLanguageModel  # Coherent Thought Model