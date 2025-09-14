"""
gpbacay_arcane.models

This module contains the neuromimetic language model architectures for the A.R.C.A.N.E. project.
Augmented Reconstruction of Consciousness through Artificial Neural Evolution.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, GlobalAveragePooling1D, Dropout, LayerNormalization, Concatenate
from tensorflow.keras import Model

from gpbacay_arcane.layers import DenseGSER, BioplasticDenseLayer


class NeuromimeticLanguageModel:
    """
    Neuromimetic Language Foundation Model
    
    A novel language model architecture that incorporates biological neural principles:
    - Spiking neural dynamics via DenseGSER layers
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
            
            # First DenseGSER layer - Primary spiking neural processing
            gser1 = DenseGSER(
                units=self.hidden_dim,
                spectral_radius=0.9,
                leak_rate=0.1,
                spike_threshold=0.35,
                activation='gelu',
                name='dense_gser_1'
            )(embedded)
            
            # Layer normalization and dropout for stability
            gser1_norm = LayerNormalization(name='layer_norm_1')(gser1)
            gser1_drop = Dropout(0.15, name='dropout_1')(gser1_norm)
            
            # Second DenseGSER layer - Secondary spiking neural processing
            gser2 = DenseGSER(
                units=self.hidden_dim,
                spectral_radius=0.8,
                leak_rate=0.12,
                spike_threshold=0.3,
                activation='gelu',
                name='dense_gser_2'
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


def load_neuromimetic_model(model_path, tokenizer_path=None):
    """
    Load a pre-trained neuromimetic language model.
    
    Args:
        model_path (str): Path to the saved model file
        tokenizer_path (str): Path to the saved tokenizer file
        
    Returns:
        tuple: (model, tokenizer) if tokenizer_path provided, else just model
    """
    # Custom objects for loading
    custom_objects = {
        'DenseGSER': DenseGSER,
        'BioplasticDenseLayer': BioplasticDenseLayer,
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