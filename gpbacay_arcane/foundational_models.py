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
    ExpandDimensionLayer,
)

class HierarchicalResonanceFoundationModel:
    """
    Hierarchical Neural Resonance Foundation Model
    
    A deep neuromimetic architecture implementing the full Hierarchical Neural Resonance
    mechanism for deliberative "System 2" reasoning. 
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
        
    def _create_resonance_level(self, x, level_idx, units, resonance_factor, spike_threshold):
        resonant = ResonantGSER(
            units=units,
            spike_threshold=spike_threshold,
            resonance_factor=resonance_factor,
            return_sequences=True,
            name=f'resonant_level_{level_idx}'
        )(x)
        self.resonant_layers.append(resonant)
        normed = LayerNormalization(epsilon=1e-6, name=f'layer_norm_{level_idx}')(resonant)
        dropped = Dropout(self.dropout_rate, name=f'dropout_{level_idx}')(normed)
        return dropped, resonant
    
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
            current = projected
            for level in range(self.num_resonance_levels):
                # Calculate level-specific parameters
                # Higher levels have higher resonance for deeper deliberation
                resonance_factor = self.resonance_factor + (level * 0.05)
                spike_threshold = 0.4 - (level * 0.05)
                level_units = self.hidden_dim
                level_out, raw_resonant = self._create_resonance_level(current, level, level_units, resonance_factor, spike_threshold)
                level_outputs.append(level_out)
                skip_connections.append(raw_resonant)
                if level > 0:
                    current = Add(name=f'skip_add_{level}')([level_out, current])
                else:
                    current = level_out
            
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
            if self.use_attention_fusion:
                gate = Dense(fused.shape[-1] or self.hidden_dim * 4, activation='sigmoid', name='pathway_gate')(fused)
                fused = Multiply(name='gated_fusion')([fused, gate])
            
            bioplastic1 = BioplasticDenseLayer(units=self.hidden_dim * 2, normalization='l2', activation='gelu', name='bioplastic_1')(fused)
            bioplastic2 = BioplasticDenseLayer(units=self.hidden_dim, normalization='l2', activation='gelu', name='bioplastic_2')(bioplastic1)
            
            dense_out = Dense(self.hidden_dim, activation='gelu', name='output_dense')(bioplastic2)
            dense_out = Dropout(self.dropout_rate / 2, name='output_dropout')(dense_out)
            outputs = Dense(self.vocab_size, activation='softmax', name='language_output')(dense_out)
            self.model = Model(inputs=inputs, outputs=outputs, name='hierarchical_resonance_foundation_model')
        return self.model
    
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
            "hidden_dim": self.hidden_dim
        }
