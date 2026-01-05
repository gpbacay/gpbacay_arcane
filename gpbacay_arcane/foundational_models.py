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
    
    def run_resonance_cycle(self, num_cycles=1):
        """
        Execute the full RSAA algorithm across the hierarchy.
        
        This implements Algorithm 1 from the RSAA paper:
        1. Project (Top-Down): Higher layers send expectations to lower layers
        2. Harmonize (Bottom-Up): Lower layers adjust to reduce divergence
        3. Check Convergence: Monitor global divergence
        
        Args:
            num_cycles: Number of full hierarchy resonance cycles to run
        
        Returns:
            List of divergence values for each cycle
        """
        if not hasattr(self, 'resonant_layer_objects'):
            raise ValueError("Model must be built before running resonance cycles")
        
        divergences = []
        
        for cycle in range(num_cycles):
            # Step A: Project (Top-Down) - from highest to lowest
            for i in range(len(self.resonant_layer_objects) - 1, 0, -1):
                self.resonant_layer_objects[i].propagate_feedback_to_lower()
            
            # Step B: Harmonize (Bottom-Up) - implicit in next forward pass
            # The harmonization happens automatically when the model processes data
            # because each cell now has its resonance_alignment set
            
            # Step C: Check Convergence
            cycle_divergence = sum([
                layer.get_divergence() 
                for layer in self.resonant_layer_objects
            ])
            divergences.append(cycle_divergence)
            
            # Early stopping if converged
            if cycle_divergence < self.resonant_layer_objects[0]._cell.convergence_epsilon:
                break
        
        return divergences
    
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
