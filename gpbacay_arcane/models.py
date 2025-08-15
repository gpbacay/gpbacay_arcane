import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RNN, Input, BatchNormalization, Flatten, LayerNormalization, RepeatVector, Reshape, Conv2D, MaxPooling2D, Dropout, Embedding, Dense
from tensorflow.keras import Model

from gpbacay_arcane.layers import GSER, HebbianHomeostaticNeuroplasticity, DenseGSER, RelationalConceptModeling, RelationalGraphAttentionReasoning, LatentTemporalCoherence, BioplasticDenseLayer


# Test Accuracy: 0.9687, Total Loss: 0.1806
class DSTSMGSER(Model):
    """
    The Dynamic Spatio-Temporal Self-Modeling Gated Spiking Elastic Reservoir (DSTSMGSER) 
    is an advanced neuromorphic architecture designed to process complex spatio-temporal patterns 
    with high adaptability and efficiency.
    """

    def __init__(self, input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, 
                 max_dynamic_reservoir_dim, output_dim, use_weighted_summary=True, d_model=128, num_heads=8,
                 num_thought_steps=15, d_coherence=256,
                 activation='gelu', momentum=0.9, learning_rate=1e-5, target_avg=0.1, homeostatic_rate=1e-5,
                 min_scale=0.1, max_scale=2.0, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.reservoir_dim = reservoir_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_dynamic_reservoir_dim = max_dynamic_reservoir_dim
        self.output_dim = output_dim
        self.use_weighted_summary = use_weighted_summary
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_thought_steps = num_thought_steps
        self.d_coherence = d_coherence
        self.activation = tf.keras.activations.get(activation)
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.target_avg = target_avg
        self.homeostatic_rate = homeostatic_rate
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        self.reservoir_layer = None
        self.model = None

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        # Preprocessing
        x = Flatten()(inputs)
        x = DenseGSER(self.d_model)(x)
        x = Reshape((1, self.d_model))(x)

        # Relational Concept Modeling (RCM)
        rcm_layer = RelationalConceptModeling(
            d_model=self.d_model, num_heads=self.num_heads, use_weighted_summary=self.use_weighted_summary
        )
        x = rcm_layer(x)
        x = BatchNormalization()(x)
        x = LayerNormalization()(x)

        # Relational Graph Attention Reasoning (RGAR)
        rdl_layer = RelationalGraphAttentionReasoning(
            d_model=self.d_model, num_heads=self.num_heads, num_classes=self.d_model
        )
        x = rdl_layer(x)
        x = LayerNormalization()(x)

        # Create a sequence of "thought steps"
        x = Flatten()(x) # Squeeze from (batch, 1, d_model) to (batch, d_model)
        x = RepeatVector(self.num_thought_steps)(x)

        # Liquid Neural Network
        self.reservoir_layer = GSER(
            input_dim=self.d_model,
            initial_reservoir_size=self.reservoir_dim,
            max_dynamic_reservoir_dim=self.max_dynamic_reservoir_dim,
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold
        )
        x = RNN(self.reservoir_layer, return_sequences=True)(x)

        # Distill GSER history into a coherent thought vector
        coherence_layer = LatentTemporalCoherence(d_coherence=self.d_coherence)
        x = coherence_layer(x)

        # Bioplastic Dense Layer with Hebbian Learning and Homeostatic Neuroplasticity
        bioplastic_layer = BioplasticDenseLayer(
            units=self.d_coherence, # Adjusted to match the output of the coherence layer
            learning_rate=self.learning_rate,
            target_avg=self.target_avg,
            homeostatic_rate=self.homeostatic_rate,
            activation=self.activation,
            min_scale=self.min_scale,
            max_scale=self.max_scale,
            momentum=self.momentum,
            normalization='l2'
        )
        x = bioplastic_layer(x)

        # Classification main task
        clf_out = DenseGSER(
            units=self.output_dim,
            input_dim=self.d_coherence, # Adjusted input dimension
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold,
            max_dynamic_units=self.max_dynamic_reservoir_dim,
            activation='softmax',
            name='clf_out'
        )(x)

        # Self-modeling auxiliary task
        sm_out = DenseGSER(
            units=np.prod(self.input_shape),
            input_dim=self.d_coherence, # Adjusted input dimension
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold,
            max_dynamic_units=self.max_dynamic_reservoir_dim,
            activation='sigmoid',
            name='sm_out'
        )(x)

        # Model Compilation
        self.model = Model(inputs=inputs, outputs=[clf_out, sm_out])

    def compile_model(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
            loss={
                'clf_out': 'categorical_crossentropy',
                'sm_out': 'mse'
            },
            loss_weights={
                'clf_out': 1.0,
                'sm_out': 0.5
            },
            metrics={
                'clf_out': 'accuracy',
                'sm_out': 'mse'
            }
        )

    def get_config(self):
        return {
            'input_shape': self.input_shape,
            'reservoir_dim': self.reservoir_dim,
            'spectral_radius': self.spectral_radius,
            'leak_rate': self.leak_rate,
            'spike_threshold': self.spike_threshold,
            'max_dynamic_reservoir_dim': self.max_dynamic_reservoir_dim,
            'output_dim': self.output_dim,
            'use_weighted_summary': self.use_weighted_summary,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_thought_steps': self.num_thought_steps,
            'd_coherence': self.d_coherence
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)







class GSERModel(Model):
    """
    A simplified neural network model that uses only GSER and DenseGSER layers.
    This model is designed for spatio-temporal data processing.
    """

    def __init__(self, input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, 
                 max_dynamic_reservoir_dim, output_dim, d_model=128, activation='gelu', **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.reservoir_dim = reservoir_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_dynamic_reservoir_dim = max_dynamic_reservoir_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.activation = tf.keras.activations.get(activation)

        # Define layers
        self.flatten = Flatten()
        self.dense_gser = DenseGSER(self.d_model)
        self.gser_layer = GSER(
            input_dim=self.d_model,
            initial_reservoir_size=self.reservoir_dim,
            max_dynamic_reservoir_dim=self.max_dynamic_reservoir_dim,
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold
        )
        self.output_layer = DenseGSER(
            units=self.output_dim,
            input_dim=self.reservoir_dim,
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold,
            max_dynamic_units=self.max_dynamic_reservoir_dim,
            activation='softmax'
        )

    def call(self, inputs):
        # Preprocessing
        x = self.flatten(inputs)
        x = self.dense_gser(x)

        # GSER Layer (Reservoir Computing)
        x = self.gser_layer(x)

        # Output Layer
        outputs = self.output_layer(x)
        return outputs

    def build_model(self):
        inputs = Input(shape=self.input_shape)
        outputs = self.call(inputs)
        self.model = Model(inputs=inputs, outputs=outputs)

    def compile_model(self):
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def get_config(self):
        return {
            'input_shape': self.input_shape,
            'reservoir_dim': self.reservoir_dim,
            'spectral_radius': self.spectral_radius,
            'leak_rate': self.leak_rate,
            'spike_threshold': self.spike_threshold,
            'max_dynamic_reservoir_dim': self.max_dynamic_reservoir_dim,
            'output_dim': self.output_dim,
            'd_model': self.d_model
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)




# Test Accuracy: 0.9801, Loss: 0.0847
class CoherentThoughtModel(Model):
    """
    A model that uses a sequence of internal "thought steps" processed by a GSER 
    reservoir, and then distills the reservoir's history into a single latent
    representation using the LatentTemporalCoherence layer for classification.
    """
    def __init__(self, input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, 
                 max_dynamic_reservoir_dim, output_dim, d_model=128, num_thought_steps=10, 
                 d_coherence=256, activation='gelu', **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.reservoir_dim = reservoir_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_dynamic_reservoir_dim = max_dynamic_reservoir_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.num_thought_steps = num_thought_steps
        self.d_coherence = d_coherence
        self.activation = tf.keras.activations.get(activation)
        
        self.model = None

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        # 1. Preprocessing
        x = Flatten()(inputs)
        x = DenseGSER(self.d_model)(x)

        # 2. Thought Generation: Repeat the state vector to create a sequence
        x = RepeatVector(self.num_thought_steps)(x)

        # 3. Dynamic Processing: Process the sequence with GSER, returning the full history
        self.reservoir_layer = GSER(
            input_dim=self.d_model,
            initial_reservoir_size=self.reservoir_dim,
            max_dynamic_reservoir_dim=self.max_dynamic_reservoir_dim,
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold
        )
        # return_sequences=True is critical for the next layer
        x = RNN(self.reservoir_layer, return_sequences=True)(x) 

        # 4. Coherence Distillation: Distill the history into a single thought vector
        coherence_layer = LatentTemporalCoherence(d_coherence=self.d_coherence)
        x = coherence_layer(x)

        # 5. Classification
        outputs = DenseGSER(
            units=self.output_dim,
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold,
            activation='softmax',
            name='clf_out'
        )(x)

        self.model = Model(inputs=inputs, outputs=outputs)

    def compile_model(self):
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def get_config(self):
        return {
            'input_shape': self.input_shape,
            'reservoir_dim': self.reservoir_dim,
            'spectral_radius': self.spectral_radius,
            'leak_rate': self.leak_rate,
            'spike_threshold': self.spike_threshold,
            'max_dynamic_reservoir_dim': self.max_dynamic_reservoir_dim,
            'output_dim': self.output_dim,
            'd_model': self.d_model,
            'num_thought_steps': self.num_thought_steps,
            'd_coherence': self.d_coherence,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class NeuromimeticLanguageModel(Model):
    """
    A small language model that uses custom neuromimetic layers for text generation.
    
    Architecture:
    - Embedding layer for token representation
    - Multi-head attention with kernel approximation for efficient sequence modeling
    - GSER (Gated Spiking Elastic Reservoir) for dynamic temporal processing
    - Latent Temporal Coherence for sequence summarization
    - Bioplastic Dense Layer with Hebbian learning for adaptive readout
    - DenseGSER for final token prediction
    
    This model is designed for small-scale language modeling tasks and can be trained
    on datasets like Shakespeare text or small corpora.
    """
    
    def __init__(
        self,
        vocab_size,
        max_sequence_length,
        embedding_dim=128,
        d_model=256,
        num_heads=8,
        reservoir_dim=512,
        spectral_radius=1.2,
        leak_rate=0.1,
        spike_threshold=0.3,
        max_dynamic_reservoir_dim=1024,
        d_coherence=128,
        num_thought_steps=8,
        activation='gelu',
        learning_rate=1e-4,
        target_avg=0.1,
        homeostatic_rate=1e-5,
        min_scale=0.1,
        max_scale=2.0,
        momentum=0.9,
        dropout_rate=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.reservoir_dim = reservoir_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_dynamic_reservoir_dim = max_dynamic_reservoir_dim
        self.d_coherence = d_coherence
        self.num_thought_steps = num_thought_steps
        self.activation = tf.keras.activations.get(activation)
        self.learning_rate = learning_rate
        self.target_avg = target_avg
        self.homeostatic_rate = homeostatic_rate
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.momentum = momentum
        self.dropout_rate = dropout_rate
        
        self.reservoir_layer = None
        self.model = None
    
    def build_model(self):
        # Input: token indices
        inputs = Input(shape=(self.max_sequence_length,), dtype='int32')
        
        # Token embedding
        x = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name='token_embedding'
        )(inputs)
        
        # Project to model dimension
        x = DenseGSER(self.d_model)(x)
        x = LayerNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Multi-head attention for sequence modeling
        from gpbacay_arcane.layers import MultiheadLinearSelfAttentionKernalization
        attention_layer = MultiheadLinearSelfAttentionKernalization(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            use_weighted_summary=False
        )
        x = attention_layer(x)
        x = LayerNormalization()(x)
        
        # Relational concept modeling to extract key concepts
        rcm_layer = RelationalConceptModeling(
            d_model=self.d_model,
            num_heads=self.num_heads,
            use_weighted_summary=True
        )
        concept_vector = rcm_layer(x)  # (batch, 1, d_model)
        concept_vector = LayerNormalization()(concept_vector)
        
        # Create thought steps for deeper processing
        thought_sequence = Flatten()(concept_vector)
        thought_sequence = RepeatVector(self.num_thought_steps)(thought_sequence)
        
        # GSER for dynamic temporal processing
        self.reservoir_layer = GSER(
            input_dim=self.d_model,
            initial_reservoir_size=self.reservoir_dim,
            max_dynamic_reservoir_dim=self.max_dynamic_reservoir_dim,
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold
        )
        reservoir_states = RNN(self.reservoir_layer, return_sequences=True)(thought_sequence)
        
        # Temporal coherence extraction
        coherence_layer = LatentTemporalCoherence(d_coherence=self.d_coherence)
        coherent_representation = coherence_layer(reservoir_states)
        
        # Bioplastic readout with Hebbian learning
        bioplastic_layer = BioplasticDenseLayer(
            units=self.d_coherence,
            learning_rate=self.learning_rate,
            target_avg=self.target_avg,
            homeostatic_rate=self.homeostatic_rate,
            activation=self.activation,
            min_scale=self.min_scale,
            max_scale=self.max_scale,
            momentum=self.momentum,
            normalization='l2',
            dropout_rate=self.dropout_rate
        )
        processed_features = bioplastic_layer(coherent_representation)
        
        # Final token prediction using DenseGSER
        output_logits = DenseGSER(
            units=self.vocab_size,
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold,
            activation='softmax',
            name='token_predictions'
        )(processed_features)
        
        self.model = Model(inputs=inputs, outputs=output_logits)
    
    def compile_model(self, learning_rate=1e-3):
        # Custom perplexity metric
        def perplexity(y_true, y_pred):
            cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            return tf.exp(cross_entropy)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', perplexity]
        )
    
    def generate_text(self, seed_text, tokenizer, max_length=100, temperature=1.0):
        """Generate text using the trained model."""
        if not self.model:
            raise ValueError("Model must be built and trained before text generation")
        
        # Convert seed text to tokens
        seed_tokens = tokenizer.texts_to_sequences([seed_text])[0]
        
        # Pad or truncate to model's expected length
        if len(seed_tokens) > self.max_sequence_length:
            seed_tokens = seed_tokens[-self.max_sequence_length:]
        else:
            seed_tokens = [0] * (self.max_sequence_length - len(seed_tokens)) + seed_tokens
        
        generated_tokens = seed_tokens.copy()
        
        for _ in range(max_length):
            # Prepare input
            input_sequence = np.array([generated_tokens[-self.max_sequence_length:]])
            
            # Predict next token
            predictions = self.model.predict(input_sequence, verbose=0)[0]
            
            # Apply temperature sampling
            predictions = predictions / temperature
            predictions = tf.nn.softmax(predictions).numpy()
            
            # Sample next token
            next_token = np.random.choice(len(predictions), p=predictions)
            
            # Add to sequence
            generated_tokens.append(next_token)
            
            # Stop if we hit end token (if using one)
            if next_token == 0:  # Assuming 0 is padding/end token
                break
        
        # Convert back to text
        generated_sequence = generated_tokens[len(seed_tokens):]
        generated_text = tokenizer.sequences_to_texts([generated_sequence])[0]
        
        return generated_text
    
    def get_config(self):
        return {
            'vocab_size': self.vocab_size,
            'max_sequence_length': self.max_sequence_length,
            'embedding_dim': self.embedding_dim,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'reservoir_dim': self.reservoir_dim,
            'spectral_radius': self.spectral_radius,
            'leak_rate': self.leak_rate,
            'spike_threshold': self.spike_threshold,
            'max_dynamic_reservoir_dim': self.max_dynamic_reservoir_dim,
            'd_coherence': self.d_coherence,
            'num_thought_steps': self.num_thought_steps,
            'dropout_rate': self.dropout_rate
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


# A.R.C.A.N.E (Augmented Reconstruction of Consciousness through Artificial Neural Evolution)

# Test Accuracy: 0.9943, Loss: 0.0386
class HighAccuracyDigitModel(Model):
    """
    A high-accuracy MNIST model that combines a CNN front-end with custom neuromimetic
    layers (DenseGSER, GSER, LatentTemporalCoherence, BioplasticDenseLayer, RCM) and
    produces both classification and self-modeling outputs.

    The model preserves the multi-output interface (`clf_out`, `sm_out`) and the
    `reservoir_layer` attribute for compatibility with existing callbacks and
    test-time code.
    """

    def __init__(
        self,
        input_shape,
        output_dim,
        reservoir_dim,
        spectral_radius,
        leak_rate,
        spike_threshold,
        max_dynamic_reservoir_dim,
        d_model=128,
        num_heads=8,
        num_thought_steps=8,
        d_coherence=256,
        use_weighted_summary=True,
        activation='gelu',
        momentum=0.9,
        learning_rate=1e-4,
        target_avg=0.1,
        homeostatic_rate=1e-5,
        min_scale=0.1,
        max_scale=2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.reservoir_dim = reservoir_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_dynamic_reservoir_dim = max_dynamic_reservoir_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_thought_steps = num_thought_steps
        self.d_coherence = d_coherence
        self.use_weighted_summary = use_weighted_summary
        self.activation = tf.keras.activations.get(activation)
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.target_avg = target_avg
        self.homeostatic_rate = homeostatic_rate
        self.min_scale = min_scale
        self.max_scale = max_scale

        self.reservoir_layer = None
        self.model = None

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        # CNN front-end
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # Project into custom DenseGSER embedding
        x = Flatten()(x)
        x = DenseGSER(self.d_model)(x)
        x = Reshape((1, self.d_model))(x)

        # Relational Concept Modeling
        rcm_layer = RelationalConceptModeling(
            d_model=self.d_model,
            num_heads=self.num_heads,
            use_weighted_summary=self.use_weighted_summary,
        )
        x = rcm_layer(x)
        x = LayerNormalization()(x)

        # Thought steps
        x = Flatten()(x)
        x = RepeatVector(self.num_thought_steps)(x)

        # GSER recurrent processing
        self.reservoir_layer = GSER(
            input_dim=self.d_model,
            initial_reservoir_size=self.reservoir_dim,
            max_dynamic_reservoir_dim=self.max_dynamic_reservoir_dim,
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold,
        )
        x = RNN(self.reservoir_layer, return_sequences=True)(x)

        # Coherence distillation
        coherence_layer = LatentTemporalCoherence(d_coherence=self.d_coherence)
        x = coherence_layer(x)

        # Bioplastic readout
        x = BioplasticDenseLayer(
            units=self.d_coherence,
            learning_rate=self.learning_rate,
            target_avg=self.target_avg,
            homeostatic_rate=self.homeostatic_rate,
            activation=self.activation,
            min_scale=self.min_scale,
            max_scale=self.max_scale,
            momentum=self.momentum,
            normalization='l2',
        )(x)

        # Outputs
        clf_out = DenseGSER(
            units=self.output_dim,
            input_dim=self.d_coherence,
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold,
            max_dynamic_units=self.max_dynamic_reservoir_dim,
            activation='softmax',
            name='clf_out',
        )(x)

        sm_out = DenseGSER(
            units=np.prod(self.input_shape),
            input_dim=self.d_coherence,
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold,
            max_dynamic_units=self.max_dynamic_reservoir_dim,
            activation='sigmoid',
            name='sm_out',
        )(x)

        self.model = Model(inputs=inputs, outputs=[clf_out, sm_out])

    def compile_model(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
            loss={
                'clf_out': 'categorical_crossentropy',
                'sm_out': 'mse',
            },
            loss_weights={
                'clf_out': 1.0,
                'sm_out': 0.3,
            },
            metrics={
                'clf_out': 'accuracy',
                'sm_out': 'mse',
            },
        )

    def get_config(self):
        return {
            'input_shape': self.input_shape,
            'output_dim': self.output_dim,
            'reservoir_dim': self.reservoir_dim,
            'spectral_radius': self.spectral_radius,
            'leak_rate': self.leak_rate,
            'spike_threshold': self.spike_threshold,
            'max_dynamic_reservoir_dim': self.max_dynamic_reservoir_dim,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_thought_steps': self.num_thought_steps,
            'd_coherence': self.d_coherence,
            'use_weighted_summary': self.use_weighted_summary,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)