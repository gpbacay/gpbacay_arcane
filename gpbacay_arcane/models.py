import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RNN, Input, BatchNormalization, Flatten, LayerNormalization
from tensorflow.keras import Model

from gpbacay_arcane.layers import GSER, HebbianHomeostaticNeuroplasticity, DenseGSER, RelationalConceptModeling, RelationalGraphAttentionReasoning



class DSTSMGSER(Model):
    """
    The Dynamic Spatio-Temporal Self-Modeling Gated Spiking Elastic Reservoir (DSTSMGSER) 
    is an advanced neuromorphic architecture designed to process complex spatio-temporal patterns 
    with high adaptability and efficiency.
    """

    def __init__(self, input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, 
                 max_dynamic_reservoir_dim, output_dim, use_weighted_summary=True, d_model=128, num_heads=8,
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

        # Liquid Neural Network
        self.reservoir_layer = GSER(
            input_dim=self.d_model,
            initial_reservoir_size=self.reservoir_dim,
            max_dynamic_reservoir_dim=self.max_dynamic_reservoir_dim,
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold
        )
        x = RNN(self.reservoir_layer)(x)

        # Hebbian Learning and Homeostatic Neuroplasticity
        hebbian_homeostatic_layer = HebbianHomeostaticNeuroplasticity(
            units=self.reservoir_dim,
            learning_rate=self.learning_rate,
            target_avg=self.target_avg,
            homeostatic_rate=self.homeostatic_rate,
            activation=self.activation,
            min_scale=self.min_scale,
            max_scale=self.max_scale,
            momentum=self.momentum
        )
        x = hebbian_homeostatic_layer(x)

        # Classification main task
        clf_out = DenseGSER(
            units=self.output_dim,
            input_dim=self.reservoir_dim,
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
            input_dim=self.reservoir_dim,
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
            optimizer='adam',
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
            'num_heads': self.num_heads
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

