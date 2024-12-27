import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RNN, Input, BatchNormalization, Flatten, LayerNormalization
from tensorflow.keras import Model

from gpbacay_arcane.layers import GSER
from gpbacay_arcane.layers import HebbianHomeostaticNeuroplasticity
from gpbacay_arcane.layers import DenseGSER
from gpbacay_arcane.layers import RelationalConceptModeling
from gpbacay_arcane.layers import RelationalGraphAttentionReasoning



class DSTSMGSER(Model):
    """
    The Dynamic Spatio-Temporal Self-Modeling Gated Spiking Elastic Reservoir (DSTSMGSER) 
    is an advanced neuromorphic architecture designed to process complex spatio-temporal patterns 
    with high adaptability and efficiency.
    """

    def __init__(self, input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, 
                 max_dynamic_reservoir_dim, output_dim, use_weighted_summary=True, d_model=128, num_heads=8):
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
            units=self.reservoir_dim
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
        self.model = tf.keras.Model(inputs=inputs, outputs=[clf_out, sm_out])

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




