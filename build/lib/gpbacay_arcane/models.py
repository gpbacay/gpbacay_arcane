import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RNN, Input, BatchNormalization, Flatten, Dropout, LayerNormalization
from gpbacay_arcane.layers import MultiheadLinearSelfAttentionKernalizationLayer
from gpbacay_arcane.layers import ExpandDimensionLayer
from gpbacay_arcane.layers import GSER
from gpbacay_arcane.layers import HebbianHomeostaticLayer
from gpbacay_arcane.layers import DenseGSER
from gpbacay_arcane.layers import SpatioTemporalSummaryMixingLayer
from gpbacay_arcane.layers import GatedMultiheadLinearSelfAttentionKernalization


class DSTSMGSER:
    def __init__(self, input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, max_dynamic_reservoir_dim, output_dim, use_weighted_summary=False):
        self.input_shape = input_shape
        self.reservoir_dim = reservoir_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_dynamic_reservoir_dim = max_dynamic_reservoir_dim
        self.output_dim = output_dim
        self.use_weighted_summary = use_weighted_summary
        self.model = None
        self.reservoir_layer = None

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        # Preprocessing
        x = BatchNormalization()(inputs)
        x = Flatten()(x)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Attention Layer
        gated_linear_attention_layer = GatedMultiheadLinearSelfAttentionKernalization(
            d_model=128, num_heads=8, use_weighted_summary=self.use_weighted_summary)
        x = ExpandDimensionLayer()(x)
        x = gated_linear_attention_layer(x)

        # Reservoir layer
        self.reservoir_layer = GSER(
            initial_reservoir_size=self.reservoir_dim,
            input_dim=x.shape[-1],
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold,
            max_dynamic_reservoir_dim=self.max_dynamic_reservoir_dim
        )
        lnn_layer = RNN(self.reservoir_layer, return_sequences=True)
        lnn_output = lnn_layer(x)

        # Hebbian homeostatic layer
        hebbian_homeostatic_layer = HebbianHomeostaticLayer(units=self.reservoir_dim, name='hebbian_homeostatic_layer')
        x = hebbian_homeostatic_layer(lnn_output)

        # Classification output
        clf_out = DenseGSER(
            units=self.output_dim,
            input_dim=x.shape[-1],
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold,
            max_dynamic_units=self.max_dynamic_reservoir_dim,
            activation='softmax',
            name='clf_out'
        )(Flatten()(x))

        # Self-modeling output
        sm_out = DenseGSER(
            units=np.prod(self.input_shape),
            input_dim=x.shape[-1],
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold,
            max_dynamic_units=self.max_dynamic_reservoir_dim,
            activation='sigmoid',
            name='sm_out'
        )(Flatten()(x))

        # Compile the model
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








# with Spatiotemporal Summary Mixing mechanism, denseGSER, and GSER
# 313/313 - 29s - 92ms/step - clf_out_accuracy: 0.9824 - clf_out_loss: 0.0773 - loss: 0.1038 - sm_out_loss: 0.0527 - sm_out_mse: 0.0527

# with Multihead Linear Self Attention Kernalization mechanism, denseGSER, and GSER
# 313/313 - 30s - 97ms/step - clf_out_accuracy: 0.9717 - clf_out_loss: 0.1470 - loss: 0.1734 - sm_out_loss: 0.0525 - sm_out_mse: 0.0525

# with Gated Multihead Linear Self Attention Kernalization mechanism, denseGSER, and GSER
# 313/313 - 31s - 98ms/step - clf_out_accuracy: 0.9764 - clf_out_loss: 0.1125 - loss: 0.1389 - sm_out_loss: 0.0526 - sm_out_mse: 0.0526

class DSTSMGSER_test1:
    def __init__(self, input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, max_dynamic_reservoir_dim, output_dim, use_weighted_summary=False):
        self.input_shape = input_shape
        self.reservoir_dim = reservoir_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_dynamic_reservoir_dim = max_dynamic_reservoir_dim
        self.output_dim = output_dim
        self.use_weighted_summary = use_weighted_summary
        self.model = None
        self.reservoir_layer = None

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        # Preprocessing
        x = BatchNormalization()(inputs)
        x = Flatten()(x)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Attention Layer
        gated_linear_attention_layer = GatedMultiheadLinearSelfAttentionKernalization(
            d_model=128, num_heads=8, use_weighted_summary=self.use_weighted_summary)
        x = ExpandDimensionLayer()(x)
        x = gated_linear_attention_layer(x)
        
        # linear_attention_layer = MultiheadLinearSelfAttentionKernalizationLayer(
        #     d_model=128, num_heads=8, use_weighted_summary=self.use_weighted_summary)
        # x = ExpandDimensionLayer()(x)
        # x = linear_attention_layer(x)
        
        # summary_mixing_layer = SpatioTemporalSummaryMixingLayer(d_model=128, use_weighted_summary=self.use_weighted_summary)
        # x = ExpandDimensionLayer()(x)
        # x = summary_mixing_layer(x)

        # Reservoir layer
        self.reservoir_layer = GSER(
            initial_reservoir_size=self.reservoir_dim,
            input_dim=x.shape[-1],
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold,
            max_dynamic_reservoir_dim=self.max_dynamic_reservoir_dim
        )
        lnn_layer = RNN(self.reservoir_layer, return_sequences=True)
        lnn_output = lnn_layer(x)

        # Hebbian homeostatic layer
        hebbian_homeostatic_layer = HebbianHomeostaticLayer(units=self.reservoir_dim, name='hebbian_homeostatic_layer')
        x = hebbian_homeostatic_layer(lnn_output)

        # Classification output
        clf_out = DenseGSER(
            units=self.output_dim,
            input_dim=x.shape[-1],
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold,
            max_dynamic_units=self.max_dynamic_reservoir_dim,
            activation='softmax',
            name='clf_out'
        )(Flatten()(x))

        # Self-modeling output
        sm_out = DenseGSER(
            units=np.prod(self.input_shape),
            input_dim=x.shape[-1],
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold,
            max_dynamic_units=self.max_dynamic_reservoir_dim,
            activation='sigmoid',
            name='sm_out'
        )(Flatten()(x))

        # Compile the model
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