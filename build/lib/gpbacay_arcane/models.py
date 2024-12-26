import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, RNN, Input, BatchNormalization, Flatten, Dropout, LayerNormalization
from tensorflow.keras import Model

from gpbacay_arcane.layers import MultiheadLinearSelfAttentionKernalizationLayer
from gpbacay_arcane.layers import ExpandDimensionLayer
from gpbacay_arcane.layers import GSER
from gpbacay_arcane.layers import HebbianHomeostaticNeuroplasticity
from gpbacay_arcane.layers import DenseGSER
from gpbacay_arcane.layers import SpatioTemporalSummaryMixingLayer
from gpbacay_arcane.layers import GatedMultiheadLinearSelfAttentionKernalization
from gpbacay_arcane.layers import SpatioTemporalSummarization
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









































# # with Spatio Temporal Summarization mechanism, denseGSER, and GSER
# # 313/313 - 8s - 25ms/step - clf_out_accuracy: 0.9823 - clf_out_loss: 0.0815 - loss: 0.1079 - sm_out_loss: 0.0525 - sm_out_mse: 0.0525
# # Test Accuracy: 0.9823, Loss: 0.1079
# class DSTSMGSER_test2:
#     """
#     The Dynamic Spatio-Temporal Self-Modeling Gated Spiking Elastic Reservoir (DSTSMGSER) is a neuromimetic RNN architecture 
#     designed to model dynamic spatio-temporal data. It integrates modified liquid neural networks (LNN) for dynamic reservoir computing,  
#     along with Hebbian learning and homeostatic neuroplasticity. By utilizing a gated spiking elastic reservoir (GSER) 
#     and spatio-temporal summarization, it captures complex patterns in sequential data. The model offers dual outputs for classification 
#     and self-modeling, providing high performance and introspective capabilities for understanding internal dynamics. 
#     It is well-suited for tasks that require spatio-temporal understanding and prediction, with enhanced interpretability.

#     Attributes:
#         input_shape (tuple): The shape of the input data (e.g., (height, width, channels) for image data).
#         reservoir_dim (int): The dimensionality of the reservoir (number of neurons in the reservoir layer).
#         spectral_radius (float): The spectral radius for the reservoir's weight matrix. It controls the dynamical
#                                  properties of the reservoir.
#         leak_rate (float): The rate at which information "leaks" out of the reservoir, influencing its memory retention.
#         spike_threshold (float): The threshold for the spike generation in the reservoir neurons.
#         max_dynamic_reservoir_dim (int): The maximum size for the dynamically growing reservoir.
#         output_dim (int): The dimensionality of the output layer for classification.
#         use_weighted_summary (bool): A flag indicating whether to use a weighted summary during spatio-temporal 
#                                       summarization.
#         model (tf.keras.Model): The Keras model that encompasses the entire architecture.
#         reservoir_layer (GSER): The custom spiking neural network reservoir layer used in the model.

#     Methods:
#         build_model(): Constructs the full model by defining input layers, preprocessing, contextualization, reservoir
#                        layers, Hebbian learning, and output layers.
#         compile_model(): Compiles the model with specified loss functions, optimizers, and metrics for training.
#         get_config(): Returns the configuration parameters of the model.
#     """

#     def __init__(self, input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, 
#                  max_dynamic_reservoir_dim, output_dim, use_weighted_summary=False):
#         """
#         Initializes the DSTSMGSER model with the given parameters.

#         Parameters:
#             input_shape (tuple): The shape of the input data.
#             reservoir_dim (int): The dimensionality of the reservoir layer.
#             spectral_radius (float): The spectral radius of the reservoir weight matrix.
#             leak_rate (float): The leak rate for the reservoir layer.
#             spike_threshold (float): The spike threshold for reservoir neurons.
#             max_dynamic_reservoir_dim (int): Maximum size of the dynamically growing reservoir.
#             output_dim (int): The output dimension for classification.
#             use_weighted_summary (bool, optional): Flag to use weighted summarization in spatio-temporal summarization.
#         """
#         self.input_shape = input_shape
#         self.reservoir_dim = reservoir_dim
#         self.spectral_radius = spectral_radius
#         self.leak_rate = leak_rate
#         self.spike_threshold = spike_threshold
#         self.max_dynamic_reservoir_dim = max_dynamic_reservoir_dim
#         self.output_dim = output_dim
#         self.use_weighted_summary = use_weighted_summary
        
#         self.model = None
#         self.reservoir_layer = None

#     def build_model(self):
#         """
#         Builds the full DSTSMGSER model, including the input preprocessing, 
#         spatio-temporal summarization, dynamic reservoir layer, Hebbian learning, 
#         and output layers for classification and self-modeling.

#         This method defines:
#             - Input layer with normalization and dropout.
#             - Spatio-temporal summarization layer (contextualization).
#             - Reservoir layer (GSER).
#             - Hebbian homeostatic learning layer.
#             - Output layers for classification (softmax) and self-modeling (sigmoid).
#         """
#         inputs = Input(shape=self.input_shape)

#         # Preprocessing
#         x = BatchNormalization()(inputs)
#         x = Flatten()(x)
#         x = LayerNormalization()(x)
#         x = Dropout(0.2)(x)
        
#         # Contextualization
#         summarization_layer = SpatioTemporalSummarization(d_model=128, use_weighted_summary=self.use_weighted_summary)
#         x = ExpandDimensionLayer()(x)
#         x = summarization_layer(x)
        
#          # summary_mixing_layer = SpatioTemporalSummaryMixingLayer(d_model=128, use_weighted_summary=self.use_weighted_summary)
#         # x = ExpandDimensionLayer()(x)
#         # x = summary_mixing_layer(x)
        
#         # gated_linear_attention_layer = GatedMultiheadLinearSelfAttentionKernalization(
#         #     d_model=128, num_heads=8, use_weighted_summary=self.use_weighted_summary)
#         # x = ExpandDimensionLayer()(x)
#         # x = gated_linear_attention_layer(x)
        
#         # Reservoir layer
#         self.reservoir_layer = GSER(
#             initial_reservoir_size=self.reservoir_dim,
#             input_dim=x.shape[-1],
#             spectral_radius=self.spectral_radius,
#             leak_rate=self.leak_rate,
#             spike_threshold=self.spike_threshold,
#             max_dynamic_reservoir_dim=self.max_dynamic_reservoir_dim
#         )
#         lnn_layer = RNN(self.reservoir_layer, return_sequences=True)
#         lnn_output = lnn_layer(x)

#         # Hebbian homeostatic layer
#         hebbian_homeostatic_layer = HebbianHomeostaticNeuroplasticity(units=self.reservoir_dim, name='hebbian_homeostatic_layer')
#         x = hebbian_homeostatic_layer(lnn_output)

#         # Classification output
#         clf_out = DenseGSER(
#             units=self.output_dim,
#             input_dim=x.shape[-1],
#             spectral_radius=self.spectral_radius,
#             leak_rate=self.leak_rate,
#             spike_threshold=self.spike_threshold,
#             max_dynamic_units=self.max_dynamic_reservoir_dim,
#             activation='softmax',
#             name='clf_out'
#         )(Flatten()(x))

#         # Self-modeling output
#         sm_out = DenseGSER(
#             units=np.prod(self.input_shape),
#             input_dim=x.shape[-1],
#             spectral_radius=self.spectral_radius,
#             leak_rate=self.leak_rate,
#             spike_threshold=self.spike_threshold,
#             max_dynamic_units=self.max_dynamic_reservoir_dim,
#             activation='sigmoid',
#             name='sm_out'
#         )(Flatten()(x))

#         # Compile the model
#         self.model = tf.keras.Model(inputs=inputs, outputs=[clf_out, sm_out])

#     def compile_model(self):
#         """
#         Compiles the DSTSMGSER model by specifying the optimizer, loss functions, 
#         loss weights, and evaluation metrics for both classification and self-modeling outputs.
#         """
#         self.model.compile(
#             optimizer='adam',
#             loss={
#                 'clf_out': 'categorical_crossentropy',
#                 'sm_out': 'mse'
#             },
#             loss_weights={
#                 'clf_out': 1.0,
#                 'sm_out': 0.5
#             },
#             metrics={
#                 'clf_out': 'accuracy',
#                 'sm_out': 'mse'
#             }
#         )
    
#     def get_config(self):
#         """
#         Returns the configuration of the DSTSMGSER model, including its parameters.

#         Returns:
#             dict: Configuration dictionary containing the model parameters.
#         """
#         return {
#             'input_shape': self.input_shape,
#             'reservoir_dim': self.reservoir_dim,
#             'spectral_radius': self.spectral_radius,
#             'leak_rate': self.leak_rate,
#             'spike_threshold': self.spike_threshold,
#             'max_dynamic_reservoir_dim': self.max_dynamic_reservoir_dim,
#             'output_dim': self.output_dim,
#             'use_weighted_summary': self.use_weighted_summary
#         }








# # with Spatio Temporal Summarization mechanism, denseGSER, and GSER
# # 313/313 - 8s - 26ms/step - clf_out_accuracy: 0.9779 - clf_out_loss: 0.0997 - loss: 0.1259 - sm_out_loss: 0.0522 - sm_out_mse: 0.0522

# # with Spatiotemporal Summary Mixing mechanism, denseGSER, and GSER
# # 313/313 - 29s - 92ms/step - clf_out_accuracy: 0.9824 - clf_out_loss: 0.0773 - loss: 0.1038 - sm_out_loss: 0.0527 - sm_out_mse: 0.0527

# # with Multihead Linear Self Attention Kernalization mechanism, denseGSER, and GSER
# # 313/313 - 30s - 97ms/step - clf_out_accuracy: 0.9717 - clf_out_loss: 0.1470 - loss: 0.1734 - sm_out_loss: 0.0525 - sm_out_mse: 0.0525

# # with Gated Multihead Linear Self Attention Kernalization mechanism, denseGSER, and GSER
# # 313/313 - 31s - 98ms/step - clf_out_accuracy: 0.9764 - clf_out_loss: 0.1125 - loss: 0.1389 - sm_out_loss: 0.0526 - sm_out_mse: 0.0526

# class DSTSMGSER_test1:
#     def __init__(self, input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, max_dynamic_reservoir_dim, output_dim, use_weighted_summary=False):
#         self.input_shape = input_shape
#         self.reservoir_dim = reservoir_dim
#         self.spectral_radius = spectral_radius
#         self.leak_rate = leak_rate
#         self.spike_threshold = spike_threshold
#         self.max_dynamic_reservoir_dim = max_dynamic_reservoir_dim
#         self.output_dim = output_dim
#         self.use_weighted_summary = use_weighted_summary
#         self.model = None
#         self.reservoir_layer = None

#     def build_model(self):
#         inputs = Input(shape=self.input_shape)

#         # Preprocessing
#         x = BatchNormalization()(inputs)
#         x = Flatten()(x)
#         x = LayerNormalization()(x)
#         x = Dropout(0.2)(x)
        
#         # Attention Layer
#         gated_linear_attention_layer = GatedMultiheadLinearSelfAttentionKernalization(
#             d_model=128, num_heads=8, use_weighted_summary=self.use_weighted_summary)
#         x = ExpandDimensionLayer()(x)
#         x = gated_linear_attention_layer(x)
        
#         summarization_layer = SpatioTemporalSummarization(d_model=128, use_weighted_summary=self.use_weighted_summary)
#         x = ExpandDimensionLayer()(x)
#         x = summarization_layer(x)
        
#         # linear_attention_layer = MultiheadLinearSelfAttentionKernalizationLayer(
#         #     d_model=128, num_heads=8, use_weighted_summary=self.use_weighted_summary)
#         # x = ExpandDimensionLayer()(x)
#         # x = linear_attention_layer(x)
        
#         # summary_mixing_layer = SpatioTemporalSummaryMixingLayer(d_model=128, use_weighted_summary=self.use_weighted_summary)
#         # x = ExpandDimensionLayer()(x)
#         # x = summary_mixing_layer(x)

#         # Reservoir layer
#         self.reservoir_layer = GSER(
#             initial_reservoir_size=self.reservoir_dim,
#             input_dim=x.shape[-1],
#             spectral_radius=self.spectral_radius,
#             leak_rate=self.leak_rate,
#             spike_threshold=self.spike_threshold,
#             max_dynamic_reservoir_dim=self.max_dynamic_reservoir_dim
#         )
#         lnn_layer = RNN(self.reservoir_layer, return_sequences=True)
#         lnn_output = lnn_layer(x)

#         # Hebbian homeostatic layer
#         hebbian_homeostatic_layer = HebbianHomeostaticNeuroplasticity(units=self.reservoir_dim, name='hebbian_homeostatic_layer')
#         x = hebbian_homeostatic_layer(lnn_output)

#         # Classification output
#         clf_out = DenseGSER(
#             units=self.output_dim,
#             input_dim=x.shape[-1],
#             spectral_radius=self.spectral_radius,
#             leak_rate=self.leak_rate,
#             spike_threshold=self.spike_threshold,
#             max_dynamic_units=self.max_dynamic_reservoir_dim,
#             activation='softmax',
#             name='clf_out'
#         )(Flatten()(x))

#         # Self-modeling output
#         sm_out = DenseGSER(
#             units=np.prod(self.input_shape),
#             input_dim=x.shape[-1],
#             spectral_radius=self.spectral_radius,
#             leak_rate=self.leak_rate,
#             spike_threshold=self.spike_threshold,
#             max_dynamic_units=self.max_dynamic_reservoir_dim,
#             activation='sigmoid',
#             name='sm_out'
#         )(Flatten()(x))

#         # Compile the model
#         self.model = tf.keras.Model(inputs=inputs, outputs=[clf_out, sm_out])

#     def compile_model(self):
#         self.model.compile(
#             optimizer='adam',
#             loss={
#                 'clf_out': 'categorical_crossentropy',
#                 'sm_out': 'mse'
#             },
#             loss_weights={
#                 'clf_out': 1.0,
#                 'sm_out': 0.5
#             },
#             metrics={
#                 'clf_out': 'accuracy',
#                 'sm_out': 'mse'
#             }
#         )
    
#     def get_config(self):
#         return {
#             'input_shape': self.input_shape,
#             'reservoir_dim': self.reservoir_dim,
#             'spectral_radius': self.spectral_radius,
#             'leak_rate': self.leak_rate,
#             'spike_threshold': self.spike_threshold,
#             'max_dynamic_reservoir_dim': self.max_dynamic_reservoir_dim,
#             'output_dim': self.output_dim,
#             'use_weighted_summary': self.use_weighted_summary
#         }