import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def efficientnet_block(inputs, filters, expansion_factor, stride):
    expanded_filters = filters * expansion_factor
    x = Conv2D(expanded_filters, kernel_size=1, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Conv2D(expanded_filters, kernel_size=3, padding='same', strides=stride, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if stride == 1 and inputs.shape[-1] == x.shape[-1]:
        x = tf.keras.layers.Add()([inputs, x])
    return x

def create_ecnn_model(input_shape, output_dim):
    inputs = Input(shape=input_shape)
    
    # EfficientNet Convolutional layers for feature extraction
    x = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = efficientnet_block(x, 16, expansion_factor=1, stride=1)
    x = efficientnet_block(x, 24, expansion_factor=6, stride=2)
    x = efficientnet_block(x, 40, expansion_factor=6, stride=2)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)

    # Final classification layer
    outputs = Dense(output_dim, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = x_train[..., tf.newaxis]
    x_val = x_val[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def main():
    input_shape = (28, 28, 1)
    output_dim = 10
    num_epochs = 10
    batch_size = 64

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
    
    # Data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    datagen.fit(x_train)

    model = create_ecnn_model(input_shape, output_dim)

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=num_epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr]
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_accuracy:.4f}')
    
    # Create Models directory if it doesn't exist
    os.makedirs('Models', exist_ok=True)

    # Save the model
    model_path = os.path.join('Models', 'cnn_model.keras')
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()



# Efficient Convolutional Neural Network (ECNN)
# python cnn.py
# Test accuracy: 0.9877



















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