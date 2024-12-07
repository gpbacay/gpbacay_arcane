import asyncio
import websockets
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RNN, Input, Dropout, Flatten, BatchNormalization, LayerNormalization, Lambda, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Custom modules
from hebbian_learning_homeostatic_neuroplasticity_mechanism import HebbianHomeostaticLayer
from spatiotemporal_summary_mixing_mechanism import SpatioTemporalSummaryMixingLayer
from gated_spiking_elastic_reservoir_computing import GatedSpikingElasticReservoirLayer
from dense_gated_spiking_elastic_reservoir_computing import DenseReservoirLayer


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

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        # Preprocessing
        x = BatchNormalization()(inputs)
        x = Flatten()(x)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)

        # SpatioTemporal Summary Mixing Layer
        summary_mixing_layer = SpatioTemporalSummaryMixingLayer(d_model=128, use_weighted_summary=self.use_weighted_summary)
        x = Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
        x = summary_mixing_layer(x)

        # Reservoir Layer
        reservoir_layer = GatedSpikingElasticReservoirLayer(
            initial_reservoir_size=self.reservoir_dim,
            input_dim=x.shape[-1],
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold,
            max_dynamic_reservoir_dim=self.max_dynamic_reservoir_dim
        )
        lnn_layer = RNN(reservoir_layer, return_sequences=True)
        lnn_output = lnn_layer(x)

        # Hebbian Homeostatic Layer
        hebbian_homeostatic_layer = HebbianHomeostaticLayer(units=self.reservoir_dim, name='hebbian_homeostatic_layer')
        hebbian_output = hebbian_homeostatic_layer(lnn_output)

        # Flatten and apply DenseReservoirLayer
        x = Flatten()(hebbian_output)

        # Classification Output
        classification_output = DenseReservoirLayer(
            units=self.output_dim,
            input_dim=x.shape[-1],
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            spike_threshold=self.spike_threshold,
            max_dynamic_units=self.max_dynamic_reservoir_dim,
            activation='softmax',
            name='classification_output'
        )(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=classification_output)

    def compile_model(self):
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )


class WebSocketServer:
    def __init__(self):
        self.clients = set()

    async def register(self, websocket):
        self.clients.add(websocket)

    async def unregister(self, websocket):
        self.clients.remove(websocket)

    async def send_update(self, data):
        if self.clients:
            message = json.dumps({'type': 'training_update', 'data': data})
            await asyncio.gather(*[client.send(message) for client in self.clients])

    async def handler(self, websocket, path):
        await self.register(websocket)
        try:
            async for _ in websocket:
                pass
        finally:
            await self.unregister(websocket)


async def train_model(server, dstsmgser, x_train, y_train, x_test, y_test):
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5)

    history = dstsmgser.model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=10,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    for epoch in range(len(history.history['accuracy'])):
        await server.send_update({
            'epoch': epoch + 1,
            'accuracy': history.history['accuracy'][epoch],
            'val_accuracy': history.history['val_accuracy'][epoch],
        })


async def main():
    server = WebSocketServer()
    websocket_server = await websockets.serve(server.handler, "localhost", 8000)

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[..., np.newaxis] / 255.0
    x_test = x_test[..., np.newaxis] / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Initialize and compile model
    input_shape = (28, 28, 1)
    dstsmgser = DSTSMGSER(input_shape, 512, 2.0, 0.2, 0.5, 4096, 10, True)
    dstsmgser.build_model()
    dstsmgser.compile_model()

    # Train in an async task
    training_task = asyncio.create_task(train_model(server, dstsmgser, x_train, y_train, x_test, y_test))

    await websocket_server.wait_closed()
    await training_task


if __name__ == "__main__":
    asyncio.run(main())




# python main.py