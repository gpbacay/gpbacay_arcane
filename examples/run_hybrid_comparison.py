import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.layers import Input, RNN, LSTMCell, Dense, LayerNormalization, Dropout, Layer
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

# Set seeds
tf.random.set_seed(42)
np.random.seed(42)

# ==============================================================================
# HYBRID RESONANT LSTM CELL
# Integrates the linear memory stability of LSTM with the
# oscillation/modulation capabilities of Resonant layers.
# ==============================================================================
class HybridResonantLSTMCell(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        # Core LSTM Engine
        self.lstm_cell = LSTMCell(units)
        
    @property
    def state_size(self):
        return self.lstm_cell.state_size # [units, units] (h, c)
        
    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        self.lstm_cell.build(input_shape)
        
        # Resonance Parameters (The "Soul" of the machine)
        self.resonance_gate = self.add_weight(
            name='resonance_gate',
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True
        )
        self.resonance_bias = self.add_weight(
            name='resonance_bias',
            shape=(self.units,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True
        )
        self.resonance_factor = 0.1 # Scaling factor for modulation
        
        self.built = True

    def call(self, inputs, states):
        # 1. Core LSTM Step (Standard Dynamics with Cell State Ct)
        # states is [h_tm1, c_tm1]
        lstm_output, [h_new, c_new] = self.lstm_cell(inputs, states)
        
        # 2. Resonance Modulation
        # We modulate the Hidden State (h_new) before it goes to the next step/output.
        # This allows the "Context/Resonance" to influence the standard LSTM flow.
        
        # Gate: Determines how much resonance to let in
        res_strength = tf.sigmoid(self.resonance_gate) * self.resonance_factor
        
        # Modulate: h_final = h_lstm * (1 + res) + bias
        # This is a multiplicative interaction (Shunting inhibition/excitation)
        h_modulated = h_new * (1.0 + res_strength) + self.resonance_bias
        
        # We do NOT modulate the Cell State (c_new) to preserve the Linear Memory Gradient Highway without interference.
        # This is the key insight: Modulate the "View" (h), keep the "Memory" (c) pure.
        
        return h_modulated, [h_modulated, c_new]

# ==============================================================================
# DATASET
# ==============================================================================
def prepare_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Batch trimming
    batch_size = 32
    train_lim = (len(x_train) // batch_size) * batch_size
    test_lim = (len(x_test) // batch_size) * batch_size
    
    return (x_train[:train_lim], y_train[:train_lim]), (x_test[:test_lim], y_test[:test_lim])

# ==============================================================================
# MODELS
# ==============================================================================
def build_standard_lstm():
    inputs = Input(shape=(28, 28))
    # Standard LSTM Wrapper
    x = RNN(LSTMCell(128), return_sequences=True)(inputs)
    x = RNN(LSTMCell(64), return_sequences=False)(x)
    outputs = Dense(10, activation='softmax')(x)
    return Model(inputs, outputs, name="Standard_LSTM")

def build_hybrid_lstm():
    inputs = Input(shape=(28, 28))
    # Hybrid Wrapper using the official ResonantGSER layer (which now wraps the Hybrid architecture)
    from gpbacay_arcane.layers import ResonantGSER
    
    # ResonantGSER is now an RNN wrapper itself
    x = ResonantGSER(128, return_sequences=True)(inputs)
    x = ResonantGSER(64, return_sequences=False)(x)
    outputs = Dense(10, activation='softmax')(x)
    return Model(inputs, outputs, name="Hybrid_Resonant_LSTM")

# ==============================================================================
# EXPERIMENT
# ==============================================================================
def run_hybrid_comparison():
    print("="*80)
    print("      RNN BATTLE: STANDARD LSTM vs HYBRID RESONANT LSTM")
    print("="*80)
    
    (x_train, y_train), (x_test, y_test) = prepare_mnist()
    EPOCHS = 5 # Short run to test convergence speed and peak
    BATCH = 32
    
    # 1. Standard
    print("\n[Standard LSTM] Training...")
    std_model = build_standard_lstm()
    std_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    t0 = time.time()
    std_hist = std_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH, verbose=1)
    std_time = time.time() - t0
    
    # 2. Hybrid
    print("\n[Hybrid Resonant LSTM] Training (with Deliberative Resonance Phase)...")
    from gpbacay_arcane.callbacks import NeuralResonanceCallback
    hyb_cb = NeuralResonanceCallback(resonance_cycles=3)
    
    hyb_model = build_hybrid_lstm()
    hyb_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    t0 = time.time()
    hyb_hist = hyb_model.fit(x_train, y_train, validation_data=(x_test, y_test), 
                             epochs=EPOCHS, batch_size=BATCH, verbose=1,
                             callbacks=[hyb_cb])
    hyb_time = time.time() - t0
    
    # 3. Results
    std_acc = std_hist.history['val_accuracy'][-1]
    hyb_acc = hyb_hist.history['val_accuracy'][-1]
    
    print("\n" + "="*80)
    print("RESULTS: HYBRID INTEGRATION")
    print("="*80)
    print(f"{'Metric':<20} | {'Standard LSTM':<15} | {'Hybrid LSTM':<15}")
    print("-" * 60)
    print(f"{'Val Accuracy':<20} | {std_acc:.4f}          | {hyb_acc:.4f}")
    print(f"{'Time':<20} | {std_time:.1f}s            | {hyb_time:.1f}s")
    print("-" * 60)
    
    if hyb_acc > std_acc:
        print(f"WINNER: Hybrid LSTM (+{(hyb_acc - std_acc)*100:.2f}%)")
        print("Success! Adding Resonance Modulation to the LSTM Hidden State improved performance.")
    else:
        print(f"WINNER: Standard LSTM (+{(std_acc - hyb_acc)*100:.2f}%)")
        print("Insight: The resonance modulation might be adding noise or unneeded complexity to the pure LSTM.")

if __name__ == "__main__":
    run_hybrid_comparison()
