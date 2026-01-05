import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tensorflow.keras.layers import Input, LSTM, Dense, LayerNormalization, Dropout
from tensorflow.keras import Model
from gpbacay_arcane.layers import ResonantGSER, BioplasticDenseLayer

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

# ==============================================================================
# 1. LIQUID MODEL DEFINITION (Self-Contained Subclass)
# ==============================================================================
class LiquidResonantGSER(ResonantGSER):
    """
    Trainable Liquid Reservoir with Resonance capabilities.
    """
    def build(self, input_shape):
        self.input_dim = input_shape[-1] if self.input_dim is None else self.input_dim
        
        # Standard Trainable Weights
        self.input_weights = self.add_weight(shape=(self.input_dim, self.units), initializer=self.kernel_initializer, trainable=True, name='input_weights')
        self.reservoir_weights = self.add_weight(shape=(self.input_dim, self.units), initializer=self.kernel_initializer, trainable=True, name='reservoir_weights') # Unfrozen
        self.gate_weights = self.add_weight(shape=(self.input_dim, 3 * self.units), initializer=self.kernel_initializer, trainable=True, name='gate_weights')
        self.gate_bias = self.add_weight(shape=(3 * self.units,), initializer=self.bias_initializer, trainable=True, name='gate_bias')
        self.output_weights = self.add_weight(shape=(self.units, self.units), initializer=self.kernel_initializer, trainable=True, name='output_weights')
        self.output_bias = self.add_weight(shape=(self.units,), initializer=self.bias_initializer, trainable=True, name='output_bias')

        # Resonance Weights
        self.resonance_gate = self.add_weight(name='resonance_gate', shape=(self.units,), initializer=tf.keras.initializers.Constant(1.0), trainable=True)
        self.resonance_bias = self.add_weight(name='resonance_bias', shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True)
        self.feedback_weights = self.add_weight(name='feedback_weights', shape=(self.units, self.input_dim), initializer='glorot_uniform', trainable=True)
        
        self.built = True

    def call(self, inputs, training=None):
        inputs = tf.cast(inputs, tf.float32)
        
        # Reservoir Dynamics
        input_part = tf.matmul(inputs, self.input_weights)
        reservoir_part = tf.matmul(inputs, self.reservoir_weights)
        gate_part = tf.matmul(inputs, self.gate_weights) + self.gate_bias
        i_gate, f_gate, o_gate = tf.split(tf.sigmoid(gate_part), 3, axis=-1)
        
        state = (1 - self.leak_rate) * (f_gate * reservoir_part) + \
                self.leak_rate * tf.tanh(i_gate * (input_part + reservoir_part))
        state = o_gate * state
        
        # Spiking threshold
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)
        
        # Output Projection
        output = tf.matmul(state, self.output_weights) + self.output_bias
        
        # Resonance Modulation
        resonance_modulation = tf.sigmoid(self.resonance_gate) * self.resonance_factor
        modulated = output * (1.0 + resonance_modulation) + self.resonance_bias
        
        if self.activation is not None:
            modulated = self.activation(modulated)
            
        return modulated

# ==============================================================================
# 2. DATA GENERATION (Non-Stationary)
# ==============================================================================
def generate_shifting_wave(n_samples=3000, seq_len=50):
    """
    Generates a sine wave that gradually increases in frequency and chaos.
    """
    t = np.linspace(0, 100, n_samples)
    
    # Frequency Drift: Starts at 1.0, drifts to 3.0
    freq_drift = np.linspace(1.0, 3.0, n_samples)
    
    # Generate Wave
    data = np.sin(t * freq_drift)
    
    # Add subtle non-linear noise
    noise = np.random.normal(0, 0.05, n_samples)
    data = data + noise
    
    # Prepare Sequences (X -> y)
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
        
    X = np.array(X)[..., np.newaxis] # (Batch, Seq, 1)
    y = np.array(y)
    
    return X, y

# ==============================================================================
# 3. MODELS
# ==============================================================================
def build_trad_model(seq_len):
    inputs = Input(shape=(seq_len, 1))
    x = LSTM(64, return_sequences=False)(inputs)
    outputs = Dense(1)(x) # Regression output
    return Model(inputs, outputs, name="Traditional_LSTM")

def build_liquid_model(seq_len):
    inputs = Input(shape=(seq_len, 1))
    # Liquid Reservoir
    x = LiquidResonantGSER(64, spectral_radius=0.95, leak_rate=0.15)(inputs)
    x = LayerNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x) # <--- ADDED POOLING
    # Bioplastic Readout (Crucial for adaptation)
    x = BioplasticDenseLayer(
        units=1, 
        activation='linear', # Regression
        learning_rate=0.01, # High plasticity for adaptation
        normalization='layer'
    )(x)
    return Model(inputs, outputs=x, name="Liquid_Resonant")

# ==============================================================================
# 4. EXPERIMENT: ONLINE ADAPTATION
# ==============================================================================
def run_adaptive_test():
    print("="*80)
    print("      ADAPTIVE LEARNING CHALLENGE: SHIFTING REALITY")
    print("="*80)
    
    # 1. Generate Data
    SEQ_LEN = 30
    X, y = generate_shifting_wave(n_samples=2000, seq_len=SEQ_LEN)
    
    # Split into Regimes
    # Phase 1: Stable (First 30%)
    split_idx = int(len(X) * 0.3)
    X_train_static = X[:split_idx]
    y_train_static = y[:split_idx]
    
    # Phase 2: Drifting/New Reality (Remaining 70%)
    X_adapt = X[split_idx:]
    y_adapt = y[split_idx:]
    
    print(f"Phase 1 (Static Training): {len(X_train_static)} samples")
    print(f"Phase 2 (Online Adaptation): {len(X_adapt)} samples")
    
    # 2. Initialize Models
    trad_model = build_trad_model(SEQ_LEN)
    trad_model.compile(optimizer='adam', loss='mse')
    
    liquid_model = build_liquid_model(SEQ_LEN)
    liquid_model.compile(optimizer='adam', loss='mse')
    
    # 3. Pre-train on Static Phase (Batch Training)
    print("\n[Phase 1] Pre-training on known physics...")
    trad_model.fit(X_train_static, y_train_static, epochs=5, batch_size=32, verbose=0)
    liquid_model.fit(X_train_static, y_train_static, epochs=5, batch_size=32, verbose=0)
    print("Pre-training complete.")
    
    # 4. Online Adaptation Loop
    # We feed the 'drift' data one batch at a time.
    # Metric: Cumulative Mean Squared Error
    
    batch_size = 1 # True Online Learning (or small batch)
    # Using small batch for stability
    batch_size = 16 
    
    trad_dists = []     # Predictions
    liquid_dists = []   # Predictions
    
    trad_errors = []
    liquid_errors = []
    
    print(f"\n[Phase 2] Entering Shifting Reality (Online Adaptation)...")
    print("Streaming data batches...")
    
    n_batches = len(X_adapt) // batch_size
    
    start_time = time.time()
    
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        
        x_batch = X_adapt[start:end]
        y_batch = y_adapt[start:end]
        
        # A. PREDICT FIRST (Evaluate current fitness)
        pred_trad = trad_model.predict_on_batch(x_batch)
        pred_liq = liquid_model.predict_on_batch(x_batch)
        
        # Calculate Error
        mse_trad = np.mean(np.square(y_batch - pred_trad.flatten()))
        mse_liq = np.mean(np.square(y_batch - pred_liq.flatten()))
        
        trad_errors.append(mse_trad)
        liquid_errors.append(mse_liq)
        
        # B. LEARN NOW (Update model on this new reality)
        # Train on the batch we just saw
        trad_model.train_on_batch(x_batch, y_batch)
        liquid_model.train_on_batch(x_batch, y_batch)
        
        if i % 10 == 0:
            print(f"Batch {i}/{n_batches} | Trad Error: {mse_trad:.4f} | Liquid Error: {mse_liq:.4f}")

    duration = time.time() - start_time
    
    # 5. Results
    total_mse_trad = np.mean(trad_errors)
    total_mse_liq = np.mean(liquid_errors)
    
    print("\n" + "="*80)
    print("RESULTS: ADAPTATION PERFORMANCE")
    print("="*80)
    print(f"{'Metric':<25} | {'Traditional LSTM':<20} | {'Liquid-Resonant':<20}")
    print("-" * 70)
    print(f"{'Avg Adaptation MSE':<25} | {total_mse_trad:.5f}               | {total_mse_liq:.5f}")
    print("-" * 70)
    
    if total_mse_liq < total_mse_trad:
        print(f"WINNER: Liquid-Resonant ({(total_mse_trad - total_mse_liq)/total_mse_trad * 100:.2f}% improvement)")
        print("Conclusion: Liquid architecture adapted faster to the shifting frequency.")
    else:
        print("WINNER: Traditional LSTM")
        print("Conclusion: LSTM remained more robust or learned closer to the mean.")

if __name__ == "__main__":
    run_adaptive_test()
