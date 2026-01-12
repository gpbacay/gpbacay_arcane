import tensorflow as tf
import numpy as np
import pytest
from gpbacay_arcane.activations import resonant_spike, homeostatic_gelu, adaptive_softplus

def test_resonant_spike_leaky_integration():
    print("\n--- Testing Resonant Spike Leaky Integration ---")
    threshold = 1.0
    leak_rate = 0.1
    state = tf.constant([0.5], dtype=tf.float32)
    inputs = tf.constant([0.0], dtype=tf.float32)
    
    # After 1 step with zero input, potential should be state * (1 - leak_rate)
    spikes, new_state = resonant_spike(inputs, state, threshold=threshold, leak_rate=leak_rate)
    expected_state = 0.5 * (1.0 - 0.1)
    
    print(f"Initial State: {state.numpy()[0]}")
    print(f"New State: {new_state.numpy()[0]}")
    assert np.isclose(new_state.numpy()[0], expected_state), f"Expected {expected_state}, got {new_state.numpy()[0]}"
    assert spikes.numpy()[0] == 0, "Should not spike with zero input and low potential"

def test_resonant_spike_firing():
    print("\n--- Testing Resonant Spike Firing and Reset ---")
    threshold = 0.5
    state = tf.constant([0.4], dtype=tf.float32)
    inputs = tf.constant([0.2], dtype=tf.float32) # Total potential 0.6
    
    spikes, new_state = resonant_spike(inputs, state, threshold=threshold, leak_rate=0.0)
    
    print(f"Potential: 0.6, Threshold: {threshold}")
    print(f"Spike: {spikes.numpy()[0]}")
    print(f"New State (after reset): {new_state.numpy()[0]}")
    
    assert spikes.numpy()[0] == 1, "Should spike when potential > threshold"
    # Reset: 0.6 - 0.5 = 0.1
    assert np.isclose(new_state.numpy()[0], 0.1), f"Expected reset state 0.1, got {new_state.numpy()[0]}"

def test_resonant_spike_resonance():
    print("\n--- Testing Resonant Spike Resonance Modulation ---")
    threshold = 1.0
    state = tf.constant([0.4], dtype=tf.float32)
    inputs = tf.constant([0.4], dtype=tf.float32) # Total potential 0.8
    resonance_factor = 0.5 # Boost by 50% -> 0.8 * 1.5 = 1.2
    
    # Without resonance, no spike (0.8 < 1.0)
    spikes_no_res, _ = resonant_spike(inputs, state, threshold=threshold, resonance_factor=0.0)
    # With resonance, spike (1.2 > 1.0)
    spikes_res, _ = resonant_spike(inputs, state, threshold=threshold, resonance_factor=resonance_factor)
    
    print(f"Potential (no res): 0.8, Spike: {spikes_no_res.numpy()[0]}")
    print(f"Potential (with res): 1.2, Spike: {spikes_res.numpy()[0]}")
    
    assert spikes_no_res.numpy()[0] == 0
    assert spikes_res.numpy()[0] == 1

def test_homeostatic_gelu_gain():
    print("\n--- Testing Homeostatic GELU Gain Adjustment ---")
    target = 0.1
    # Activity below target -> Gain should increase
    low_activity = 0.05
    out_low = homeostatic_gelu(tf.constant([1.0]), tf.constant([low_activity]), target_activity=target, adaptation_rate=0.1)
    
    # Activity above target -> Gain should decrease
    high_activity = 0.2
    out_high = homeostatic_gelu(tf.constant([1.0]), tf.constant([high_activity]), target_activity=target, adaptation_rate=0.1)
    
    print(f"Low Activity Output: {out_low.numpy()[0]}")
    print(f"High Activity Output: {out_high.numpy()[0]}")
    
    # GELU(1.0) is approx 0.8413. 
    # Low activity gain: 1 + 0.1*(0.1-0.05) = 1.005. Out: GELU(1.005) > GELU(1.0)
    # High activity gain: 1 + 0.1*(0.1-0.2) = 0.99. Out: GELU(0.99) < GELU(1.0)
    assert out_low.numpy()[0] > out_high.numpy()[0], "Low activity should result in higher gain than high activity"

def test_adaptive_softplus():
    print("\n--- Testing Adaptive Softplus ---")
    x = tf.constant([-1.0, 0.0, 1.0, 2.0], dtype=tf.float32)
    out = adaptive_softplus(x, threshold=1.0, sharpness=2.0)
    print(f"Inputs: {x.numpy()}")
    print(f"Outputs: {out.numpy()}")
    # At x=1.0 (threshold), value should be softplus(0)/2 = log(2)/2 approx 0.346
    assert np.isclose(out.numpy()[2], np.log(2)/2.0), f"Expected {np.log(2)/2.0}, got {out.numpy()[2]}"

if __name__ == "__main__":
    pytest.main([__file__])
