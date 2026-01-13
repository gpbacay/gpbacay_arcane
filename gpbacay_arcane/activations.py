import tensorflow as tf
import numpy as np

"""
gpbacay_arcane.activations

This module contains neuromimetic activation functions for the ARCANE project.
These activations go beyond simple non-linearities by incorporating principles of 
neural resonance, spiking dynamics, and homeostatic regulation.
"""

def resonant_spike(x, state, threshold=0.5, leak_rate=0.1, resonance_factor=0.0):
    """
    Resonant Spiking Activation (RSA)
    
    A stateful neuromimetic activation that integrates inputs over time, 
    modulates them via resonance, and fires discrete spikes.
    
    Args:
        x: Input tensor (current stimulus)
        state: Previous internal potential (membrane state)
        threshold: Firing threshold
        leak_rate: Rate at which potential decays over time
        resonance_factor: Top-down modulation from higher hierarchical layers
        
    Returns:
        spikes: Binary or graded spikes
        new_state: Updated internal potential
    """
    # 1. Leaky Integration: Combine current input with decayed previous state
    # This mimics the biological membrane potential.
    integrated_potential = x + (state * (1.0 - leak_rate))
    
    # 2. Resonance Modulation: Scale the potential based on hierarchical alignment
    # If the input aligns with global expectations, the potential is amplified.
    modulated_potential = integrated_potential * (1.0 + resonance_factor)
    
    # 3. Spiking Logic: Fire if the modulated potential exceeds the threshold
    spikes = tf.cast(tf.greater(modulated_potential, threshold), dtype=tf.float32)
    
    # 4. Refractory Reset: Subtract the threshold if a spike occurred
    # This prevents the neuron from firing continuously without new input.
    new_state = tf.where(spikes > 0, modulated_potential - threshold, modulated_potential)
    
    return spikes, new_state

def homeostatic_gelu(x, activity_history, target_activity=0.12, adaptation_rate=0.01):
    """
    Homeostatic GELU (h-GELU)
    
    A modified GELU activation that self-regulates its sensitivity based on 
    historical activity levels to prevent runaway excitation or silence.
    
    Args:
        x: Input tensor
        activity_history: Moving average of recent activity
        target_activity: The desired average firing rate
        adaptation_rate: How quickly the activation adapts to activity shifts
        
    Returns:
        activated: Regulated GELU output
    """
    # Calculate the homeostatic gain factor
    # If activity is too high, gain decreases. If too low, gain increases.
    gain = 1.0 + adaptation_rate * (target_activity - activity_history)
    gain = tf.maximum(0.1, gain) # Ensure the gain doesn't become negative
    
    # Apply standard GELU scaled by homeostatic gain
    return tf.nn.gelu(x * gain)

def adaptive_softplus(x, threshold=1.0, sharpness=1.0):
    """
    Adaptive Softplus
    
    A smooth activation that mimics the firing rate of biological neurons 
    with a tunable saturation threshold.
    """
    return tf.math.softplus(sharpness * (x - threshold)) / sharpness

class NeuromimeticActivation(tf.keras.layers.Layer):
    """
    A Keras-compatible wrapper for stateful neuromimetic activations.
    Can be used as a layer within a recurrent or feed-forward model.
    """
    def __init__(self, activation_type='resonant_spike', **kwargs):
        super().__init__(**kwargs)
        self.activation_type = activation_type
        self.activation_params = kwargs
        
    def build(self, input_shape):
        self.state = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=False,
            name='potential_state'
        )
        super().build(input_shape)
        
    def call(self, inputs, resonance_factor=0.0):
        if self.activation_type == 'resonant_spike':
            spikes, new_state = resonant_spike(
                inputs, 
                self.state, 
                resonance_factor=resonance_factor,
                **self.activation_params
            )
            self.state.assign(tf.reduce_mean(new_state, axis=0))
            return spikes
        
        return tf.nn.gelu(inputs)
