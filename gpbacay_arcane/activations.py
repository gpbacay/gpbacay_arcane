import tensorflow as tf

"""
gpbacay_arcane.activations

Neuromimetic activation functions for ARCANE: leaky integrate-and-fire
spiking, homeostatic GELU, and adaptive firing-rate maps.
"""


def straight_through_spike(potential, threshold=0.5, sharpness=10.0):
    """
    Binary spike in the forward pass with a sigmoid gradient in the backward pass.

    Hard thresholding alone has zero gradient almost everywhere, so training
    through spike outputs requires a straight-through estimator (STE).
    """
    hard = tf.cast(tf.greater(potential, threshold), dtype=potential.dtype)
    soft = tf.nn.sigmoid(sharpness * (potential - threshold))
    return tf.stop_gradient(hard - soft) + soft


def resonant_spike(x, state, threshold=0.5, leak_rate=0.1, resonance_factor=0.0, sharpness=10.0):
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
        sharpness: STE sigmoid slope for differentiable spikes

    Returns:
        spikes: Differentiable spike tensor (hard forward, sigmoid backward)
        new_state: Updated internal potential after subtractive reset
    """
    leak_rate = tf.clip_by_value(tf.cast(leak_rate, x.dtype), 0.0, 1.0)
    integrated_potential = x + (state * (1.0 - leak_rate))
    modulated_potential = integrated_potential * (1.0 + resonance_factor)
    spikes = straight_through_spike(modulated_potential, threshold=threshold, sharpness=sharpness)
    new_state = modulated_potential - spikes * threshold
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
    gain = 1.0 + adaptation_rate * (target_activity - activity_history)
    gain = tf.clip_by_value(gain, 0.1, 10.0)
    return tf.nn.gelu(x * gain)


def adaptive_softplus(x, threshold=1.0, sharpness=1.0):
    """
    Adaptive Softplus

    A smooth activation that mimics the firing rate of biological neurons
    with a tunable saturation threshold.
    """
    sharpness = tf.maximum(tf.cast(sharpness, x.dtype), 1e-6)
    return tf.math.softplus(sharpness * (x - threshold)) / sharpness


class NeuromimeticActivation(tf.keras.layers.Layer):
    """
    Keras wrapper for stateful neuromimetic activations.

    Membrane potential is tracked as a running feature-wise average so the
    layer can sit in a feed-forward graph without collapsing the batch into
    the activation parameters passed to `resonant_spike`.
    """

    def __init__(
        self,
        activation_type="resonant_spike",
        threshold=0.5,
        leak_rate=0.1,
        target_activity=0.12,
        adaptation_rate=0.01,
        sharpness=10.0,
        state_momentum=0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation_type = activation_type
        self.threshold = threshold
        self.leak_rate = leak_rate
        self.target_activity = target_activity
        self.adaptation_rate = adaptation_rate
        self.sharpness = sharpness
        self.state_momentum = state_momentum

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.potential_state = self.add_weight(
            shape=(feature_dim,),
            initializer="zeros",
            trainable=False,
            name="potential_state",
        )
        self.activity_trace = self.add_weight(
            shape=(),
            initializer="zeros",
            trainable=False,
            name="activity_trace",
        )
        super().build(input_shape)

    def _feature_mean(self, tensor):
        rank = tf.rank(tensor)
        reduce_axes = tf.range(rank - 1)
        return tf.reduce_mean(tensor, axis=reduce_axes)

    def call(self, inputs, resonance_factor=0.0):
        if self.activation_type == "resonant_spike":
            spikes, new_state = resonant_spike(
                inputs,
                self.potential_state,
                threshold=self.threshold,
                leak_rate=self.leak_rate,
                resonance_factor=resonance_factor,
                sharpness=self.sharpness,
            )
            feature_mean = self._feature_mean(new_state)
            self.potential_state.assign(
                self.state_momentum * self.potential_state
                + (1.0 - self.state_momentum) * feature_mean
            )
            return spikes

        if self.activation_type == "homeostatic_gelu":
            activated = homeostatic_gelu(
                inputs,
                self.activity_trace,
                target_activity=self.target_activity,
                adaptation_rate=self.adaptation_rate,
            )
            mean_activity = tf.reduce_mean(tf.abs(activated))
            self.activity_trace.assign(
                self.state_momentum * self.activity_trace
                + (1.0 - self.state_momentum) * mean_activity
            )
            return activated

        if self.activation_type == "adaptive_softplus":
            return adaptive_softplus(inputs, threshold=self.threshold, sharpness=self.sharpness)

        return tf.nn.gelu(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "activation_type": self.activation_type,
                "threshold": self.threshold,
                "leak_rate": self.leak_rate,
                "target_activity": self.target_activity,
                "adaptation_rate": self.adaptation_rate,
                "sharpness": self.sharpness,
                "state_momentum": self.state_momentum,
            }
        )
        return config
