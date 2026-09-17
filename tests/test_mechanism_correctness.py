import numpy as np
import tensorflow as tf

from gpbacay_arcane.activations import (
    NeuromimeticActivation,
    resonant_spike,
    straight_through_spike,
)
from gpbacay_arcane.callbacks import DynamicSelfModelingReservoirCallback
from gpbacay_arcane.layers import (
    BioplasticDenseLayer,
    DenseGSER,
    HebbianHomeostaticNeuroplasticity,
    PositionalEncodingLayer,
    ResonantGSER,
)
from gpbacay_arcane.mechanisms import (
    AttentionResidual,
    GSER,
    MultiheadLinearSelfAttentionKernalization,
    SpatioTemporalSummaryMixingLayer,
)


def test_straight_through_spike_has_gradient():
    x = tf.Variable([0.4, 0.6, 1.2], dtype=tf.float32)
    with tf.GradientTape() as tape:
        spikes = straight_through_spike(x, threshold=0.5, sharpness=8.0)
        loss = tf.reduce_sum(spikes)
    grad = tape.gradient(loss, x)
    assert grad is not None
    assert tf.reduce_sum(tf.abs(grad)).numpy() > 0
    np.testing.assert_allclose(spikes.numpy(), [0.0, 1.0, 1.0])


def test_neuromimetic_activation_accepts_keras_kwargs():
    layer = NeuromimeticActivation(activation_type="resonant_spike", name="rsa", threshold=0.5)
    out = layer(tf.ones((2, 4)))
    assert out.shape == (2, 4)
    cfg = layer.get_config()
    assert cfg["activation_type"] == "resonant_spike"
    assert cfg["name"] == "rsa"


def test_resonant_gser_skips_zero_alignment():
    cell_layer = ResonantGSER(units=8, resonance_factor=0.5, resonance_cycles=5, return_sequences=False)
    inputs = tf.random.normal((3, 4, 6))
    out = cell_layer(inputs)
    assert out.shape == (3, 8)
    assert float(cell_layer.cell.alignment_set.numpy()) == 0.0


def test_gser_semantic_gate_with_partial_reservoir():
    cell = GSER(
        input_dim=4,
        initial_reservoir_size=6,
        max_dynamic_reservoir_dim=12,
        spectral_radius=0.9,
        leak_rate=0.2,
        spike_threshold=0.5,
    )
    x = tf.random.normal((2, 4))
    state = cell.get_initial_state(batch_size=2)
    out, new_state = cell(x, state)
    assert out.shape == (2, 12)
    assert new_state[0].shape == (2, 12)
    rho = float(np.max(np.abs(np.linalg.eigvals(cell.spatiotemporal_reservoir_weights.numpy()))))
    assert abs(rho - 0.9) < 0.15


def test_linear_attention_is_sequence_linear():
    layer = MultiheadLinearSelfAttentionKernalization(d_model=16, num_heads=4, dropout_rate=0.0)
    short = layer(tf.random.normal((2, 8, 16)), training=False)
    long = layer(tf.random.normal((2, 64, 16)), training=False)
    assert short.shape == (2, 8, 16)
    assert long.shape == (2, 64, 16)


def test_positional_encoding_is_not_identity():
    layer = PositionalEncodingLayer(max_position=16, d_model=8)
    x = tf.zeros((2, 16, 8))
    y = layer(x)
    assert y.shape == x.shape
    assert float(tf.reduce_sum(tf.abs(y))) > 0
    # Positions 0 and 1 must differ.
    assert float(tf.reduce_sum(tf.abs(y[:, 0, :] - y[:, 1, :]))) > 0


def test_summary_mixing_attention_weights_over_sequence():
    layer = SpatioTemporalSummaryMixingLayer(d_model=8, dropout_rate=0.0, use_weighted_summary=True)
    y = layer(tf.random.normal((2, 10, 8)), training=False)
    assert y.shape == (2, 10, 8)


def test_attention_residual_uniform_at_init():
    layer = AttentionResidual(d_model=4)
    h0 = tf.constant([[1.0, 0.0, 0.0, 0.0]])
    h1 = tf.constant([[0.0, 1.0, 0.0, 0.0]])
    out = layer([h0, h1])
    expected = 0.5 * (h0 + h1)
    np.testing.assert_allclose(out.numpy(), expected.numpy(), atol=1e-5)


def test_dense_gser_uses_spike_threshold():
    low = DenseGSER(units=4, leak_rate=0.1, spike_threshold=-10.0, use_conceptual_gate=False)
    high = DenseGSER(units=4, leak_rate=0.1, spike_threshold=10.0, use_conceptual_gate=False)
    x = tf.ones((2, 3))
    low.build(x.shape)
    high.build(x.shape)
    high.kernel.assign(low.kernel)
    high.bias.assign(low.bias)
    low_out = low(x)
    high_out = high(x)
    assert float(tf.reduce_mean(tf.abs(low_out))) > float(tf.reduce_mean(tf.abs(high_out)))


def test_bioplastic_bcm_updates_plastic_kernel():
    layer = BioplasticDenseLayer(
        units=3,
        enable_inference_plasticity=True,
        dropout_rate=0.0,
        activation="linear",
        bcm_tau=2.0,
        learning_rate=0.5,
    )
    x = tf.ones((4, 2))
    layer.build(x.shape)
    before = layer.plastic_kernel.numpy().copy()
    layer(x, training=False)
    after = layer.plastic_kernel.numpy()
    assert np.linalg.norm(after - before) > 0


def test_hebbian_layer_inference_matches_dense():
    layer = HebbianHomeostaticNeuroplasticity(units=3, learning_rate=0.1)
    x = tf.ones((2, 4))
    layer.build(x.shape)
    y = layer(x, training=False)
    expected = tf.matmul(x, layer.kernel) + layer.bias
    np.testing.assert_allclose(y.numpy(), expected.numpy(), atol=1e-6)


def test_dynamic_self_modeling_callback_constructs():
    cell = GSER(
        input_dim=3,
        initial_reservoir_size=4,
        max_dynamic_reservoir_dim=8,
        spectral_radius=0.8,
        leak_rate=0.2,
        spike_threshold=0.4,
    )
    x = tf.zeros((1, 3))
    cell(x, cell.get_initial_state(batch_size=1))
    cb = DynamicSelfModelingReservoirCallback(reservoir_layer=cell, growth_rate=1)
    cb.on_epoch_end(0, logs={"val_accuracy": 0.1})
    cb.on_epoch_end(1, logs={"val_accuracy": 0.5})
    assert cb.stagnation_counter >= 0


def test_resonant_spike_backward_compatible_reset():
    spikes, new_state = resonant_spike(
        tf.constant([0.2]), tf.constant([0.4]), threshold=0.5, leak_rate=0.0
    )
    assert float(spikes.numpy()[0]) == 1.0
    np.testing.assert_allclose(new_state.numpy()[0], 0.1, atol=1e-6)
