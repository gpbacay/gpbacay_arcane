import numpy as np
import tensorflow as tf

from gpbacay_arcane.activations import (
    NeuromimeticActivation,
    graded_spike,
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



def test_graded_spike_counts_scale_and_gradients():
    x = tf.Variable([[0.1, -2.0, 0.9, 3.0]])  # mean|x| = 1.5 -> V_th = 0.75 at threshold 0.5
    theta = tf.Variable(0.5)
    with tf.GradientTape(persistent=True) as tape:
        y = graded_spike(x, theta)
    np.testing.assert_allclose(y.numpy(), [[0.0, -2.25, 0.75, 3.0]], atol=1e-5)  # counts 0, -3, 1, 4
    np.testing.assert_allclose(tape.gradient(y, x).numpy(), np.ones((1, 4)))  # no dead band
    assert abs(float(tape.gradient(y, theta))) > 0  # learnable thresholds still train
    np.testing.assert_allclose(graded_spike(10.0 * x, 0.5).numpy(), 10.0 * y.numpy(), rtol=1e-5)


def test_field_resonance_learns_spike_params_and_loads_older_weights():
    from gpbacay_arcane.mechanisms import FieldResonance
    layer = FieldResonance(d_model=4)
    x = tf.random.normal((2, 5, 4))
    with tf.GradientTape() as tape:
        y = layer(x)
    assert all(g is not None for g in tape.gradient(y, [layer.threshold, layer.leak_logit]))
    old = {str(i): v.numpy() + 1.0 for i, v in enumerate(layer.trainable_variables[:3])}
    layer.load_own_variables(old)  # checkpoint from before the threshold existed
    np.testing.assert_allclose(layer.resonance_bias.numpy(), old["2"])
    np.testing.assert_allclose(layer.threshold.numpy(), 0.4)


def test_recurrent_cells_learn_spike_params_and_load_older_weights():
    from gpbacay_arcane.mechanisms import PredictiveResonantCell, ResonantGSERCell
    for cell in (ResonantGSERCell(units=4, spike_threshold=0.5), PredictiveResonantCell(units=4, spike_threshold=0.5),
                 GSER(input_dim=3, initial_reservoir_size=4, max_dynamic_reservoir_dim=4, spectral_radius=0.9,
                      leak_rate=0.2, spike_threshold=0.5)):
        layer = tf.keras.layers.RNN(cell)
        x = tf.random.normal((2, 5, 3))
        with tf.GradientTape() as tape:
            y = layer(x)
        assert all(g is not None for g in tape.gradient(y, cell.spike_params))
        own = cell._trainable_variables + cell._non_trainable_variables
        older = [v for v in own if not any(v is p for p in cell.spike_params)]  # layout before spike params
        floats = [v for v in older if "float" in str(v.dtype)]
        cell.load_own_variables({str(i): np.full(v.shape, 0.7, "float32") if "float" in str(v.dtype) else v.numpy()
                                 for i, v in enumerate(older)})
        for v in floats:
            np.testing.assert_allclose(v.numpy(), 0.7, rtol=1e-6)
        np.testing.assert_allclose(tf.sigmoid(cell.leak_logit).numpy(), 1 / (1 + np.exp(3.0)), rtol=1e-5)
