import numpy as np
import tensorflow as tf

from gpbacay_arcane.layers import ResonantGSER
from gpbacay_arcane.foundational_models import HierarchicalResonanceFoundationModel
from gpbacay_arcane.models import NeuromimeticSemanticModel
from gpbacay_arcane.test_time_adaptation import TestTimeAdapter
from gpbacay_arcane.ablation import run_resonance_ablation


def test_harmonize_states_applies_rsaa_update():
    layer = ResonantGSER(
        units=4,
        resonance_factor=0.2,
        resonance_cycles=1,
        return_sequences=False,
    )
    layer(tf.ones((1, 2, 3)), training=False)
    current = layer.cell.last_h.numpy().copy()
    projection = tf.constant([[0.5, -0.25, 0.75, 0.0]], dtype=tf.float32)

    layer.harmonize_states(projection)

    # alpha = resonance_factor + the cell's default semantic weight.
    expected = current - 0.3 * (current - projection.numpy()[0])
    np.testing.assert_allclose(
        layer.cell.resonance_alignment.numpy(), expected, atol=1e-6
    )


def test_predict_with_resonance_refines_current_input_and_is_episodic():
    tf.keras.utils.set_random_seed(123)
    wrapper = NeuromimeticSemanticModel(
        vocab_size=31,
        seq_len=6,
        embed_dim=8,
        hidden_dim=12,
    )
    wrapper.build_model()
    initial_weights = wrapper.model.get_weights()
    for layer in wrapper.resonant_layer_objects:
        # Exercise every requested cycle instead of allowing an early stop.
        layer.cell.convergence_epsilon = -1.0

    current = np.array([[7, 8, 9, 10, 11, 12]], dtype=np.int32)
    previous_a = np.array([[1, 1, 1, 1, 1, 1]], dtype=np.int32)
    previous_b = np.array([[20, 21, 22, 23, 24, 25]], dtype=np.int32)

    wrapper.model(previous_a, training=False)
    output_a = wrapper.predict_with_resonance(
        current, resonance_cycles=3, reset_state=True
    )
    alignment_a = wrapper.last_resonance_targets[0].numpy().mean(axis=0)

    assert output_a.shape == (1, 31)
    assert len(wrapper.last_resonance_divergences) == 3
    assert 1 <= wrapper.last_resonance_best_cycle <= 3
    assert np.linalg.norm(alignment_a) > 0.0

    # Restoring the same model and changing an unrelated preceding request must
    # not change an episodic resonance prediction for the current input.
    wrapper.model.set_weights(initial_weights)
    wrapper.model(previous_b, training=False)
    output_b = wrapper.predict_with_resonance(
        current, resonance_cycles=3, reset_state=True
    )
    np.testing.assert_allclose(output_a, output_b, atol=1e-6)


def test_resonance_does_not_leak_into_later_plain_calls():
    tf.keras.utils.set_random_seed(654)
    wrapper = NeuromimeticSemanticModel(
        vocab_size=17, seq_len=5, embed_dim=6, hidden_dim=8
    )
    wrapper.build_model()
    x = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=np.int32)

    before = wrapper.model(x, training=False).numpy()
    wrapper.predict_with_resonance(x, resonance_cycles=3)
    after = wrapper.model(x, training=False).numpy()

    np.testing.assert_allclose(before, after, atol=1e-6)
    for layer in wrapper.resonant_layer_objects:
        assert float(layer.cell.alignment_set.numpy()) == 0.0


def test_resonance_target_depends_on_current_input():
    tf.keras.utils.set_random_seed(321)
    wrapper = NeuromimeticSemanticModel(
        vocab_size=31,
        seq_len=6,
        embed_dim=8,
        hidden_dim=12,
    )
    wrapper.build_model()
    initial_weights = wrapper.model.get_weights()

    x = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int32)
    y = np.array([[20, 21, 22, 23, 24, 25]], dtype=np.int32)
    wrapper.predict_with_resonance(x, resonance_cycles=1, reset_state=True)
    target_x = wrapper.last_resonance_targets[0].numpy().mean(axis=0)

    wrapper.model.set_weights(initial_weights)
    wrapper.predict_with_resonance(y, resonance_cycles=1, reset_state=True)
    target_y = wrapper.last_resonance_targets[0].numpy().mean(axis=0)

    assert np.linalg.norm(target_x - target_y) > 1e-6


def test_hierarchical_foundation_public_resonance_api_runs():
    tf.keras.utils.set_random_seed(456)
    wrapper = HierarchicalResonanceFoundationModel(
        vocab_size=23,
        seq_len=4,
        embed_dim=6,
        hidden_dim=8,
        num_resonance_levels=2,
        use_temporal_coherence=False,
        use_attention_fusion=False,
        dropout_rate=0.0,
    )
    wrapper.build_model()
    inputs = np.array([[1, 2, 3, 4]], dtype=np.int32)

    output = wrapper.predict_with_resonance(inputs, resonance_cycles=2)

    assert output.shape == (1, 23)
    assert wrapper.last_resonance_divergences
    assert 1 <= wrapper.last_resonance_best_cycle <= 2


def test_per_example_targets_remove_cobatching_contamination():
    tf.keras.utils.set_random_seed(789)
    wrapper = NeuromimeticSemanticModel(
        vocab_size=29, seq_len=4, embed_dim=6, hidden_dim=8
    )
    wrapper.build_model()
    first = np.array([[1, 2, 3, 4]], dtype=np.int32)
    other = np.array([[20, 21, 22, 23]], dtype=np.int32)

    alone = wrapper.predict_with_resonance(first, resonance_cycles=2)
    together = wrapper.predict_with_resonance(
        np.concatenate([first, other], axis=0), resonance_cycles=2
    )

    np.testing.assert_allclose(alone[0], together[0], atol=1e-6)
    assert wrapper.last_resonance_targets[0].shape == (2, 8)
    assert wrapper.last_resonance_targets[1] is None


def test_feedback_projection_has_real_gradient_path():
    tf.keras.utils.set_random_seed(901)
    wrapper = NeuromimeticSemanticModel(
        vocab_size=19, seq_len=4, embed_dim=5, hidden_dim=7
    )
    wrapper.build_model()
    x = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int32)
    y = np.array([2, 3], dtype=np.int32)
    before = wrapper.resonant_layer_objects[1].cell.projection_kernel.numpy().copy()

    metrics = wrapper.train_feedback_step(
        x,
        y,
        alignment_weight=1.0,
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2),
    )
    after = wrapper.resonant_layer_objects[1].cell.projection_kernel.numpy()

    assert metrics["feedback_alignment_loss"] >= 0.0
    assert metrics["projection_gradient_norms"]["resonant_gser_2"] > 0.0
    assert np.linalg.norm(after - before) > 0.0


def test_test_time_adapter_is_bounded_and_episodic():
    inputs = tf.keras.Input((3,))
    normalized = tf.keras.layers.LayerNormalization()(inputs)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(normalized)
    model = tf.keras.Model(inputs, outputs)
    x = np.array([[2.0, -1.0, 0.5], [-0.5, 1.5, 0.25]], dtype=np.float32)
    adapter = TestTimeAdapter(model, variables="normalization")
    before = [value.numpy().copy() for value in adapter.variables]

    result = adapter.adapt(
        x,
        objective="entropy",
        steps=3,
        learning_rate=1e-2,
        max_update_norm=1e-3,
        max_total_delta_norm=2e-3,
        episodic=True,
    )

    assert result.best_objective <= result.baseline_objective + 1e-7
    assert all(norm <= 1e-3 + 1e-8 for norm in result.update_norms)
    for variable, value in zip(adapter.variables, before):
        np.testing.assert_allclose(variable.numpy(), value, atol=1e-8)


def test_test_time_adapter_rolls_back_a_worse_objective():
    inputs = tf.keras.Input((1,))
    outputs = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer="ones"
    )(inputs)
    model = tf.keras.Model(inputs, outputs)
    x = np.ones((1, 1), dtype=np.float32)
    adapter = TestTimeAdapter(model, variables="all")

    def deliberately_mismatched_objective(current_model, unused_inputs, training):
        del unused_inputs
        weight = current_model.trainable_variables[0]
        return -tf.reduce_sum(weight) if training else tf.reduce_sum(tf.square(weight))

    result = adapter.adapt(
        x,
        objective=deliberately_mismatched_objective,
        steps=2,
        learning_rate=0.1,
        max_update_norm=0.1,
        max_total_delta_norm=0.2,
    )

    assert result.rolled_back
    assert result.accepted_steps == 0
    assert result.final_objective == result.baseline_objective


def test_ablation_restores_the_same_checkpoint_and_reports_sensitivities():
    tf.keras.utils.set_random_seed(902)
    wrapper = NeuromimeticSemanticModel(
        vocab_size=11, seq_len=3, embed_dim=4, hidden_dim=5
    )
    wrapper.build_model()
    x = np.array([[1, 2, 3], [3, 2, 1]], dtype=np.int32)
    y = np.array([1, 2], dtype=np.int32)
    checkpoint = [weight.copy() for weight in wrapper.model.get_weights()]

    report = run_resonance_ablation(
        wrapper,
        x,
        y,
        cycles=(0, 1),
        seeds=(0,),
        batch_size=2,
    )

    assert len(report.records) == 2
    assert set(report.summary) == {"0", "1"}
    assert report.records[1].batch_size_sensitivity < 1e-6
    for actual, expected in zip(wrapper.model.get_weights(), checkpoint):
        np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_hierarchical_model_wires_enhanced_homeostasis_signals():
    tf.keras.utils.set_random_seed(903)
    wrapper = HierarchicalResonanceFoundationModel(
        vocab_size=13,
        seq_len=3,
        embed_dim=4,
        hidden_dim=6,
        num_resonance_levels=2,
        use_temporal_coherence=False,
        use_attention_fusion=True,
        dropout_rate=0.0,
        enable_inference_plasticity=True,
        use_hierarchical_homeostasis=True,
        homeostasis_config={
            "homeostatic_window_size": 4,
            "slow_bcm_tau": 20.0,
            "adaptive_target_rate": 0.5,
            "resonance_coupling": 1.0,
        },
    )
    wrapper.build_model()
    inputs = np.array([[1, 2, 3], [3, 2, 1]], dtype=np.int32)

    predictions = wrapper.predict_with_resonance(inputs, resonance_cycles=1)
    first_plastic = wrapper.model.get_layer("bioplastic_1")

    assert predictions.shape == (2, 13)
    assert wrapper.model.get_layer("hierarchical_homeostasis") is not None
    assert int(first_plastic.window_count.numpy()) > 0
    assert first_plastic.slow_trace is not None
    assert first_plastic.adaptive_target is not None
