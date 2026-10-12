"""Per-example hierarchical resonance and feedback-alignment training helpers."""

from __future__ import annotations

import threading
from contextlib import contextmanager

import tensorflow as tf


def attach_resonance_probe(owner, model_input, prediction, state_sequences):
    """Attach a model returning predictions plus every resonant sequence."""
    owner.resonance_probe_model = tf.keras.Model(
        model_input,
        [prediction, *state_sequences],
        name=f"{owner.model.name}_resonance_probe",
    )
    # Runtime targets live on cells for only one eager call.  Serializing calls
    # through this lock prevents two requests from swapping those tensors.
    owner._resonance_lock = threading.RLock()


@contextmanager
def _runtime_targets(owner, targets):
    layers = owner.resonant_layer_objects
    for index, layer in enumerate(layers):
        target = None if targets is None or index >= len(targets) else targets[index]
        layer.set_runtime_alignment(target)
    try:
        yield
    finally:
        for layer in layers:
            layer.clear_runtime_alignment()


def forward_with_states(owner, inputs, targets=None, training=False):
    """Run one forward pass and return per-example state prototypes."""
    if not hasattr(owner, "resonance_probe_model"):
        raise ValueError("Build the model before requesting resonance states")
    with _runtime_targets(owner, targets):
        values = owner.resonance_probe_model(inputs, training=training)
    prediction, sequences = values[0], values[1:]
    # A sequence mean retains one prototype per example.  No batch reduction is
    # performed here; this is the central isolation guarantee of this module.
    states = [tf.reduce_mean(sequence, axis=1) for sequence in sequences]
    return prediction, states


def make_feedback_targets(owner, states):
    """Compute simultaneous, per-example top-down harmonization targets."""
    targets = [None] * len(states)
    for upper_index in range(1, len(states)):
        lower_index = upper_index - 1
        lower_layer = owner.resonant_layer_objects[lower_index]
        upper_layer = owner.resonant_layer_objects[upper_index]
        projection = upper_layer.project_feedback(states[upper_index])
        lower = states[lower_index]
        alpha = tf.clip_by_value(
            tf.cast(lower_layer.resonance_factor, lower.dtype)
            + tf.cast(lower_layer.cell.semantic_divergence_weight, lower.dtype),
            0.0,
            0.99,
        )
        targets[lower_index] = lower - alpha * (lower - projection)
    return targets


def hierarchy_divergence(owner, states):
    """Mean squared hierarchy mismatch, preserving examples until reduction."""
    terms = []
    for upper_index in range(1, len(states)):
        projection = owner.resonant_layer_objects[upper_index].project_feedback(
            states[upper_index]
        )
        terms.append(tf.reduce_mean(tf.square(states[upper_index - 1] - projection)))
    if not terms:
        return tf.constant(0.0, dtype=tf.float32)
    return tf.add_n(terms)


def _publish_compatibility_state(owner, states, divergence):
    """Publish monitoring summaries (prototype, divergence) for old APIs.

    ``resonance_alignment``/``alignment_set`` are deliberately not written:
    the non-explicit forward path reads them as a live target, so leaving a
    request's mean target there would perturb every later plain call. The
    per-example targets are available as ``owner.last_resonance_targets``.
    """
    for layer, state in zip(owner.resonant_layer_objects, states):
        layer.cell.last_h.assign(tf.reduce_mean(state, axis=0))
        layer.cell.global_divergence.assign(tf.cast(divergence, tf.float32))


def run_per_example_resonance(owner, inputs, num_cycles=1, reset_state=True):
    """Refine a batch with isolated targets and retain the best cycle."""
    if num_cycles < 0:
        raise ValueError("num_cycles must be non-negative")
    if reset_state:
        owner.reset_resonance_state(clear_prototypes=True)

    lock = getattr(owner, "_resonance_lock", None)
    if lock is None:
        raise ValueError("Build the model before running resonance")

    with lock:
        prediction, states = forward_with_states(owner, inputs, targets=None)
        if num_cycles == 0:
            owner._last_resonance_predictions = prediction
            owner.last_resonance_divergences = []
            owner.last_resonance_best_cycle = 0
            owner.last_resonance_targets = [None] * len(states)
            return []

        divergences = []
        best_divergence = float("inf")
        best_prediction = tf.identity(prediction)
        best_states = states
        best_targets = [None] * len(states)
        best_cycle = 0

        for cycle in range(num_cycles):
            targets = make_feedback_targets(owner, states)
            prediction, states = forward_with_states(owner, inputs, targets=targets)
            divergence_tensor = hierarchy_divergence(owner, states)
            divergence = float(divergence_tensor.numpy())
            divergences.append(divergence)
            if divergence < best_divergence:
                best_divergence = divergence
                best_prediction = tf.identity(prediction)
                best_states = [tf.identity(state) for state in states]
                best_targets = [
                    None if target is None else tf.identity(target)
                    for target in targets
                ]
                best_cycle = cycle + 1
            if divergence < owner.resonant_layer_objects[0].cell.convergence_epsilon:
                break

        _publish_compatibility_state(owner, best_states, tf.constant(best_divergence))
        owner._last_resonance_predictions = best_prediction
        owner.last_resonance_divergences = divergences
        owner.last_resonance_best_cycle = best_cycle
        owner.last_resonance_targets = best_targets
        return divergences


def feedback_alignment_loss(owner, states, stop_target_gradient=True):
    """Reconstruct each lower representation through the upper feedback map."""
    losses = []
    for upper_index in range(1, len(states)):
        target = states[upper_index - 1]
        if stop_target_gradient:
            target = tf.stop_gradient(target)
        projection = owner.resonant_layer_objects[upper_index].project_feedback(
            states[upper_index]
        )
        losses.append(tf.reduce_mean(tf.square(target - projection)))
    if not losses:
        return tf.constant(0.0, dtype=tf.float32)
    return tf.add_n(losses) / tf.cast(len(losses), losses[0].dtype)


def train_feedback_step(
    owner,
    inputs,
    labels=None,
    alignment_weight=0.1,
    optimizer=None,
    task_loss_fn=None,
    stop_target_gradient=True,
):
    """Take one end-to-end task + feedback reconstruction training step.

    Unlike assignment-based feedback, the projection operation is executed
    inside the gradient tape, so ``projection_kernel`` and upstream features
    receive a real gradient path.
    """
    optimizer = optimizer or getattr(owner.model, "optimizer", None)
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(1e-3)
    if task_loss_fn is None and labels is not None:
        task_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    with tf.GradientTape() as tape:
        predictions, states = forward_with_states(owner, inputs, training=True)
        alignment_loss = feedback_alignment_loss(
            owner, states, stop_target_gradient=stop_target_gradient
        )
        task_loss = (
            tf.cast(task_loss_fn(labels, predictions), alignment_loss.dtype)
            if labels is not None
            else tf.zeros((), dtype=alignment_loss.dtype)
        )
        regularization = (
            tf.add_n(owner.model.losses)
            if owner.model.losses
            else tf.zeros((), dtype=alignment_loss.dtype)
        )
        total_loss = task_loss + alignment_weight * alignment_loss + regularization

    variables = owner.model.trainable_variables
    gradients = tape.gradient(total_loss, variables)
    pairs = [(g, v) for g, v in zip(gradients, variables) if g is not None]
    optimizer.apply_gradients(pairs)
    projection_gradient_norms = {}
    for layer in owner.resonant_layer_objects[1:]:
        variable = layer.cell.projection_kernel
        gradient = next(
            (g for g, candidate in zip(gradients, variables) if candidate is variable),
            None,
        )
        projection_gradient_norms[layer.name] = (
            0.0 if gradient is None else float(tf.linalg.global_norm([gradient]).numpy())
        )
    return {
        "loss": float(total_loss.numpy()),
        "task_loss": float(task_loss.numpy()),
        "feedback_alignment_loss": float(alignment_loss.numpy()),
        "projection_gradient_norms": projection_gradient_norms,
    }
