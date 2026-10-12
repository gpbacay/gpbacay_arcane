"""Objective-driven, bounded, reversible test-time adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np
import tensorflow as tf


@dataclass
class ModelSnapshot:
    """Values of a concrete set of model variables."""

    variables: Sequence[tf.Variable]
    values: Sequence[np.ndarray]


@dataclass
class AdaptationResult:
    """Outcome and diagnostics from one episodic adaptation request."""

    predictions: np.ndarray
    baseline_objective: float
    best_objective: float
    final_objective: float
    accepted_steps: int
    attempted_steps: int
    rolled_back: bool
    update_norms: Sequence[float]


Objective = Callable[[tf.keras.Model, object, bool], tf.Tensor]


class TestTimeAdapter:
    """Adapt selected model weights on an unlabeled current batch.

    The adapter is episodic by default: it snapshots selected variables,
    optimizes a measurable objective, restores the best observed point, emits
    predictions, then restores the original checkpoint unless ``episodic`` is
    disabled explicitly.
    """

    __test__ = False

    def __init__(
        self,
        model: tf.keras.Model,
        variables: Union[str, Iterable[tf.Variable]] = "normalization",
    ):
        self.model = model
        self.variables = self._select_variables(variables)
        if not self.variables:
            raise ValueError(
                "No variables matched the adaptation policy; pass variables='all' "
                "or an explicit variable iterable"
            )
        self._initial_snapshot = self.snapshot()

    def _select_variables(self, policy):
        if not isinstance(policy, str):
            return list(policy)
        trainable = list(self.model.trainable_variables)
        if policy == "all":
            return trainable
        if policy == "normalization":
            selected = []
            for variable in trainable:
                path = getattr(variable, "path", variable.name).lower()
                if any(token in path for token in ("normalization", "layer_norm", "batch_norm")):
                    selected.append(variable)
            return selected
        if policy == "feedback":
            return [
                variable
                for variable in trainable
                if "projection_kernel" in getattr(variable, "path", variable.name)
            ]
        raise ValueError("variables must be 'normalization', 'feedback', 'all', or an iterable")

    def snapshot(self) -> ModelSnapshot:
        return ModelSnapshot(
            variables=tuple(self.variables),
            values=tuple(np.array(variable.numpy(), copy=True) for variable in self.variables),
        )

    @staticmethod
    def restore(snapshot: ModelSnapshot) -> None:
        for variable, value in zip(snapshot.variables, snapshot.values):
            variable.assign(value)

    def reset(self) -> None:
        """Restore the variable values captured when this adapter was created."""
        self.restore(self._initial_snapshot)

    @staticmethod
    def entropy_objective(from_logits=False) -> Objective:
        """Return mean predictive entropy (minimization objective)."""

        def objective(model, inputs, training):
            outputs = model(inputs, training=training)
            probabilities = tf.nn.softmax(outputs, axis=-1) if from_logits else outputs
            probabilities = tf.clip_by_value(probabilities, 1e-7, 1.0)
            return tf.reduce_mean(
                -tf.reduce_sum(probabilities * tf.math.log(probabilities), axis=-1)
            )

        return objective

    @staticmethod
    def consistency_objective() -> Objective:
        """Return a two-stochastic-view consistency objective.

        This is useful when the model contains dropout or another stochastic
        augmentation in its training path.
        """

        def objective(model, inputs, training):
            del training
            first = model(inputs, training=True)
            second = model(inputs, training=True)
            target = tf.stop_gradient(0.5 * (first + second))
            return 0.5 * (
                tf.reduce_mean(tf.square(first - target))
                + tf.reduce_mean(tf.square(second - target))
            )

        return objective

    @staticmethod
    def reconstruction_objective(targets, loss_fn=None) -> Objective:
        """Return an output reconstruction objective for a supplied target."""
        target_tensor = tf.convert_to_tensor(targets)
        loss_fn = loss_fn or tf.keras.losses.MeanSquaredError()

        def objective(model, inputs, training):
            outputs = model(inputs, training=training)
            return loss_fn(tf.cast(target_tensor, outputs.dtype), outputs)

        return objective

    @staticmethod
    def masked_token_objective(target_token_ids, mask, from_logits=True) -> Objective:
        """Return sparse token loss evaluated only where ``mask`` is true."""
        targets = tf.convert_to_tensor(target_token_ids)
        mask_tensor = tf.cast(mask, tf.float32)
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits, reduction="none"
        )

        def objective(model, inputs, training):
            outputs = model(inputs, training=training)
            losses = cross_entropy(targets, outputs)
            typed_mask = tf.cast(mask_tensor, losses.dtype)
            return tf.reduce_sum(losses * typed_mask) / tf.maximum(
                tf.reduce_sum(typed_mask), tf.cast(1.0, losses.dtype)
            )

        return objective

    def adapt(
        self,
        inputs,
        objective: Union[str, Objective] = "entropy",
        steps: int = 1,
        learning_rate: float = 1e-4,
        max_update_norm: float = 1e-3,
        max_total_delta_norm: float = 5e-3,
        rollback_on_worsening: bool = True,
        min_improvement: float = 0.0,
        episodic: bool = True,
        from_logits: bool = False,
    ) -> AdaptationResult:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if learning_rate <= 0 or max_update_norm <= 0 or max_total_delta_norm <= 0:
            raise ValueError("learning rate and update bounds must be positive")
        if isinstance(objective, str):
            if objective == "entropy":
                objective_fn = self.entropy_objective(from_logits=from_logits)
            elif objective == "consistency":
                objective_fn = self.consistency_objective()
            else:
                raise ValueError(
                    "Built-in objectives are 'entropy' and 'consistency'; pass a "
                    "callable for reconstruction or masked-token objectives"
                )
        else:
            objective_fn = objective

        request_snapshot = self.snapshot()
        baseline = float(objective_fn(self.model, inputs, False).numpy())
        best_objective = baseline
        best_snapshot = request_snapshot
        accepted_steps = 0
        attempted_steps = 0
        rolled_back = False
        update_norms = []

        for _ in range(steps):
            attempted_steps += 1
            with tf.GradientTape() as tape:
                loss = objective_fn(self.model, inputs, True)
            gradients = tape.gradient(loss, self.variables)
            updates = []
            update_variables = []
            for gradient, variable in zip(gradients, self.variables):
                if gradient is not None:
                    updates.append(-tf.cast(learning_rate, gradient.dtype) * gradient)
                    update_variables.append(variable)
            if not updates:
                rolled_back = True
                break
            updates, raw_norm = tf.clip_by_global_norm(updates, max_update_norm)
            applied_norm = min(float(raw_norm.numpy()), max_update_norm)
            update_norms.append(applied_norm)
            for variable, update in zip(update_variables, updates):
                variable.assign_add(tf.cast(update, variable.dtype))

            # Enforce a trust region around the request snapshot in addition to
            # bounding each individual step.
            deltas = [
                variable - tf.cast(start, variable.dtype)
                for variable, start in zip(self.variables, request_snapshot.values)
            ]
            total_delta = tf.linalg.global_norm(deltas)
            if float(total_delta.numpy()) > max_total_delta_norm:
                scale = max_total_delta_norm / (float(total_delta.numpy()) + 1e-12)
                for variable, start, delta in zip(
                    self.variables, request_snapshot.values, deltas
                ):
                    variable.assign(tf.cast(start, variable.dtype) + scale * delta)

            candidate = float(objective_fn(self.model, inputs, False).numpy())
            is_finite = np.isfinite(candidate)
            improved = is_finite and candidate <= best_objective - min_improvement
            if improved:
                best_objective = candidate
                best_snapshot = self.snapshot()
                accepted_steps += 1
            elif rollback_on_worsening:
                self.restore(best_snapshot)
                rolled_back = True
                break

        self.restore(best_snapshot)
        final_objective = float(objective_fn(self.model, inputs, False).numpy())
        predictions = np.asarray(self.model(inputs, training=False))
        if episodic:
            self.restore(request_snapshot)

        return AdaptationResult(
            predictions=predictions,
            baseline_objective=baseline,
            best_objective=best_objective,
            final_objective=final_objective,
            accepted_steps=accepted_steps,
            attempted_steps=attempted_steps,
            rolled_back=rolled_back,
            update_norms=tuple(update_norms),
        )


def adapt_at_inference(owner, inputs, variables="normalization", **kwargs):
    """Convenience entry point for ARCANE wrapper classes."""
    if owner.model is None:
        raise ValueError("Build the model before test-time adaptation")
    adapter = TestTimeAdapter(owner.model, variables=variables)
    result = adapter.adapt(inputs, **kwargs)
    owner.last_adaptation_result = result
    return result
