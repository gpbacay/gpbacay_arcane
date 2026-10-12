"""Controlled same-checkpoint evaluation of inference-time resonance."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import tensorflow as tf


@dataclass
class AblationRecord:
    seed: int
    cycles: int
    accuracy: float
    negative_log_likelihood: float
    expected_calibration_error: float
    latency_ms_per_example: float
    forgetting_prediction_drift: float
    weight_drift_l2: float
    order_sensitivity: float
    batch_size_sensitivity: float
    shifted_accuracy: Optional[float]
    shift_accuracy_drop: Optional[float]


@dataclass
class AblationReport:
    records: Sequence[AblationRecord]
    summary: dict

    def to_dict(self):
        return {
            "records": [asdict(record) for record in self.records],
            "summary": self.summary,
        }

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent)


def _labels_as_indices(labels):
    labels = np.asarray(labels)
    return np.argmax(labels, axis=-1) if labels.ndim > 1 else labels.astype(np.int64)


def _classification_metrics(probabilities, labels, ece_bins):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = _labels_as_indices(labels)
    predictions = np.argmax(probabilities, axis=-1)
    confidence = np.max(probabilities, axis=-1)
    accuracy = float(np.mean(predictions == labels))
    chosen = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    nll = float(-np.mean(np.log(chosen)))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, ece_bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if np.any(mask):
            bin_accuracy = np.mean(predictions[mask] == labels[mask])
            ece += np.mean(mask) * abs(bin_accuracy - np.mean(confidence[mask]))
    return accuracy, nll, float(ece)


def _predict_in_batches(owner, inputs, cycles, batch_size):
    outputs = []
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size]
        outputs.append(
            owner.predict_with_resonance(
                batch, resonance_cycles=cycles, reset_state=True, verbose=0
            )
        )
    return np.concatenate(outputs, axis=0)


def _weight_drift(before, after):
    squared = sum(
        float(np.sum(np.square(np.asarray(left) - np.asarray(right))))
        for left, right in zip(before, after)
    )
    return float(np.sqrt(squared))


def run_resonance_ablation(
    owner,
    inputs,
    labels,
    cycles=(0, 1, 2, 4, 8),
    seeds=(0, 1, 2),
    batch_size=32,
    ece_bins=15,
    shift_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    latency_repeats=1,
):
    """Compare resonance depths while restoring one checkpoint each time.

    ``shift_fn`` receives ``(inputs, seed)`` and should return a shifted copy.
    The harness measures accuracy, NLL, ECE, latency, forgetting, order and
    batch-size sensitivity, plus shifted accuracy when a shift is supplied.
    """
    if owner.model is None:
        raise ValueError("Build/load a checkpoint before running the ablation")
    inputs = np.asarray(inputs)
    labels = np.asarray(labels)
    if len(inputs) != len(labels) or len(inputs) == 0:
        raise ValueError("inputs and labels must have the same non-zero length")
    if batch_size < 1 or latency_repeats < 1:
        raise ValueError("batch_size and latency_repeats must be positive")

    checkpoint = [np.array(value, copy=True) for value in owner.model.get_weights()]
    records = []
    try:
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)
            for cycle_count in cycles:
                owner.model.set_weights(checkpoint)
                owner.reset_resonance_state(clear_prototypes=True)
                before_trainable = [
                    np.array(variable.numpy(), copy=True)
                    for variable in owner.model.trainable_variables
                ]

                start = time.perf_counter()
                for _ in range(latency_repeats):
                    probabilities = _predict_in_batches(
                        owner, inputs, int(cycle_count), batch_size
                    )
                elapsed = time.perf_counter() - start
                latency = 1000.0 * elapsed / (len(inputs) * latency_repeats)
                accuracy, nll, ece = _classification_metrics(
                    probabilities, labels, ece_bins
                )

                reversed_probabilities = _predict_in_batches(
                    owner, inputs[::-1], int(cycle_count), batch_size
                )[::-1]
                order_sensitivity = float(
                    np.mean(np.abs(probabilities - reversed_probabilities))
                )
                singleton_probabilities = _predict_in_batches(
                    owner, inputs, int(cycle_count), 1
                )
                batch_sensitivity = float(
                    np.mean(np.abs(probabilities - singleton_probabilities))
                )

                after_trainable = [
                    np.array(variable.numpy(), copy=True)
                    for variable in owner.model.trainable_variables
                ]
                weight_drift = _weight_drift(before_trainable, after_trainable)
                owner.reset_resonance_state(clear_prototypes=True)
                post_predictions = np.asarray(owner.model(inputs, training=False))
                owner.model.set_weights(checkpoint)
                owner.reset_resonance_state(clear_prototypes=True)
                reference_predictions = np.asarray(owner.model(inputs, training=False))
                forgetting = float(
                    np.mean(np.abs(post_predictions - reference_predictions))
                )

                shifted_accuracy = None
                shift_drop = None
                if shift_fn is not None:
                    shifted_inputs = np.asarray(shift_fn(np.array(inputs, copy=True), seed))
                    shifted_probabilities = _predict_in_batches(
                        owner, shifted_inputs, int(cycle_count), batch_size
                    )
                    shifted_accuracy, _, _ = _classification_metrics(
                        shifted_probabilities, labels, ece_bins
                    )
                    shift_drop = accuracy - shifted_accuracy

                records.append(
                    AblationRecord(
                        seed=int(seed),
                        cycles=int(cycle_count),
                        accuracy=accuracy,
                        negative_log_likelihood=nll,
                        expected_calibration_error=ece,
                        latency_ms_per_example=latency,
                        forgetting_prediction_drift=forgetting,
                        weight_drift_l2=weight_drift,
                        order_sensitivity=order_sensitivity,
                        batch_size_sensitivity=batch_sensitivity,
                        shifted_accuracy=shifted_accuracy,
                        shift_accuracy_drop=shift_drop,
                    )
                )
    finally:
        owner.model.set_weights(checkpoint)

    summary = {}
    for cycle_count in cycles:
        group = [record for record in records if record.cycles == int(cycle_count)]
        numeric_fields = (
            "accuracy",
            "negative_log_likelihood",
            "expected_calibration_error",
            "latency_ms_per_example",
            "forgetting_prediction_drift",
            "weight_drift_l2",
            "order_sensitivity",
            "batch_size_sensitivity",
            "shifted_accuracy",
            "shift_accuracy_drop",
        )
        cycle_summary = {}
        for field in numeric_fields:
            values = [getattr(record, field) for record in group]
            values = [value for value in values if value is not None]
            if values:
                cycle_summary[field] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
        summary[str(int(cycle_count))] = cycle_summary
    return AblationReport(records=tuple(records), summary=summary)
