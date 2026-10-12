# ARCANE 4.1: Testable Adaptive Inference and Homeostatic Plasticity

ARCANE is an experimental TensorFlow library for recurrent state refinement, local plasticity, spiking transforms, and activity regulation. Version 4.1 makes an important distinction explicit: refining hidden state with fixed weights is not the same operation as learning at inference time.

## Per-example resonance state

`predict_with_resonance` now carries top-down targets with shape `(batch, units)` through every refinement cycle. Each example therefore retains its own target instead of being attracted toward a batch-mean prototype. The legacy callback remains available for compatibility, but it uses a previous-batch summary and is not a current-input inner loop.

The model records divergence on every cycle and returns the prediction associated with the lowest observed divergence. Lower divergence is an internal diagnostic, not proof of higher task quality.

## Feedback projections with a training objective

Learned top-down maps need a gradient path. `train_feedback_step` combines the task loss with an explicit alignment or reconstruction objective between lower states and projections of higher states. Identity feedback remains a useful initialization when adjacent dimensions match, while learned feedback can now be evaluated rather than remaining effectively untrained.

## Objective-driven test-time adaptation

ARCANE exposes optional test-time weight adaptation with entropy, consistency, reconstruction, masked-token, or custom objectives. The adapter provides:

- checkpoint snapshots and reset;
- per-step and total update-norm limits;
- rollback when the objective worsens; and
- episodic operation, which restores the original checkpoint after prediction.

BCM fast weights remain available as a separate unsupervised correlation rule. BCM activity updates should not be interpreted as evidence that predictions improve.

## Expanded homeostatic plasticity

`BioplasticDenseLayer` now supports exact sliding activity windows, fast and slow traces, global/local/hybrid regulation, adaptive activity targets, attention-guided updates, resonance coupling, and hierarchy-aware targets. `HierarchicalHomeostaticController` accepts two or more representation tensors and returns bounded regulated tensors plus imbalance and target signals.

These options are disabled by default to preserve checkpoint behavior and because their effect is task-dependent. Enabling more adaptive mechanisms does not guarantee better accuracy.

## Controlled evaluation

The resonance ablation harness compares one checkpoint at 0, 1, 2, 4, and 8 cycles over multiple seeds. It reports accuracy, negative log-likelihood, expected calibration error, latency, forgetting, weight drift, order sensitivity, batch-size sensitivity, and optional distribution-shift results.

This evaluation is the standard needed to determine whether additional inference cycles or fast-weight updates help a specific task. Historical Tiny Shakespeare and MNIST examples are exploratory and do not isolate inference-time learning.

## Scope

ARCANE's biological vocabulary describes sources of design inspiration. The implementation does not establish biological equivalence, human-like deliberation, reduced energy use, resistance to catastrophic forgetting, or improved reasoning. Those remain empirical questions.
