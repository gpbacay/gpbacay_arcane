# Inference-Time Resonance in ARCANE

## Overview

Inference-Time Resonance is ARCANE's state-alignment path for adaptive System 1 inference. It uses top-down feedback to revisit a trained model's state before choosing an output, with explicit controls for cycle count, objectives, and reset behavior.

## Core API

### 1. `run_resonance_cycle()`

Runs one explicit feedback cycle in `HierarchicalResonanceFoundationModel`: higher-level state is projected downward, then each lower state is harmonized toward its per-example target.

**Location:** `gpbacay_arcane/foundational_models.py`

### 2. `propagate_feedback_to_lower()`

Propagates a `ResonantGSER` layer's feedback to the lower layer connected through `set_lower_layer()`.

**Location:** `gpbacay_arcane/layers.py`

### 3. `predict_with_resonance()`

Runs bounded resonance cycles before returning a prediction from `HierarchicalResonanceFoundationModel` or `NeuromimeticSemanticModel`.

**Locations:**
- `gpbacay_arcane/foundational_models.py`
- `gpbacay_arcane/models.py`

### 4. Hierarchical references

Layers support name-based and direct references, so hierarchical connections remain explicit when model references are unavailable.

### 5. Current-input Refinement Cycles

Every outer cycle now includes a complete forward refinement of the current input. Projection operations are independent, but no O(1) cross-layer parallelism is claimed.

## Usage

### Basic Usage: Automatic Resonance

```python
from gpbacay_arcane import HierarchicalResonanceFoundationModel
import numpy as np

# Create and build model
model = HierarchicalResonanceFoundationModel(vocab_size=1000, seq_len=32)
model.build_model()
model.compile_model()

# Load weights (if available)
# model.model.load_weights('path/to/weights')

# Inference with automatic resonance cycles
input_data = np.random.randint(0, 1000, size=(1, 32))
predictions = model.predict_with_resonance(
    input_data, 
    resonance_cycles=5,  # Number of resonance cycles
    verbose=1            # Show progress
)
```

### Advanced Usage: Manual Control

```python
# Run resonance cycles manually
divergences = model.run_resonance_cycle(
    num_cycles=10,
    inputs=input_data,
    reset_state=True,
)
print(f"Convergence history: {divergences}")

# The best-cycle prediction is retained by the wrapper
predictions = model._last_resonance_predictions.numpy()
```

### Comparison: Standard vs Resonant Inference

```python
# Standard inference (single forward pass, no resonance)
predictions_standard = model.model.predict(input_data, verbose=0)

# Resonant inference (runs resonance cycles first, then forward pass)
predictions_resonant = model.predict_with_resonance(
    input_data, 
    resonance_cycles=5
)

# Compare results
top_token_standard = np.argmax(predictions_standard[0])
top_token_resonant = np.argmax(predictions_resonant[0])
print(f"Standard: {top_token_standard}, Resonant: {top_token_resonant}")
```

## How It Works

### 1. Resonance Cycle Process

When `run_resonance_cycle()` or `predict_with_resonance()` is called:

1. **Episodic reset**: By default, transient prototypes and alignments from earlier requests are cleared.
2. **Forward initialization**: The current input is encoded once and a `(batch, units)` state is extracted for every layer.
3. **Top-down projection**: Higher layers project each example's state to the layer below.
4. **State harmonization**: Each lower state takes an RSAA step toward its own projection. The resulting batch-shaped tensor is installed only for the next forward call.
5. **Forward refinement**: The same current input is re-encoded with those targets active.
6. **Fresh convergence check**: Divergence is measured between the refined lower states and newly projected higher states.
7. **Iteration**: Steps 3–6 repeat. The lowest-divergence cycle is retained because nonlinear/spiking dynamics need not improve monotonically.

Targets are per-example. They are never saved as layer weights, and a request lock prevents concurrent calls from exchanging ephemeral targets. The old `(units,)` variables remain as monitoring/backward-compatibility summaries and are not read by the explicit per-example loop.

### 2. Forward Pass Integration

During every resonance cycle:

- Each `ResonantGSERCell` uses an explicit batch-shaped runtime target in its internal `resonance_loop()`.
- The complete model is re-run on the current input, allowing lower-layer changes to affect higher layers in the next cycle.
- `last_resonance_divergences` records every attempted cycle and `last_resonance_best_cycle` identifies the committed result.

### 3. Key Differences from Training-Time Resonance

| Aspect | Training-Time (Callback) | Inference-Time (Manual) |
|--------|-------------------------|--------------------------|
| **When** | Callback before each batch | Explicit current-input prediction |
| **Input represented** | Previous batch prototype | Current input |
| **Weight Updates** | Normal optimizer step follows | None |
| **Full inner loop?** | No; callback cannot re-run the current batch | Yes |

### 4. PredictiveResonantLayer and BioplasticDenseLayer Inference-Time Features

**PredictiveResonantLayer** (local predictive resonance) and **BioplasticDenseLayer** support additional inference-time behavior without manual resonance cycles:

- **PredictiveResonantLayer with `persist_alignment=True`**: The layer keeps a slow-moving alignment state across separate forward passes. Each call updates an internal alignment memory used as the initial resonance target for the next call, so repeated inference (e.g. on the same or new inputs) exhibits stateful resonance across calls. Weights are unchanged; only the non-trainable alignment state evolves.

- **BioplasticDenseLayer with `enable_inference_plasticity=True`**: During inference, BCM uses a running squared-activity threshold $\theta$ to update `plastic_kernel` only: $\Delta W \propto x^{\top}[y \odot (y-\theta)]$, with fast-weight decay and activity scaling. The output is recomputed with the adapted weights; `reset_plasticity()` starts a new episode. The gradient `kernel` is unchanged.

Use these options in sequence models (e.g. MNIST classifier with PredictiveResonantLayer and BioplasticDenseLayer) when you want the model to adapt its internal state or readout during repeated inference.

## Benefits

### 1. Additional latent computation

The same weights receive more compute, while higher-level state can influence the next lower-level pass.

### 2. Measurable coherence

The API reports an explicit hierarchy-divergence trace. Whether lower divergence improves task quality must be established by an ablation on the target task.

### 3. Flexible Control

You can choose:
- **No resonance**: Fast inference, single forward pass
- **Few cycles (2-5)**: Candidate refinements with bounded overhead
- **Many cycles**: Research use only; more cycles are not assumed to be better

### 4. Research Applications

Enables studying:
- How many resonance cycles are optimal for different tasks
- The relationship between divergence and prediction quality
- The trade-off between inference speed and reasoning quality

## Training the feedback maps

Assignment-based feedback has no gradient path. Use the explicit reconstruction step to train it:

```python
metrics = model.train_feedback_step(
    input_data,
    labels,
    alignment_weight=0.1,
)
print(metrics["feedback_alignment_loss"])
print(metrics["projection_gradient_norms"])
```

This minimizes the task loss plus `||stop_gradient(S_lower) - P(S_higher)||^2` inside one gradient tape. Consequently the feedback projection and the features producing the higher state receive gradients. Set `stop_target_gradient=False` to train both sides of the reconstruction target.

## Objective-driven test-time adaptation

Resonance changes activations but does not by itself show that predictions improve. For an actual unlabeled test-time objective, use:

```python
result = model.adapt_at_inference(
    input_data,
    objective="entropy",       # or "consistency"
    steps=4,
    learning_rate=1e-4,
    max_update_norm=1e-3,
    max_total_delta_norm=5e-3,
    episodic=True,
)
predictions = result.predictions
```

`TestTimeAdapter` snapshots selected normalization variables, bounds every update and the total displacement, retains only objective improvements, rolls back a worsening/non-finite step, and restores the checkpoint after emitting predictions when `episodic=True`. Reconstruction and masked-token objectives can be supplied as callables.

## Controlled ablation

```python
from gpbacay_arcane import run_resonance_ablation

report = run_resonance_ablation(
    model,
    test_x,
    test_y,
    cycles=(0, 1, 2, 4, 8),
    seeds=(0, 1, 2, 3, 4),
    batch_size=32,
    shift_fn=my_distribution_shift,
)
print(report.to_json())
```

Every condition starts from the same saved weights. The report includes accuracy, negative log-likelihood, expected calibration error, latency, prediction/weight forgetting, order sensitivity, batch-size sensitivity, and shifted accuracy. Use a real trained checkpoint and held-out data; unit tests only validate the harness, not a task-performance claim.

## Performance Considerations

### Computational Cost

- **Standard Inference**: 1 forward pass
- **Resonant Inference**: N + 1 full forward passes, plus inexpensive projections
- **Overhead**: approximately N extra model evaluations

### Execution

Projection operations are independent, but the Python implementation does not promise O(1) execution across layers. The forward refinements dominate runtime and remain sequential across resonance cycles.

### When to Use

**Use inference-time resonance when:**
- You can measure whether added cycles improve the task objective
- Latency from extra forward evaluations is acceptable
- You want to study state refinement or top-down alignment
- You will compare against zero-cycle inference from the same checkpoint

**Use standard inference when:**
- Speed is critical
- Tasks are simple/straightforward
- Real-time applications
- Batch processing many examples

## Example Output

```
Running 5 resonance cycles for inference-time alignment...
Resonance finished. Best divergence: 0.000234 (cycle 3)
Prediction shape: (1, 1000)
Top 5 predicted tokens: [342, 891, 123, 567, 234]
```

## Technical Details

### Method Signatures

```python
# HierarchicalResonanceFoundationModel
def run_resonance_cycle(self, num_cycles=1, inputs=None, reset_state=True) -> List[float]
def predict_with_resonance(self, inputs, resonance_cycles=5, verbose=0, reset_state=True) -> np.ndarray

# NeuromimeticSemanticModel  
def run_resonance_cycle(self, num_cycles=1, inputs=None, reset_state=True) -> List[float]
def predict_with_resonance(self, inputs, resonance_cycles=5, verbose=0, reset_state=True) -> np.ndarray

# ResonantGSER Layer
def propagate_feedback_to_lower(self) -> None
def project_feedback(self, representation=None) -> tf.Tensor
def harmonize_states(self, projection) -> None
def get_divergence(self) -> float
```

### Internal State Management

- Each `ResonantGSERCell` maintains:
  - ephemeral runtime alignment: per-example target used by explicit resonance
  - `resonance_alignment`: legacy target for non-explicit calls and the training callback; explicit resonance never writes it, so later plain calls are unaffected (per-example targets: `last_resonance_targets`)
  - `last_h`: compatibility/monitoring batch mean
  - `global_divergence`: Current divergence metric

- Resonance cycles carry batch-shaped target tensors without modifying weights.
- `reset_state=True` prevents unrelated prior requests from supplying the initial state.

## See Also

- `examples/inference_time_resonance_example.py` - Complete working example
- `docs/NEURAL_RESONANCE.md` - Detailed explanation of resonance mechanism
- `docs/RESONANT_GSER.md` - Technical details on ResonantGSER layers

## Remaining research work

The implementation now makes the proposed mechanisms testable; it does not establish a gain. The next result should come from the controlled ablation above on a trained checkpoint, with uncertainty across seeds and a pre-declared primary metric.

---

*Implementation revised: October 2026*
