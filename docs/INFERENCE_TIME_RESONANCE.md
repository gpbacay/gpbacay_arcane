# Inference-Time Resonance in ARCANE

## Overview

Inference-Time Resonance enables ARCANE models to perform deliberative "System 2" reasoning **during inference**, not just during training. This allows models to align their hierarchical internal states before making predictions, leading to more coherent and contextually-aware outputs.

## What Was Implemented

### 1. Fixed `run_resonance_cycle()` Method

The `run_resonance_cycle()` method in `HierarchicalResonanceFoundationModel` was calling a non-existent `propagate_feedback_to_lower()` method. This has been fixed to use the existing `project_feedback()` and `harmonize_states()` methods, matching the pattern used in the training callback.

**Location:** `gpbacay_arcane/foundational_models.py`

### 2. Added `propagate_feedback_to_lower()` Convenience Method

Added a convenience method to `ResonantGSER` layer that automatically propagates feedback to the lower layer using the hierarchical connections established via `set_lower_layer()`.

**Location:** `gpbacay_arcane/layers.py`

### 3. Added `predict_with_resonance()` Method

Added a convenient method to both `HierarchicalResonanceFoundationModel` and `NeuromimeticSemanticModel` that automatically runs resonance cycles before prediction, making inference-time resonance easy to use.

**Locations:**
- `gpbacay_arcane/foundational_models.py`
- `gpbacay_arcane/models.py`

### 4. Enhanced Layer References

Improved the layer reference system to support both name-based and direct references, ensuring robust hierarchical connections even when model references aren't available.

### 5. Parallelized Resonance Cycles

Added parallelization within each phase of the resonance cycle:
- **Projection Phase**: All layers compute projections simultaneously (parallel)
- **Harmonization Phase**: All layers apply harmonizations simultaneously (parallel)
- **Divergence Computation**: All divergence values computed in parallel

This provides significant performance improvements, especially for models with many resonance levels. TensorFlow's graph executor automatically parallelizes independent operations when they're collected together.

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
divergences = model.run_resonance_cycle(num_cycles=10)
print(f"Convergence history: {divergences}")

# Then predict (resonance_alignment is already set)
predictions = model.model.predict(input_data, verbose=0)
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

1. **Top-Down Projection**: Higher layers project their internal states down to lower layers using `project_feedback()`.
2. **Harmonization**: Lower layers receive these projections and update their `resonance_alignment` targets using `harmonize_states()`.
3. **Convergence Check**: The system measures prediction divergence across all layers.
4. **Iteration**: Steps 1-3 repeat for the specified number of cycles (or until convergence).

### 2. Forward Pass Integration

After resonance cycles complete, the `resonance_alignment` values are set in each layer's cell. When the forward pass runs:

- Each `ResonantGSERCell` uses its `resonance_alignment` in its internal `resonance_loop()`.
- The cell harmonizes its state with the top-down expectation during the forward pass.
- This creates a coherent hierarchical representation aligned with higher-level context.

### 3. Key Differences from Training-Time Resonance

| Aspect | Training-Time (Callback) | Inference-Time (Manual) |
|--------|-------------------------|--------------------------|
| **When** | Automatically before each batch | Manually before prediction |
| **Purpose** | Align states before weight updates | Align states before output |
| **Weight Updates** | Yes (after resonance) | No (inference only) |
| **Automatic** | Yes (via callback) | No (must call explicitly) |

### 4. PredictiveResonantLayer and BioplasticDenseLayer Inference-Time Features

**PredictiveResonantLayer** (local predictive resonance) and **BioplasticDenseLayer** support additional inference-time behavior without manual resonance cycles:

- **PredictiveResonantLayer with `persist_alignment=True`**: The layer keeps a slow-moving alignment state across separate forward passes. Each call updates an internal alignment memory used as the initial resonance target for the next call, so repeated inference (e.g. on the same or new inputs) exhibits stateful resonance across calls. Weights are unchanged; only the non-trainable alignment state evolves.

- **BioplasticDenseLayer with `enable_inference_plasticity=True`**: During inference, the layer applies a lightweight Hebbian-style update to a non-trainable plastic weight component. The effective weights are `kernel + plastic_kernel`; only `plastic_kernel` is updated at inference time, so gradient-based training remains unchanged. This gives inference-time learning (e.g. confidence or predictions adapting over repeated calls) without a separate training step.

Use these options in sequence models (e.g. MNIST classifier with PredictiveResonantLayer and BioplasticDenseLayer) when you want the model to adapt its internal state or readout during repeated inference.

## Benefits

### 1. Deliberative Reasoning

Models can "think" before responding, resolving ambiguities through hierarchical state alignment.

### 2. Improved Coherence

Top-down context influences lower-level feature extraction, leading to more contextually-aware predictions.

### 3. Flexible Control

You can choose:
- **No resonance**: Fast inference, single forward pass
- **Few cycles (3-5)**: Balanced speed and deliberation
- **Many cycles (10+)**: Maximum deliberation for complex tasks

### 4. Research Applications

Enables studying:
- How many resonance cycles are optimal for different tasks
- The relationship between divergence and prediction quality
- The trade-off between inference speed and reasoning quality

## Performance Considerations

### Computational Cost

- **Standard Inference**: 1 forward pass
- **Resonant Inference**: N resonance cycles + 1 forward pass
- **Overhead**: ~N × (number of layers) projection/harmonization operations

### Parallelization

The inference-time resonance system uses **parallelization within each phase**:

1. **Projection Phase (Parallelized)**: All layers compute their projections simultaneously since they're independent operations. TensorFlow's graph executor automatically parallelizes these when operations are collected together.

2. **Harmonization Phase (Parallelized)**: All layers apply harmonizations simultaneously since each layer only modifies its own `resonance_alignment` variable.

3. **Divergence Computation (Parallelized)**: All divergence values are computed in parallel.

**Performance Benefit**: With L layers, parallelization reduces the time complexity from O(L) sequential operations to O(1) parallel operations per phase (limited by hardware parallelism). This provides significant speedup, especially for models with many resonance levels.

**Note**: True parallelism depends on TensorFlow's graph execution and hardware capabilities (GPU/TPU). On CPU, operations may still execute sequentially but benefit from optimized graph execution.

### When to Use

**Use inference-time resonance when:**
- Quality is more important than speed
- Tasks require complex reasoning
- You want to study deliberative behavior
- Contextual coherence is critical

**Use standard inference when:**
- Speed is critical
- Tasks are simple/straightforward
- Real-time applications
- Batch processing many examples

## Example Output

```
Running 5 resonance cycles for inference-time alignment...
Resonance converged. Final divergence: 0.000234
Prediction shape: (1, 1000)
Top 5 predicted tokens: [342, 891, 123, 567, 234]
```

## Technical Details

### Method Signatures

```python
# HierarchicalResonanceFoundationModel
def run_resonance_cycle(self, num_cycles=1) -> List[float]
def predict_with_resonance(self, inputs, resonance_cycles=5, verbose=0) -> np.ndarray

# NeuromimeticSemanticModel  
def run_resonance_cycle(self, num_cycles=1) -> List[float]
def predict_with_resonance(self, inputs, resonance_cycles=5, verbose=0) -> np.ndarray

# ResonantGSER Layer
def propagate_feedback_to_lower(self) -> None
def project_feedback(self, representation=None) -> tf.Tensor
def harmonize_states(self, projection) -> None
def get_divergence(self) -> float
```

### Internal State Management

- Each `ResonantGSERCell` maintains:
  - `resonance_alignment`: Target state from higher layer
  - `last_h`: Last hidden state for projection
  - `global_divergence`: Current divergence metric

- Resonance cycles update `resonance_alignment` without modifying weights.

## See Also

- `examples/inference_time_resonance_example.py` - Complete working example
- `docs/NEURAL_RESONANCE.md` - Detailed explanation of resonance mechanism
- `docs/RESONANT_GSER.md` - Technical details on ResonantGSER layers

## Future Enhancements

Potential improvements:
- Automatic cycle count optimization based on convergence
- Parallelized resonance cycles across layers
- Adaptive resonance cycles (more for difficult inputs)
- Integration with generation loops for autoregressive models

---

*Implementation completed: February 2026*
*ARCANE v3.0.0 - Inference-Time Resonance Support*
