# Neuromimetic Activation Functions in ARCANE

## Overview

ARCANE provides stateful activation functions inspired by leaky integration, spiking dynamics, and homeostatic regulation. The names describe computational inspiration rather than biological equivalence or verified cognitive behavior.

---

## Mathematical Foundation

Neuromimetic activations shift the paradigm from static mappings ($y = f(x)$) to dynamic state transitions.

### 1. Resonant Spiking Dynamics
The `resonant_spike` function is grounded in the **Leaky Integrate-and-Fire (LIF)** model, augmented by the **Resonant State Alignment Algorithm (RSAA)**. 

The internal potential $V$ at time step $t$ is defined as:

$$V_{integrated}(t) = I(t) + V(t-1) \cdot (1 - \lambda)$$

Where:
- $I(t)$ is the incoming stimulus.
- $\lambda$ is the `leak_rate`, representing the passive decay of the membrane potential.

The potential is then modulated by the **Resonance Factor** $\rho$, which represents top-down hierarchical expectations:

$$V_{res}(t) = V_{integrated}(t) \cdot (1 + \rho)$$

The output spike $S(t)$ and the subsequent state reset follow:

$$S(t) = \begin{cases} 1 & \text{if } V_{res}(t) > \theta \\ 0 & \text{otherwise} \end{cases}$$
$$V(t) = V_{res}(t) - S(t) \cdot \theta$$

Where $\theta$ is the firing threshold. The **forward** pass is this hard threshold. The **backward** pass uses a straight-through estimator (STE): a sigmoid with slope `sharpness` so spike outputs can train. Without STE, $\partial S / \partial V$ is zero almost everywhere.

### 2. Homeostatic Activity Scaling
The `homeostatic_gelu` implements a synaptic-scaling-inspired gain that moves activity toward a configured target. The gain $G$ is regulated by:

$$G(t) = \mathrm{clip}\bigl(1 + \eta \cdot (A_{target} - \bar{A}),\; 0.1,\; 10\bigr)$$

Where:
- $\eta$ is the `adaptation_rate`.
- $A_{target}$ is the goal activity level.
- $\bar{A}$ is the moving average of historical activity.

The final activated output is:

$$y = \text{GELU}(x \cdot G(t))$$

---

## Scope and Trade-offs

### Stateful versus memoryless mappings
Standard activations such as ReLU and GELU are memoryless. ARCANE's stateful wrappers can integrate activity across calls or accept an external resonance factor. This adds state-management cost and does not imply deliberation or superior task performance.

### Implemented behavior
The activation functions provide three directly testable behaviors:

1.  **Leaky integration**: a membrane variable carries a decayed contribution from its previous value.
2.  **Thresholded output with gradients**: the forward pass is discrete while the backward pass uses a straight-through estimator.
3.  **Bounded homeostatic gain**: activity history adjusts gain within configured limits. It does not guarantee maximum entropy or eliminate dead or exploding units.

---

## Available Activations

### 1. Resonant Spiking Activation (RSA)
Implements a leaky integrate-and-fire-style state with top-down modulation.
**Usage:**
```python
spikes, new_state = resonant_spike(inputs, current_state, threshold=0.5, leak_rate=0.1, resonance_factor=0.2)
```

### 2. Graded Spike
The spiking step used by every ARCANE resonance layer. `threshold` is relative: each row fires against `V_th = threshold * mean|potential|`, so the firing rate stays steady however the potential is scaled. The neuron emits a signed spike count `round(potential / V_th)` (negative = inhibitory, large values burst) and transmits `count * V_th`. The gradient is the identity w.r.t. the potential, and a learnable threshold still trains.

An optional per-channel `leak` (0..1) passes that fraction of the sub-quantum remainder through as a graded potential. `FieldResonance`, `ResonantSequenceMixer`, `ResonantGSERCell` and `PredictiveResonantCell` learn both a threshold and a leak per channel (`GSER` learns a leak alongside its existing per-unit threshold, and keeps its membrane potential unrounded between steps) (initialised at `spike_threshold` and about 0.05), so each channel settles on its own resolution during training.
**Usage:**
```python
out = graded_spike(potential, threshold=0.4)
out = graded_spike(potential, threshold=per_channel_threshold, leak=per_channel_leak)
```

### 3. Homeostatic GELU (h-GELU)
Self-regulates sensitivity to prevent runaway excitation. Gain is clipped to `[0.1, 10]`.
**Usage:**
```python
activated = homeostatic_gelu(inputs, moving_average_activity, target_activity=0.12)
```

### 4. Adaptive Softplus
Smooth firing-rate map with a tunable threshold and sharpness.

### 5. `NeuromimeticActivation` layer
Keras wrapper around the functions above. Supports `resonant_spike`, `homeostatic_gelu`, and `adaptive_softplus`. Membrane / activity traces are feature-wise running averages (not per-sample RNN state). Keras kwargs such as `name=` are **not** forwarded into `resonant_spike`.

```python
from gpbacay_arcane import NeuromimeticActivation

layer = NeuromimeticActivation(activation_type="resonant_spike", threshold=0.5, name="rsa")
spikes = layer(inputs, resonance_factor=0.2)
```

---

## Validation and Performance

The activation implementations are covered by unit tests for their stated numerical behavior.

### 1. Resonance Convergence
The `resonant_spike` tests verify threshold crossing and reset behavior:
- **Baseline Potential**: 0.8 (Threshold: 1.0) -> No spike.
- **Resonant Potential**: 1.2 (via $\rho = 0.5$) -> Successful spike.
- **State Reset**: Potential resets by exactly $\theta$ after firing.

### 2. Homeostatic Stability
The `homeostatic_gelu` function successfully demonstrates self-tuning gain:
- **Under-activity Scenario**: Gain automatically increases by $\eta \cdot \Delta A$, boosting signal sensitivity.
- **Over-activity Scenario**: Gain decreases, preventing semantic saturation and maintaining high entropy in the latent space.

### 3. Temporal Leaky Integration
Tests confirm that internal membrane potential $V$ decays exponentially at the `leak_rate`, ensuring that old, irrelevant stimuli do not cause ghost spikes in future time steps.

## Running Validations

To reproduce the activation function tests, run:

```bash
python -m pytest tests/test_activations.py tests/test_mechanism_correctness.py -q
```

`tests/test_mechanism_correctness.py` also checks that STE spikes have a non-zero gradient and that `NeuromimeticActivation` accepts Keras layer kwargs.

---

## References

1.  **Turrigiano, G. G. (2008).** *The self-tuning neuron: synaptic scaling of excitatory synapses.* Cell, 135(3), 422-435.
2.  **Rao, R. P., & Ballard, D. H. (1999).** *Nature Neuroscience, 2(1), 79-87.*
3.  **Grossberg, S. (2013).** *Adaptive Resonance Theory: How a brain learns to consciously attend, learn, and recognize a changing world.* Frontiers in Psychology, 4, 301.
4.  **Gerstner, W., & Kistler, W. M. (2002).** *Spiking Neuron Models: Single Neurons, Populations, Plasticity.* Cambridge University Press.

---

*This page documents the activation behavior shipped with ARCANE v4.1.0.*
