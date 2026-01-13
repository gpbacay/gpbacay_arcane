# Neuromimetic Activation Functions in ARCANE

## Overview

In the ARCANE framework, activation functions are not merely static mathematical non-linearities. They are designed to mirror the dynamic and adaptive nature of biological neurons. While standard AI models use functions like ReLU or GELU to provide differentiability, ARCANE introduces **Neuromimetic Activations** that incorporate **Inference-Time State Adaptation**, **Spiking Dynamics**, and **Homeostatic Regulation**.

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

Where $\theta$ is the firing threshold. This ensures that the unit only communicates information when its internal state aligns with or significantly exceeds hierarchical expectations.

### 2. Homeostatic Activity Scaling
The `homeostatic_gelu` implements **Synaptic Scaling**, a mechanism observed in the neocortex to maintain firing rates within an optimal information-theoretic range. The gain $G$ of the activation is regulated by:

$$G(t) = 1 + \eta \cdot (A_{target} - \bar{A})$$

Where:
- $\eta$ is the `adaptation_rate`.
- $A_{target}$ is the goal activity level.
- $\bar{A}$ is the moving average of historical activity.

The final activated output is:

$$y = \text{GELU}(x \cdot G(t))$$

---

## The Argument for Neuromimetic Activations

### Why Standard Activations are Insufficient
Standard activations like ReLU or GELU are **memoryless mappings**. They treat every input in isolation and have no capacity for "deliberation." In deep hierarchies, this leads to a "System 1" reactive loop where the model is forced to commit to a feature representation before it has resolved potential ambiguities with higher-level context.

### The Deliberative Advantage
Neuromimetic activations transform the unit from a passive gate into an **active agent**. By integrating **Inference-Time State Adaptation**, these functions provide three critical advantages:

1.  **Temporal Coherence**: By maintaining a membrane potential ($V$), the neuron can integrate evidence over multiple time steps, naturally filtering out high-frequency noise that standard "static" models often succumb to.
2.  **Deliberative Thresholding**: The spiking mechanism ensures that "weak" or "ambiguous" signals are suppressed until hierarchical resonance provides enough gain to push the potential past the threshold. This creates a natural "Thinking Phase" at the per-neuron level.
3.  **Structural Stability through Homeostasis**: Traditional deep networks suffer from "dead neurons" (dying ReLU) or "exploding activations." ARCANE’s homeostatic gain ensures that every neuron stays in its responsive "sweet spot," maximizing the entropy and information capacity of the semantic space.

---

## Available Activations

### 1. Resonant Spiking Activation (RSA)
Mimics a biological neuron's membrane potential with top-down modulation.
**Usage:**
```python
spikes, new_state = resonant_spike(inputs, current_state, threshold=0.5, leak_rate=0.1, resonance_factor=0.2)
```

### 2. Homeostatic GELU (h-GELU)
Self-regulates sensitivity to prevent runaway excitation.
**Usage:**
```python
activated = homeostatic_gelu(inputs, moving_average_activity, target_activity=0.12)
```

---

## Validation and Performance

The neuromimetic activations in ARCANE have been rigorously validated through unit tests and behavioral analysis.

### 1. Resonance Convergence
In testing, the `resonant_spike` function demonstrated perfect convergence with hierarchical signals:
- **Baseline Potential**: 0.8 (Threshold: 1.0) -> No spike.
- **Resonant Potential**: 1.2 (via $\rho = 0.5$) -> Successful spike.
- **State Reset**: Potential consistently resets by exactly $\theta$ after firing, maintaining biological energy constraints.

### 2. Homeostatic Stability
The `homeostatic_gelu` function successfully demonstrates self-tuning gain:
- **Under-activity Scenario**: Gain automatically increases by $\eta \cdot \Delta A$, boosting signal sensitivity.
- **Over-activity Scenario**: Gain decreases, preventing semantic saturation and maintaining high entropy in the latent space.

### 3. Temporal Leaky Integration
Tests confirm that internal membrane potential $V$ decays exponentially at the `leak_rate`, ensuring that old, irrelevant stimuli do not cause ghost spikes in future time steps.

## Running Validations

To reproduce the activation function tests, run:

```bash
python -m pytest tests/test_activations.py
```

---

## References

1.  **Turrigiano, G. G. (2008).** *The self-tuning neuron: synaptic scaling of excitatory synapses.* Cell, 135(3), 422-435.
2.  **Rao, R. P., & Ballard, D. H. (1999).** *Nature Neuroscience, 2(1), 79-87.*
3.  **Grossberg, S. (2013).** *Adaptive Resonance Theory: How a brain learns to consciously attend, learn, and recognize a changing world.* Frontiers in Psychology, 4, 301.
4.  **Gerstner, W., & Kistler, W. M. (2002).** *Spiking Neuron Models: Single Neurons, Populations, Plasticity.* Cambridge University Press.

---

*This module is part of the ARCANE v3.0.0 semantic engineering library.*
