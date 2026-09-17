# ResonantGSER: Hierarchical Neural Resonance in ARCANE

## Overview

ResonantGSER (Resonant Gated Spiking Elastic Reservoir) is the core mechanism implementing **Hierarchical Neural Resonance** in the ARCANE (Augmented Reconstruction of Consciousness through Artificial Neural Evolution) framework. This biologically-inspired mechanism enables **Inference-Time Learning** and **Inference-Time State Adaptation**, allowing multi-layered neural systems to iteratively synchronize their internal semantic representations through continuous feedback loops and state alignment, addressing fundamental limitations in traditional neural architectures.

## Biological Inspiration

ResonantGSER draws inspiration from several key biological neural processes:

### Predictive Coding
The brain constantly generates predictions about sensory inputs and compares them with actual inputs, minimizing prediction errors through hierarchical feedback loops.

### Hierarchical Processing
Cortical processing occurs across multiple hierarchical levels, with higher areas providing contextual expectations that modulate lower-level processing.

### Adaptive Resonance Theory (ART)
Neural systems maintain plasticity while preserving stability through resonance-based learning, allowing adaptation to new information without catastrophic forgetting.

### Bi-directional Cortical Communication
The neocortex features extensive feedback connections that are as numerous as feedforward connections, enabling top-down contextual influence.

## Core Concept

ResonantGSER implements a **"Thinking Phase"** where neural representations are iteratively refined before final outputs are produced. Instead of single-pass feedforward processing, the system engages in multiple cycles of internal communication:

1. **Feedforward Pass**: Initial processing through hierarchical layers
2. **Resonance Cycles**: Iterative feedback and harmonization
3. **State Alignment**: Layers synchronize representations
4. **Semantic Optimization**: Direct optimization of latent space meanings

## Mathematical Formulation

Harmonization is one EMA step toward the top-down projection $P$:

$$
S \leftarrow S - \alpha \cdot (S - P), \qquad \alpha = \mathrm{clip}(\gamma + \beta,\, 0,\, 0.99)
$$

$N$ cycles have the closed form

$$
S_N = (1-\alpha)^N S_0 + \bigl(1-(1-\alpha)^N\bigr) P
$$

The cell evaluates this closed form rather than an unrolled Python loop. If no alignment has been set (zero vector and `alignment_set == 0`), the resonance step is skipped so the first forward pass is not attracted to the origin.

`project_feedback(state)` maps within hidden space (`projection_kernel`). `project_feedback(state, to_input_space=True)` uses `feedback_weights` to reconstruct the cell's input dim.

**Scope:** `resonance_alignment` is a single vector of shape `(units,)`, typically the mean of the previous batch's hidden state. This is a prototype / attractor, not a per-example top-down code. Per-example alignment lives in `PredictiveResonantLayer`.

### Resonance Cycle (callback / `run_resonance_cycle`)

At each outer cycle the orchestrator performs:

$$
\begin{aligned}
&\text{1. Feedback Projection: } P_{i \rightarrow i-1}^{(n)} = f_{proj}(S_i^{(n-1)}; W_{proj}) \\
&\text{2. Prediction Divergence: } \Delta_{i-1}^{(n)} = S_{i-1}^{(n-1)} - P_{i \rightarrow i-1}^{(n)} \\
&\text{3. State Harmonization: } S_{i-1}^{(n)} = S_{i-1}^{(n-1)} - \alpha \cdot \Delta_{i-1}^{(n)}
\end{aligned}
$$

`NeuralResonanceCallback` runs this on `on_train_batch_begin`, so it uses the **previous** batch prototype. A true inner-loop RSAA step would re-run the stack on the current batch (custom `train_step`).

Where:
- $S_i$: Semantic representation at layer $i$
- $P_{i \rightarrow i-1}$: Top-down projection from layer $i$ to $i-1$
- $\Delta$: Prediction divergence (error signal)
- $\gamma$: Resonance factor
- $\beta$: Semantic divergence weight
- $\alpha$: Clipped step size $\mathrm{clip}(\gamma+\beta, 0, 0.99)$

### Spiking Mechanism

ResonantGSER uses a subtractive reset with a straight-through estimator so the spike decision can train:

$$
\begin{aligned}
&h_{mod} = h_{res} \cdot (1.0 + \sigma(g) \cdot \gamma) + b_{res} \\
&s = \mathrm{STE}\bigl(\mathbb{I}(h_{mod} > \theta)\bigr) \\
&h_{final} = h_{mod} - s \cdot \theta
\end{aligned}
$$

## Implementation in ARCANE

### ResonantGSERCell

The fundamental building block implementing resonance dynamics:

```python
cell = ResonantGSERCell(
    units=128,                    # Hidden units
    resonance_factor=0.2,         # Resonance strength (γ)
    spike_threshold=0.5,          # Spiking threshold (θ)
    resonance_cycles=5,           # Maximum resonance iterations
    convergence_epsilon=1e-4,     # Convergence criterion (ε)
    semantic_divergence_weight=0.1 # Semantic weighting (β)
)
```

**Key Components:**
- **LSTM Base**: Standard LSTM for temporal processing
- **Resonance Gate**: Learned modulation of resonance strength
- **Projection kernel**: Hidden-space top-down map used by `project_feedback()`
- **Feedback weights**: Input-space reconstruction via `project_feedback(..., to_input_space=True)`
- **State Tracking**: `last_h` is a slow EMA of the batch mean, used as a prototype for the next callback cycle

### ResonantGSER Layer

RNN wrapper providing hierarchical integration:

```python
layer = ResonantGSER(
    units=128,
    resonance_factor=0.2,
    resonance_cycles=5,
    return_sequences=False,     # Single output or sequence
    return_state=False         # Include final states
)
```

**Hierarchical Features:**
- **Layer Linking**: `set_higher_layer()` and `set_lower_layer()`
- **Feedback Projection**: `project_feedback()` for top-down signals
- **State Harmonization**: `harmonize_states()` for bottom-up alignment

### PredictiveResonantLayer: Local Predictive Resonance

For sequence models that do not require cross-layer hierarchical wiring, ARCANE provides **PredictiveResonantLayer**, which implements *local* predictive resonance in a self-contained RNN:

- **Per-example alignment**: Alignment is part of the recurrent state `(h, c, align)`, not a single global vector. Each sequence has its own resonance target.
- **No external callbacks**: The layer does not depend on model references or custom training callbacks; it resonates toward an internal slow-moving prediction of future activity.
- **Optional stateful resonance**: Set `persist_alignment=True` so alignment state carries across separate forward passes (e.g. repeated inference on the same or different inputs), enabling inference-time state adaptation.
- **Typical use**: Sequence classification (e.g. MNIST as 28 time steps), when you want predictive-coding-style resonance without configuring a hierarchy of ResonantGSER layers.

```python
from gpbacay_arcane import PredictiveResonantLayer

layer = PredictiveResonantLayer(
    units=128,
    resonance_cycles=3,
    resonance_step_size=0.2,
    spike_threshold=0.4,
    return_sequences=True,
    persist_alignment=False   # True for stateful resonance across calls
)
```

Combine with `BioplasticDenseLayer(..., enable_inference_plasticity=True)` for inference-time Hebbian plasticity in the same model.

## Integration in Models

### NeuromimeticSemanticModel

ResonantGSER layers are integrated throughout the semantic foundation model:

```python
# Multi-layer hierarchical resonance
gser1 = ResonantGSER(units=256, resonance_factor=0.15, resonance_cycles=3)
gser2 = ResonantGSER(units=128, resonance_factor=0.2, resonance_cycles=5)
gser3 = ResonantGSER(units=64, resonance_factor=0.25, resonance_cycles=7)

# Establish hierarchical relationships
gser1.set_higher_layer(gser2)
gser2.set_lower_layer(gser1)
gser2.set_higher_layer(gser3)
gser3.set_lower_layer(gser2)

# Model architecture
x = gser1(inputs)
x = gser2(x)
outputs = gser3(x)
```

### Neural Resonance Training

Integration with the `NeuralResonanceCallback` for training:

```python
# During training, resonance cycles are orchestrated
callback = NeuralResonanceCallback(
    resonance_cycles=5,
    learning_rate=0.01,
    resonant_layers=[gser1, gser2, gser3]
)

model.fit(x_train, y_train, callbacks=[callback])
```

## Validation and Testing

Unit tests in `tests/test_resonant_gser.py` and `tests/test_mechanism_correctness.py` cover:

```
ResonantGSER Cell Basic Functionality: state management
Resonance Convergence: divergence decreases over harmonization steps
Hierarchical Resonance: set_higher_layer / set_lower_layer after build (Keras 3)
Divergence Computation: Δ = S - P
Layer Integration: compiles inside a Keras Model
Zero-alignment skip: first forward pass is not pulled toward the origin
```

Run:

```bash
python -m pytest tests/test_resonant_gser.py tests/test_mechanism_correctness.py -q
```

`convergence_epsilon` is stored on the cell and used by `HierarchicalResonanceFoundationModel.run_resonance_cycle()` for outer early-stop. The inner cell step is closed-form and does not early-break.

### Parameter Sensitivity

| Parameter | Practical Range | Effect |
|-----------|-----------------|---------|
| `resonance_factor` | 0.15 - 0.30 | Larger α moves faster toward $P$; clipped below 0.99 |
| `resonance_cycles` | 3 - 8 | Appears in the closed-form exponent $N$ |
| `spike_threshold` | 0.3 - 0.7 | Subtractive reset after STE spike |
| `convergence_epsilon` | 1e-6 - 1e-3 | Outer-loop stop in `run_resonance_cycle` |

## Visual Analysis

### Convergence Trajectories

The test suite generates detailed visualizations showing:

#### Resonance Cycle Convergence
- Divergence reduction over 15 iterations
- Exponential convergence to target alignment
- Parameter sensitivity across different configurations

#### Hierarchical State Alignment
- Multi-layer representation synchronization
- Feedback projection effectiveness
- Cross-layer coherence development

#### Parameter Optimization
- Convergence rate vs resonance factor relationships
- Error decay analysis with logarithmic scaling
- Statistical performance summaries

## Benefits for Neural Networks

### Overcoming Feedforward Limitations

1. **Iterative alignment**: Hidden states can move toward a top-down prototype before the next forward pass
2. **Hierarchical wiring**: `set_higher_layer` / `set_lower_layer` connect projection and harmonization
3. **Local predictive variant**: `PredictiveResonantLayer` keeps alignment per example in RNN state

The closed-form step is a smoother, not an inner optimizer. Do not treat Tiny Shakespeare or MNIST deltas as evidence of System-2 reasoning.

### Enhanced Capabilities

1. **Latent Space Reasoning**: Directly optimizes semantic representations
2. **Direct Semantic Optimization**: Minimizes prediction errors locally and immediately
3. **Unified Multi-Modal Space**: Creates coherent representations across modalities
4. **Surface Variability Abstraction**: Focuses on essential semantic content

### Biological Advantages

1. **Efficient Learning**: Avoids catastrophic forgetting through resonance
2. **Energy Efficiency**: Local error correction reduces global backpropagation costs
3. **Robust Adaptation**: Maintains stability during concept shifts
4. **Hierarchical Intelligence**: Supports complex reasoning through layered resonance

## Usage Guidelines

### Architecture Design

1. **Layer Hierarchy**: Design 3-5 layer hierarchies for optimal resonance
2. **Parameter Tuning**: Start with `resonance_factor=0.2`, adjust based on convergence
3. **Cycle Count**: Balance 3-5 cycles for most applications
4. **Integration Points**: Use in intermediate layers for semantic processing

### Training Considerations

1. **Callback Integration**: Always use `NeuralResonanceCallback` during training
2. **Learning Rate**: Coordinate with overall model learning rate
3. **Batch Size**: Smaller batches (4-16) work better for resonance
4. **Sequence Length**: Moderate lengths (16-64) for optimal temporal processing

### Performance Optimization

1. **Closed-form inner step**: The cell no longer unrolls $N$ Python iterations.
2. **Outer early stop**: `run_resonance_cycle` can halt when summed divergence $<$ `convergence_epsilon`.
3. **Do not pull to zero**: Resonance is skipped until `harmonize_states` (or a non-zero alignment) has run.

## What this mechanism is not

ResonantGSER does **not** re-run the network on the current batch inside `NeuralResonanceCallback`. Alignment is a `(units,)` prototype. Claims of System-2 reasoning, multi-modal unification, or large gains in continual learning are hypotheses, not results of the unit tests.

## Future work

1. Custom `train_step` that re-forwards the current batch after each projection.
2. Per-example top-down targets (already closer in `PredictiveResonantLayer`).
3. Graph-safe `GSER.prune_neurons` (still a Python swap-remove).

## References

### Biological Foundations
1. **Predictive Coding**: Rao, R. P., & Ballard, D. H. (1999). Nature Neuroscience
2. **Hierarchical Processing**: Felleman, D. J., & Van Essen, D. C. (1991). Cerebral Cortex
3. **Adaptive Resonance**: Grossberg, S. (2013). Frontiers in Psychology

### Computational Implementations
1. **Hierarchical Resonance**: RSAA notes in `docs/NEURAL_RESONANCE.md`
2. **Linear kernel attention**: Katharopoulos et al., Transformers are RNNs (2020)

---

*This documentation matches the ResonantGSER implementation in ARCANE after the 2026 mechanism corrections.*
