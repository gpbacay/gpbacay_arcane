# Homeostatic Plasticity in ARCANE

## Overview

ARCANE implements homeostatic-plasticity-inspired mechanisms for measuring activity and applying bounded corrections toward configured targets. These controls can reduce a measured activity mismatch, but they do not guarantee training stability, prevent dead units, or improve task performance.

## Biological Inspiration

Homeostatic plasticity is inspired by biological neural systems where neurons maintain stable activity levels despite changing synaptic inputs. This mechanism is crucial for:

- **Activity Stabilization**: Preventing neurons from becoming too active (runaway excitation) or too quiet (silence)
- **Metaplasticity**: Allowing synapses to adjust their plasticity rules based on overall network activity
- **Network Stability**: Maintaining balanced information flow throughout the neural system
- **Learning Adaptation**: Enabling networks to adapt to new learning contexts while preserving stability

## Core Concept

The implementation tracks activity with exponential traces, optional exact windows, or both. Depending on the layer, it adjusts a plastic weight component, a bounded gain, or the representation itself. Learned task weights are not silently overwritten.

## Mathematical Formulation

### BCM plasticity (BioplasticDenseLayer)

When `enable_inference_plasticity=True` and `training=False`, the **plastic** kernel (not the gradient kernel) is updated with a BCM rule. By default, the sliding threshold $\theta$ is the exponential running trace of postsynaptic activity:

\[
\theta \leftarrow \bigl(1 - 1/\tau\bigr)\,\theta + (1/\tau)\,\overline{y^2}
\]
\[
\Delta W_{\text{plastic}} \propto x^{\top}\bigl[y \odot (y - \theta)\bigr]
\]

The anti-Hebbian term decays the fast component instead of biasing all correlations negative. Activity homeostasis then scales that plastic kernel toward `target_avg`, and the kernel is clipped by global norm (5.0). The output is recomputed with the new plastic weights, so adaptation affects the current inference. Gradient descent still updates `kernel`; the two do not overwrite each other. Call `reset_plasticity()` to begin a new episode.

The threshold can instead use an exact finite window with `homeostatic_window_size`, and can blend that fast estimate with a second exponential trace configured by `slow_bcm_tau` and `slow_trace_weight`. Attention weights, when supplied, weight both the activity statistics and BCM correlation.

Plasticity is **off** during `fit()` unless you call the layer with `training=False` and `enable_inference_plasticity=True`.

### Homeostatic gain (HebbianHomeostaticNeuroplasticity)

During `training=True`, a Hebbian update is applied to a non-trainable `plastic_kernel`, and a scalar `gain` moves toward `target_activity`:

\[
g \leftarrow \mathrm{clip}\bigl(g + \eta \cdot (a_{\text{target}} - \bar{|y|}),\; 0.1,\; 10\bigr)
\]

At inference (`training=False`) the layer is a dense map `(kernel + plastic_kernel)` times `gain`. Existing tests that scale `kernel` by hand still pass because inference does not touch `kernel`.

## Implementation in ARCANE Layers

### BioplasticDenseLayer

The `BioplasticDenseLayer` incorporates homeostatic plasticity alongside Hebbian learning:

```python
layer = BioplasticDenseLayer(
    units=64,
    learning_rate=1e-3,
    target_avg=0.12,
    homeostatic_rate=5e-5,
    bcm_tau=800.0,
    homeostatic_window_size=32,
    slow_bcm_tau=4000.0,
    slow_trace_weight=0.5,
    homeostasis_mode='hybrid',
    local_homeostasis_weight=0.5,
    adaptive_target_rate=0.05,
    resonance_coupling=0.5,
    enable_inference_plasticity=True,  # required for BCM / homeostasis at inference
    activation='gelu'
)
```

**Parameters:**
- `target_avg`: Target average activity level (default: 0.12)
- `homeostatic_rate`: Rate of activity scaling on `plastic_kernel`
- `bcm_tau`: Time constant of the sliding BCM threshold
- `enable_inference_plasticity`: If False (default), the layer is a dense map during both train and infer
- `learning_rate`: Step size of the BCM update
- `homeostatic_window_size`: Exact recent-activity window; zero keeps the EMA-only behavior
- `slow_bcm_tau`: Optional second, slower activity trace
- `homeostasis_mode`: `global`, `local`, or `hybrid` activity-error regulation
- `local_homeostasis_weight`: Local contribution in hybrid mode
- `adaptive_target_rate`: Update rate toward an externally supplied target signal
- `target_bounds`: Lower and upper limits for adaptive targets
- `resonance_coupling`: Strength by which hierarchy mismatch increases regulation
- `hierarchy_level`, `num_hierarchy_levels`, `hierarchy_target_slope`: Optional level-specific target profile

### Context-guided update

The enhanced controls are explicit call-time inputs:

```python
output = layer(
    features,
    training=False,
    attention_weights=token_or_sample_importance,
    resonance_signal=hierarchy_divergence,
    target_signal=task_conditioned_activity_target,
)
```

- `attention_weights` determines which samples or tokens contribute to activity and BCM statistics.
- `resonance_signal` scales the homeostatic correction rate. Larger hierarchy mismatch produces stronger bounded regulation when `resonance_coupling > 0`.
- `target_signal` updates the per-neuron adaptive target. Without this signal, the configured target remains stable.

### HierarchicalHomeostaticController

`HierarchicalHomeostaticController` accepts two or more representation levels. It measures per-example activity at every level, creates a configurable level target profile, applies bounded gains to each representation, and returns both hierarchy imbalance and adaptive-target signals for downstream plastic layers.

```python
regulated_low, regulated_high, imbalance, target = (
    HierarchicalHomeostaticController(
        target_activity=0.12,
        regulation_rate=0.1,
        level_target_slope=0.1,
        adaptive_target_rate=0.1,
    )([low_level, high_level])
)
```

### HebbianHomeostaticNeuroplasticity

The `HebbianHomeostaticNeuroplasticity` layer keeps a trainable dense kernel plus a plastic kernel and homeostatic gain. Hebbian / gain updates run only when `training=True`.

```python
layer = HebbianHomeostaticNeuroplasticity(
    units=64,
    learning_rate=1e-3,
    target_activity=0.1    # Target activity level
)
```

**Parameters:**
- `target_activity`: Target neural activity level (default: 0.1)
- `learning_rate`: Rate of homeostatic adjustment

## Integration in Models

### NeuromimeticSemanticModel

Homeostatic plasticity is integrated into the main ARCANE semantic model:

```python
model = NeuromimeticSemanticModel(
    vocab_size=10000,
    seq_len=16,
    embed_dim=32,
    hidden_dim=64,
    enable_inference_plasticity=True,
    use_hierarchical_homeostasis=True,
    homeostasis_config={
        "homeostasis_mode": "hybrid",
        "homeostatic_window_size": 32,
        "slow_bcm_tau": 4000.0,
        "resonance_coupling": 0.5,
    },
)

# The model includes BioplasticDenseLayer with homeostatic regulation
bioplastic = BioplasticDenseLayer(
    units=model.hidden_dim * 2,
    learning_rate=1.5e-3,
    target_avg=0.11,
    homeostatic_rate=8e-5,
    activation='gelu'
)
```

## Validation and Testing

### Test Results

`tests/test_homeostatic_plasticity.py` still demonstrates **manual** kernel scaling toward a target (the tests assign `layer.kernel` themselves). That is a behavioral illustration, not a test of BCM.

`tests/test_mechanism_correctness.py` checks the implemented rules:

- `test_bioplastic_bcm_updates_plastic_kernel`: with `enable_inference_plasticity=True`, `plastic_kernel` changes at inference
- `test_bioplastic_finite_window_and_attention_weighted_activity`: the exact window records attention-weighted activity
- `test_bioplastic_local_and_global_regulation_are_distinct`: local and global controllers produce different update structures
- `test_bioplastic_adaptive_target_and_resonance_coupling`: targets adapt and hierarchy mismatch changes regulation strength
- `test_hierarchical_homeostatic_controller_regulates_each_level`: low- and high-activity levels receive bounded corrections
- `test_hebbian_layer_inference_matches_dense`: at inference, output equals `(kernel + 0) @ x + bias`

`tests/test_inference_time_resonance.py` verifies that the hierarchical model wires controller, pathway-gating, target, and resonance signals into its plastic layers.

```bash
python -m pytest tests/test_homeostatic_plasticity.py tests/test_mechanism_correctness.py -q
```

### Test Implementation

The mechanism is validated through targeted unit tests:

```python
def test_bioplastic_dense_layer_homeostasis():
    """Test homeostatic regulation in BioplasticDenseLayer"""
    layer = BioplasticDenseLayer(
        units=10,
        target_avg=0.5,
        homeostatic_rate=0.1
    )

    # Initialize with high activity weights
    layer.kernel.assign(tf.ones_like(layer.kernel) * 10.0)

    # Apply homeostatic adjustments over iterations
    for _ in range(100):
        output = layer(inputs)
        current_activity = tf.reduce_mean(output).numpy()

        if current_activity != 0:
            desired_scale = target_avg / current_activity
            adjustment_factor = 1 + homeostatic_rate * (desired_scale - 1)
            layer.kernel.assign(layer.kernel * adjustment_factor)

    # Verify convergence to target
    assert np.isclose(final_activity, target_avg, atol=0.2)
```

## Visual Analysis

The test suite generates comprehensive visualizations showing:

### Individual Layer Convergence
- Activity curves over 100 iterations
- Target activity reference lines
- Smooth convergence trajectories

### Comparative Analysis
- Side-by-side comparison of different implementations
- Error convergence plots (log scale)
- Performance metrics comparison

## Measurements and Hypotheses

### Stability measurements
- Activity error relative to the configured target
- Oscillation and saturation across adaptation steps
- Gradient and activation norms during training

### Adaptation measurements
- Objective change before and after plastic updates
- Retention after distribution or task changes
- Order and batch-size sensitivity

### Scope of the analogy
- Sliding thresholds and bounded activity targets are biologically inspired
- The implementation is an engineering abstraction, not a biological simulation
- Cognitive behavior and biological fidelity are not inferred from mechanism names

## Usage Guidelines

### Parameter Selection

1. **Target Activity**: Choose based on activation function range
   - Sigmoid/Tanh: 0.1-0.5
   - ReLU: 0.1-1.0
   - GELU: 0.05-0.3

2. **Homeostatic Rate**: Balance convergence speed vs. stability
   - Too high: Oscillatory behavior
   - Too low: Slow convergence
   - Recommended: 1e-5 to 1e-1

3. **Learning Rate**: Coordinate with overall model learning rate
   - Should be similar magnitude to other layer learning rates

### Integration Best Practices

1. **Layer Placement**: Use in intermediate layers for feature stability
2. **Regularization**: Combine with other regularization techniques
3. **Monitoring**: Track activity levels during training
4. **Tuning**: Adjust parameters based on validation performance

## Advanced Applications

### Continual Learning
Homeostatic plasticity can be evaluated as one component of continual-learning systems. ARCANE does not establish that it prevents catastrophic forgetting; report retention and forward/backward transfer explicitly.

### Meta-Learning
Adaptive targets and fast weights provide mechanisms for meta-learning experiments, but rapid adaptation and knowledge retention are task-dependent outcomes.

### Robustness
Potential robustness evaluations include:
- Input distribution shifts
- Adversarial perturbations
- Long training sequences

## Implemented Advanced Mechanisms

- **Finite-window and multi-timescale homeostasis**: Exact activity windows can be combined with fast and slow exponential traces.
- **Local and global regulation**: Regulation can operate per neuron, layer-wide, or as a configurable hybrid.
- **Adaptive targets**: Per-neuron targets track an explicit task or hierarchy target signal within configured bounds.
- **Resonance-coupled homeostasis**: Hierarchy mismatch controls the strength of plastic regulation.
- **Hierarchical regulation**: A differentiable controller regulates multiple representation levels and supplies imbalance and target signals to plastic layers.
- **Attention-guided homeostasis**: Token or sample importance weights the activity and correlation statistics.

All enhanced mechanisms are opt-in. Existing models retain their original behavior unless inference plasticity or hierarchical homeostasis is enabled. This preserves checkpoint compatibility and prevents stateful, order-dependent updates from being introduced silently; enable each mechanism for a controlled task-level comparison.

## Remaining Research Directions

- Learn target policies from an outer validation or reinforcement objective instead of supplying target signals directly.
- Evaluate whether controller gains and adaptive targets improve calibration, robustness, or continual learning on held-out tasks.
- Compare finite-window, dual-timescale, and EMA-only regulation under controlled distribution shifts.
- Extend hierarchical regulation to generation-time token states and persistent streaming identities.

## References

1. **Biological Homeostasis**: Turrigiano, G. G. (2008). The self-tuning neuron: synaptic scaling of excitatory synapses. Cell.
2. **Neural Stability**: Zenke, F., et al. (2015). Diverse synaptic plasticity mechanisms orchestrated to form and retrieve memories in spiking neural networks. Frontiers in Computational Neuroscience.
3. **Homeostatic Learning**: Zenke, F., et al. (2017). Continual learning through synaptic intelligence. International Conference on Machine Learning.

---

*This documentation describes the current implementation. Performance benefits require task-level ablation rather than mechanism-level unit tests alone.*
