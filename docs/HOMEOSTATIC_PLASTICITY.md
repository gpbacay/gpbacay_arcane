# Homeostatic Plasticity in ARCANE

## Overview

Homeostatic Plasticity is a critical neuromimetic mechanism in the ARCANE (Augmented Reconstruction of Consciousness through Artificial Neural Evolution) framework that enables stable and adaptive neural activity regulation. This mechanism ensures that neural networks maintain appropriate activity levels over time, preventing issues like runaway activation or dead neurons.

## Biological Inspiration

Homeostatic plasticity is inspired by biological neural systems where neurons maintain stable activity levels despite changing synaptic inputs. This mechanism is crucial for:

- **Activity Stabilization**: Preventing neurons from becoming too active (runaway excitation) or too quiet (silence)
- **Metaplasticity**: Allowing synapses to adjust their plasticity rules based on overall network activity
- **Network Stability**: Maintaining balanced information flow throughout the neural system
- **Learning Adaptation**: Enabling networks to adapt to new learning contexts while preserving stability

## Core Concept

Homeostatic plasticity works by continuously monitoring neural activity levels and adjusting synaptic weights to maintain target activity levels. When activity is too high, weights are reduced; when activity is too low, weights are increased.

## Mathematical Formulation

### BCM plasticity (BioplasticDenseLayer)

When `enable_inference_plasticity=True` and `training=False`, the **plastic** kernel (not the gradient kernel) is updated with a BCM rule. The sliding threshold $\theta$ is the running trace of postsynaptic activity:

\[
\theta \leftarrow \bigl(1 - 1/\tau\bigr)\,\theta + (1/\tau)\,\bar{y}
\]
\[
\Delta W_{\text{plastic}} \propto x^{\top}\bigl[y \odot (y - \theta)\bigr]
\]

Activity homeostasis then scales that plastic kernel toward `target_avg`. The plastic kernel is clipped by global norm (5.0). Gradient descent still updates `kernel`; the two do not overwrite each other.

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
    hidden_dim=64
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
- `test_hebbian_layer_inference_matches_dense`: at inference, output equals `(kernel + 0) @ x + bias`

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

## Benefits for Neural Networks

### Stability
- Prevents gradient explosion/collapse
- Maintains network responsiveness
- Enables long-term learning without degradation

### Adaptability
- Allows networks to adapt to new tasks
- Maintains plasticity while ensuring stability
- Supports continual learning scenarios

### Biological Fidelity
- Mirrors real neural homeostatic mechanisms
- Supports complex cognitive behaviors
- Enables more brain-like information processing

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
Homeostatic plasticity enables networks to learn new tasks without catastrophic forgetting by maintaining stable representations.

### Meta-Learning
The mechanism supports rapid adaptation to new learning contexts while preserving previously learned knowledge.

### Robustness
Networks with homeostatic plasticity are more robust to:
- Input distribution shifts
- Adversarial perturbations
- Long training sequences

## Future Directions

### Enhanced Mechanisms
- **Sliding Window Homeostasis**: Activity regulation over time windows
- **Local vs Global Regulation**: Neuron-specific vs. layer-wide control
- **Adaptive Targets**: Dynamic target adjustment based on task requirements

### Integration with Other Mechanisms
- **Neural Resonance**: Coordinating homeostatic stability with dynamic adaptation
- **Hierarchical Regulation**: Multi-scale homeostatic control
- **Attention-Guided Homeostasis**: Task-specific activity regulation

## References

1. **Biological Homeostasis**: Turrigiano, G. G. (2008). The self-tuning neuron: synaptic scaling of excitatory synapses. Cell.
2. **Neural Stability**: Zenke, F., et al. (2015). Diverse synaptic plasticity mechanisms orchestrated to form and retrieve memories in spiking neural networks. Frontiers in Computational Neuroscience.
3. **Homeostatic Learning**: Zenke, F., et al. (2017). Continual learning through synaptic intelligence. International Conference on Machine Learning.

---

*This documentation covers the Homeostatic Plasticity mechanism as implemented in ARCANE v3.0.0. For the latest updates and additional features, refer to the main project repository.*
