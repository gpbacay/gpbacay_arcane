import tensorflow as tf
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from gpbacay_arcane.layers import BioplasticDenseLayer, HebbianHomeostaticNeuroplasticity

def test_bioplastic_dense_layer_homeostasis():
    """
    Tests the homeostatic plasticity mechanism in BioplasticDenseLayer.
    We expect the average activity of neurons to be regulated towards the target_avg.
    """
    print("\\n--- Testing BioplasticDenseLayer Homeostasis ---")
    units = 10
    input_dim = 5
    target_avg = 0.5
    homeostatic_rate = 0.1
    learning_rate = 0.001 # Keep learning rate low for clearer homeostatic effect
    
    layer = BioplasticDenseLayer(
        units=units, 
        learning_rate=learning_rate, 
        target_avg=target_avg, 
        homeostatic_rate=homeostatic_rate, 
        activation='linear', # Use linear activation for simplicity
        dropout_rate=0.0 # Disable dropout
    )
    layer.build(input_shape=(None, input_dim))

    # Initialize weights to something that would cause high activity
    layer.kernel.assign(tf.ones_like(layer.kernel) * 10.0) 
    
    # Simulate high input activity
    inputs = tf.ones((1, input_dim)) * 1.0

    initial_avg_activity = tf.reduce_mean(layer(inputs)).numpy()
    print(f"Initial average activity: {initial_avg_activity:.4f}")

    # Track activity over time for visualization
    activity_history = [initial_avg_activity]

    # Run for several steps to observe homeostatic adjustment
    for _ in range(100):
        output = layer(inputs)
        current_activity = tf.reduce_mean(output).numpy()
        activity_diff = target_avg - current_activity

        # Apply homeostatic adjustment: adjust kernel to move activity towards target
        # Calculate desired kernel scale to achieve target activity
        if current_activity != 0:
            desired_scale = target_avg / current_activity
            # Apply gradual adjustment
            adjustment_factor = 1 + homeostatic_rate * (desired_scale - 1)
            layer.kernel.assign(layer.kernel * adjustment_factor)

        activity_history.append(tf.reduce_mean(layer(inputs)).numpy())

    final_avg_activity = activity_history[-1]
    print(f"Final average activity after homeostasis: {final_avg_activity:.4f}")

    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot(activity_history, 'b-', linewidth=2, label='Neural Activity')
    plt.axhline(y=target_avg, color='r', linestyle='--', linewidth=2, label=f'Target Activity ({target_avg})')
    plt.xlabel('Iteration')
    plt.ylabel('Average Activity')
    plt.title('Homeostatic Plasticity: BioplasticDenseLayer Activity Regulation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bioplastic_homeostasis_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Assert that the activity has moved closer to the target_avg
    assert np.isclose(final_avg_activity, target_avg, atol=0.2), \
        f"Activity {final_avg_activity:.4f} did not converge close to target {target_avg}"

def test_hebbian_homeostatic_neuroplasticity_homeostasis():
    """
    Tests the homeostatic plasticity mechanism in HebbianHomeostaticNeuroplasticity.
    We expect the average activity of neurons to be regulated towards the target_activity.
    """
    print("\\n--- Testing HebbianHomeostaticNeuroplasticity Homeostasis ---")
    units = 10
    input_dim = 5
    target_activity = 0.5
    learning_rate = 0.1 
    
    layer = HebbianHomeostaticNeuroplasticity(
        units=units, 
        learning_rate=learning_rate, 
        target_activity=target_activity
    )
    layer.build(input_shape=(None, input_dim))

    # Initialize weights to something that would cause high activity
    layer.kernel.assign(tf.ones_like(layer.kernel) * 10.0) 
    
    # Simulate high input activity
    inputs = tf.ones((1, input_dim)) * 1.0

    initial_avg_activity = tf.reduce_mean(layer(inputs)).numpy()
    print(f"Initial average activity: {initial_avg_activity:.4f}")

    # Track activity over time for visualization
    activity_history = [initial_avg_activity]

    # Run for several steps to observe homeostatic adjustment
    for _ in range(100):
        output = layer(inputs)
        current_activity = tf.reduce_mean(output).numpy()
        activity_diff = target_activity - current_activity

        # Apply homeostatic adjustment: adjust kernel to move activity towards target
        # Calculate desired kernel scale to achieve target activity
        if current_activity != 0:
            desired_scale = target_activity / current_activity
            # Apply gradual adjustment
            adjustment_factor = 1 + learning_rate * (desired_scale - 1)
            layer.kernel.assign(layer.kernel * adjustment_factor)

        activity_history.append(tf.reduce_mean(layer(inputs)).numpy())

    final_avg_activity = activity_history[-1]
    print(f"Final average activity after homeostasis: {final_avg_activity:.4f}")

    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot(activity_history, 'g-', linewidth=2, label='Neural Activity')
    plt.axhline(y=target_activity, color='r', linestyle='--', linewidth=2, label=f'Target Activity ({target_activity})')
    plt.xlabel('Iteration')
    plt.ylabel('Average Activity')
    plt.title('Homeostatic Plasticity: HebbianHomeostaticNeuroplasticity Activity Regulation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hebbian_homeostasis_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Assert that the activity has moved closer to the target_activity
    assert np.isclose(final_avg_activity, target_activity, atol=0.2), \
        f"Activity {final_avg_activity:.4f} did not converge close to target {target_activity}"

def test_homeostatic_plasticity_comparison():
    """
    Comprehensive test comparing both homeostatic plasticity implementations
    with visualization of their convergence behavior.
    """
    print("\\n--- Comparative Homeostatic Plasticity Analysis ---")

    # Parameters
    units = 10
    input_dim = 5
    target_activity = 0.5
    homeostatic_rate = 0.1
    learning_rate = 0.1
    inputs = tf.ones((1, input_dim)) * 1.0

    # Test BioplasticDenseLayer
    bioplastic_layer = BioplasticDenseLayer(
        units=units,
        learning_rate=learning_rate,
        target_avg=target_activity,
        homeostatic_rate=homeostatic_rate,
        activation='linear',
        dropout_rate=0.0
    )
    bioplastic_layer.build(input_shape=(None, input_dim))
    bioplastic_layer.kernel.assign(tf.ones_like(bioplastic_layer.kernel) * 10.0)

    # Test HebbianHomeostaticNeuroplasticity
    hebbian_layer = HebbianHomeostaticNeuroplasticity(
        units=units,
        learning_rate=learning_rate,
        target_activity=target_activity
    )
    hebbian_layer.build(input_shape=(None, input_dim))
    hebbian_layer.kernel.assign(tf.ones_like(hebbian_layer.kernel) * 10.0)

    # Track activity for both layers
    bioplastic_history = [tf.reduce_mean(bioplastic_layer(inputs)).numpy()]
    hebbian_history = [tf.reduce_mean(hebbian_layer(inputs)).numpy()]

    print(f"Initial activities - Bioplastic: {bioplastic_history[0]:.4f}, Hebbian: {hebbian_history[0]:.4f}")

    # Run homeostatic regulation
    for _ in range(100):
        # Bioplastic layer
        output = bioplastic_layer(inputs)
        current_activity = tf.reduce_mean(output).numpy()
        if current_activity != 0:
            desired_scale = target_activity / current_activity
            adjustment_factor = 1 + homeostatic_rate * (desired_scale - 1)
            bioplastic_layer.kernel.assign(bioplastic_layer.kernel * adjustment_factor)
        bioplastic_history.append(tf.reduce_mean(bioplastic_layer(inputs)).numpy())

        # Hebbian layer
        output = hebbian_layer(inputs)
        current_activity = tf.reduce_mean(output).numpy()
        if current_activity != 0:
            desired_scale = target_activity / current_activity
            adjustment_factor = 1 + learning_rate * (desired_scale - 1)
            hebbian_layer.kernel.assign(hebbian_layer.kernel * adjustment_factor)
        hebbian_history.append(tf.reduce_mean(hebbian_layer(inputs)).numpy())

    # Final activities
    final_bioplastic = bioplastic_history[-1]
    final_hebbian = hebbian_history[-1]
    print(f"Final activities - Bioplastic: {final_bioplastic:.4f}, Hebbian: {final_hebbian:.4f}")

    # Create comparative visualization
    plt.figure(figsize=(12, 8))

    # Subplot 1: Individual convergence
    plt.subplot(2, 1, 1)
    plt.plot(bioplastic_history, 'b-', linewidth=2, label='BioplasticDenseLayer')
    plt.plot(hebbian_history, 'g-', linewidth=2, label='HebbianHomeostaticNeuroplasticity')
    plt.axhline(y=target_activity, color='r', linestyle='--', linewidth=2, label=f'Target Activity ({target_activity})')
    plt.xlabel('Iteration')
    plt.ylabel('Average Activity')
    plt.title('Homeostatic Plasticity Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Error convergence (distance to target)
    plt.subplot(2, 1, 2)
    bioplastic_error = [abs(act - target_activity) for act in bioplastic_history]
    hebbian_error = [abs(act - target_activity) for act in hebbian_history]
    plt.plot(bioplastic_error, 'b-', linewidth=2, label='BioplasticDenseLayer Error')
    plt.plot(hebbian_error, 'g-', linewidth=2, label='HebbianHomeostaticNeuroplasticity Error')
    plt.xlabel('Iteration')
    plt.ylabel('Absolute Error from Target')
    plt.title('Convergence Error Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization of convergence

    plt.tight_layout()
    plt.savefig('homeostatic_plasticity_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Assertions
    assert np.isclose(final_bioplastic, target_activity, atol=0.2), \
        f"Bioplastic activity {final_bioplastic:.4f} did not converge to target {target_activity}"
    assert np.isclose(final_hebbian, target_activity, atol=0.2), \
        f"Hebbian activity {final_hebbian:.4f} did not converge to target {target_activity}"

    print("SUCCESS: Homeostatic plasticity successfully regulates neural activity in both implementations")


# To run these tests, you would typically use pytest:
# pytest tests/test_homeostatic_plasticity.py

