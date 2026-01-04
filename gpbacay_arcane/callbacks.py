import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class NeuralResonanceCallback(Callback):
    """
    Orchestrates Neural Resonance cycles within the A.R.C.A.N.E. architecture, enabling
    Latent Space Reasoning and Direct Semantic Optimization. It performs multiple 'Resonance Cycles'
    to align internal states across the semantic hierarchy, fostering a Unified Multi-Modal Semantic Space
    and enhancing the Abstraction of Surface-Level Conceptual Variability before synaptic weights are modified.
    """
    def __init__(self, resonance_cycles=5, learning_rate=0.01):
        super(NeuralResonanceCallback, self).__init__()
        self.resonance_cycles = resonance_cycles
        self.lr = learning_rate

    def on_train_batch_begin(self, batch, logs=None):
        """
        The Semantic Resonance Phase: Synchronizes neural representations by iteratively aligning
        projections from higher-level semantic layers, fostering Direct Semantic Optimization and
        Abstraction of Surface-Level Conceptual Variability before synaptic weights are modified.
        """
        if not hasattr(self, 'model') or self.model is None:
            return

        # Identify ResonantGSER layers in the architecture
        resonant_layers = []
        for layer in self.model.layers:
            if 'ResonantGSER' in str(type(layer)):
                resonant_layers.append(layer)
        
        if not resonant_layers:
            return

        # Resonance Cycle: Prospective State Alignment
        for _ in range(self.resonance_cycles):
            # 1. Emit feedback projections from higher layers
            projections = {}
            for i in range(len(resonant_layers) - 1, 0, -1):
                # Layer i projects feedback to layer i-1
                projections[i-1] = resonant_layers[i].project_feedback()

            # 2. Harmonize internal states based on projections
            for i, layer in enumerate(resonant_layers):
                incoming_proj = projections.get(i)
                if incoming_proj is not None:
                    layer.harmonize_states(incoming_proj)


class DynamicSelfModelingReservoirCallback(Callback):
    """
    A callback for dynamic self-modeling and adaptation of reservoir-based layers,
    contributing to Latent Space Reasoning and Abstraction of Surface-Level Conceptual Variability.
    It dynamically adjusts the reservoir's size (neurogenesis and pruning) based on performance metrics,
    optimizing the model's capacity for Direct Semantic Optimization and efficient processing
    within a Unified Multi-Modal Semantic Space.
    """

    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.performance_metric, 0)
        self.performance_history.append(current_metric)

        # Calculate the rate of change in performance over the last epoch
        if len(self.performance_history) > 1:
            improvement_rate = current_metric - self.performance_history[-2]
        else:
            improvement_rate = 0

        # If performance improvement is above the threshold, trigger growth
        if improvement_rate > self.performance_threshold:
            self.reservoir_layer.add_neurons(self.growth_rate)
            print(f" - Growing reservoir by {self.growth_rate} neurons.")
            self.stagnation_counter = 0 # Reset stagnation counter on improvement
        else:
            self.stagnation_counter += 1

        # Trigger pruning if needed based on the prune rate
        if improvement_rate < self.performance_threshold:
            self.reservoir_layer.prune_connections(self.prune_rate)
            print(f" - Pruned connections by {self.prune_rate * 100}%.")
        
        # If performance has stagnated, trigger apoptosis
        if self.stagnation_counter >= self.stagnation_epochs:
            self.reservoir_layer.prune_neurons(self.apoptosis_rate)
            print(f" - Performance stagnated. Pruning {self.apoptosis_rate} neuron(s) via apoptosis.")
            self.stagnation_counter = 0 # Reset after apoptosis

        # If the current metric has reached the target, allow for reservoir growth or pruning
        if current_metric >= self.target_metric:
            print(f" - Performance metric {self.performance_metric} reached target {self.target_metric}.")
            if improvement_rate > self.performance_threshold:
                self.reservoir_layer.add_neurons(self.growth_rate)
            elif improvement_rate < self.performance_threshold:
                self.reservoir_layer.prune_connections(self.prune_rate)

        # Optionally print or log the training progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: {self.performance_metric} = {current_metric:.4f}")

    def reset(self):
        """Resets the monitoring mechanism for a new training session."""
        self.performance_history = []
        self.stagnation_counter = 0

    def get_config(self):
        """Returns the configuration of the callback."""
        config = {
            'reservoir_layer': self.reservoir_layer,
            'performance_metric': self.performance_metric,
            'target_metric': self.target_metric,
            'growth_rate': self.growth_rate,
            'prune_rate': self.prune_rate,
            'performance_threshold': self.performance_threshold,
            'growth_phase_length': self.growth_phase_length,
            'pruning_phase_length': self.pruning_phase_length,
            'stagnation_epochs': self.stagnation_epochs,
            'apoptosis_rate': self.apoptosis_rate
        }
        return config

    @classmethod
    def from_config(cls, config):
        """Creates the callback from its configuration."""
        return cls(**config)