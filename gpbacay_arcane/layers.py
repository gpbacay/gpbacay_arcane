import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, Flatten
from typing import Union, Callable



class ExpandDimensionLayer(Layer):
    """
    A custom layer that expands the dimensions of the input tensor along a specified axis.
    This layer is useful for adding a new axis to the input tensor, which can be necessary 
    for reshaping the input to be compatible with subsequent layers or operations.

    Attributes:
        axis (int): The axis along which the dimension will be expanded. Default is 1.
    """
    
    def __init__(self, axis=1, **kwargs):
        super(ExpandDimensionLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({
            'axis': self.axis
        })
        return config



class GSER(Layer):
    """
    The Gated Spiking Elastic Reservoir (GSER) Layer is an innovative neural network layer that combines dynamic reservoir sizing, 
    spiking neuron behavior, and adaptive gating mechanisms to enhance temporal sequence processing. 
    It features elastic reservoir growth, spiking neurons that trigger upon threshold exceedance, 
    and three gating mechanisms (input, forget, output) for precise memory control. 
    Supporting neurogenesis (adding new neurons) and synaptogenesis (pruning connections), 
    GSER can self-organize to optimize performance, balancing long-term and short-term memory retention. 
    Its scalability, adaptability, and efficiency make it ideal for complex, event-driven learning tasks in dynamic environments.
    
    Attributes:
        initial_reservoir_size (int): Initial number of neurons in the reservoir.
        input_dim (int): Dimension of the input data.
        spectral_radius (float): Spectral radius of the reservoir weight matrix, influencing its stability.
        leak_rate (float): Initial rate at which the state of the reservoir decays over time. Now trainable.
        spike_threshold (float): Initial threshold for spikes. Now trainable.
        max_dynamic_reservoir_dim (int): Maximum dynamic size of the reservoir.
        state_size (list): Size of the reservoir state.
        output_size (int): Size of the output of the reservoir layer.
    """

    def __init__(self, input_dim, initial_reservoir_size, max_dynamic_reservoir_dim, spectral_radius, leak_rate, spike_threshold, neurogenesis_rate=0.05, pruning_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.initial_reservoir_size = initial_reservoir_size
        self.max_dynamic_reservoir_dim = max_dynamic_reservoir_dim
        self.spectral_radius = spectral_radius
        self.initial_leak_rate = leak_rate
        self.initial_spike_threshold = spike_threshold
        self.neurogenesis_rate = neurogenesis_rate
        self.pruning_rate = pruning_rate
        
        # State tracking for elastic reservoir (will be created in build method)
        self.initial_reservoir_size_value = initial_reservoir_size
        self.current_reservoir_size = None
        
        self.state_size = [self.max_dynamic_reservoir_dim]
        self.output_size = self.max_dynamic_reservoir_dim
        
    def build(self, input_shape):
        """Build the layer and initialize all weights on the correct device."""
        super().build(input_shape)
        
        # Create the current reservoir size variable in build method to ensure proper device placement
        self.current_reservoir_size = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_reservoir_size_value),
            trainable=False,
            dtype=tf.int32,
            name='current_reservoir_size'
        )
        
        # Initialize all other weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initializes all weights to their max dimension for efficient growth (pre-allocation)."""
        # Reservoir weights (pre-allocated to max size)
        self.spatiotemporal_reservoir_weights = self.add_weight(
            shape=(self.max_dynamic_reservoir_dim, self.max_dynamic_reservoir_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=False,
            name='spatiotemporal_reservoir_weights'
        )
        
        # Input weights (pre-allocated to max size)
        self.spatiotemporal_input_weights = self.add_weight(
            shape=(self.max_dynamic_reservoir_dim, self.input_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=False,
            name='spatiotemporal_input_weights'
        )
        
        # Spiking gate weights (pre-allocated to max size)
        self.spiking_gate_weights = self.add_weight(
            shape=(3 * self.max_dynamic_reservoir_dim, self.input_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=False,
            name='spiking_gate_weights'
        )

        # Trainable, per-neuron leak rate
        self.leak_rate_param = self.add_weight(
            shape=(self.max_dynamic_reservoir_dim,),
            initializer=tf.keras.initializers.Constant(np.log(self.initial_leak_rate / (1 - self.initial_leak_rate))), # Inverse sigmoid
            trainable=True,
            name='leak_rate_param'
        )

        # Trainable, per-neuron spike threshold
        self.spike_threshold_param = self.add_weight(
            shape=(self.max_dynamic_reservoir_dim,),
            initializer=tf.keras.initializers.Constant(np.log(np.exp(self.initial_spike_threshold) - 1)), # Inverse softplus
            trainable=True,
            name='spike_threshold_param'
        )


    def add_neurons(self, new_neurons_count):
        """Efficiently adds new neurons by incrementing the current size."""
        new_size = tf.minimum(self.current_reservoir_size + new_neurons_count, self.max_dynamic_reservoir_dim)
        self.current_reservoir_size.assign(new_size)

    def prune_connections(self, pruning_threshold=0.1):
        """Prunes weak connections within the active part of the reservoir."""
        active_size = self.current_reservoir_size
        
        # Create a mask for the active reservoir
        active_weights = self.spatiotemporal_reservoir_weights[:active_size, :active_size]
        mask = tf.abs(active_weights) < pruning_threshold
        
        # Apply pruning only to the active part
        pruned_weights = tf.where(mask, tf.zeros_like(active_weights), active_weights)
        
        # Update the weight matrix
        self.spatiotemporal_reservoir_weights.assign(tf.tensor_scatter_nd_update(
            self.spatiotemporal_reservoir_weights,
            tf.where(tf.ones((active_size, active_size), dtype=tf.bool)),
            tf.reshape(pruned_weights, [-1])
        ))

    def prune_neurons(self, num_to_prune):
        """Prunes inactive neurons using the efficient 'swap and pop' method."""
        active_size = self.current_reservoir_size
        if active_size <= num_to_prune:
            return # Avoid pruning too many neurons

        # Calculate neuron activity (e.g., sum of absolute outgoing weights)
        activity = tf.reduce_sum(tf.abs(self.spatiotemporal_reservoir_weights[:active_size, :active_size]), axis=1)
        
        # Get indices of neurons to prune (least active)
        _, indices_to_prune = tf.nn.top_k(-activity, k=num_to_prune)

        for idx_to_prune in tf.sort(indices_to_prune, direction='DESCENDING'):
            last_active_idx = self.current_reservoir_size - 1
            if idx_to_prune >= last_active_idx:
                 # If it's the last one, just decrement size
                self.current_reservoir_size.assign_sub(1)
                continue

            # Efficiently swap the neuron to be pruned with the last active neuron
            
            # 1. Swap rows in the reservoir weights matrix
            p = tf.stack([idx_to_prune, last_active_idx])
            q = tf.stack([last_active_idx, idx_to_prune])
            
            # Swap rows
            temp_weights = tf.tensor_scatter_nd_update(self.spatiotemporal_reservoir_weights, tf.expand_dims(p, axis=1), tf.gather(self.spatiotemporal_reservoir_weights, q))
            # Swap columns
            temp_weights = tf.transpose(temp_weights)
            temp_weights = tf.tensor_scatter_nd_update(temp_weights, tf.expand_dims(p, axis=1), tf.gather(temp_weights, q))
            self.spatiotemporal_reservoir_weights.assign(tf.transpose(temp_weights))

            # 2. Swap the corresponding rows in the input weights
            self.spatiotemporal_input_weights.assign(
                tf.tensor_scatter_nd_update(self.spatiotemporal_input_weights, tf.expand_dims(p, axis=1), tf.gather(self.spatiotemporal_input_weights, q))
            )
            
            # 3. Swap trainable parameters
            self.leak_rate_param.assign(
                tf.tensor_scatter_nd_update(self.leak_rate_param, tf.expand_dims(p, axis=1), tf.gather(self.leak_rate_param, q))
            )
            self.spike_threshold_param.assign(
                tf.tensor_scatter_nd_update(self.spike_threshold_param, tf.expand_dims(p, axis=1), tf.gather(self.spike_threshold_param, q))
            )

            # 4. Decrement the reservoir size
            self.current_reservoir_size.assign_sub(1)


    def call(self, inputs, states):
        """
        The forward pass for the Gated Spiking Elastic Reservoir Layer. The method computes
        the new state of the reservoir based on the previous state and the input.
        
        Parameters:
            inputs (tensor): The current input to the layer.
            states (list): The previous state of the reservoir.

        Returns:
            tensor: The updated state of the reservoir after processing the input.
            list: The updated state of the reservoir for use in the next time step.
        """
        # Convert to float32 for computation to avoid mixed precision issues
        inputs = tf.cast(inputs, tf.float32)
        
        active_size = self.current_reservoir_size
        prev_state = tf.cast(states[0][:, :active_size], tf.float32)

        # Get active weights using slicing
        active_input_weights = self.spatiotemporal_input_weights[:active_size, :]
        active_reservoir_weights = self.spatiotemporal_reservoir_weights[:active_size, :active_size]
        active_gate_weights = self.spiking_gate_weights[:3 * active_size, :]
        
        # Get active trainable parameters and apply constraining activations
        leak_rate = tf.sigmoid(self.leak_rate_param[:active_size])
        spike_threshold = tf.nn.softplus(self.spike_threshold_param[:active_size])

        # Compute the input part
        input_part = tf.matmul(inputs, active_input_weights, transpose_b=True)
        
        # Compute the reservoir part
        reservoir_part = tf.matmul(prev_state, active_reservoir_weights)
        
        # Compute the gate part
        gate_part = tf.matmul(inputs, active_gate_weights, transpose_b=True)

        # Split the gate part into three separate gates (input, forget, and output gates)
        i_gate, f_gate, o_gate = tf.split(tf.sigmoid(gate_part), 3, axis=-1)

        # Update the state using the gating mechanism and reservoir dynamics
        state = (1 - leak_rate) * (f_gate * prev_state) + leak_rate * tf.tanh(i_gate * (input_part + reservoir_part))
        state = o_gate * state

        # Generate spikes if the state exceeds the spike threshold
        spikes = tf.cast(tf.greater(state, spike_threshold), dtype=tf.float32)
        
        # If a spike occurs, reset the state by subtracting the spike threshold
        state = tf.where(spikes > 0, state - spike_threshold, state)

        # Pad the state to match the max dimension for the RNN layer
        padded_state = tf.pad(state, [[0, 0], [0, self.max_dynamic_reservoir_dim - active_size]])

        # Explicitly set the shape to fix RNN layer issues with dynamic shapes
        padded_state.set_shape([None, self.max_dynamic_reservoir_dim])

        return padded_state, [padded_state]

    def get_config(self):
        """Returns the configuration of the layer, useful for model serialization."""
        config = super().get_config()
        config.update({
            'initial_reservoir_size': self.initial_reservoir_size,
            'input_dim': self.input_dim,
            'spectral_radius': self.spectral_radius,
            'leak_rate': self.initial_leak_rate,
            'spike_threshold': self.initial_spike_threshold,
            'max_dynamic_reservoir_dim': self.max_dynamic_reservoir_dim,
            'neurogenesis_rate': self.neurogenesis_rate,
            'pruning_rate': self.pruning_rate
        })
        return config




class DenseGSER(Layer):
    """
    A dynamic, dense reservoir layer that integrates custom mechanisms for adaptive state updating 
    and memory retention. It combines traditional dense layer functionality with advanced features 
    like gated memory, leak-based state evolution, and spike thresholding. By leveraging input weights, 
    reservoir weights, and gating mechanisms (input, forget, and output gates), it facilitates more 
    complex, temporally-aware learning processes, making it particularly suitable for tasks involving 
    sequential data or long-term dependencies. This layer provides a novel approach by incorporating 
    spiking behaviors and dynamic state updating, offering an advantage over conventional dense layers 
    in memory-intensive or adaptive models.

    Attributes:
    - units: Number of units (neurons) in the layer.
    - input_dim: Dimensionality of the input (optional, inferred from input shape if not provided).
    - spectral_radius: Controls the stability of the reservoir by adjusting the spectral radius of the reservoir weights.
    - leak_rate: Rate at which past states decay, balancing the influence of previous states and new inputs.
    - spike_threshold: Threshold for triggering spikes in the state, enabling spiking behavior.
    - max_dynamic_units: Optional maximum limit for dynamic unit expansion, allowing for adaptive layer size (currently unused).
    - activation: Activation function applied to the output (e.g., 'gelu', 'relu').
    - kernel_initializer: Initializer for the input and reservoir weight matrices.
    - bias_initializer: Initializer for the bias vectors.
    - kernel_regularizer: Regularizer applied to the input and reservoir weight matrices.
    - bias_regularizer: Regularizer applied to the bias vectors.
    """

    def __init__(self, units, input_dim=None, spectral_radius=0.9, leak_rate=0.1, spike_threshold=0.5, 
                 max_dynamic_units=None, activation='gelu', kernel_initializer='glorot_uniform', 
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, name=None, **kwargs):
        super(DenseGSER, self).__init__(name=name, **kwargs)
        
        # Parameters for custom mechanisms
        self.units = units
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_dynamic_units = max_dynamic_units
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

    def build(self, input_shape):
        # Ensure that input_dim matches the expected input shape dimension
        self.input_dim = input_shape[-1] if self.input_dim is None else self.input_dim
        
        # Initialize the input, reservoir, and gate weights, and biases
        self.input_weights = self.add_weight(
            shape=(self.input_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name='input_weights'
        )
        
        self.reservoir_weights = self.add_weight(
            shape=(self.input_dim, self.units),
            initializer=self.kernel_initializer,
            trainable=False,
            name='reservoir_weights'
        )
        
        self.gate_weights = self.add_weight(
            shape=(self.input_dim, 3 * self.units),  # Corrected shape
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name='gate_weights'
        )
        
        self.gate_bias = self.add_weight(
            shape=(3 * self.units,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            trainable=True,
            name='gate_bias'
        )

        # Weight matrix for final transformation
        self.output_weights = self.add_weight(
            shape=(self.units, self.units),  # New weight matrix
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name='output_weights'
        )

        # Bias for the output layer
        self.output_bias = self.add_weight(
            shape=(self.units,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            trainable=True,
            name='output_bias'
        )

        self.built = True

    def call(self, inputs):
        # Convert to float32 for computation to avoid mixed precision issues
        inputs = tf.cast(inputs, tf.float32)
        
        # Ensure that input is of the expected shape
        input_part = tf.matmul(inputs, self.input_weights)
        reservoir_part = tf.matmul(inputs, self.reservoir_weights)
        gate_part = tf.matmul(inputs, self.gate_weights) + self.gate_bias

        # Split the gates into input, forget, and output gates
        i_gate, f_gate, o_gate = tf.split(tf.sigmoid(gate_part), 3, axis=-1)

        # Compute the state update using the gates and leak rate
        state = (1 - self.leak_rate) * (f_gate * reservoir_part) + self.leak_rate * tf.tanh(i_gate * (input_part + reservoir_part))
        state = o_gate * state

        # Apply spike thresholding
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)

        # Final transformation using output weights and bias
        output = tf.matmul(state, self.output_weights) + self.output_bias

        # Apply the activation function
        output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'input_dim': self.input_dim,
            'spectral_radius': self.spectral_radius,
            'leak_rate': self.leak_rate,
            'spike_threshold': self.spike_threshold,
            'max_dynamic_units': self.max_dynamic_units,
            'activation': tf.keras.activations.serialize(self.activation),
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['activation'] = tf.keras.activations.deserialize(config['activation'])
        config['kernel_initializer'] = tf.keras.initializers.deserialize(config['kernel_initializer'])
        config['bias_initializer'] = tf.keras.initializers.deserialize(config['bias_initializer'])
        config['kernel_regularizer'] = tf.keras.regularizers.deserialize(config['kernel_regularizer'])
        config['bias_regularizer'] = tf.keras.regularizers.deserialize(config['bias_regularizer'])
        return cls(**config)




class RelationalConceptModeling(Layer):
    """
    Relational Concept Modeling (RCM) is an encoder layer that applies multi-head self-attention to capture token-level relations, 
    uses spiking-inspired DenseGSER pooling to condense these into compact concept vectors,
    refines inter-concept interactions via a second attention stage, and projects the result into a final `(batch, 1, d_model)` 
    embedding for efficient, end-to-end summarization and hierarchical reasoning over sequential or structured data.
    
    Attributes:
        d_model (int): Dimensionality of input and output features.
        num_heads (int): Number of attention heads for multi-head attention.
        dropout_rate (float): Dropout rate for regularization.
        use_weighted_summary (bool): Flag to enable learnable summary weighting.
        eps (float): Small constant for numerical stability.
    """
    def __init__(self, d_model, num_heads, dropout_rate=0.1, use_weighted_summary=False, eps=1e-6, **kwargs):
        super(RelationalConceptModeling, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_weighted_summary = use_weighted_summary
        self.eps = eps
        
        # Attention mechanism to model token-level relationships
        self.attention_layer = MultiheadLinearSelfAttentionKernalization(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            use_weighted_summary=use_weighted_summary,
            eps=eps
        )
        
        # Replace Dense with DenseGSER for pooling concept representations
        self.concept_pooling = DenseGSER(
            units=d_model,
            spectral_radius=0.9,
            leak_rate=0.1,
            spike_threshold=0.5,
            activation="gelu"
        )
        
        # Interaction attention layer to model the relationships between pooled concepts
        self.interaction_attention = MultiheadLinearSelfAttentionKernalization(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            use_weighted_summary=use_weighted_summary,
            eps=eps
        )
        
        # Output projection to map concepts into final output space
        self.output_projection = DenseGSER(
            units=d_model,
            spectral_radius=0.9,
            leak_rate=0.1,
            spike_threshold=0.5,
            activation=None  # No activation for the projection layer
        )

    def build(self, input_shape):
        # Build all child layers
        self.attention_layer.build(input_shape)
        
        # For concept pooling, we need to calculate the pooled shape
        pooled_shape = (input_shape[0], 1, self.d_model)
        self.concept_pooling.build(pooled_shape)
        
        # For interaction attention, use the same pooled shape
        self.interaction_attention.build(pooled_shape)
        
        # For output projection, use the pooled shape as well
        self.output_projection.build(pooled_shape)
        
        super(RelationalConceptModeling, self).build(input_shape)

    def call(self, inputs, training=False):
        # Step 1: Extract token-level relationships using attention
        token_relations = self.attention_layer(inputs, training=training)
        
        # Step 2: Pool concepts by averaging token relations and applying DenseGSER
        pooled_concepts = tf.reduce_mean(token_relations, axis=1, keepdims=True)
        pooled_concepts = self.concept_pooling(pooled_concepts)
        
        # Step 3: Model interactions between pooled concepts using attention
        refined_concepts = self.interaction_attention(pooled_concepts, training=training)
        
        # Step 4: Project concepts to the output space for further tasks
        output = self.output_projection(refined_concepts)
        return output

    def compute_output_shape(self, input_shape):
        # The output is always a single concept vector.
        return (input_shape[0], 1, self.d_model)

    def get_config(self):
        # Return configuration to recreate the model
        config = super(RelationalConceptModeling, self).get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "use_weighted_summary": self.use_weighted_summary,
            "eps": self.eps
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            dropout_rate=config["dropout_rate"],
            use_weighted_summary=config["use_weighted_summary"],
            eps=config["eps"],
        )




# Test Accuracy: 0.9687, Total Loss: 0.1806
class RelationalGraphAttentionReasoning(Layer):
    """
    RelationalGraphAttentionReasoning (RGAR) is an encoder layer that processes a set of node or concept embeddings via 
    multi-head self-attention to perform dynamic message passing, and then produces task-specific graph-level predictions 
    through a spiking-inspired DenseGSER readout. It accepts an input tensor of shape `(batch_size, N, d_model)`, 
    refines inter-node relations without fixed aggregation rules, and outputs a `(batch_size, num_classes)` 
    tensor of class scores for graph-level classification or regression tasks.

    Attributes:
        d_model (int): Dimensionality of input node embeddings.
        num_heads (int): Number of attention heads for message passing.
        num_classes (int): Number of output classes for predictions.
    """
    def __init__(self, d_model, num_heads, num_classes, **kwargs):
        super(RelationalGraphAttentionReasoning, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_classes = num_classes

        # Relational Entity Graph message passing layer (GNN-like operation)
        self.message_passing_layer = MultiheadLinearSelfAttentionKernalization(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=0.1
        )

        # Output layer for task-specific predictions
        self.output_layer = DenseGSER(
            units=num_classes,
            spectral_radius=0.9,
            leak_rate=0.1,
            spike_threshold=0.5,
            activation="gelu"
        )
        
        self.flatten = Flatten()

    def build(self, input_shape):
        """
        Build the internal components of the model.
        """
        # Call the build methods of child layers to ensure all variables are initialized
        self.message_passing_layer.build(input_shape)
        message_passing_output_shape = self.message_passing_layer.compute_output_shape(input_shape)
        self.output_layer.build(message_passing_output_shape)

    def call(self, inputs, training=None):
        """
        Forward pass of the model.
        """
        # Step 1: Perform message passing on relational graph using the output from RCM
        graph_relations = self.message_passing_layer(inputs, training=training)

        # Step 2: Produce task-specific predictions and flatten the output
        output = self.output_layer(graph_relations)
        output = self.flatten(output)
        return output

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the model.
        """
        return (input_shape[0], self.num_classes)

    def get_config(self):
        """
        Return the configuration of the model for serialization.
        """
        config = {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_classes": self.num_classes
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_classes=config["num_classes"]
        )




class BioplasticDenseLayer(tf.keras.layers.Layer):
    """
    Advanced Keras layer implementing biologically-inspired plasticity mechanisms
    including Hebbian/anti-Hebbian learning, BCM metaplasticity, homeostatic scaling,
    and structural plasticity for adaptive neural computation.

    Key Improvements:
      - BCM-style metaplasticity with sliding threshold for more stable learning
      - Structural plasticity with synaptic pruning and sprouting
      - Anti-Hebbian inhibitory plasticity for better balance
      - Sparse activity regularization for energy efficiency
      - Multiple normalization schemes (batch, layer, adaptive)
      - Better numerical stability and performance optimizations
      - Comprehensive logging and visualization support

    Biological Mechanisms:
      - Hebbian strengthening: LTP-like potentiation for correlated activity
      - Anti-Hebbian weakening: LTD-like depression for uncorrelated activity
      - BCM rule: Activity-dependent plasticity threshold prevents runaway dynamics
      - Homeostatic scaling: Maintains target firing rates across populations
      - Structural plasticity: Dynamic pruning/sprouting based on activity

    Parameters
    ----------
    units : int
        Number of output neurons
    learning_rate : float, default=1e-4
        Base learning rate for Hebbian updates
    anti_hebbian_rate : float, default=0.1
        Relative strength of anti-Hebbian (LTD-like) plasticity
    target_avg : float, default=0.15
        Target average activity for homeostatic regulation
    homeostatic_rate : float, default=1e-4
        Learning rate for homeostatic scaling adjustments
    bcm_tau : float, default=1000.0
        Time constant for BCM threshold adaptation (batches)
    structural_rate : float, default=1e-6
        Rate of structural plasticity changes
    sparsity_target : float, default=0.05
        Target fraction of active neurons (for sparse coding)
    activation : str or callable, default='swish'
        Activation function (swish is more biologically plausible)
    normalization : str, default='adaptive'
        Weight normalization scheme: 'l2', 'batch', 'layer', 'adaptive', or None
    pruning_threshold : float, default=0.01
        Threshold below which synapses are pruned
    max_weight : float, default=5.0
        Maximum absolute weight value
    momentum : float, default=0.95
        Momentum for Hebbian delta smoothing
    ema_alpha : float, default=0.05
        EMA factor for activity tracking (lower = more stable)
    use_bias : bool, default=True
        Whether to include trainable bias terms
    dropout_rate : float, default=0.0
        Synaptic dropout rate during training
    enable_logging : bool, default=False
        Enable detailed logging of plasticity dynamics
    """

    def __init__(
        self,
        units: int,
        learning_rate: float = 1e-4,
        anti_hebbian_rate: float = 0.1,
        target_avg: float = 0.15,
        homeostatic_rate: float = 1e-4,
        bcm_tau: float = 1000.0,
        structural_rate: float = 1e-6,
        sparsity_target: float = 0.05,
        min_scale: float = 0.1,
        max_scale: float = 2.0,
        activation: Union[str, Callable] = 'swish',
        normalization: str = 'adaptive',
        pruning_threshold: float = 0.01,
        max_weight: float = 5.0,
        momentum: float = 0.95,
        ema_alpha: float = 0.05,
        use_bias: bool = True,
        dropout_rate: float = 0.0,
        enable_logging: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Core parameters
        self.units = int(units)
        self.learning_rate = float(learning_rate)
        self.anti_hebbian_rate = float(anti_hebbian_rate)
        self.target_avg = float(target_avg)
        self.homeostatic_rate = float(homeostatic_rate)
        self.bcm_tau = float(bcm_tau)
        self.structural_rate = float(structural_rate)
        self.sparsity_target = float(sparsity_target)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        
        # Regularization and stability
        self.normalization = normalization
        self.pruning_threshold = float(pruning_threshold)
        self.max_weight = float(max_weight)
        self.momentum = float(momentum)
        self.ema_alpha = float(ema_alpha)
        self.use_bias = bool(use_bias)
        self.dropout_rate = float(dropout_rate)
        self.enable_logging = bool(enable_logging)
        
        # Activation function
        if isinstance(activation, str):
            if activation == 'swish':
                self.activation = lambda x: x * tf.nn.sigmoid(x)
            else:
                self.activation = tf.keras.activations.get(activation)
        else:
            self.activation = activation
            
        # Internal constants
        self._eps = 1e-8
        self._step_counter = 0

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        
        # Xavier/Glorot initialization for better convergence
        fan_in, fan_out = feature_dim, self.units
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        
        # Main synaptic weight matrix
        self.kernel = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=tf.keras.initializers.RandomUniform(-limit, limit),
            trainable=True,
            name='kernel',
        )
        
        # Bias terms
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
                name='bias',
            )
        
        # Homeostatic scaling factors (per-unit)
        self.activity_scale = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.Ones(),
            trainable=False,
            name='activity_scale',
        )
        
        # BCM plasticity thresholds (per-unit)
        self.bcm_threshold = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(self.target_avg),
            trainable=False,
            name='bcm_threshold',
        )
        
        # Activity tracking variables
        self.avg_activity = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=False,
            name='avg_activity',
        )
        
        self.activity_variance = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.Ones(),
            trainable=False,
            name='activity_variance',
        )
        
        # Hebbian momentum terms
        self.hebbian_momentum = self.add_weight(
            shape=self.kernel.shape,
            initializer=tf.keras.initializers.Zeros(),
            trainable=False,
            name='hebbian_momentum',
        )
        
        # Structural plasticity masks
        self.synaptic_strength = self.add_weight(
            shape=self.kernel.shape,
            initializer=tf.keras.initializers.Ones(),
            trainable=False,
            name='synaptic_strength',
        )
        
        # Adaptive normalization parameters
        if self.normalization == 'adaptive':
            self.norm_scale = self.add_weight(
                shape=(self.units,),
                initializer=tf.keras.initializers.Ones(),
                trainable=False,
                name='norm_scale',
            )
        
        # Logging variables (if enabled)
        if self.enable_logging:
            self.log_weights_norm = self.add_weight(
                shape=(),
                initializer=tf.keras.initializers.Zeros(),
                trainable=False,
                name='log_weights_norm',
            )
            
            self.log_plasticity_rate = self.add_weight(
                shape=(),
                initializer=tf.keras.initializers.Zeros(),
                trainable=False,
                name='log_plasticity_rate',
            )

        super().build(input_shape)

    def _apply_normalization(self, weights: tf.Tensor, inputs: tf.Tensor) -> tf.Tensor:
        """Apply the specified normalization scheme to weights."""
        if self.normalization == 'l2':
            return tf.nn.l2_normalize(weights, axis=0, epsilon=self._eps)
        elif self.normalization == 'batch':
            # Batch-dependent normalization
            batch_std = tf.math.reduce_std(tf.reduce_mean(tf.abs(weights), axis=0))
            return weights / (batch_std + self._eps)
        elif self.normalization == 'layer':
            # Layer normalization
            mean, var = tf.nn.moments(weights, axes=[0, 1], keepdims=True)
            return (weights - mean) / tf.sqrt(var + self._eps)
        elif self.normalization == 'adaptive':
            # Adaptive normalization based on input RMS (per-feature) and per-unit scaling
            feature_dim = tf.shape(weights)[0]
            unit_dim = tf.shape(weights)[1]
            input_rms = tf.sqrt(tf.reduce_mean(tf.square(inputs), axis=0) + self._eps)  # (feature_dim,)
            input_scale = tf.reshape(input_rms, (feature_dim, 1))  # (feature_dim, 1)
            unit_scale = tf.reshape(self.norm_scale, (1, unit_dim))  # (1, units)
            scaled = weights * input_scale * unit_scale
            return scaled
        else:
            return weights

    def _compute_bcm_plasticity(self, pre: tf.Tensor, post: tf.Tensor, 
                               post_squared: tf.Tensor) -> tf.Tensor:
        """Compute BCM-style plasticity with sliding threshold."""
        # BCM rule: Δw ∝ post * (post - θ) * pre
        # where θ is the sliding threshold
        post_minus_threshold = post - tf.expand_dims(self.bcm_threshold, 0)
        plasticity_signal = post * post_minus_threshold
        
        # Compute weight changes
        batch_size = tf.cast(tf.shape(pre)[0], tf.float32)
        delta = tf.matmul(tf.transpose(pre), plasticity_signal) / (batch_size + self._eps)
        
        # Update BCM threshold (sliding average of post^2)
        new_threshold = (
            (1.0 - 1.0/self.bcm_tau) * self.bcm_threshold + 
            (1.0/self.bcm_tau) * tf.reduce_mean(post_squared, axis=0)
        )
        self.bcm_threshold.assign(new_threshold)
        
        return delta

    def _apply_structural_plasticity(self, weights: tf.Tensor) -> tf.Tensor:
        """Apply synaptic pruning and sprouting based on activity."""
        # Compute synaptic efficacy (running average of absolute weights)
        abs_weights = tf.abs(weights)
        new_strength = (
            (1.0 - self.structural_rate) * self.synaptic_strength + 
            self.structural_rate * abs_weights
        )
        self.synaptic_strength.assign(new_strength)
        
        # Pruning: set very weak synapses to zero
        pruning_mask = tf.cast(new_strength > self.pruning_threshold, tf.float32)
        
        # Sprouting: random reconnection of pruned synapses (small probability)
        sprouting_prob = self.structural_rate * 10.0  # Higher rate for sprouting
        random_vals = tf.random.uniform(tf.shape(weights))
        sprouting_mask = tf.cast(random_vals < sprouting_prob, tf.float32)
        
        # Combine masks: keep existing strong connections, add new random ones
        final_mask = tf.maximum(pruning_mask, sprouting_mask * (1.0 - pruning_mask))
        
        return weights * final_mask

    def _compute_sparsity_loss(self, outputs: tf.Tensor) -> tf.Tensor:
        """Compute sparsity regularization loss."""
        activity_rate = tf.reduce_mean(tf.cast(outputs > 0.1, tf.float32), axis=0)
        sparsity_error = activity_rate - self.sparsity_target
        return tf.reduce_mean(tf.square(sparsity_error))

    def call(self, inputs, training=None):
        """Enhanced forward pass with multiple plasticity mechanisms."""
        if training is None:
            training = tf.keras.backend.learning_phase()
        
        # Convert to float32 for computation to avoid mixed precision issues
        inputs = tf.cast(inputs, tf.float32)
            
        # Handle temporal sequences
        original_shape = tf.shape(inputs)
        if len(inputs.shape) == 3:
            batch_size, seq_len = original_shape[0], original_shape[1]
            inputs = tf.reshape(inputs, [-1, inputs.shape[-1]])
        else:
            batch_size, seq_len = original_shape[0], None

        # Apply synaptic dropout during training
        if training and self.dropout_rate > 0:
            dropout_mask = tf.nn.dropout(tf.ones_like(self.kernel), self.dropout_rate)
            effective_kernel = self.kernel * dropout_mask
        else:
            effective_kernel = self.kernel

        # Apply structural plasticity mask
        effective_kernel = effective_kernel * self.synaptic_strength

        # Normalize inputs and weights
        inputs_norm = tf.nn.l2_normalize(inputs, axis=-1, epsilon=self._eps)
        kernel_norm = self._apply_normalization(effective_kernel, inputs_norm)

        # Forward computation (with safe matmul)
        pre_activation = tf.matmul(inputs_norm, kernel_norm)
        pre_activation = tf.clip_by_value(pre_activation, -1e6, 1e6)
        if self.use_bias:
            pre_activation += self.bias

        # Apply homeostatic scaling
        scaled_pre = pre_activation * self.activity_scale
        
        # Activation
        if self.activation is not None:
            outputs = self.activation(scaled_pre)
        else:
            outputs = scaled_pre
        outputs = tf.clip_by_value(outputs, -50.0, 50.0)

        # Plasticity updates (only during training)
        if training:
            self._step_counter += 1
            
            # Track activities for homeostasis
            batch_activity = tf.reduce_mean(outputs, axis=0)
            batch_activity_sq = tf.reduce_mean(tf.square(outputs), axis=0)
            
            # Update activity statistics
            new_avg = (1.0 - self.ema_alpha) * self.avg_activity + self.ema_alpha * batch_activity
            new_var = (1.0 - self.ema_alpha) * self.activity_variance + self.ema_alpha * batch_activity_sq
            
            self.avg_activity.assign(new_avg)
            self.activity_variance.assign(new_var)

            # BCM-style Hebbian plasticity
            if self.learning_rate > 0:
                hebbian_delta = self._compute_bcm_plasticity(
                    inputs_norm, outputs, tf.square(outputs)
                )
                
                # Add anti-Hebbian component (decorrelation)
                anti_hebbian_delta = -self.anti_hebbian_rate * tf.matmul(
                    tf.transpose(inputs_norm), 
                    tf.nn.relu(outputs - tf.expand_dims(new_avg, 0))
                )
                
                total_delta = tf.clip_by_value(hebbian_delta + anti_hebbian_delta, -1e3, 1e3)
                
                # Apply momentum
                new_momentum = (
                    self.momentum * self.hebbian_momentum + 
                    (1.0 - self.momentum) * total_delta
                )
                self.hebbian_momentum.assign(new_momentum)
                
                # Update weights with clipping
                new_kernel = tf.clip_by_value(
                    self.kernel + self.learning_rate * new_momentum,
                    -self.max_weight, self.max_weight
                )
                
                # Apply structural plasticity
                new_kernel = self._apply_structural_plasticity(new_kernel)
                self.kernel.assign(new_kernel)

            # Homeostatic scaling update
            if self.homeostatic_rate > 0:
                scale_error = self.target_avg - new_avg
                scale_adjustment = self.homeostatic_rate * scale_error
                new_scale = tf.clip_by_value(
                    self.activity_scale + scale_adjustment,
                    self.min_scale,
                    self.max_scale
                )
                self.activity_scale.assign(new_scale)

            # Update adaptive normalization
            if self.normalization == 'adaptive':
                input_rms = tf.sqrt(tf.reduce_mean(tf.square(inputs_norm), axis=0))
                new_norm_scale = (
                    0.9 * self.norm_scale + 0.1 * (1.0 / (input_rms + self._eps))
                )
                self.norm_scale.assign(new_norm_scale)

            # Logging updates
            if self.enable_logging:
                weights_norm = tf.reduce_mean(tf.square(self.kernel))
                plasticity_rate = tf.reduce_mean(tf.abs(new_momentum)) if 'new_momentum' in locals() else 0.0
                
                self.log_weights_norm.assign(weights_norm)
                self.log_plasticity_rate.assign(plasticity_rate)

        # Reshape back to original if needed
        if seq_len is not None:
            outputs = tf.reshape(outputs, [batch_size, seq_len, self.units])

        return outputs

    def get_plasticity_stats(self) -> dict:
        """Return current plasticity statistics for monitoring."""
        stats = {
            'avg_activity': self.avg_activity.numpy(),
            'activity_scale': self.activity_scale.numpy(),
            'bcm_threshold': self.bcm_threshold.numpy(),
            'synaptic_density': tf.reduce_mean(
                tf.cast(self.synaptic_strength > self.pruning_threshold, tf.float32)
            ).numpy(),
        }
        
        if self.enable_logging:
            stats.update({
                'weights_norm': self.log_weights_norm.numpy(),
                'plasticity_rate': self.log_plasticity_rate.numpy(),
            })
            
        return stats

    def reset_plasticity(self):
        """Reset all plasticity-related variables to initial state."""
        self.activity_scale.assign(tf.ones_like(self.activity_scale))
        self.bcm_threshold.assign(tf.ones_like(self.bcm_threshold) * self.target_avg)
        self.avg_activity.assign(tf.zeros_like(self.avg_activity))
        self.activity_variance.assign(tf.ones_like(self.activity_variance))
        self.hebbian_momentum.assign(tf.zeros_like(self.hebbian_momentum))
        self.synaptic_strength.assign(tf.ones_like(self.synaptic_strength))

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 3:
            return (input_shape[0], input_shape[1], self.units)
        return (input_shape[0], self.units)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'learning_rate': self.learning_rate,
            'anti_hebbian_rate': self.anti_hebbian_rate,
            'target_avg': self.target_avg,
            'homeostatic_rate': self.homeostatic_rate,
            'bcm_tau': self.bcm_tau,
            'structural_rate': self.structural_rate,
            'sparsity_target': self.sparsity_target,
            'activation': 'swish' if callable(self.activation) else tf.keras.activations.serialize(self.activation),
            'normalization': self.normalization,
            'pruning_threshold': self.pruning_threshold,
            'max_weight': self.max_weight,
            'momentum': self.momentum,
            'ema_alpha': self.ema_alpha,
            'use_bias': self.use_bias,
            'dropout_rate': self.dropout_rate,
            'enable_logging': self.enable_logging,
        })
        return config





class HebbianHomeostaticNeuroplasticity(Layer):
    """
    This layer integrates Hebbian learning with homeostatic scaling to stabilize neural activity.
    It adapts synaptic weights based on local neuron correlations while dynamically adjusting
    activity levels to maintain balance in high-dimensional or temporal input scenarios.

    Attributes:
        units (int): The number of neurons in the layer.
        learning_rate (float): The learning rate for Hebbian weight updates.
        target_avg (float): The target average activity level for homeostatic scaling.
        homeostatic_rate (float): The rate for adjusting activity scaling.
        activation (str or callable): The activation function for the layer output.
        min_scale (float): Minimum value for activity scaling.
        max_scale (float): Maximum value for activity scaling.
    """

    def __init__(self, units, learning_rate=1e-5, target_avg=0.1, homeostatic_rate=1e-5,
                 activation='gelu', min_scale=0.1, max_scale=2.0, momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.learning_rate = learning_rate
        self.target_avg = target_avg
        self.homeostatic_rate = homeostatic_rate
        self.activation = tf.keras.activations.get(activation)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.momentum = momentum
        self.ema_alpha = 0.1  # EMA smoothing factor for avg_activity

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.001)

        self.kernel = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=initializer,
            trainable=True,
            name='kernel'
        )

        self.activity_scale = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.Ones(),
            trainable=False,
            name='activity_scale'
        )

        # Initialize EMA for average activity
        self.avg_activity = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=False,
            name='avg_activity'
        )

    def call(self, inputs):
        original_shape = tf.shape(inputs)
        if len(inputs.shape) == 3:
            inputs = tf.reshape(inputs, [-1, inputs.shape[-1]])

        # Normalize inputs and weights
        inputs = tf.nn.l2_normalize(inputs, axis=-1)
        normalized_kernel = tf.nn.l2_normalize(self.kernel, axis=0)

        # Compute raw outputs and apply homeostatic scaling
        raw_outputs = tf.matmul(inputs, normalized_kernel)
        scaled_outputs = raw_outputs * self.activity_scale

        # Apply activation
        outputs = self.activation(scaled_outputs) if self.activation else scaled_outputs

        # Hebbian weight update
        if self.learning_rate > 0:
            delta_weights = tf.matmul(
                tf.transpose(inputs),
                raw_outputs
            ) * self.learning_rate / tf.cast(tf.shape(inputs)[0], tf.float32)

            # Momentum smoothing for weight updates
            delta_weights = self.momentum * delta_weights + (1 - self.momentum) * delta_weights

            # Update and normalize weights
            new_kernel = self.kernel + delta_weights
            self.kernel.assign(tf.clip_by_norm(new_kernel, 1.0))

        # Homeostatic scaling update
        batch_avg_activity = tf.reduce_mean(raw_outputs, axis=0)
        self.avg_activity.assign(
            self.ema_alpha * batch_avg_activity + (1 - self.ema_alpha) * self.avg_activity
        )

        scale_adjustment = self.homeostatic_rate * (self.target_avg - self.avg_activity)
        new_scale = tf.clip_by_value(
            self.activity_scale + scale_adjustment,
            self.min_scale,
            self.max_scale
        )
        self.activity_scale.assign(new_scale)

        # Reshape outputs back if inputs were reshaped
        if len(inputs.shape) == 3:
            outputs = tf.reshape(outputs, [original_shape[0], original_shape[1], self.units])

        return outputs

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 3:
            return (input_shape[0], input_shape[1], self.units)
        return (input_shape[0], self.units)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'learning_rate': self.learning_rate,
            'target_avg': self.target_avg,
            'homeostatic_rate': self.homeostatic_rate,
            'activation': tf.keras.activations.serialize(self.activation),
            'min_scale': self.min_scale,
            'max_scale': self.max_scale,
            'momentum': self.momentum
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['activation'] = tf.keras.activations.deserialize(config['activation'])
        return cls(**config)





class SpatioTemporalSummaryMixingLayer(Layer):
    """
    The SpatioTemporalSummaryMixingLayer enhances the processing of spatio-temporal data by integrating local and global context, 
    making it ideal for tasks like video processing and time-series forecasting. It addresses the challenge of efficiently 
    capturing long-range dependencies by combining GLU and GELU activations, enabling both local interactions and high-level summaries. 
    The optional weighted summary mechanism dynamically adjusts token importance, improving flexibility and performance. 
    This layer improves computational efficiency while maintaining the ability to process complex sequences, offering a scalable 
    solution for real-time applications.

    Attributes:
        d_model: Dimensionality of the model (output size).
        dropout_rate: Rate for dropout regularization.
        use_weighted_summary: Boolean to control the use of learnable summary weights.
    """
    
    def __init__(self, d_model, dropout_rate=0.1, use_weighted_summary=False, **kwargs):
        super(SpatioTemporalSummaryMixingLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.use_weighted_summary = use_weighted_summary

    def build(self, input_shape):
        # Local processing with GLU
        self.local_dense1 = Dense(4 * self.d_model)  # GLU will be applied here
        self.local_dense2 = Dense(self.d_model)
        self.local_dropout = Dropout(self.dropout_rate)

        # Summary processing with GELU
        self.summary_dense1 = Dense(4 * self.d_model, activation='gelu')
        self.summary_dense2 = Dense(self.d_model)
        self.summary_dropout = Dropout(self.dropout_rate)

        if self.use_weighted_summary:
            self.summary_weights = Dense(1, activation='softmax')  # Learnable weights

        # Combining layers with GELU
        self.combiner_dense1 = Dense(4 * self.d_model, activation='gelu')
        self.combiner_dense2 = Dense(self.d_model)
        self.combiner_dropout = Dropout(self.dropout_rate)

        # Dynamic dense layer (potentially using GLU for dynamic gating)
        self.dynamic_dense = Dense(self.d_model)
        
        # Layer normalization
        self.layer_norm = LayerNormalization(epsilon=1e-6)

        super(SpatioTemporalSummaryMixingLayer, self).build(input_shape)

    def call(self, inputs, training=False):
        # Apply GLU for local processing (using split for gating mechanism)
        local_output = self.local_dense1(inputs)
        local_output, gate = tf.split(local_output, 2, axis=-1)  # Split for GLU
        local_output = local_output * tf.sigmoid(gate)  # GLU activation: element-wise multiplication
        local_output = self.local_dense2(local_output)
        local_output = self.local_dropout(local_output, training=training)
        
        # Summary processing with GELU
        summary = self.summary_dense1(inputs)
        summary = self.summary_dense2(summary)
        summary = self.summary_dropout(summary, training=training)

        if self.use_weighted_summary:
            weights = self.summary_weights(summary)  # Learnable token weights
            weighted_summary = tf.reduce_sum(summary * weights, axis=1, keepdims=True)
        else:
            weighted_summary = tf.reduce_mean(summary, axis=1, keepdims=True)

        weighted_summary = tf.tile(weighted_summary, [1, tf.shape(inputs)[1], 1])
        
        # Combine local output and weighted summary
        combined = tf.concat([local_output, weighted_summary], axis=-1)
        output = self.combiner_dense1(combined)
        output = self.combiner_dense2(output)
        output = self.combiner_dropout(output, training=training)

        # Apply dynamic dense layer (optional GLU or GELU, you could try both)
        inputs = self.dynamic_dense(inputs)

        # Return the final output with layer normalization
        return self.layer_norm(inputs + output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate,
            'use_weighted_summary': self.use_weighted_summary,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            d_model=config['d_model'],
            dropout_rate=config['dropout_rate'],
            use_weighted_summary=config['use_weighted_summary'],
            **{key: value for key, value in config.items() if key not in ['d_model', 'dropout_rate', 'use_weighted_summary']}
        )




class SpatioTemporalSummarization(Layer):
    """
    The SpatioTemporalSummarization layer enhances the processing of spatio-temporal data by integrating local and 
    global context, making it ideal for tasks like video processing and time-series forecasting. It addresses the challenge 
    of efficiently capturing long-range dependencies by combining GLU and GELU activations, enabling both local interactions 
    and high-level summaries. The optional weighted summary mechanism dynamically adjusts token importance, improving 
    flexibility and performance. This layer improves computational efficiency while maintaining the ability to process 
    complex sequences, offering a scalable solution for real-time applications.

    Attributes:
        d_model: Dimensionality of the model (output size).
        dropout_rate: Rate for dropout regularization.
        use_weighted_summary: Boolean to control the use of learnable summary weights.
    """
    
    def __init__(self, d_model, dropout_rate=0.1, use_weighted_summary=False, **kwargs):
        super(SpatioTemporalSummarization, self).__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.use_weighted_summary = use_weighted_summary

    def build(self, input_shape):
        # Local processing with DenseGSER (replaces GLU)
        self.local_dense = DenseGSER(self.d_model)
        self.local_dropout = Dropout(self.dropout_rate)

        # Summary processing with DenseGSER (replaces GELU)
        self.summary_dense = DenseGSER(self.d_model)
        self.summary_dropout = Dropout(self.dropout_rate)

        if self.use_weighted_summary:
            self.summary_weights = Dense(self.d_model, activation='softmax')  # Learnable weights

        # Combining layers with DenseGSER
        self.combiner_dense = DenseGSER(self.d_model)
        self.combiner_dropout = Dropout(self.dropout_rate)

        # Dynamic dense layer with DenseGSER
        self.dynamic_dense = DenseGSER(self.d_model)
        
        # Layer normalization
        self.layer_norm = LayerNormalization(epsilon=1e-6)

        super(SpatioTemporalSummarization, self).build(input_shape)

    def call(self, inputs, training=False):
        # Apply DenseGSER for local processing
        local_output = self.local_dense(inputs)
        local_output = self.local_dropout(local_output, training=training)
        
        # Summary processing with DenseGSER
        summary = self.summary_dense(inputs)
        summary = self.summary_dropout(summary, training=training)

        if self.use_weighted_summary:
            weights = self.summary_weights(summary)  # Learnable token weights
            weighted_summary = tf.reduce_sum(summary * weights, axis=1, keepdims=True)
        else:
            weighted_summary = tf.reduce_mean(summary, axis=1, keepdims=True)

        weighted_summary = tf.tile(weighted_summary, [1, tf.shape(inputs)[1], 1])
        
        # Combine local output and weighted summary
        combined = tf.concat([local_output, weighted_summary], axis=-1)
        combined_output = self.combiner_dense(combined)
        combined_output = self.combiner_dropout(combined_output, training=training)

        # Apply DenseGSER for dynamic transformation
        inputs = self.dynamic_dense(inputs)

        # Return the final output with layer normalization
        return self.layer_norm(inputs + combined_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate,
            'use_weighted_summary': self.use_weighted_summary,
        })
        return config




class MultiheadLinearSelfAttentionKernalization(Layer):
    """
    A multi-head linear self-attention layer with kernel approximation. The MultiheadLinearSelfAttentionKernalization (MLSAK)
    replaces the quadratic QK^T computation of traditional mechanisms with positive activations and key normalization. 
    This approach achieves linear complexity O(n), addressing the inefficiencies of standard attention for long sequences 
    and enabling scalable, real-time processing without compromising performance.

    Attributes:
        d_model (int): The dimension of the model (input and output space).
        num_heads (int): The number of attention heads in the multi-head attention mechanism.
        dropout_rate (float): The rate of dropout to apply to the output of the attention mechanism to prevent overfitting.
        use_weighted_summary (bool): Whether to use a weighted summary of the attention output.
        eps (float): A small constant added for numerical stability.
    """
    def __init__(self, d_model, num_heads, dropout_rate=0.1, use_weighted_summary=False, eps=1e-6, **kwargs):
        super(MultiheadLinearSelfAttentionKernalization, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_weighted_summary = use_weighted_summary
        self.eps = eps

        # Ensure d_model is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads

        self.layer_norm = LayerNormalization(epsilon=eps)
        self.dropout = Dropout(self.dropout_rate)

    def build(self, input_shape):
        d_model = self.d_model

        # Initialize weights for Q, K, V projections
        self.query_weight = self.add_weight(
            name='query_weight',
            shape=(d_model, d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        self.query_bias = self.add_weight(
            name='query_bias',
            shape=(d_model,),
            initializer='zeros',
            trainable=True
        )

        self.key_weight = self.add_weight(
            name='key_weight',
            shape=(d_model, d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        self.key_bias = self.add_weight(
            name='key_bias',
            shape=(d_model,),
            initializer='zeros',
            trainable=True
        )

        self.value_weight = self.add_weight(
            name='value_weight',
            shape=(d_model, d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        self.value_bias = self.add_weight(
            name='value_bias',
            shape=(d_model,),
            initializer='zeros',
            trainable=True
        )

        self.output_weight = self.add_weight(
            name='output_weight',
            shape=(d_model, d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        self.output_bias = self.add_weight(
            name='output_bias',
            shape=(d_model,),
            initializer='zeros',
            trainable=True
        )

        if self.use_weighted_summary:
            self.summary_weight = self.add_weight(
                name='summary_weight',
                shape=(d_model, 1),
                initializer='glorot_uniform',
                trainable=True
            )
            self.summary_bias = self.add_weight(
                name='summary_bias',
                shape=(1,),
                initializer='zeros',
                trainable=True
            )

        # Explicitly build LayerNormalization
        self.layer_norm.build(input_shape)
        super(MultiheadLinearSelfAttentionKernalization, self).build(input_shape)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Linear projections
        queries = tf.matmul(inputs, self.query_weight) + self.query_bias
        keys = tf.matmul(inputs, self.key_weight) + self.key_bias
        values = tf.matmul(inputs, self.value_weight) + self.value_bias

        # Split heads
        queries = self.split_heads(queries, batch_size)
        keys = self.split_heads(keys, batch_size)
        values = self.split_heads(values, batch_size)

        # Apply kernel trick with ELU activation
        queries = tf.nn.elu(queries) + 1.0
        keys = tf.nn.elu(keys) + 1.0

        # Normalize keys
        key_norm = tf.sqrt(tf.reduce_sum(tf.square(keys), axis=-1, keepdims=True) + self.eps)
        keys = keys / key_norm

        # Compute attention scores
        scores = tf.einsum("bhqd,bhkd->bhqk", queries, keys)

        # Apply attention to values
        attention_output = tf.einsum("bhqk,bhvd->bhqd", scores, values)

        # Merge heads back
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, seq_len, self.d_model))

        # Optional weighted summary
        if self.use_weighted_summary:
            weights = tf.nn.sigmoid(tf.matmul(attention_output, self.summary_weight) + self.summary_bias)
            attention_output = attention_output * weights

        # Final linear projection
        output = tf.matmul(attention_output, self.output_weight) + self.output_bias
        output = self.dropout(output, training=training)

        # Residual connection and normalization
        return self.layer_norm(inputs + output)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(MultiheadLinearSelfAttentionKernalization, self).get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "use_weighted_summary": self.use_weighted_summary,
            "eps": self.eps,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)





class PositionalEncodingLayer(Layer):
    def __init__(self, max_position, d_model, **kwargs):
        super(PositionalEncodingLayer, self).__init__(**kwargs)
        self.max_position = max_position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(max_position, d_model)
        
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, max_position, d_model):
        angle_rads = self.get_angles(np.arange(max_position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'max_position': self.max_position,
            'd_model': self.d_model
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(config['max_position'], config['d_model'])





class LatentTemporalCoherence(Layer):
    """Distills a sequence of neural states into a single "thought vector" by measuring temporal coherence.

    This layer processes the output history of a recurrent layer (e.g., GSER with 
    `return_sequences=True`) to compute a fixed-size latent representation. It operates 
    by calculating a weighted inner product on the activation histories of sampled neuron pairs.
    A learnable decay rate for each pair allows the model to dynamically weigh the importance 
    of recent versus long-term temporal correlations.

    **Importance:**
    Instead of using only the final state of a recurrent layer, this provides a richer, 
    high-level summary of a neural system's entire temporal dynamics. This "coherence vector" 
    is crucial for tasks requiring reasoning about processes over time and is a core component 
    for modeling emergent consciousness in the A.R.C.A.N.E. architecture.

    Attributes:
        d_coherence (int): Dimensionality of the output coherence vector, which corresponds 
                         to the number of neuron pairs to sample.
    """
    def __init__(self, d_coherence, **kwargs):
        super(LatentTemporalCoherence, self).__init__(**kwargs)
        self.d_coherence = d_coherence

    def build(self, input_shape):
        num_neurons = input_shape[-1]
        if num_neurons is None:
            raise ValueError("The number of neurons (last dimension of input) must be defined for LatentTemporalCoherence layer.")

        # Generate all unique pairs of neuron indices (i, j) where i < j
        indices = tf.range(num_neurons)
        i_indices, j_indices = tf.meshgrid(indices, indices)
        mask = i_indices < j_indices
        pair_i = tf.boolean_mask(i_indices, mask)
        pair_j = tf.boolean_mask(j_indices, mask)
        
        num_possible_pairs = tf.shape(pair_i)[0]

        if self.d_coherence > num_possible_pairs:
            raise ValueError(f"d_coherence ({self.d_coherence}) cannot be greater than the number of unique neuron pairs ({num_possible_pairs}).")

        # We sample a fixed set of pairs once during layer creation.
        # These pairs will remain constant throughout training.
        shuffled_pair_indices = tf.random.shuffle(tf.range(num_possible_pairs))
        sampled_indices = shuffled_pair_indices[:self.d_coherence]

        self.neuron_pair_i = tf.gather(pair_i, sampled_indices)
        self.neuron_pair_j = tf.gather(pair_j, sampled_indices)

        # Create learnable decay rates for each sampled neuron pair.
        self.decay_rates = self.add_weight(
            name='decay_rates',
            shape=(self.d_coherence,),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            trainable=True
        )
        super(LatentTemporalCoherence, self).build(input_shape)

    def call(self, inputs):
        # Convert to float32 for computation to avoid mixed precision issues
        inputs = tf.cast(inputs, tf.float32)
        
        # inputs shape: (batch_size, num_ticks, num_neurons)
        num_ticks = tf.shape(inputs)[1]

        # 1. Gather the activation histories for the sampled pairs.
        # Shape of both: (batch_size, num_ticks, d_coherence)
        history_i = tf.gather(inputs, self.neuron_pair_i, axis=2)
        history_j = tf.gather(inputs, self.neuron_pair_j, axis=2)

        # 2. Create the exponential decay weights for the time dimension.
        # `tau` represents the time distance from the present.
        # For a history of length T, tau will be [T-1, T-2, ..., 0]
        tau = tf.range(tf.cast(num_ticks, tf.float32), 0, -1) - 1.0
        tau = tf.reshape(tau, (1, num_ticks, 1)) # Reshape for broadcasting

        # Constrain decay_rates to be positive using softplus
        r = tf.nn.softplus(self.decay_rates)
        r = tf.reshape(r, (1, 1, self.d_coherence)) # Reshape for broadcasting
        
        # Calculate decay weights: exp(-r_ij * tau)
        # Shape: (1, num_ticks, d_coherence)
        decay_weights = tf.exp(-r * tau)
        
        # 3. Calculate the weighted inner product over time.
        # Element-wise product of histories for each pair
        product_over_time = history_i * history_j
        
        # Apply the temporal decay weights
        weighted_product = product_over_time * decay_weights
        
        # Sum over the time dimension to get the final synchronization value for each pair
        # Shape: (batch_size, d_coherence)
        coherence_values = tf.reduce_sum(weighted_product, axis=1)

        # 4. Normalize the resulting coherence vector for each batch item.
        # This becomes the latent representation S_t
        denom = tf.sqrt(tf.reduce_sum(tf.square(coherence_values), axis=-1, keepdims=True) + 1e-8)
        S_t = coherence_values / denom
        
        return S_t

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.d_coherence)

    def get_config(self):
        config = super(LatentTemporalCoherence, self).get_config()
        config.update({
            'd_coherence': self.d_coherence,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



# A.R.C.A.N.E (Augmented Reconstruction of Consciousness through Artificial Neural Evolution)