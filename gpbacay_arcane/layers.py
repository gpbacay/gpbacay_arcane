import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization



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
        leak_rate (float): Rate at which the state of the reservoir decays over time.
        spike_threshold (float): Threshold above which a spike occurs in the reservoir neurons.
        max_dynamic_reservoir_dim (int): Maximum dynamic size of the reservoir.
        state_size (list): Size of the reservoir state.
        output_size (int): Size of the output of the reservoir layer.
    """

    def __init__(self, initial_reservoir_size, input_dim, spectral_radius, leak_rate, spike_threshold, max_dynamic_reservoir_dim, neurogenesis_rate=0.05, pruning_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.initial_reservoir_size = initial_reservoir_size
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_dynamic_reservoir_dim = max_dynamic_reservoir_dim
        self.neurogenesis_rate = neurogenesis_rate  # Rate of new neurons added
        self.pruning_rate = pruning_rate  # Rate of pruning connections
        
        self.state_size = [self.max_dynamic_reservoir_dim]  # Define the size of the state
        self.output_size = self.max_dynamic_reservoir_dim  # Output size is the dynamic reservoir size
        
        # Initialize weights and reservoirs
        self.initialize_weights()

    def initialize_weights(self):
        """Initializes the weights for the spatiotemporal reservoir, input weights, and spiking gate weights."""
        # Initialize reservoir weights (connections between neurons in the reservoir)
        self.spatiotemporal_reservoir_weights = self.add_weight(
            shape=(self.initial_reservoir_size, self.initial_reservoir_size),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=False,
            name='spatiotemporal_reservoir_weights'
        )
        
        # Initialize input weights for mapping the input to the reservoir
        self.spatiotemporal_input_weights = self.add_weight(
            shape=(self.initial_reservoir_size, self.input_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=False,
            name='spatiotemporal_input_weights'
        )
        
        # Initialize spiking gate weights
        self.spiking_gate_weights = self.add_weight(
            shape=(3 * self.initial_reservoir_size, self.input_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=False,
            name='spiking_gate_weights'
        )

    def add_neurons(self, new_neurons_count):
        """Add new neurons to the reservoir."""
        if self.initial_reservoir_size + new_neurons_count <= self.max_dynamic_reservoir_dim:
            # Update the reservoir size
            self.initial_reservoir_size += new_neurons_count
            self.state_size[0] = self.initial_reservoir_size  # Update the state size

            # Expand reservoir weights with appropriate padding
            new_row_block = tf.zeros([self.spatiotemporal_reservoir_weights.shape[0], new_neurons_count])
            new_col_block = tf.zeros([new_neurons_count, self.spatiotemporal_reservoir_weights.shape[1] + new_neurons_count])
            
            self.spatiotemporal_reservoir_weights = tf.concat([ 
                tf.concat([self.spatiotemporal_reservoir_weights, new_row_block], axis=1),
                new_col_block
            ], axis=0)

            # Reinitialize the new neurons' input connections
            new_input_weights = tf.zeros([new_neurons_count, self.input_dim])
            self.spatiotemporal_input_weights = tf.concat([
                self.spatiotemporal_input_weights,
                new_input_weights
            ], axis=0)

            # Reset the spiking gate weights for the new neurons
            new_gate_weights = tf.zeros([3 * new_neurons_count, self.input_dim])
            self.spiking_gate_weights = tf.concat([
                self.spiking_gate_weights,
                new_gate_weights
            ], axis=0)
    
    def prune_connections(self, pruning_threshold=0.1):
        """Prune connections with small weights, pruning weak or redundant synaptic connections."""
        # Mask weak or redundant connections based on a pruning threshold
        mask = tf.abs(self.spatiotemporal_reservoir_weights) < pruning_threshold
        self.spatiotemporal_reservoir_weights = tf.where(mask, tf.zeros_like(self.spatiotemporal_reservoir_weights), self.spatiotemporal_reservoir_weights)

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
        prev_state = states[0][:, :self.initial_reservoir_size]

        # Compute the input part (mapping inputs to the reservoir state)
        input_part = tf.matmul(inputs, self.spatiotemporal_input_weights, transpose_b=True)
        
        # Compute the reservoir part (feedback from the reservoir state)
        reservoir_part = tf.matmul(prev_state, self.spatiotemporal_reservoir_weights)
        
        # Compute the gate part (gating mechanism to control the input flow)
        gate_part = tf.matmul(inputs, self.spiking_gate_weights, transpose_b=True)

        # Split the gate part into three separate gates (input, forget, and output gates)
        i_gate, f_gate, o_gate = tf.split(tf.sigmoid(gate_part), 3, axis=-1)

        # Update the state using the gating mechanism and reservoir dynamics
        state = (1 - self.leak_rate) * (f_gate * prev_state) + self.leak_rate * tf.tanh(i_gate * (input_part + reservoir_part))
        state = o_gate * state

        # Generate spikes if the state exceeds the spike threshold
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        
        # If a spike occurs, reset the state by subtracting the spike threshold
        state = tf.where(spikes > 0, state - self.spike_threshold, state)

        # Padding the state to ensure the dynamic reservoir size is met
        padded_state = tf.concat([state, tf.zeros([tf.shape(state)[0], self.max_dynamic_reservoir_dim - tf.shape(state)[-1]])], axis=1)

        return padded_state, [padded_state]

    def get_config(self):
        """Returns the configuration of the layer, useful for model serialization."""
        config = super().get_config()
        config.update({
            'initial_reservoir_size': self.initial_reservoir_size,
            'input_dim': self.input_dim,
            'spectral_radius': self.spectral_radius,
            'leak_rate': self.leak_rate,
            'spike_threshold': self.spike_threshold,
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
        return (input_shape[0], self.units)

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
    The Relational Concept Modeling (RCM) mechanism effectively captures and models hierarchical relationships among 
    conceptual representations in sequential or structured data by integrating multi-head self-attention with 
    spiking-inspired DenseGSER layers. It addresses the challenge of efficiently summarizing token-level interactions while 
    refining high-level concept relationships through attention-based extraction, dynamic concept pooling, and interaction modeling. 
    This mechanism enhances the ability to process complex spatio-temporal and hierarchical data, improving both 
    computational efficiency and the expressiveness of conceptual representations. 
    It is designed for tasks such as reasoning, relationship extraction, and structured data understanding, 
    offering a scalable solution for handling large or structured input sequences.
    
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
        self.attention_layer = MultiheadLinearSelfAttentionKernalizationLayer(
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
        self.interaction_attention = MultiheadLinearSelfAttentionKernalizationLayer(
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





class RelationalGraphAttentionReasoning(Layer):
    """
    RelationalGraphAttentionReasoning (RGAR) is a custom Keras layer designed to address the challenges of relational reasoning 
    in graph-based data by leveraging attention mechanisms for efficient message passing and task-specific predictions. 
    It combines the benefits of graph neural networks (GNNs) and multi-head attention to process relational embeddings 
    derived from prior layers, such as RelationalConceptModeling (RCM). 
    Unlike traditional GNNs that rely on fixed aggregation functions, RGAR introduces dynamic, 
    attention-driven reasoning to enhance expressiveness and adaptability. 
    Its novelty lies in the integration of DenseGSER, a spiking-inspired dense layer, 
    for precise, low-latency output generation, offering improved performance and efficiency for 
    relational reasoning tasks in neuromorphic and attention-driven architectures.
    """
    def __init__(self, d_model, num_heads, num_classes, **kwargs):
        super(RelationalGraphAttentionReasoning, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_classes = num_classes

        # Relational Entity Graph message passing layer (GNN-like operation)
        self.message_passing_layer = MultiheadLinearSelfAttentionKernalizationLayer(
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

    def build(self, input_shape):
        """
        Build the internal components of the model.
        """
        # Call the build methods of child layers to ensure all variables are initialized
        self.message_passing_layer.build(input_shape)
        message_passing_output_shape = (input_shape[0], self.d_model)
        self.output_layer.build(message_passing_output_shape)

    def call(self, inputs, training=None):
        """
        Forward pass of the model.
        """
        # Step 1: Perform message passing on relational graph using the output from RCM
        graph_relations = self.message_passing_layer(inputs, training=training)

        # Step 2: Produce task-specific predictions
        output = self.output_layer(graph_relations)
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





class HebbianHomeostaticNeuroplasticity(Layer):
    """
    The HebbianHomeostaticNeuroplasticity integrates Hebbian learning with homeostatic scaling to stabilize neural activity.
    It adapts the synaptic weights based on local neuron correlations, while dynamically adjusting the activity level
    to maintain balance in high-dimensional or temporal input scenarios. This approach provides self-regulating
    neural networks that do not rely on reward-based mechanisms, enhancing unsupervised learning and efficiency.

    Attributes:
        units (int): The number of units (neurons) in the layer.
        learning_rate (float): The learning rate for the Hebbian weight update.
        target_avg (float): The target average activity level for homeostatic scaling.
        homeostatic_rate (float): The rate at which activity scaling is adjusted.
        activation (str or function): The activation function to apply to the output.
    """
    
    def __init__(self, units, learning_rate=0.00001, target_avg=0.1, homeostatic_rate=0.00001, activation='gelu', **kwargs):
        super(HebbianHomeostaticNeuroplasticity, self).__init__(**kwargs)
        self.units = units
        self.learning_rate = learning_rate
        self.target_avg = target_avg
        self.homeostatic_rate = homeostatic_rate
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # Handle input shape properly
        if len(input_shape) == 3:  # (batch, timesteps, features)
            feature_dim = input_shape[-1]
        else:  # (batch, features)
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
            trainable=False,  # Homeostatic scaling is not trainable by gradients
            name='activity_scale'
        )
        self.built = True

    def call(self, inputs):
        # Handle input shape
        original_shape = tf.shape(inputs)
        if len(inputs.shape) == 3:
            # Reshape (batch, timesteps, features) to (batch * timesteps, features)
            inputs = tf.reshape(inputs, [-1, inputs.shape[-1]])

        # Normalize inputs
        inputs = tf.nn.l2_normalize(inputs, axis=-1)

        # Normalize weights
        normalized_kernel = tf.nn.l2_normalize(self.kernel, axis=0)

        # Compute outputs
        raw_outputs = tf.matmul(inputs, normalized_kernel)

        # Apply homeostatic scaling
        outputs = raw_outputs * self.activity_scale

        # Apply optional activation
        if self.activation is not None:
            outputs = self.activation(outputs)

        # Hebbian update
        if self.learning_rate > 0:
            # Compute hebbian update
            delta_weights = tf.matmul(
                tf.transpose(inputs), 
                raw_outputs
            ) * self.learning_rate / tf.cast(tf.shape(inputs)[0], tf.float32)
            
            # Update weights with normalization
            new_kernel = self.kernel + delta_weights
            self.kernel.assign(tf.clip_by_norm(new_kernel, 1.0))

        # Homeostatic update
        avg_activity = tf.reduce_mean(raw_outputs, axis=0)
        scale_adjustment = self.homeostatic_rate * (self.target_avg - avg_activity)
        new_scale = tf.clip_by_value(
            self.activity_scale + scale_adjustment,
            0.1,
            2.0
        )
        self.activity_scale.assign(new_scale)

        # Reshape output back to original shape if necessary
        if len(inputs.shape) == 3:
            outputs = tf.reshape(outputs, [original_shape[0], original_shape[1], self.units])

        return outputs

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 3:  # (batch, timesteps, features)
            return (input_shape[0], input_shape[1], self.units)
        return (input_shape[0], self.units)  # (batch, units)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'learning_rate': self.learning_rate,
            'target_avg': self.target_avg,
            'homeostatic_rate': self.homeostatic_rate,
            'activation': tf.keras.activations.serialize(self.activation),
        })
        return config




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




class MultiheadLinearSelfAttentionKernalizationLayer(Layer):
    """
    A multi-head linear self-attention layer with kernel approximation. The MultiheadLinearSelfAttentionKernalizationLayer (MLSAK)
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
        super(MultiheadLinearSelfAttentionKernalizationLayer, self).__init__(**kwargs)
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
        super(MultiheadLinearSelfAttentionKernalizationLayer, self).build(input_shape)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]

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
        attention_output = tf.reshape(attention_output, (batch_size, -1, self.d_model))

        # Optional weighted summary
        if self.use_weighted_summary:
            weights = tf.nn.sigmoid(tf.matmul(attention_output, self.summary_weight) + self.summary_bias)
            attention_output = attention_output * weights

        # Final linear projection
        output = tf.matmul(attention_output, self.output_weight) + self.output_bias
        output = self.dropout(output, training=training)

        # Residual connection and normalization
        return self.layer_norm(inputs + output)

    def get_config(self):
        config = super(MultiheadLinearSelfAttentionKernalizationLayer, self).get_config()
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




class GatedMultiheadLinearSelfAttentionKernalization(Layer):
    """
    A gated multi-head linear self-attention layer with kernel approximation. This layer extends the traditional
    attention mechanism by introducing a gating mechanism that dynamically modulates the flow of information
    through the attention outputs and weighted summary. By combining kernel approximation with gating,
    the layer achieves linear time complexity O(n), making it efficient for long sequences while enhancing
    its adaptability and robustness.
    
    MLSAK vs. GMLSAK:
    MLSAK is simpler and more direct, focusing on optimizing attention for long sequences with kernel approximations 
    and key normalization. It is a more efficient approach when scalability is prioritized.
    
    GMLSAK, on the other hand, enhances adaptability and robustness with the introduction of gating mechanisms. 
    This layer may be more appropriate when the model requires dynamic control over attention outputs and summaries, 
    making it more flexible but computationally more complex.
    """

    def __init__(self, d_model, num_heads, dropout_rate=0.1, use_weighted_summary=False, eps=1e-6, **kwargs):
        super(GatedMultiheadLinearSelfAttentionKernalization, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_weighted_summary = use_weighted_summary
        self.eps = eps

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads

    def build(self, input_shape):
        # Query, Key, and Value projection layers for multi-head attention
        self.query_dense = DenseGSER(self.d_model)
        self.key_dense = DenseGSER(self.d_model)
        self.value_dense = DenseGSER(self.d_model)

        # Output projection layer
        self.output_dense = DenseGSER(self.d_model)

        # Dropout layer
        self.dropout = Dropout(self.dropout_rate)

        # Optional weighted summary
        if self.use_weighted_summary:
            self.summary_weights = DenseGSER(1, activation="softmax")

        # Gating mechanisms
        self.gate_attention_dense = DenseGSER(self.d_model)
        self.gate_summary_dense = DenseGSER(self.d_model)

        # Layer normalization
        self.layer_norm = LayerNormalization(epsilon=self.eps)

        super(GatedMultiheadLinearSelfAttentionKernalization, self).build(input_shape)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth) and transpose for parallel processing.
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # Shape: (batch_size, num_heads, seq_len, depth)

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]

        # Project inputs to query, key, and value spaces
        queries = self.query_dense(inputs)
        keys = self.key_dense(inputs)
        values = self.value_dense(inputs)

        # Split into multiple heads
        queries = self.split_heads(queries, batch_size)
        keys = self.split_heads(keys, batch_size)
        values = self.split_heads(values, batch_size)

        # Apply kernel trick for linear attention
        queries = tf.nn.elu(queries) + 1
        keys = tf.nn.elu(keys) + 1

        # Compute linear attention
        key_sum = tf.reduce_sum(keys, axis=2, keepdims=True)
        scores = tf.einsum("bhqd,bhkd->bhqk", queries, keys) / (key_sum + self.eps)
        attention_output = tf.einsum("bhqk,bhvd->bhqd", scores, values)

        # Merge heads back
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, -1, self.d_model))

        # Optionally extract a weighted summary
        if self.use_weighted_summary:
            weights = self.summary_weights(attention_output)
            weighted_summary = tf.reduce_sum(attention_output * weights, axis=1, keepdims=True)
        else:
            weighted_summary = tf.reduce_mean(attention_output, axis=1, keepdims=True)

        # Expand summary to match sequence length
        weighted_summary = tf.tile(weighted_summary, [1, tf.shape(inputs)[1], 1])

        # Apply gating mechanism
        gate_attention = tf.sigmoid(self.gate_attention_dense(inputs))
        gated_output = gate_attention * attention_output

        gate_summary = tf.sigmoid(self.gate_summary_dense(inputs))
        gated_summary = gate_summary * weighted_summary

        # Combine gated outputs
        combined_output = tf.concat([gated_output, gated_summary], axis=-1)
        combined_output = self.output_dense(combined_output)
        combined_output = self.dropout(combined_output, training=training)

        # Residual connection and layer normalization
        inputs_projected = self.query_dense(inputs)
        return self.layer_norm(inputs_projected + combined_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "use_weighted_summary": self.use_weighted_summary,
            "eps": self.eps,
        })
        return config




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