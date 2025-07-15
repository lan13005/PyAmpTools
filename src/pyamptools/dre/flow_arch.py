import jax
import jax.numpy as jnp
from flax import nnx

class MADE(nnx.Module):
    """
    Masked Autoregressive Density Estimator for MAF implementation.
    
    Implements the autoregressive network that outputs the shift and scale
    parameters for the normalizing flow.
    """
    
    def __init__(self, input_dim, hidden_dims, *, rngs, reverse=False):
        """
        Initialize a MADE network.
        
        Args:
            input_dim: Input dimensionality
            hidden_dims: List of hidden layer dimensions
            rngs: nnx.Rngs object for initialization
            reverse: Whether to reverse the ordering of dependencies
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = input_dim * 2  # For mean and log_scale
        
        # Create masks to enforce autoregressive property
        self.masks = self._create_masks(reverse)

        # Initialize network parameters
        #   Define dimensions for (input, hidden, and output) layers
        layer_dims = [input_dim] + hidden_dims + [self.output_dim]
        
        # Recommended initializers
        he_init = nnx.initializers.variance_scaling(
            scale=2.0,
            mode='fan_in',
            distribution='truncated_normal'
        )
        zeros_init = nnx.initializers.zeros

        # Initialize weights and biases
        self.layers = []
        for l in range(len(layer_dims) - 1):
            w_key, b_key = rngs.split(2)
            W = nnx.Param(he_init(w_key, (layer_dims[l], layer_dims[l + 1])))
            b = nnx.Param(zeros_init(b_key, (layer_dims[l + 1],)))
            self.layers.append({
                'w': W,
                'b': b,
                'mask': self.masks[l]
            })
    
    def _create_masks(self, reverse):
        """
        Create the autoregressive masks for MADE.
        
        Args:
            reverse: Whether to reverse the order of dependencies
        
        Returns:
            List of masks for each layer
        """
        # Assign degrees to input nodes
        degrees = jnp.arange(1, self.input_dim + 1)
        if reverse:
            degrees = jnp.flip(degrees)
        
        masks = []
        
        # Input to first hidden layer
        m = jnp.zeros((self.input_dim, self.hidden_dims[0]))
        for i in range(self.input_dim):
            for j in range(self.hidden_dims[0]):
                m = m.at[i, j].set(degrees[i] <= j % self.input_dim + 1)
        masks.append(m)
        
        # Hidden to hidden layers
        for l in range(len(self.hidden_dims) - 1):
            m = jnp.zeros((self.hidden_dims[l], self.hidden_dims[l+1]))
            for i in range(self.hidden_dims[l]):
                for j in range(self.hidden_dims[l+1]):
                    m = m.at[i, j].set((i % self.input_dim + 1) <= j % self.input_dim + 1)
            masks.append(m)
        
        # Hidden to output layer
        # For MAF, output needs two parameters per input (shift and scale)
        m = jnp.zeros((self.hidden_dims[-1], self.output_dim))
        for i in range(self.hidden_dims[-1]):
            for j in range(self.input_dim):  # half for means
                m = m.at[i, j].set((i % self.input_dim + 1) < (j % self.input_dim + 1))
            for j in range(self.input_dim):  # half for log scales
                m = m.at[i, j + self.input_dim].set((i % self.input_dim + 1) < (j % self.input_dim + 1))
        masks.append(m)
        
        return masks
    
    def __call__(self, x):
        """
        Forward pass through the MADE network.
        
        Args:
            x: Input data of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (means, log_scales) for the autoregressive transformation
        """
        h = x
        
        # Apply all layers with leaky ReLU activation except for the last one
        for layer_idx, layer_params in enumerate(self.layers):
            w = layer_params['w'] * layer_params['mask']
            b = layer_params['b']
            
            h = jnp.dot(h, w) + b
            
            # Apply activation to all layers except the last one
            if layer_idx < len(self.layers) - 1:
                h = jax.nn.leaky_relu(h, negative_slope=0.01)
        
        # Split output into means and log_scales
        means = h[..., :self.input_dim]
        log_scales = h[..., self.input_dim:]
        
        # Constrain scales to prevent numerical issues
        log_scales = 2 * jnp.tanh(log_scales / 2)
        
        return means, jnp.exp(log_scales)

class MAF(nnx.Module):
    """
    Masked Autoregressive Flow implementation.
    
    This flow transforms between data space and latent space through a series
    of autoregressive transformations.
    """
    
    def __init__(self, input_dim, hidden_dims, num_flows, *, rngs):
        """
        Initialize a MAF model.
        
        Args:
            input_dim: Input dimensionality
            hidden_dims: List of hidden layer dimensions for each MADE
            num_flows: Number of flow layers
            rngs: nnx.Rngs object for initialization
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_flows = num_flows
        
        # Initialize layers
        made_rngs = rngs.split(num_flows)
        self.made_layers = []
        self.batch_norms = []
        
        for i in range(num_flows):
            # Alternate between forward and reverse ordering
            reverse = i % 2 == 1
            self.made_layers.append(MADE(input_dim, hidden_dims, rngs=made_rngs[i], reverse=reverse))
            if i < num_flows - 1:
                bn_rng = rngs.split(1)[0]
                self.batch_norms.append(nnx.BatchNorm(num_features=input_dim, rngs=bn_rng))
    
    def forward(self, x, training=True):
        """
        Transform data from input space to latent space (forward flow).
        
        Args:
            x: Input data of shape (batch_size, input_dim)
            training: Whether to use training mode for batch norm
            
        Returns:
            z: Transformed data in latent space
            log_det: Log determinant of the Jacobian
        """
        z = x
        log_det_sum = jnp.zeros(x.shape[0])
        
        for i in range(self.num_flows):
            # Get means and scales from MADE
            means, scales = self.made_layers[i](z)
            
            # Transform
            z = (z - means) / scales
            
            # Compute log determinant
            log_det = -jnp.sum(jnp.log(scales), axis=1)
            log_det_sum += log_det
            
            # Permute dimensions for the next flow (except for the last one)
            if i < self.num_flows - 1:
                z = self.batch_norms[i](z, use_running_average=not training)  # Apply BatchNorm
                z = jnp.flip(z, axis=-1) # Simple reversing permutation
        
        return z, log_det_sum
    
    def inverse(self, z, training=True):
        """
        Transform data from latent space to input space (inverse flow).
        
        Args:
            z: Latent variables of shape (batch_size, input_dim)
            training: Whether to use training mode for batch norm
            
        Returns:
            x: Transformed data in input space
            log_det: Log determinant of the Jacobian
        """
        x = z
        log_det_sum = jnp.zeros(z.shape[0])
        
        # Apply flows in reverse order
        for i in range(self.num_flows - 1, -1, -1):
            if i < self.num_flows - 1:
                x = jnp.flip(x, axis=-1)
                x = self.batch_norms[i].inverse(x, use_running_average=not training)
            
            # Get means and scales for this input
            means, scales = self.made_layers[i](x)
            
            # Inverse transform: z = (x - μ)/σ => x = z*σ + μ
            x = x * scales + means
            
            # Compute log determinant (positive for inverse transform)
            log_det = jnp.sum(jnp.log(scales), axis=1)
            log_det_sum += log_det
        
        return x, log_det_sum
    
    def log_prob(self, x, training=True):
        """
        Compute log probability of data under the MAF model.
        
        Args:
            x: Input data of shape (batch_size, input_dim)
            training: Whether to use training mode for batch norm
            
        Returns:
            A tuple of (log_prob, log_det):
                - log_prob: Total log probability of each input point
                - log_det: Log determinant of the Jacobian for each input
        """
        z, log_det = self.forward(x, training=training)
        # Log probability of standard normal distribution N(0,I)
        log_prior = -0.5 * jnp.sum(z**2, axis=1) - 0.5 * self.input_dim * jnp.log(2 * jnp.pi)
        log_prob = log_prior + log_det
        return log_prob, log_det

    # def _apply_made_layer(self, params, made_layer, x):
    #     """Apply a MADE layer with its parameters."""
    #     h = x
        
    #     # Apply all layers with ReLU activation except for the last one
    #     for layer_idx, layer_params in enumerate(params):
    #         w = layer_params['w'] * made_layer.masks[layer_idx]
    #         b = layer_params['b']
            
    #         # Check if we need to handle multi-device parameters.
    #         # - Weight matrix should only have 2 dimensions.
    #         # - Bias vector should only have 1 dimension.
    #         if w.ndim > 2: w = w[0]
    #         if b.ndim > 1: b = b[0]
            
    #         h = jnp.dot(h, w) + b
            
    #         # Apply ReLU to all layers except the last one
    #         if layer_idx < len(params) - 1:
    #             # h = jax.nn.relu(h)
    #             h = jax.nn.leaky_relu(h, negative_slope=0.01)
        
    #     # Split output into means and log_scales
    #     means = h[..., :self.input_dim]
    #     log_scales = h[..., self.input_dim:]
        
    #     # Constrain scales to prevent numerical issues
    #     # log_scales = jnp.clip(log_scales, -5, 5)
    #     log_scales = 2 * jnp.tanh(log_scales / 2) # could be more stable?
        
    #     return means, jnp.exp(log_scales)
    