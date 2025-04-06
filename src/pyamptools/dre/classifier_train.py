import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from pyamptools.dre.classifier_loss import likelihood_ratio_grad_regularized_loss_fn, convert_to_probabilities

@nnx.jit
def classifier_train_step(model, optimizer, batch, loss_type_code=0, 
               reg_strength=0.0, transition_sensitivity=0.5):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(likelihood_ratio_grad_regularized_loss_fn, has_aux=True)
    (loss, (logits, main_loss, grad_loss)), grads = grad_fn(
        model, batch, loss_type_code, reg_strength, transition_sensitivity
    )    
    optimizer.update(grads)
    probs = convert_to_probabilities(logits, loss_type_code)
    accuracy = jnp.mean((probs > 0.5) == batch['y'])
    
    return loss, accuracy, main_loss, grad_loss

# TODO: Do I need to set model.eval() here?
@nnx.jit
def classifier_eval_step(model, batch, loss_type_code=0, 
              reg_strength=0.0, transition_sensitivity=0.5):
    """Evaluate for a single step."""
    loss, (logits, main_loss, grad_loss) = likelihood_ratio_grad_regularized_loss_fn(
        model, batch, loss_type_code, reg_strength, transition_sensitivity
    )
    
    # Calculate accuracy
    probs = convert_to_probabilities(logits, loss_type_code)
    accuracy = jnp.mean((probs > 0.5) == batch['y'])
    
    return loss, accuracy, main_loss, grad_loss, probs

@nnx.jit
def classifier_predict(model, x, loss_type_code=0):
    """Generate predictions from the model."""
    logits = model(x)
    return convert_to_probabilities(logits, loss_type_code)

class DensityRatioEstimator(nnx.Module):
    
    def __init__(self, dims, dropout_rate=0.2, rngs=None):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.layers = []
        for din, dout in zip(dims[:-1], dims[1:]):
            self.layers.append(nnx.Linear(din, dout, rngs=rngs))
        self.dropout = nnx.Dropout(rate=self.dropout_rate, rngs=rngs)
        self.output_layer = nnx.Linear(dims[-1], 1, rngs=rngs)

    def __call__(self, x):
        for layer in self.layers:
            x = nnx.relu(self.dropout(layer(x)))
        return self.output_layer(x)

class DataLoader:
    """
    Iterator that yields batches across multiple epochs.
    
    This class handles data shuffling, batching, padding, and sharding across multiple devices for JAX-based training.
    """
    def __init__(self, X_train, y_train, weights_train, batch_size, num_devices, 
                 data_sharding_2d, data_sharding_1d, num_epochs, start_epoch=0):
        """
        Initialize the iterator.
        
        Args:
            X_train: Training features
            y_train: Training labels
            weights_train: Training weights
            batch_size: Desired batch size
            num_devices: Number of devices for parallelization
            data_sharding_2d: JAX Sharding spec for 2D arrays
            data_sharding_1d: JAX Sharding spec for 1D arrays
            num_epochs: Total number of epochs to iterate
            start_epoch: Starting epoch (for resuming training)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.weights_train = weights_train
        self.batch_size = batch_size
        self.num_devices = num_devices
        self.data_sharding_2d = data_sharding_2d
        self.data_sharding_1d = data_sharding_1d
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        
        # Calculate effective batch size (divisible by num_devices)
        if batch_size % num_devices != 0:
            self.effective_batch_size = (batch_size // num_devices + 1) * num_devices
        else:
            self.effective_batch_size = batch_size
        
        self.num_batches = int(np.ceil(len(X_train) / self.effective_batch_size))
    
    def __iter__(self):
        """Return iterator over epochs and batches."""
        for epoch in range(self.start_epoch, self.num_epochs):
            # Shuffle data for this epoch
            indices = np.random.permutation(len(self.X_train))
            X_shuffled = self.X_train[indices]
            y_shuffled = self.y_train[indices]
            weights_shuffled = self.weights_train[indices]
            
            # Signal start of new epoch
            yield {'epoch_start': epoch, 'num_batches': self.num_batches}
            
            # Generate batches for this epoch
            for batch_idx in range(self.num_batches):
                start_idx = batch_idx * self.effective_batch_size
                end_idx = min((batch_idx + 1) * self.effective_batch_size, len(self.X_train))
                
                # Get batch data
                batch_x = X_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]
                batch_weights = weights_shuffled[start_idx:end_idx]
                
                # Pad batch to be divisible by num_devices if needed
                if len(batch_x) % self.num_devices != 0:
                    pad_size = self.num_devices - (len(batch_x) % self.num_devices)
                    batch_x = np.concatenate([batch_x, batch_x[:pad_size]])
                    batch_y = np.concatenate([batch_y, batch_y[:pad_size]])
                    batch_weights = np.concatenate([batch_weights, batch_weights[:pad_size]])
                
                # Shard the data across devices based on array rank
                batch_x = jax.device_put(batch_x, self.data_sharding_2d)
                batch_y = jax.device_put(batch_y, self.data_sharding_1d)
                batch_weights = jax.device_put(batch_weights, self.data_sharding_1d)
                
                # Create batch dictionary with metadata
                batch = {
                    'x': batch_x,
                    'y': batch_y,
                    'weights': batch_weights,
                    'batch_idx': batch_idx,
                    'epoch': epoch
                }
                
                yield batch
            
            # Signal end of epoch
            yield {'epoch_end': epoch}