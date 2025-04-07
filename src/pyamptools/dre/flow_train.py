import numpy as np
import jax.numpy as jnp
from flax import nnx
import jax
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pickle
from .flow_arch import MAF
from .flow_utils import create_corner_plot
from .diagnostics import create_entropy_plot, create_logdet_plot

def weighted_maximum_likelihood_loss(params, maf, x, sample_weights=None, training=True):
    """
    Compute negative log likelihood loss, optionally weighted.
    
    Args:
        params: MAF model parameters
        maf: MAF model
        x: Input data
        sample_weights: Optional weights for each sample
    
    Returns:
        Negative log likelihood (weighted if weights provided)
    """
    log_probs, _ = maf.log_prob(params, x, training=training)
    
    if sample_weights is not None:
        # Use normalized weights to make loss function ~scale-invariant
        #   Optimizers are sensitive to size of gradients
        normalized_weights = sample_weights / jnp.sum(sample_weights)
        return -jnp.sum(log_probs * normalized_weights)
    else:
        return -jnp.mean(log_probs)

def batch_transform_maf(maf, params, data, batch_size=1024, direction="forward", rngs=None, training=False):
    """
    Evaluation Mode: Transform data using MAF in batches to avoid memory issues.
    
    Args:
        maf: The MAF model
        params: Model parameters
        data: Input data to transform
        batch_size: Size of batches to process
        direction: "forward" for data->latent or "inverse" for latent->data
        rngs: Optional PRNG streams for sampling
        
    Returns:
        Transformed data and log determinants
    """
    
    # Convert input to numpy for easier handling
    data_np = np.array(data)
    data_size = len(data_np)
    
    # Initialize output arrays
    result = np.zeros((data_size, data_np.shape[1]), dtype=np.float32)
    log_dets = np.zeros(data_size, dtype=np.float32)
    
    # Process in batches
    for i in range(0, data_size, batch_size):
        end_idx = min(i + batch_size, data_size)
        batch = jnp.array(data_np[i:end_idx])
        
        # Transform batch
        if direction == "forward":
            batch_result, batch_log_dets = maf.forward(params, batch, training=training)
        elif direction == "inverse":
            batch_result, batch_log_dets = maf.inverse(params, batch, training=training)
        elif direction == "sample" and rngs is not None:
            raise NotImplementedError("Sampling not implemented yet")
            # TODO: I think we need to pass in the nnx.rngs object to streams could be extracted
            #       for each batch
            # batch_size_actual = end_idx - i
            # batch_result, batch_log_dets = maf.sample(params, rngs, batch_size_actual)
        else: 
            raise ValueError(f"Unknown direction: {direction}")
            
        # Store results
        result[i:end_idx]   = np.array(batch_result)
        log_dets[i:end_idx] = np.array(batch_log_dets)

    return result, log_dets

def fit_maf(maf, params, data, rngs, batch_size=1024, learning_rate=1e-3, weight_decay=1e-6, num_epochs=100, patience=10,
           sample_weights=None, plot_latent_frequency=None, diagnostic_batch_sample_percentage=5, checkpoint_dir=None,
           feature_names=None, X_gen=None, X_acc=None, use_gpu=False,
           clip_norm=0.5, adam_b1=0.9, adam_b2=0.999, adam_eps=1e-8):
    
    """
    Training Mode: Fit the MAF model to the data.
    
    Args:
        maf: The MAF model
        params: Model parameters
        data: Input data for the model to train on
        rngs: nnx.rngs object for random number generation
        batch_size: Size of batches to process
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for the optimizer
        num_epochs: Number of epochs to train for
        patience: Patience for early stopping (number of epochs to wait before stopping if no improvement)
        sample_weights: Sample weights for the data
        plot_latent_frequency: Frequency (in epochs) of plotting diagnostics
        diagnostic_batch_sample_percentage: Percentage of batches to sample for logging logdet and entropy statistics
        checkpoint_dir: Directory to save checkpoints
        feature_names: Feature names for the data
        X_gen: Generated data for diagnostic plots
        X_acc: Accepted data for diagnostic plots
        use_gpu: Whether to use GPU for training
        clip_norm: Clip norm for the optimizer
        adam_b1: Adam beta 1 for the optimizer
        adam_b2: Adam beta 2 for the optimizer
        adam_eps: Adam epsilon for the optimizer
        
    Returns:
        Dictionary containing the trained MAF model, parameters, and losses
    """

    # Create optimizer with AdamW (includes weight decay)
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adamw(learning_rate, b1=adam_b1, b2=adam_b2, eps=adam_eps, weight_decay=weight_decay)
    )    
    opt_state = optimizer.init(params)
    
    # Get data size and prepare for batching
    data_size = data.shape[0]
    if batch_size == -1: # -1 is reserved for batch size = data size (if your data is smaller relative to RAM size)
        steps_per_epoch = 1
        batch_size = data_size
    else:
        steps_per_epoch = data_size // batch_size
        last_batch_size = data_size - (steps_per_epoch * batch_size)
        if last_batch_size > 0:
            steps_per_epoch += 1
    
    epoch_losses = [] # track losses for plotting
    logdet_stats = {
        'median': [],
        'percentile_2.5': [],
        'percentile_16': [],
        'percentile_84': [],
        'percentile_97.5': [],
        'epoch': []
    }  # track logdet statistics for plotting
    
    # Check for GPU availability
    if use_gpu and jax.devices('gpu'):
        print(f"Training MAF using GPU: {jax.devices('gpu')[0]}")
        # Define a jitted update function for single GPU
        @nnx.jit
        def update_step(params, opt_state, batch_x, batch_weights):
            def loss_fn(p):
                return weighted_maximum_likelihood_loss(p, maf, batch_x, batch_weights, training=False)
            
            loss_value, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss_value
    else:
        # Get number of devices for parallelization
        num_devices = jax.device_count()
        print(f"Training MAF using {num_devices} CPU cores")
        
        # Define a pmapped update function to parallelize across devices
        @nnx.pmap
        def update_step_parallel(params_device, opt_state_device, batch_x_device, batch_weights_device):
            def loss_fn(p):
                return weighted_maximum_likelihood_loss(p, maf, batch_x_device, batch_weights_device, training=False)
            
            loss_value, grads = jax.value_and_grad(loss_fn)(params_device)
            updates, new_opt_state = optimizer.update(grads, opt_state_device, params_device)
            new_params = optax.apply_updates(params_device, updates)
            return new_params, new_opt_state, loss_value
        
        # Replicate parameters and optimizer state across devices
        params = jax.device_put_replicated(params, jax.devices())
        opt_state = jax.device_put_replicated(opt_state, jax.devices())
    
    progress_bar = tqdm(total=num_epochs, desc=f"Training MAF")
    
    best_loss = float('inf')
    patience_counter = 0
    
    batch_sample_frequency = max(1, int(steps_per_epoch * diagnostic_batch_sample_percentage / 100))
    print(f"Sampling metrics for diagnostic plots (logdet) every {batch_sample_frequency} batches")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_logdets = []
        
        # Create a new random permutation for each epoch
        perm = jax.random.permutation(rngs['permutation'](), data_size)
        
        # Process each batch
        for idx in range(steps_per_epoch):
            
            # Calculate batch start/end indices
            start_idx = idx * batch_size
            end_idx = min(start_idx + batch_size, data_size)
            actual_batch_size = end_idx - start_idx            
            batch_idx = perm[start_idx:end_idx]
            batch_x = data[batch_idx]
            
            # Extract batch weights if provided
            batch_weights = None
            if sample_weights is not None:
                batch_weights = sample_weights[batch_idx]
            else:
                batch_weights = jnp.ones(actual_batch_size)
            
            if use_gpu and jax.devices('gpu'):
                # Single GPU update
                params, opt_state, loss_value = update_step(
                    params, opt_state, batch_x, batch_weights
                )
            else:
                # Multi-CPU update
                # Pad batch to be divisible by number of devices
                if actual_batch_size % num_devices != 0:
                    pad_size = num_devices - (actual_batch_size % num_devices)
                    batch_x = jnp.pad(batch_x, ((0, pad_size), (0, 0)), mode='constant')
                    batch_weights = jnp.pad(batch_weights, (0, pad_size), mode='constant')
                
                # Reshape for pmap
                batch_x_shaped = batch_x.reshape(num_devices, -1, batch_x.shape[1])
                batch_weights_shaped = batch_weights.reshape(num_devices, -1)
                
                # Update model in parallel across devices
                params, opt_state, losses_device = update_step_parallel(
                    params, opt_state, batch_x_shaped, batch_weights_shaped
                )
                
                # Average loss across devices
                loss_value = jnp.mean(losses_device)
            
            # Check for NaN loss and break if detected
            if jnp.isnan(loss_value):
                print(f"\nWARNING: NaN loss detected at epoch {epoch+1}, batch {idx+1}. Stopping training.")
                progress_bar.close()
                
                # Get the parameters from the last stable iteration
                if not use_gpu or not jax.devices('gpu'):
                    params = jax.tree_map(lambda x: x[0], params)
                
                maf_trained = MAF(maf.input_dim, maf.hidden_dims, maf.num_flows, rngs=rngs)
                maf_trained.made_layers = maf.made_layers  # Copy the architecture
                
                return {"maf": maf_trained, "params": params, "losses": epoch_losses, "logdet_stats": logdet_stats}
            
            epoch_loss += loss_value
            
            # Collect statistics for a subset of batches
            if idx % batch_sample_frequency == 0:
                subset_size = min(256, actual_batch_size)
                if use_gpu and jax.devices('gpu'):
                    # For GPU, just take the first subset_size samples
                    subset_x = batch_x[:subset_size]
                    
                    # Get log probabilities and log determinants
                    log_probs, batch_logdets = maf.log_prob(params, subset_x, training=False)
                    epoch_logdets.extend(jnp.array(batch_logdets))                    
                else:
                    # For multi-CPU, extract data from the first device
                    current_params = jax.tree_map(lambda x: x[0], params)
                    subset_x = batch_x_shaped[0, :subset_size]
                    
                    # Get log probabilities and log determinants in one call
                    log_probs, batch_logdets = maf.log_prob(current_params, subset_x, training=False)
                    epoch_logdets.extend(jnp.array(batch_logdets))
                                
        # Record average epoch loss
        avg_loss = epoch_loss / steps_per_epoch
        epoch_losses.append(float(avg_loss))
        
        # Record logdet statistics for this epoch
        if epoch_logdets:
            epoch_logdets = jnp.array(epoch_logdets)
            logdet_stats['median'].append(float(jnp.median(epoch_logdets)))
            logdet_stats['percentile_2.5'].append(float(jnp.percentile(epoch_logdets, 2.5)))
            logdet_stats['percentile_16'].append(float(jnp.percentile(epoch_logdets, 16)))
            logdet_stats['percentile_84'].append(float(jnp.percentile(epoch_logdets, 84)))
            logdet_stats['percentile_97.5'].append(float(jnp.percentile(epoch_logdets, 97.5)))
            logdet_stats['epoch'].append(epoch+1)
                
        # Update progress bar with epoch info (only once per epoch)
        progress_bar.update(1)
        progress_bar.set_description(f"Training MAF [Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}]")
        
        # Print epoch summary
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"\nEpoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_params = jax.tree_map(lambda x: jnp.copy(x), params) # save best model
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}. No improvement for {patience} epochs.")
            break
        
        # Generate diagnostic plots if requested
        if X_gen is not None and X_acc is not None:
            if plot_latent_frequency is not None and checkpoint_dir is not None and (epoch + 1) % plot_latent_frequency == 0:
                try:
                    # Get current parameters
                    current_params = params 
                    if not use_gpu or not jax.devices('gpu'):
                        current_params = jax.tree_map(lambda x: x[0], params)
                    
                    # Use batched transformation instead of all-at-once
                    print(f"Transforming subsampled data to latent space...")
                    X_gen_latent_np, _ = batch_transform_maf(
                        maf, current_params, X_gen, batch_size=1000, direction="forward"
                    )
                    X_acc_latent_np, _ = batch_transform_maf(
                        maf, current_params, X_acc, batch_size=1000, direction="forward"
                    )
                    
                    # Create corner plot for transformed data
                    print(f"\nCreating corner plot at epoch {epoch+1}...")
                    create_corner_plot(
                        X_gen_latent_np, X_acc_latent_np,
                        labels=['Generated', 'Accepted'],
                        feature_names=feature_names,
                        filename=f'latent_corner_plot_epoch_{epoch+1}.png',
                        checkpoint_dir=checkpoint_dir
                    )
                           
                except Exception as e:
                    print(f"Error creating diagnostic plots: {e}")
                finally:
                    # Free memory explicitly
                    X_gen_latent_np = None
                    X_acc_latent_np = None
                    plt.close('all')
                    
            if checkpoint_dir is not None:
                # These are cheap to save, attempt to save every epoch
                plt.figure(figsize=(10, 6))
                plt.plot(epoch_losses)
                min_loss = min(epoch_losses[0], epoch_losses[-1])
                max_loss = max(epoch_losses[0], epoch_losses[-1])
                plt.ylim(min_loss*1.15, max_loss*0.85)
                plt.xlabel('Epoch')
                plt.ylabel('Negative Log-Likelihood')
                plt.title('MAF Training Loss')
                plt.grid(True, alpha=0.3)
                plt.savefig(f'{checkpoint_dir}/maf_loss.png')
                plt.close()
                
                # Add logdet statistics plot
                if logdet_stats:
                    try:
                        create_logdet_plot(logdet_stats, checkpoint_dir, f"maf_logdet.png")
                    except Exception as e:
                        print(f"Error creating logdet plot: {e}")
    
    progress_bar.close()
    
    # Get final parameters (extract from first device if using multi-CPU)
    if not use_gpu or not jax.devices('gpu'):
        params = jax.tree_map(lambda x: x[0], params)
    
    maf_trained = MAF(maf.input_dim, maf.hidden_dims, maf.num_flows, rngs=rngs)
    maf_trained.made_layers = maf.made_layers  # Copy the architecture
    
    # Parameters can be replicated across devices (for multi-CPU training)
    #   We have to unreplicate them before saving
    def unreplicate_params(params):
        """Fully unreplicate parameters no matter how many levels of replication exist."""
        def _unreplicate(x):
            while hasattr(x, 'ndim') and x.ndim > 2:
                x = x[0]
            return x
        return jax.tree_map(_unreplicate, params)

    # Use this before saving or returning
    best_params = unreplicate_params(best_params)
    
    # Create a final logdet plot if we have data
    if logdet_stats and checkpoint_dir:
        create_logdet_plot(logdet_stats, checkpoint_dir)
        
    # Return the best model parameters with added stats
    return {"maf": maf_trained, "params": best_params, "losses": epoch_losses, 
            "logdet_stats": logdet_stats}

def save_maf_model(maf_result, filepath):
    """
    Save MAF model to disk.
    
    Args:
        maf_result: Dictionary containing the MAF model and trained parameters
        filepath: Path to save the model
    """
    directory = os.path.dirname(filepath)
    if directory: os.makedirs(directory, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(maf_result, f)
    print(f"Successfully saved MAF model to {filepath}")

def load_maf_model(filepath):
    """
    Load MAF model from disk.

    Args:
        filepath: Path to the saved model

    Returns:
        Dictionary containing the MAF model and trained parameters
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"MAF model file not found: {filepath}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)