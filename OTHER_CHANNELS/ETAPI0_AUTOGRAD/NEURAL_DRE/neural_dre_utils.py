import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=24"

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
from orbax import checkpoint as orbax_checkpoint
import matplotlib.pyplot as plt
import pickle
import optax
from tqdm import tqdm

loss_type_map = {"bce": 0, "mse": 1, "mlc": 2, "sqrt": 3}
loss_type_map_reverse = {v: k for k, v in loss_type_map.items()}

##########################################################################
######################### LOSS FUNCTIONS #################################
##########################################################################

def likelihood_ratio_loss(model_outputs, labels, loss_type_code=0, weights=None):
    """
    Unified loss function for likelihood ratio estimation
    This function roughly follows the parameterization in:
        - Learning Likelihood Ratios with Neural Network Classifiers
        - https://arxiv.org/pdf/2305.10500v2

    Args:
        model_outputs: Raw model outputs (before activation).
        labels: Ground truth labels (0 for p(x|θ1), 1 for p(x|θ0)).
        loss_type_code: integer code for the loss type
        weights: Optional sample weights (default is uniform).
    
    Returns:
        Scalar loss value.
    """
    model_outputs = jnp.squeeze(model_outputs)
    labels = jnp.squeeze(labels)

    # We later clip the loss type labels so if out of bounds then it will revert to the limiting cases
    preds = lax.cond(
        loss_type_code < 2,
        lambda _: jnp.clip(jax.nn.sigmoid(model_outputs), 1e-7, 1 - 1e-7),  # Maps to (0,1)
        lambda _: jnp.clip(jnp.exp(model_outputs), 1e-7, 1e7),  # Maps to (0,∞) with clipping to prevent explosions
        operand=None
    )
    
    # These definitions are from Table 1 in https://arxiv.org/pdf/2305.10500v2
    #   Compute loss terms A(f) and B(f) in a jax safe way
    def loss_0(_): # BCE where A(f) = log(f) and B(f) = log(1 - f)
        return jnp.log(preds), jnp.log(1 - preds)

    def loss_1(_): # MSE where A(f) = -(1 - f)^2 and B(f) = f^2
        return -(1 - preds) ** 2, -(preds ** 2)

    def loss_2(_): # MLC where A(f) = log(f) and B(f) = 1 - f
        return jnp.log(preds), 1 - preds

    def loss_3(_): # SQRT where A(f) = -1/sqrt(f) and B(f) = -sqrt(f)
        return -1 / jnp.sqrt(preds), -jnp.sqrt(preds)

    pos_loss, neg_loss = lax.switch(
        jnp.clip(loss_type_code, 0, 3),  # Ensure loss_type_code is in valid range
        [loss_0, loss_1, loss_2, loss_3],
        operand=None
    )

    # Assign loss based on labels
    loss = -1 * (labels * pos_loss + (1 - labels) * neg_loss)

    # Apply sample weights if provided
    if weights is None:
        return jnp.mean(loss)
    else:
        weights = jnp.squeeze(weights)
        return jnp.mean(weights * loss)

# Define adaptive gradient regularization loss
def adaptive_gradient_regularization_loss(model, params, x, transition_sensitivity=0.5, rngs=None):
    """
    Compute adaptive gradient regularization loss with reduced penalty in high-gradient regions.
    
    Args:
        model: The neural network model
        params: Model parameters
        x: Input data
        transition_sensitivity: Controls how quickly regularization decreases in high-gradient regions
        rngs: Random number generators for stochastic operations
    
    Returns:
        Adaptive regularization loss
    """
    def model_fn(x_sample):
        return model.apply(params, x_sample, training=False, rngs=rngs)
    
    # Compute gradients of model output with respect to inputs
    batch_gradients = jax.vmap(jax.grad(lambda x_i: jnp.sum(model_fn(x_i))), in_axes=0)(x)
    
    # Compute L2 norm of gradients with clipping
    batch_gradients = jnp.clip(batch_gradients, -1e3, 1e3)
    gradient_norms = jnp.sum(batch_gradients**2, axis=1)
    
    # Apply adaptive weighting that reduces penalty for high gradients
    suppression_factor = jnp.exp(-transition_sensitivity * jnp.sqrt(gradient_norms))
    
    # Return weighted mean of gradient norms
    return jnp.mean(suppression_factor * gradient_norms)

# Combined loss function with adaptive gradient regularization
def combined_loss_adaptive(model_outputs, labels, model, params, x, reg_strength=0.01, 
                          transition_sensitivity=0.5, weights=None, rngs=None,
                          loss_type_code=0):
    """
    Compute combined loss with selected loss function and adaptive gradient regularization.
    
    Args:
        model_outputs: Model output
        labels: Ground truth labels
        model: The neural network model
        params: Model parameters (should be direct params, not wrapped in {'params': ...})
        x: Input data
        reg_strength: Regularization strength hyperparameter
        transition_sensitivity: Controls how quickly regularization decreases in high-gradient regions
        weights: Optional sample weights
        rngs: Random number generators for stochastic operations
        loss_type_code: integer code for the loss type
    
    Returns:
        Combined loss value with adaptive gradient regularization
    """
    
    # Calculate main loss
    main_loss = likelihood_ratio_loss(model_outputs, labels, loss_type_code=loss_type_code, weights=weights)
    
    # Calculate gradient regularization loss if needed
    grad_reg_loss = lax.cond(
        reg_strength > 0,
        lambda _: adaptive_gradient_regularization_loss(
            model, params, x, transition_sensitivity, rngs
        ),
        lambda _: jnp.array(0.0, dtype=jnp.float32),
        operand=None
    )
    
    # Return combined loss and components
    return main_loss + reg_strength * grad_reg_loss, (main_loss, grad_reg_loss)

def weighted_maximum_likelihood_loss(params, maf, x, sample_weights=None):
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
    log_probs = maf.log_prob(params, x)
    
    if sample_weights is not None:
        # Use normalized weights to make loss function ~scale-invariant
        #   Optimizers are sensitive to size of gradients
        normalized_weights = sample_weights / jnp.sum(sample_weights)
        return -jnp.sum(log_probs * normalized_weights)
    else:
        return -jnp.mean(log_probs)

##########################################################################
############################ UTILITY FUNCTIONS ############################
##########################################################################

# Checkpointing utilities
def save_checkpoint(state, step, checkpoint_dir=None):
    """Save the model checkpoint at a particular step"""
    # Saving only the first device's parameters since they are synchronized
    def get_first_device(x):
        if hasattr(x, '__getitem__') and not isinstance(x, (str, bytes)):
            try:
                return x[0]  # arrays / lists
            except (IndexError, TypeError):
                return x  # scalar
        else:
            return x # scalar
    
    single_device_state = jax.tree_util.tree_map(get_first_device, state)
    
    checkpointer = orbax_checkpoint.PyTreeCheckpointer()
    options = orbax_checkpoint.CheckpointManagerOptions(
        max_to_keep=5,
        create=True
    )
    checkpoint_manager = orbax_checkpoint.CheckpointManager(
        directory=checkpoint_dir,
        checkpointers=checkpointer,
        options=options
    )
    
    # Create save_args with only the supported parameters
    save_args = orbax_checkpoint.SaveArgs(
        aggregate=False,
        dtype=None
    )
    save_args = jax.tree_util.tree_map(
        lambda _: save_args,
        single_device_state
    )
    
    checkpoint_manager.save(
        step,
        single_device_state,
        save_kwargs={'save_args': save_args}
    )
    print(f"Saved checkpoint at step {step}")
    
    return checkpoint_manager

def load_checkpoint(checkpoint_dir, step=None):
    """Load a model checkpoint"""
    checkpointer = orbax_checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax_checkpoint.CheckpointManager(
        directory=checkpoint_dir, 
        checkpointers=checkpointer
    )
    
    if step is None:
        step = checkpoint_manager.latest_step()
        if step is None:
            raise ValueError("No checkpoints found in the specified directory")
    
    # Load the checkpoint with the newer API
    restored_state = checkpoint_manager.restore(step)
    
    print(f"Loaded checkpoint from step {step}")
    
    return restored_state

# Load and use a saved model
# TODO: change name for DRE classifier
def load_and_use_model(model, state, X_data, checkpoint_dir, step=None, loss_type_code=0):
    """
    Load a model from checkpoint and use it for inference
    
    Args:
        model: The neural network model
        state: Current model state
        X_data: Input data for inference
        checkpoint_dir: Directory containing checkpoints
        step: Specific checkpoint step to load (None for latest)
        loss_type_code: integer code for the loss type
        
    Returns:
        For "bce"/"mse": Probabilities in range (0,1)
        For "mlc"/"sqrt": Direct likelihood ratios in range (0,∞)
    """
    # Store original input shape to ensure output matches
    original_sample_count = X_data.shape[0]
    print(f"Original input sample count: {original_sample_count}")
    
    # Get the number of devices being used
    num_devices = jax.device_count()
    
    # Extract parameters from state
    if isinstance(state, dict) and 'params' in state:
        model_params = state['params']
    else:
        model_params = state.params
    
    # Check if we need to extract from first device
    if hasattr(model_params, '__len__') and not isinstance(model_params, dict):
        model_params = jax.tree_map(lambda x: x[0], model_params)
    
    # Ensure we have the inner params, not nested params
    if isinstance(model_params, dict) and 'params' in model_params:
        model_params = model_params['params']
    
    print(f"Input data shape: {X_data.shape}")
    print(f"Model structure: {jax.tree_map(lambda x: x.shape, model_params)}")
    
    try:
        # For multi-device setup, we need to replicate parameters across devices
        if num_devices > 1:
            # Define a pmapped inference function
            @jax.pmap
            def infer_batch(params, batch):
                outputs = model.apply({'params': params}, batch, training=False)
                # Return raw outputs, process later based on loss type
                return outputs
            
            # Calculate padding needed for even distribution
            pad_size = 0
            if X_data.shape[0] % num_devices != 0:
                pad_size = num_devices - (X_data.shape[0] % num_devices)
                X_data_padded = jnp.pad(X_data, ((0, pad_size), (0, 0)), mode='edge')
                print(f"Padded input data from {X_data.shape} to {X_data_padded.shape} for even device distribution")
            else:
                X_data_padded = X_data
            
            # Create properly shaped parameters for each device by replicating the single device parameters
            replicated_params = {}
            for layer_name, layer_params in model_params.items():
                replicated_params[layer_name] = {}
                for param_name, param_value in layer_params.items():
                    # Create a stack of identical parameters for each device
                    replicated_params[layer_name][param_name] = jnp.stack([param_value] * num_devices)
            
            print(f"Replicated parameter structure: {jax.tree_map(lambda x: x.shape, replicated_params)}")
                
            # Reshape for pmap
            batched_data = X_data_padded.reshape(num_devices, -1, X_data_padded.shape[1])
            
            # Run inference across devices
            device_outputs = infer_batch(replicated_params, batched_data)
            
            # Combine results
            raw_outputs = device_outputs.reshape(-1, 1)
            
            # Remove padding to match original input size
            if pad_size > 0:
                raw_outputs = raw_outputs[:original_sample_count]
                
            print(f"Raw outputs shape after multi-device inference: {raw_outputs.shape}")
                
        else:
            # Single device case - simpler
            print("Using single device for inference")
            raw_outputs = model.apply({'params': model_params}, X_data, training=False)
        
        # Use lax.cond to handle different loss types
        preds = lax.cond(
            loss_type_code < 2,
            lambda _: jnp.clip(jax.nn.sigmoid(raw_outputs), 1e-7, 1 - 1e-7),  # Maps to (0,1)
            lambda _: jnp.exp(raw_outputs),                                   # Maps to (0,∞),
            operand=None
        )
        
        # Ensure output has correct shape
        if len(preds.shape) > 1 and preds.shape[1] == 1:
            preds = preds.reshape(-1)
            
        # Final check to ensure output matches input sample count
        if preds.shape[0] != original_sample_count:
            print(f"WARNING: Output sample count {preds.shape[0]} doesn't match input {original_sample_count}")
            # Truncate or pad to match original size
            if preds.shape[0] > original_sample_count:
                preds = preds[:original_sample_count]
            else:
                # This shouldn't happen, but just in case
                preds = jnp.pad(preds, (0, original_sample_count - preds.shape[0]), mode='edge')
                
        print(f"Final output shape: {preds.shape}")
        return preds
            
    except Exception as e:
        print(f"Error applying model: {e}")
        print("Model parameter structure doesn't match expected shape. Initializing fresh model.")
        
        # Initialize a fresh model for inference
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, X_data.shape[1]))
        variables = model.init(key, dummy_input, training=False)
        fresh_params = variables['params']
        
        print(f"Fresh model structure: {jax.tree_map(lambda x: x.shape, fresh_params)}")
        
        # Apply the fresh model
        raw_outputs = model.apply({'params': fresh_params}, X_data, training=False)
        
        # Use lax.cond to handle different loss types
        preds = lax.cond(
            loss_type_code < 2,
            lambda _: jnp.clip(jax.nn.sigmoid(raw_outputs), 1e-7, 1 - 1e-7),  # Maps to (0,1)
            lambda _: jnp.exp(raw_outputs),                                   # Maps to (0,∞),
            operand=None
        )
        
        # Ensure output has correct shape
        if len(preds.shape) > 1 and preds.shape[1] == 1:
            preds = preds.reshape(-1)
            
        print(f"Final output shape from fresh model: {preds.shape}")
        return preds

def batch_transform_maf(maf, params, data, batch_size=1024, direction="forward", key=None):
    """
    Transform data using MAF in batches to avoid memory issues.
    
    Args:
        maf: The MAF model
        params: Model parameters
        data: Input data to transform
        batch_size: Size of batches to process
        direction: "forward" for data->latent or "inverse" for latent->data
        key: Optional PRNG key for sampling
        
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
            batch_result, batch_log_dets = maf.forward(params, batch)
        elif direction == "inverse":
            batch_result, batch_log_dets = maf.inverse(params, batch)
        elif direction == "sample" and key is not None:
            batch_size_actual = end_idx - i
            batch_key = jax.random.split(key, batch_size_actual)
            batch_result, batch_log_dets = maf.sample(params, batch_key)
        else:
            raise ValueError(f"Unknown direction: {direction}")
            
        # Store results
        result[i:end_idx] = np.array(batch_result)
        log_dets[i:end_idx] = np.array(batch_log_dets)
        
        # # Optional progress reporting for large datasets
        # if data_size > 10000 and i % (10 * batch_size) == 0:
        #     print(f"Processed {i}/{data_size} samples ({i/data_size*100:.1f}%)")
    
    return result, log_dets

def fit_maf(maf, params, data, key, batch_size=1024, learning_rate=1e-3, num_epochs=100, patience=10,
           sample_weights=None, plot_frequency=None, checkpoint_dir=None,
           feature_names=None, X_gen=None, X_acc=None, use_gpu=False,
           n_samples_flow_diagnostic=10000, 
           clip_norm=0.5, adam_b1=0.9, adam_b2=0.999, adam_eps=1e-8,
           weight_decay=1e-5, warmup_steps=500):

    # Create a learning rate schedule with warmup
    if warmup_steps > 0:
        schedule_fn = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=num_epochs * (data.shape[0] // batch_size),
            end_value=learning_rate / 10
        )
    else:
        schedule_fn = learning_rate
    
    # Add L2 regularization through AdamW
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adamw(schedule_fn, b1=adam_b1, b2=adam_b2, eps=adam_eps, weight_decay=weight_decay)
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
    
    # Check for GPU availability
    if use_gpu and jax.devices('gpu'):
        print(f"Training MAF using GPU: {jax.devices('gpu')[0]}")
        # Define a jitted update function for single GPU
        @jax.jit
        def update_step(params, opt_state, batch_x, batch_weights):
            def loss_fn(p):
                return weighted_maximum_likelihood_loss(p, maf, batch_x, batch_weights)
            
            loss_value, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss_value
    else:
        # Get number of devices for parallelization
        num_devices = jax.device_count()
        print(f"Training MAF using {num_devices} CPU cores")
        
        # Define a pmapped update function to parallelize across devices
        @jax.pmap
        def update_step_parallel(params_device, opt_state_device, batch_x_device, batch_weights_device):
            def loss_fn(p):
                return weighted_maximum_likelihood_loss(p, maf, batch_x_device, batch_weights_device)
            
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
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        # Create a new random key for each epoch
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, data_size)
        
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
                
                maf_trained = MAF(maf.input_dim, maf.hidden_dims, maf.num_flows, key=key)
                maf_trained.made_layers = maf.made_layers  # Copy the architecture
                
                return {"maf": maf_trained, "params": params, "losses": epoch_losses}
            
            epoch_loss += loss_value
            
        # Record average epoch loss
        avg_loss = epoch_loss / steps_per_epoch
        epoch_losses.append(float(avg_loss))  # Convert to Python float for better serialization
        
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
        if plot_frequency is not None and checkpoint_dir is not None and (epoch + 1) % plot_frequency == 0:
            if X_gen is not None and X_acc is not None:
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
                    
                    # Free memory explicitly
                    X_gen_latent_np = None
                    X_acc_latent_np = None
                    
                    # Also save the loss curve
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
                    
                except Exception as e:
                    print(f"Error creating diagnostic plots: {e}")
    
    progress_bar.close()
    
    # Get final parameters (extract from first device if using multi-CPU)
    if not use_gpu or not jax.devices('gpu'):
        params = jax.tree_map(lambda x: x[0], params)
    
    maf_trained = MAF(maf.input_dim, maf.hidden_dims, maf.num_flows, key=key)
    maf_trained.made_layers = maf.made_layers  # Copy the architecture
    
    # Return the best model parameters
    return {"maf": maf_trained, "params": best_params, "losses": epoch_losses}

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
    
##########################################################################
######################### PLOTTING FUNCTIONS #############################
##########################################################################

# Calculate and plot 1D efficiency for each variable
def plot_efficiency_by_variable(X_data, model_output, feature_names=None, nbins=50, figsize=(15, 15), 
                                weight_rescaling=1.0, efficiency_dict=None, checkpoint_dir=None, suffix=None,
                                loss_type_code=0, metric_type=None, plot_generated=False, plot_predicted=False):
    """
    Plot efficiency as a function of each variable
    
    Args:
        X_data: Input feature data with shape (n_samples, n_features)
        model_output: Model predictions:
                     - For "bce"/"mse": Probabilities in range (0,1)
                     - For "mlc"/"sqrt": Direct likelihood ratios in range (0,∞)
        feature_names: List of feature names (optional)
        nbins: Number of bins for each variable
        figsize: Figure size for the plots
        weight_rescaling: Rescaling factor for the weights (optional)
        efficiency_dict: Dictionary containing reference efficiencies for comparison (optional)
        checkpoint_dir: Directory to save the plots (optional)
        suffix: Suffix to add to the plot filename (optional)
        loss_type_code: integer code for the loss type
        metric_type: Metric to plot. If None, uses "standard". 
                     Supported types: "standard", "relative"
        plot_generated: Whether to create additional plots for the generated efficiency
        plot_predicted: Whether to create additional plots for the predicted efficiency
        
    Returns:
        list: Relative MAE for each feature (relative to the generated nominal efficiency)
    """
    
    # Validate metric types
    valid_metric_types = ["standard", "relative"]
    metric_type_labels = {
        "standard": "absolute difference",
        "relative": "absolute relative difference"
    }
    if metric_type not in valid_metric_types:
        raise ValueError(f"Invalid metric_type: {metric_type}, must be one of {valid_metric_types}")
    
    n_features = X_data.shape[1]
    
    # Check input shapes
    if model_output.shape[0] != X_data.shape[0]:
        raise ValueError("model_output and X_data must have the same number of samples")
    if feature_names is None:
        raise ValueError("No feature names provided, using default names")
    
    # Ensure model_output is flattened to 1D
    model_output = np.asarray(model_output).flatten()
    
    # Compute density ratio based on loss type
    density_ratios = lax.cond(
        loss_type_code < 2,
        lambda _: np.clip(model_output, 1e-6, 1 - 1e-6) / (1 - np.clip(model_output, 1e-6, 1 - 1e-6)),
        lambda _: model_output,
        operand=None
    )
    density_ratios = density_ratios * weight_rescaling
    density_ratios = np.asarray(density_ratios).flatten()
    
    # Metric to plot on off-diagonal (correlation plots)
    metrics = [] # list of average metrics for each feature, 1 dimensional
    
    # Pre-compute bins for each feature (more efficient)
    bins_list = []
    bin_centers_list = []
    for i in range(n_features):
        print(f"Plotting efficiency for feature {feature_names[i]}")
        
        # Extract the feature and create bins
        feature_values = X_data[:, i]
        bins = np.linspace(np.min(feature_values), np.max(feature_values), nbins+1)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        bins_list.append(bins)
        bin_centers_list.append(bin_centers)
    
    # Calculate 1D efficiencies for all features at once (more efficient)
    feature_efficiencies = []
    feature_errors = []
    
    for i in range(n_features):
        print(f"Calculating efficiency for feature {feature_names[i]}")
        feature_values = X_data[:, i]
        bins = bins_list[i]
        
        # Use numpy's digitize for faster binning
        bin_indices = np.digitize(feature_values, bins) - 1
        
        # Clip to valid range
        bin_indices = np.clip(bin_indices, 0, nbins-1)
        
        # Initialize arrays
        efficiencies = np.zeros(nbins)
        errors = np.zeros(nbins)
        counts = np.zeros(nbins, dtype=int)
        
        # Use numpy's bincount for faster counting
        for bin_idx in range(nbins):
            mask = bin_indices == bin_idx
            counts[bin_idx] = np.sum(mask)
            if counts[bin_idx] > 0:
                bin_ratios = density_ratios[mask]
                efficiencies[bin_idx] = np.mean(bin_ratios)
                errors[bin_idx] = np.std(bin_ratios) / np.sqrt(counts[bin_idx]) if counts[bin_idx] > 1 else 0
        
        feature_efficiencies.append(efficiencies)
        feature_errors.append(errors)
    
    # Pre-compute 2D histograms for all feature pairs
    calc_hist_dict = {}
    ref_hist_dict = {}
    
    if efficiency_dict is not None:
        for i in range(n_features):
            for j in range(i+1, n_features):
                key = (i, j)
                print(f"Pre-computing 2D histograms for features {feature_names[i]} vs {feature_names[j]}")
                
                x_bins = bins_list[j]
                y_bins = bins_list[i]
                
                # Digitize data for both dimensions
                x_indices = np.digitize(X_data[:, j], x_bins) - 1
                y_indices = np.digitize(X_data[:, i], y_bins) - 1
                
                # Clip to valid range
                x_indices = np.clip(x_indices, 0, nbins-1)
                y_indices = np.clip(y_indices, 0, nbins-1)
                
                # Initialize 2D histogram for calculated efficiencies
                calc_hist = np.zeros((nbins, nbins))
                counts = np.zeros((nbins, nbins), dtype=int)
                
                # Fill histogram
                for idx in range(len(density_ratios)):
                    x_idx = x_indices[idx]
                    y_idx = y_indices[idx]
                    calc_hist[y_idx, x_idx] += density_ratios[idx]
                    counts[y_idx, x_idx] += 1
                
                # Average by counts
                with np.errstate(divide='ignore', invalid='ignore'):
                    calc_hist = np.where(counts > 0, calc_hist / counts, 0)
                
                calc_hist_dict[key] = calc_hist
                
                # Get 2D reference efficiency directly from efficiency_dict
                key_2d = f"{feature_names[i]}_{feature_names[j]}"
                
                if key_2d in efficiency_dict:
                    ref_hist = efficiency_dict[key_2d][0]
                    ref_y_bins = efficiency_dict[key_2d][1]
                    ref_x_bins = efficiency_dict[key_2d][2]
                    
                    # Interpolate if binning is different
                    if (len(ref_x_bins) != len(x_bins) or 
                        len(ref_y_bins) != len(y_bins) or 
                        not np.allclose(ref_x_bins, x_bins) or 
                        not np.allclose(ref_y_bins, y_bins)):
                        
                        from scipy.interpolate import RegularGridInterpolator
                        ref_y_centers = 0.5 * (ref_y_bins[1:] + ref_y_bins[:-1])
                        ref_x_centers = 0.5 * (ref_x_bins[1:] + ref_x_bins[:-1])
                        
                        # Create interpolator for 2D reference efficiency
                        interp_func = RegularGridInterpolator(
                            (ref_y_centers, ref_x_centers), 
                            ref_hist,
                            bounds_error=False, 
                            fill_value=None
                        )
                        
                        # Create meshgrid of bin centers for target grid
                        x_centers = bin_centers_list[j]
                        y_centers = bin_centers_list[i]
                        Y, X = np.meshgrid(y_centers, x_centers, indexing='ij')
                        points = np.column_stack([Y.flatten(), X.flatten()])
                        
                        # Interpolate to new grid
                        ref_hist_interp = interp_func(points).reshape(nbins, nbins)
                        ref_hist = ref_hist_interp
                else:
                    raise ValueError(f"Error: 2D reference efficiency not available for {key_2d}. Using product of 1D efficiencies.")
                
                ref_hist_dict[key] = ref_hist
    
    # Loop through each metric type and create separate plots
    metric_types = [metric_type]
    if plot_generated: metric_types.append("generated")
    if plot_predicted: metric_types.append("predicted")
    for current_metric_type in metric_types:
        print(f"Creating plots for metric type: {current_metric_type}")
        
        # Create square matrix of subplots
        fig, axes = plt.subplots(n_features, n_features, figsize=figsize)
        
        # Plot 1D efficiencies (diagonal)
        for i in range(n_features):
            print(f"Plotting 1D efficiency for feature {feature_names[i]}")
            
            efficiencies = feature_efficiencies[i]
            errors = feature_errors[i]
            bin_centers = bin_centers_list[i]
            
            # Plot efficiency vs. variable on diagonal
            ax = axes[i, i]
            
            # Scale efficiencies if efficiency_dict is provided
            metric = 0
            if efficiency_dict is not None:
                gen_eff = efficiency_dict[feature_names[i]][0]
                gen_bin_centers = efficiency_dict[feature_names[i]][1]
                
                # Scale the calculated efficiencies to match the maximum of the reference
                max_eff = np.max(efficiencies) if np.any(efficiencies) else 1.0
                max_gen = np.max(gen_eff) if np.any(gen_eff) else 1.0
                scale = max_gen / max_eff if max_eff > 0 else 1.0
                efficiencies = efficiencies * scale
                
                # Plot reference efficiency)
                ax.plot(gen_bin_centers, gen_eff, 'k--', label='Reference', zorder=10)
                
                # Calculate metric_type
                if len(gen_bin_centers) != len(bin_centers) or not np.allclose(gen_bin_centers, bin_centers): # interpolate if binning is different
                    from scipy.interpolate import interp1d
                    interp_func = interp1d(gen_bin_centers, gen_eff, bounds_error=False, fill_value="extrapolate")
                    gen_eff_interp = interp_func(bin_centers)
                    valid_bins = ~np.isnan(efficiencies) & ~np.isnan(gen_eff_interp) & (gen_eff_interp > 0)
                    if np.sum(valid_bins) > 0:
                        # Use absolute difference for both metrics
                        metric = np.mean(np.abs(efficiencies[valid_bins] - gen_eff_interp[valid_bins]))
                        if current_metric_type == "relative":
                            # For relative, divide the absolute differences by reference values
                            metric = np.mean(np.abs(efficiencies[valid_bins] - gen_eff_interp[valid_bins]) / gen_eff_interp[valid_bins])
                else: # binning is the same
                    valid_bins = ~np.isnan(efficiencies) & ~np.isnan(gen_eff) & (gen_eff > 0)
                    if np.sum(valid_bins) > 0:
                        # Use absolute difference for both metrics
                        metric = np.mean(np.abs(efficiencies[valid_bins] - gen_eff[valid_bins]))
                        if current_metric_type == "relative":
                            # For relative, divide the absolute differences by reference values
                            metric = np.mean(np.abs(efficiencies[valid_bins] - gen_eff[valid_bins]) / gen_eff[valid_bins])
                
                if current_metric_type in ["standard", "relative"]:
                    metrics.append(metric)
                
                # Set y-axis limit based on max of both efficiencies
                max_y = max(np.max(gen_eff) * 1.2, np.max(efficiencies) * 1.2) if np.any(gen_eff) and np.any(efficiencies) else 1.0
            else:
                max_y = np.max(efficiencies) * 1.2 if np.any(efficiencies) else 1.0
            
            # Plot calculated efficiency
            ax.errorbar(bin_centers, efficiencies, yerr=errors, fmt='o-', capsize=3, label='Calculated')
            
            ax.set_xlabel(feature_names[i])
            ax.set_ylabel('Efficiency')
            ax.set_title(f'{feature_names[i]} (MAE: {metric:.3f})')
            ax.set_ylim(0, max_y)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize='small')
        
        # Plot 2D efficiency maps (upper triangle)
        if efficiency_dict is not None:
            from matplotlib.colors import TwoSlopeNorm
            
            # Find global max for generated and predicted plots
            if any(mt in ["generated", "predicted"] for mt in metric_types):
                global_max = 0
                for key in calc_hist_dict:
                    global_max = max(global_max, np.nanmax(calc_hist_dict[key]), np.nanmax(ref_hist_dict[key]))
            
            for i in range(n_features):
                for j in range(i+1, n_features):
                    print(f"Plotting 2D efficiency for features {feature_names[i]} vs {feature_names[j]}")
                    
                    ax = axes[i, j]
                    key = (i, j)
                    
                    # Get pre-computed histograms
                    calc_hist = calc_hist_dict[key]
                    ref_hist = ref_hist_dict[key]
                    
                    x_bins = bins_list[j]
                    y_bins = bins_list[i]
                    
                    # Determine what to plot based on metric type
                    if current_metric_type == "standard":
                        # Standard difference
                        metric_hist = calc_hist - ref_hist
                        cmap = 'coolwarm'
                        threshold = 1
                        vcenter = 0.0
                        vmin = np.abs(np.nanmin(metric_hist))
                        vmax = np.abs(np.nanmax(metric_hist))
                        extrema = max(vmin, vmax)
                        extrema = np.clip(extrema, 0, threshold)
                        norm = TwoSlopeNorm(vmin=-1*extrema, vcenter=vcenter, vmax=extrema)
                        title = 'delta eff.'
                        
                    elif current_metric_type == "relative":
                        # Relative ratio
                        with np.errstate(divide='ignore', invalid='ignore'):
                            metric_hist = np.where((ref_hist > 0) & (calc_hist > 0), calc_hist / ref_hist, np.nan)
                        cmap = 'coolwarm'
                        threshold = 10
                        vcenter = 1.0
                        vmin = np.abs(np.nanmin(metric_hist))
                        vmax = np.abs(np.nanmax(metric_hist))
                        extrema = max(vmin, vmax)
                        extrema = np.clip(extrema, 0, threshold)
                        norm = TwoSlopeNorm(vmin=-1*extrema, vcenter=vcenter, vmax=extrema)
                        title = 'rel. eff.'
                        
                    elif current_metric_type == "generated":
                        # Generated (reference) histogram
                        metric_hist = ref_hist
                        cmap = 'plasma'
                        vmax = global_max
                        vmin = 0
                        norm = None
                        title = 'gen. eff.'
                        
                    elif current_metric_type == "predicted":
                        # Predicted (calculated) histogram
                        metric_hist = calc_hist
                        cmap = 'plasma'
                        vmax = global_max
                        vmin = 0
                        norm = None
                        title = 'pred. eff.'
                    
                    # Plot the appropriate histogram
                    if norm is not None:
                        im = ax.pcolormesh(x_bins, y_bins, metric_hist, cmap=cmap, norm=norm)
                    else:
                        im = ax.pcolormesh(x_bins, y_bins, metric_hist, cmap=cmap, vmin=vmin, vmax=vmax)
                    
                    ax.set_xlabel(feature_names[j])
                    ax.set_ylabel(feature_names[i])
                    ax.set_title(title)
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax)
        
        # Remove lower triangle plots
        for i in range(n_features):
            for j in range(i):
                fig.delaxes(axes[i, j])
        
        plt.tight_layout()
        if suffix is not None and isinstance(suffix, str) and suffix[0] != "_":
            suffix_str = "_" + suffix
        else:
            suffix_str = "" if suffix is None else suffix
            
        # Add metric type to filename
        metric_suffix = f"_{current_metric_type}{suffix_str}"
        
        if checkpoint_dir is not None:
            plt.savefig(f'{checkpoint_dir}/efficiency_plots{metric_suffix}.png', dpi=300)
            print(f"Efficiency plots saved to 'efficiency_plots{metric_suffix}.png'")
        plt.close()
    
    # Calculate average (of averages) across features - only for standard/relative metrics
    if metrics:
        avg_metric = np.mean(metrics) if metrics else 0
        metric_type_str = metric_type_labels[metric_type]
        print(f"Average '{metric_type_str}' across all features: {avg_metric:.6f}")
        print(f"Feature-wise '{metric_type_str}' metrics: {[f'{name}: {metric:.6f}' for name, metric in zip(feature_names, metrics)]}")
    
    return metrics

def create_corner_plot(X1, X2, labels, feature_names, filename, checkpoint_dir):
    """
    Create a corner plot showing pairwise relationships between dimensions.
    - Diagonal: 1D histograms showing marginal distributions
    - Upper triangle: Pairwise scatter plots
    
    Args:
        X1, X2: Arrays of shape (samples, n_dims)
        labels: List of labels for X1 and X2, i.e. ['accepted', 'generated']
        feature_names: List of feature names for each dimension
        filename: Output filename
        checkpoint_dir: Directory to save the plot
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    n_dims = X1.shape[1]
    fig, axes = plt.subplots(n_dims, n_dims, figsize=(15, 15))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    # Specs for the two distributions
    colors = ['black', 'orange']
    alphas = [0.5, 0.3]
    zorders = [1, 2]
    
    # Begin plotting
    for i in range(n_dims):
        for j in range(n_dims):
            ax = axes[i, j]
            
            # Turn off lower triangle
            if i > j:
                ax.set_visible(False)
                continue
                
            # Diagonal: 1D histograms
            if i == j:
                ax.hist(X1[:, i], bins=50, alpha=alphas[0], color=colors[0], density=True, label=labels[0], zorder=zorders[0])
                ax.hist(X2[:, i], bins=50, alpha=alphas[1], color=colors[1], density=True, label=labels[1], zorder=zorders[1])
                ax.set_ylabel('Density')    
                ax.set_yticks([]) # remove ticks since we already know its a density
                ax.set_xlabel(feature_names[j])
            
            # Upper triangle: Scatter plots
            elif i < j:
                ax.scatter(X1[:, j], X1[:, i], s=5, alpha=alphas[0], color=colors[0], label=labels[0], zorder=zorders[0])
                ax.scatter(X2[:, j], X2[:, i], s=5, alpha=alphas[1], color=colors[1], label=labels[1], zorder=zorders[1])
                ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Add legend to the top-right plot
    handles, labels_legend = axes[0, n_dims-1].get_legend_handles_labels()
    if handles:  # Check if legend handles exist
        fig.legend(handles, labels_legend, loc='upper right', bbox_to_anchor=(0.99, 0.99), prop={'size': 12})
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f'{checkpoint_dir}/{filename}', dpi=150)
    print(f"Corner plot saved to {checkpoint_dir}/{filename}")
    fig = None
    axes = None
    plt.close('all')

##############################################
# NORMALIZING FLOW IMPLEMENTATION
##############################################

class MADE:
    """
    Masked Autoregressive Density Estimator for MAF implementation.
    
    Implements the autoregressive network that outputs the shift and scale
    parameters for the normalizing flow.
    """
    
    def __init__(self, input_dim, hidden_dims, key, reverse=False):
        """
        Initialize a MADE network.
        
        Args:
            input_dim: Input dimensionality
            hidden_dims: List of hidden layer dimensions
            key: PRNG key for initialization
            reverse: Whether to reverse the ordering of dependencies
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = input_dim * 2  # For mean and log_scale
        
        # Create masks to enforce autoregressive property
        self.masks = self._create_masks(reverse)
        
        # Initialize network parameters
        keys = jax.random.split(key, len(hidden_dims) + 1)
        self.params = []
        
        # Input to first hidden layer
        self.params.append({
            'w': jax.random.normal(keys[0], (input_dim, hidden_dims[0])) * 0.01,
            'b': jax.random.normal(keys[0], (hidden_dims[0],)) * 0.01,
            'mask': self.masks[0]
        })
        
        # Hidden to hidden layers
        for l in range(len(hidden_dims) - 1):
            self.params.append({
                'w': jax.random.normal(keys[l+1], (hidden_dims[l], hidden_dims[l+1])) * 0.01,
                'b': jax.random.normal(keys[l+1], (hidden_dims[l+1],)) * 0.01,
                'mask': self.masks[l+1]
            })
        
        # Hidden to output layer
        self.params.append({
            'w': jax.random.normal(keys[-1], (hidden_dims[-1], self.output_dim)) * 0.01,
            'b': jax.random.normal(keys[-1], (self.output_dim,)) * 0.01,
            'mask': self.masks[-1]
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

class MAF:
    """
    Masked Autoregressive Flow implementation.
    
    This flow transforms between data space and latent space through a series
    of autoregressive transformations.
    """
    
    def __init__(self, input_dim, hidden_dims, num_flows, key):
        """
        Initialize a MAF model.
        
        Args:
            input_dim: Input dimensionality
            hidden_dims: List of hidden layer dimensions for each MADE
            num_flows: Number of flow layers
            key: PRNG key for initialization
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_flows = num_flows
        
        # Initialize MADE layers
        keys = jax.random.split(key, num_flows)
        self.made_layers = []
        
        for i in range(num_flows):
            # Alternate between forward and reverse ordering
            reverse = i % 2 == 1
            self.made_layers.append(MADE(input_dim, hidden_dims, keys[i], reverse=reverse))
    
    def forward(self, params, x):
        """
        Transform data from input space to latent space (forward flow).
        
        Args:
            params: The model parameters
            x: Input data of shape (batch_size, input_dim)
            
        Returns:
            z: Transformed data in latent space
            log_det: Log determinant of the Jacobian
        """
        z = x
        log_det_sum = jnp.zeros(x.shape[0])
        
        for i in range(self.num_flows):
            # Get means and scales from MADE using the parameters for this layer
            means, scales = self._apply_made_layer(params[i], self.made_layers[i], z)
            
            # Transform
            z = (z - means) / scales
            
            # Compute log determinant
            log_det = -jnp.sum(jnp.log(scales), axis=1)
            log_det_sum += log_det
            
            # Permute dimensions for the next flow (except for the last one)
            if i < self.num_flows - 1:
                # Simple reversing permutation
                z = jnp.flip(z, axis=-1)
        
        return z, log_det_sum
    
    def _apply_made_layer(self, params, made_layer, x):
        """Apply a MADE layer with its parameters."""
        h = x
        
        # Apply all layers except the last one with ReLU activation
        for layer_idx, layer_params in enumerate(params[:-1]):
            h = jnp.dot(h, layer_params['w'] * made_layer.masks[layer_idx]) + layer_params['b']
            h = jax.nn.relu(h)
        
        # Apply the last layer (output layer) without activation
        layer_params = params[-1]
        output = jnp.dot(h, layer_params['w'] * made_layer.masks[-1]) + layer_params['b']
        
        # Split output into means and log_scales
        means = output[..., :self.input_dim]
        log_scales = output[..., self.input_dim:]
        
        # Constrain scales to prevent numerical issues
        log_scales = jnp.clip(log_scales, -5, 5)
        
        return means, jnp.exp(log_scales)
    
    def inverse(self, params, z):
        """
        Transform data from latent space to input space (inverse flow).
        
        Args:
            params: The model parameters
            z: Latent variables of shape (batch_size, input_dim)
            
        Returns:
            x: Transformed data in input space
            log_det: Log determinant of the Jacobian
        """
        x = z
        log_det_sum = jnp.zeros(z.shape[0])
        
        # Apply flows in reverse order
        for i in range(self.num_flows - 1, -1, -1):
            # Inverse permutation (if not the last flow)
            if i < self.num_flows - 1:
                x = jnp.flip(x, axis=-1)
            
            # Get means and scales
            means, scales = self._apply_made_layer(params[i], self.made_layers[i], x)
            
            # Inverse transform
            x = x * scales + means
            
            # Compute log determinant
            log_det = jnp.sum(jnp.log(scales), axis=1)
            log_det_sum += log_det
        
        return x, log_det_sum
    
    def log_prob(self, params, x):
        """
        Compute log probability of data under the MAF model.
        
        Args:
            params: The model parameters
            x: Input data of shape (batch_size, input_dim)
            
        Returns:
            Log probability of each input point
        """
        z, log_det = self.forward(params, x)
        # Prior is a standard normal
        log_prior = -0.5 * jnp.sum(z**2, axis=1) - 0.5 * self.input_dim * jnp.log(2 * jnp.pi)
        return log_prior + log_det
    
    def init_params(self):
        """
        Initialize the parameters for the MAF model.
        
        Returns:
            A list of parameters for each MADE layer.
        """
        return [layer.params for layer in self.made_layers]
