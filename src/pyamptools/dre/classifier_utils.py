import jax
import jax.numpy as jnp
import jax.lax as lax
from orbax import checkpoint as orbax_checkpoint
import os
from flax import nnx
from pyamptools.utility.general import console_print


# Checkpointing utilities
def save_checkpoint(state, step, checkpoint_dir=None, console=None):
    """Save the model checkpoint at a particular step"""
    
    # NOTE: state['model'] should be an nnx.Module, do not nnx.state(model) it. rng_state will not be split correctly
    #       Try to split the model into graph definition, RNG state, and other state
    #       Use ... to match all remaining elements in the filter
    if 'model' in state:
        model = state['model']
        graphdef, rng_state, other_state = nnx.split(model, nnx.RngState, ...)
        state = {**state, 'model': other_state} # overwrite model with the non-rng state part

    # Saving only the first device's parameters since they are synchronized
    def get_first_device(x):
        # in python str is iterable so we need to explicitly check for this
        if hasattr(x, '__getitem__') and not isinstance(x, (str, bytes)):
            try:
                return x[0]  # if x is an array / list
            except (IndexError, TypeError):
                return x  # if x is a scalar
        else:
            return x # if x is a scalar
    
    single_device_state = jax.tree_util.tree_map(get_first_device, state)
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpointer = orbax_checkpoint.StandardCheckpointer()    
    checkpointer.save(f"{checkpoint_dir}/{step}", single_device_state)
    console_print(f"Saved checkpoint at step {step}", console=console)
    
    return checkpointer

def load_checkpoint(checkpoint_dir, step=None, console=None):
    """Load a model checkpoint"""
    checkpointer = orbax_checkpoint.StandardCheckpointer()
    
    # If step is not specified, find the latest checkpoint
    #   listing all directories in checkpoint_dir that can be converted to integers
    if step is None:
        checkpoint_steps = []
        if os.path.exists(checkpoint_dir):
            for item in os.listdir(checkpoint_dir):
                item = item.lstrip('epoch_')
                try:
                    checkpoint_steps.append(int(item))
                except ValueError:
                    continue
        if not checkpoint_steps:
            raise ValueError("No checkpoints found in the specified directory")
        step = max(checkpoint_steps)
    
    # Load the checkpoint
    checkpoint_path = f"{checkpoint_dir}/{step}"
    restored_state = checkpointer.restore(checkpoint_path)
    
    console_print(f"Loaded checkpoint from step {step}", console=console)
    
    return restored_state

def try_resume_from_checkpoint(model, optimizer, checkpoint_dir, console):
    try:
        latest_checkpoint = None
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_")]
            if checkpoints:
                # Extract epoch numbers and find the latest
                epoch_nums = [int(cp.split("_")[1]) for cp in checkpoints]
                latest_epoch = max(epoch_nums)
                latest_checkpoint = f"epoch_{latest_epoch}"
        
        if latest_checkpoint:
            console_print(f"Found checkpoint {latest_checkpoint}. Attempting to resume...", console=console)
            
            restored_state = load_checkpoint(checkpoint_dir, latest_checkpoint)
                
            # Restore model and optimizer parameters
            # Convert the dictionary to a state object that nnx.update can use
            model_state = nnx.state(model)
            for key, value in restored_state['model'].items():
                if key in model_state:
                    model_state[key] = value
            
            # Optax optimizer's first elements contains actual parameter statistics
            #   Other elements are auxillary
            # Currently, optimizer.opt_state returns [Dict, None, None]
            optimizer_state = nnx.state(optimizer)
            for opt_dict in restored_state['optimizer']:
                if not isinstance(opt_dict, dict): continue
                for key, value in opt_dict.items():
                    if key in optimizer_state:
                        optimizer_state[key] = value
            
            # Get starting epoch
            start_epoch = restored_state.get('epoch', 0)
            loss_type_code = restored_state.get('loss_type_code', None)
            console_print(f"Successfully resumed from epoch {start_epoch}", console=console)
            return start_epoch, loss_type_code
        else:
            console_print("No checkpoint found. Starting training from scratch.", console=console)
            return 0, None
    except Exception as e:
        console_print(f"Error resuming from checkpoint: {e}", console=console)
        console_print("Starting training from scratch.", console=console)
        return 0, None

# Load and use a saved model
def load_and_use_model(model, state, X_data, checkpoint_dir, step=None, loss_type_code=0, console=None):
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
    try:
        # Load the checkpoint
        restored_state = load_checkpoint(checkpoint_dir, step)
        if restored_state is None:
            raise ValueError("Failed to load checkpoint - restored state is None")
        model_params = restored_state['params']['params'] # extract params
    except Exception as e:
        console_print(f"Error loading model from checkpoint: {e}", console=console)
        console_print("Falling back to current model parameters", console=console)
        model_params = jax.tree_util.tree_map(lambda x: x[0], state.params)['params']
        
    model_outputs = model.apply({'params': model_params}, X_data, training=False)
    
    preds = lax.cond(
        loss_type_code < 2,
        lambda _: jnp.clip(jax.nn.sigmoid(model_outputs), 1e-7, 1 - 1e-7),  # Maps to (0,1)
        lambda _: jnp.exp(model_outputs),                                   # Maps to (0,∞),
        operand=None
    )
    return preds

def calculate_entropy(probabilities):
    """
    Calculate entropy of probability distributions.
    
    Args:
        probabilities: Array of probability distributions or binary predictions.
                       If binary predictions in range (0,1), will convert to [p, 1-p] form.
        
    Returns:
        Array of entropy values with shape (batch_size,)
    """
    # Check if input is already a distribution or just a probability
    if len(probabilities.shape) == 1 or probabilities.shape[1] == 1:
        p = jnp.clip(probabilities.reshape(-1), 1e-10, 1.0 - 1e-10)
        p_binary = jnp.stack([p, 1.0 - p], axis=-1)
    else: # Already a distribution
        p_binary = jnp.clip(probabilities, 1e-10, 1.0)
    
    # Calculate entropy
    log_probs = jnp.log(p_binary)
    entropy = -jnp.sum(p_binary * log_probs, axis=-1)
    return entropy
