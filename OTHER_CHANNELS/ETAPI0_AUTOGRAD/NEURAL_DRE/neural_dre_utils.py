import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
import numpy as np
from orbax import checkpoint as orbax_checkpoint
import matplotlib.pyplot as plt

# Define loss function (binary cross-entropy)
def binary_cross_entropy(logits, labels, weights=None):
    logits = jnp.squeeze(logits)
    if weights is not None:
        return -jnp.mean(weights * (labels * jax.nn.log_sigmoid(logits) + 
                         (1 - labels) * jax.nn.log_sigmoid(-logits)))
    else:
        return -jnp.mean(labels * jax.nn.log_sigmoid(logits) + 
                            (1 - labels) * jax.nn.log_sigmoid(-logits))

# Checkpointing utilities
def save_checkpoint(state, step, checkpoint_dir=None):
    """Save the model checkpoint at a particular step"""
    # We're saving only the first device's parameters since they're synchronized
    single_device_state = jax.tree_util.tree_map(lambda x: x[0], state)
    
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
        checkpointers=checkpointer  # Remove the dict wrapper
    )
    
    if step is None:
        step = checkpoint_manager.latest_step()
        if step is None:
            raise ValueError("No checkpoints found in the specified directory")
    
    # Load the checkpoint with the newer API
    restored_state = checkpoint_manager.restore(step)  # Remove the items parameter
    
    print(f"Loaded checkpoint from step {step}")
    
    return restored_state

# Load and use a saved model
def load_and_use_model(model, state, X_data, checkpoint_dir, step=None):
    """Load a model from checkpoint and use it for inference"""
    try:
        # Load the checkpoint
        restored_state = load_checkpoint(checkpoint_dir, step)
        
        if restored_state is None:
            raise ValueError("Failed to load checkpoint - restored state is None")
            
        # Extract just the params from the restored state
        # Remove the extra params layer
        model_params = restored_state['params']['params']
            
        # Use the model for inference with the correct params structure
        print("Begin applying model onto the data...")
        logits = model.apply({'params': model_params}, X_data, training=False)
        probabilities = jax.nn.sigmoid(logits)
        
        return probabilities
        
    except Exception as e:
        print(f"Error using model from checkpoint: {e}")
        print("Falling back to current model parameters")
        # Fall back to using the current model state
        single_device_params = jax.tree_util.tree_map(lambda x: x[0], state.params)['params']  # Remove extra params layer
        logits = model.apply({'params': single_device_params}, X_data, training=False)
        probabilities = jax.nn.sigmoid(logits)
        return probabilities

# Calculate and plot 1D efficiency for each variable
def plot_efficiency_by_variable(X_data, probabilities, feature_names=None, nbins=50, figsize=(15, 10), weight_rescaling=1.0, efficiency_dict=None, checkpoint_dir=None, suffix=None):
    """
    Plot efficiency as a function of each variable by integrating over other dimensions.
    
    Args:
        X_data: Input feature data with shape (n_samples, n_features)
        probabilities: Model predictions with shape (n_samples,)
        feature_names: List of feature names (optional)
        nbins: Number of bins for each variable
        figsize: Figure size for the plots
        weight_rescaling: Rescaling factor for the weights (optional)
        efficiency_dict: Dictionary containing reference efficiencies for comparison (optional)
        
    Returns:
        list: Relative MAE for each feature (relative to the generated nominal efficiency)
    """
    n_features = X_data.shape[1]
    
    # Check input shapes
    if probabilities.shape[0] != X_data.shape[0]:
        raise ValueError("probabilities and X_data must have the same number of samples")
    if feature_names is None:
        raise ValueError("No feature names provided, using default names")
    
    # Create subplots
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Clip probabilities to avoid division by zero
    eps = 1e-6
    probabilities = np.clip(probabilities, eps, 1 - eps)

    # Compute density ratio (efficiency function)
    density_ratios = probabilities / (1 - probabilities)

    # Track relative MAE for each feature
    feature_rel_maes = []
    
    for i in range(n_features):
        print(f"Plotting efficiency for feature {feature_names[i]}")
        
        # Extract the feature and create bins
        feature_values = X_data[:, i]
        bins = np.linspace(np.min(feature_values), np.max(feature_values), nbins+1)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        
        # Calculate efficiency in each bin
        efficiencies = []
        errors = []
        
        for j in range(nbins):
            # print(f"Calculating efficiency for bin {j}")
            # Find samples in this bin
            in_bin = (feature_values >= bins[j]) & (feature_values < bins[j+1])
            if np.sum(in_bin) > 0:
                # print(f"Found {np.sum(in_bin)} samples in bin {j}")

                # Compute efficiency as density ratio
                bin_efficiency = np.mean(density_ratios[in_bin]) * weight_rescaling
                
                if np.sum(in_bin) > 1:
                    bin_error = np.std(density_ratios[in_bin]) / np.sqrt(np.sum(in_bin))
                else:
                    bin_error = 0

                # print(f"Bin efficiency: {bin_efficiency}, bin error: {bin_error}")
                efficiencies.append(bin_efficiency)
                errors.append(bin_error)
            else:
                efficiencies.append(0)
                errors.append(0)
                
        efficiencies = np.array(efficiencies)
        errors = np.array(errors)
        
        # Plot efficiency vs. variable
        ax = axes[i]
        
        max_eff = np.max(efficiencies)
        max_gen = np.max(efficiency_dict[feature_names[i]][0])
        scale = max_gen / max_eff
        efficiencies = efficiencies * scale  # Rescale efficiencies to match the maximum of the generated efficiencies
        max_eff = max(max_gen*1.2, max_eff)
        
        gen_eff = 0
        feature_rel_mae = 0
        if efficiency_dict is not None:
            gen_eff = efficiency_dict[feature_names[i]][0]
            gen_bin_centers = efficiency_dict[feature_names[i]][1]
            ax.plot(gen_bin_centers, gen_eff, 'k--', label='Efficiency')
            
            # Calculate relative MAE for this feature
            # Interpolate generated efficiencies to match bin centers if needed
            if len(gen_bin_centers) != len(bin_centers) or not np.allclose(gen_bin_centers, bin_centers):
                from scipy.interpolate import interp1d
                interp_func = interp1d(gen_bin_centers, gen_eff, bounds_error=False, fill_value="extrapolate")
                gen_eff_interp = interp_func(bin_centers)
                valid_bins = ~np.isnan(efficiencies) & ~np.isnan(gen_eff_interp) & (gen_eff_interp > 0)
                if np.sum(valid_bins) > 0:
                    # Calculate relative MAE (normalized by the generated efficiency)
                    abs_errors = np.abs(efficiencies[valid_bins] - gen_eff_interp[valid_bins])
                    rel_errors = abs_errors / gen_eff_interp[valid_bins]
                    feature_rel_mae = np.mean(rel_errors)
            else:
                valid_bins = ~np.isnan(efficiencies) & ~np.isnan(gen_eff) & (gen_eff > 0)
                if np.sum(valid_bins) > 0:
                    abs_errors = np.abs(efficiencies[valid_bins] - gen_eff[valid_bins])
                    rel_errors = abs_errors / gen_eff[valid_bins]
                    feature_rel_mae = np.mean(rel_errors)
            
            feature_rel_maes.append(feature_rel_mae)
            
        ax.errorbar(bin_centers, efficiencies, yerr=errors, fmt='o-', capsize=3)    
            
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel('Efficiency')
        ax.set_title(f'Efficiency vs {feature_names[i]} (Rel. MAE: {feature_rel_mae:.3f})')
        ax.set_ylim(0, max_eff * 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    # Remove unused subplots
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    if suffix is not None and isinstance(suffix, str) and suffix[0] != "_":
        suffix = "_" + suffix
    if checkpoint_dir is not None:
        plt.savefig(f'{checkpoint_dir}/efficiency_plots{suffix}.png', dpi=300)
    plt.close()
    
    print(f"Efficiency plots saved to 'efficiency_plots{suffix}.png'")
    
    # Calculate average relative MAE across all features
    avg_rel_mae = np.mean(feature_rel_maes) if feature_rel_maes else 0
    print(f"Average relative MAE across all features: {avg_rel_mae:.6f}")
    print(f"Feature-wise relative MAEs: {[f'{name}: {mae:.6f}' for name, mae in zip(feature_names, feature_rel_maes)]}")
    
    return feature_rel_maes