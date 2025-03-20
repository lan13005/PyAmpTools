import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
import numpy as np
from orbax import checkpoint as orbax_checkpoint
import matplotlib.pyplot as plt

# Define gradient regularization loss
def gradient_regularization_loss(model, params, x, rngs=None):
    """
    Compute gradient regularization loss to enforce smoothness in model outputs.
    
    Args:
        model: The neural network model
        params: Model parameters
        x: Input data
        rngs: Random number generators for stochastic operations
    
    Returns:
        Regularization loss that penalizes large gradients in the output
    """
    def model_fn(x_sample):
        return model.apply(params, x_sample, training=False, rngs=rngs)
    
    # Compute gradients of model output with respect to inputs
    batch_gradients = jax.vmap(jax.grad(lambda x_i: jnp.sum(model_fn(x_i))), in_axes=0)(x)
    
    # Compute L2 norm of gradients
    gradient_norms = jnp.sum(batch_gradients**2, axis=1)
    
    # Return mean of gradient norms as the regularization loss
    return jnp.mean(gradient_norms)


# Define loss function (binary cross-entropy)
def binary_cross_entropy(logits, labels, weights=None):
    logits = jnp.squeeze(logits)
    labels = jnp.squeeze(labels)
    if weights is not None:
        return -jnp.mean(weights * (labels * jax.nn.log_sigmoid(logits) + 
                         (1 - labels) * jax.nn.log_sigmoid(-logits)))
    else:
        return -jnp.mean(labels * jax.nn.log_sigmoid(logits) + 
                            (1 - labels) * jax.nn.log_sigmoid(-logits))
        
# # Signed loss function, separates positive and negative contributions (performs worse)
# def binary_cross_entropy(logits, labels, weights):
#     logits = jnp.squeeze(logits)
#     labels = jnp.squeeze(labels)

#     log_prob_pos = jax.nn.log_sigmoid(logits)   # log(p)
#     log_prob_neg = jax.nn.log_sigmoid(-logits)  # log(1 - p)

#     pos_loss = weights * labels * log_prob_pos
#     neg_loss = weights * (1 - labels) * log_prob_neg

#     # Ensure numerical stability by separating positive/negative contributions
#     loss = -jnp.mean(jnp.where(weights >= 0, pos_loss + neg_loss, -pos_loss - neg_loss))

#     return loss


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
def plot_efficiency_by_variable(X_data, probabilities, feature_names=None, nbins=50, figsize=(15, 15), 
                                weight_rescaling=1.0, efficiency_dict=None, checkpoint_dir=None, suffix=None,
                                metric_type="standard"):
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
        checkpoint_dir: Directory to save the plots (optional)
        suffix: Suffix to add to the plot filename (optional)
        metric_type: standard  difference between calculated and reference efficiencies or relative
    Returns:
        list: Relative MAE for each feature (relative to the generated nominal efficiency)
    """
    
    if metric_type not in ["standard", "relative"]:
        raise ValueError(f"Invalid metric_type: {metric_type}, must be 'standard' or 'relative'")
    
    n_features = X_data.shape[1]
    
    # Check input shapes
    if probabilities.shape[0] != X_data.shape[0]:
        raise ValueError("probabilities and X_data must have the same number of samples")
    if feature_names is None:
        raise ValueError("No feature names provided, using default names")
    
    # Create square matrix of subplots
    fig, axes = plt.subplots(n_features, n_features, figsize=figsize)
    
    # Clip probabilities to avoid division by zero
    eps = 1e-6
    probabilities = np.clip(probabilities, eps, 1 - eps)

    # Compute density ratio (this is our efficiency function!)
    density_ratios = probabilities / (1 - probabilities) * weight_rescaling

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
                    if metric_type == "relative":
                        # For relative, divide the absolute differences by reference values
                        metric = np.mean(np.abs(efficiencies[valid_bins] - gen_eff_interp[valid_bins]) / gen_eff_interp[valid_bins])
            else: # binning is the same
                valid_bins = ~np.isnan(efficiencies) & ~np.isnan(gen_eff) & (gen_eff > 0)
                if np.sum(valid_bins) > 0:
                    # Use absolute difference for both metrics
                    metric = np.mean(np.abs(efficiencies[valid_bins] - gen_eff[valid_bins]))
                    if metric_type == "relative":
                        # For relative, divide the absolute differences by reference values
                        metric = np.mean(np.abs(efficiencies[valid_bins] - gen_eff[valid_bins]) / gen_eff[valid_bins])
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
    
    # Plot 2D efficiency ratios (upper triangle)
    if efficiency_dict is not None:
        from matplotlib.colors import TwoSlopeNorm
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                print(f"Plotting 2D efficiency ratio for features {feature_names[i]} vs {feature_names[j]}")
                
                ax = axes[i, j]
                
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

                
                # Calculate signed metric, unlike in the 1D plot which was used to measure a average deviance
                #    Here we care about the sign to highlight over or under estimation in these heatmaps
                if metric_type == "standard":
                    metric_hist = calc_hist - ref_hist
                elif metric_type == "relative":
                    with np.errstate(divide='ignore', invalid='ignore'):
                        metric_hist = np.where((ref_hist > 0) & (calc_hist > 0), calc_hist / ref_hist, np.nan)
                
                # Plot 2D histogram with coolwarm colormap as we are interested in deviance around a reference
                # 1. Relative has a center of 1 since a ratio of 1 is perfect agreement, clip at 10x factor between estimated and reference
                # 2. Standard has a center of 0 since a difference of 0 is perfect agreement, clip at 0.1 absolute deviation
                threshold = 10 if metric_type == "relative" else 1
                vcenter  = 1.0 if metric_type == "relative" else 0.0
                vmin = np.abs(np.nanmin(metric_hist))
                vmax = np.abs(np.nanmax(metric_hist))
                extrema = max(vmin, vmax)
                extrema = np.clip(extrema, 0, threshold) # extrema should already by positive
                
                norm = TwoSlopeNorm(vmin=-1*extrema, vcenter=vcenter, vmax=extrema)
                im = ax.pcolormesh(x_bins, y_bins, metric_hist, cmap='coolwarm', norm=norm)
                
                ax.set_xlabel(feature_names[j])
                ax.set_ylabel(feature_names[i])
                metric_type_str = "rel." if metric_type == "relative" else "delta"
                ax.set_title(f'{metric_type_str} eff.')
                
                # Add colorbar
                plt.colorbar(im, ax=ax)
    
    # Remove lower triangle plots
    for i in range(n_features):
        for j in range(i):
            fig.delaxes(axes[i, j])
    
    plt.tight_layout()
    if suffix is not None and isinstance(suffix, str) and suffix[0] != "_":
        suffix = "_" + suffix
    if checkpoint_dir is not None:
        plt.savefig(f'{checkpoint_dir}/efficiency_plots{suffix}.png', dpi=300)
    plt.close()
    
    print(f"Efficiency plots saved to 'efficiency_plots{suffix}.png'")
    
    # Calculate average (of averages) across features
    avg_metric = np.mean(metrics) if metrics else 0
    metric_type_str = "absolute relative difference" if metric_type == "relative" else "absolute difference"
    print(f"Average '{metric_type_str}' across all features: {avg_metric:.6f}")
    print(f"Feature-wise '{metric_type_str}' metrics: {[f'{name}: {metric:.6f}' for name, metric in zip(feature_names, metrics)]}")
    
    return metrics

# Define adaptive gradient regularization loss
def adaptive_gradient_regularization_loss(model, params, x, transition_sensitivity=0.5, rngs=None):
    """
    Compute adaptive gradient regularization loss that reduces penalty in regions with legitimate sharp transitions.
    
    Args:
        model: The neural network model
        params: Model parameters
        x: Input data
        transition_sensitivity: Controls how quickly regularization decreases in high-gradient regions
        rngs: Random number generators for stochastic operations
    
    Returns:
        Adaptive regularization loss that penalizes large gradients less in transition regions
    """
    def model_fn(x_sample):
        return model.apply(params, x_sample, training=False, rngs=rngs)
    
    # Compute gradients of model output with respect to inputs
    batch_gradients = jax.vmap(jax.grad(lambda x_i: jnp.sum(model_fn(x_i))), in_axes=0)(x)
    
    # Compute gradient magnitudes (L2 norm for each sample)
    gradient_norms = jnp.sum(batch_gradients**2, axis=1)
    
    # Compute transition-sensitive weights: reduce regularization in high-gradient regions
    # Higher transition_sensitivity means sharper distinction between regions
    suppression_factor = jnp.exp(-transition_sensitivity * jnp.sqrt(gradient_norms))
    
    # Apply adaptive regularization: multiply gradient norms by suppression factor
    adaptive_loss = jnp.mean(suppression_factor * gradient_norms)
    
    return adaptive_loss

# Combined loss function with adaptive gradient regularization
def combined_loss_adaptive(logits, labels, model, params, x, reg_strength=0.01, 
                          transition_sensitivity=0.5, weights=None, rngs=None):
    """
    Compute combined loss with BCE and adaptive gradient regularization.
    
    Args:
        logits: Model output logits
        labels: Ground truth labels
        model: The neural network model
        params: Model parameters
        x: Input data
        reg_strength: Regularization strength hyperparameter
        transition_sensitivity: Controls how quickly regularization decreases in high-gradient regions
        weights: Optional sample weights
        rngs: Random number generators for stochastic operations
    
    Returns:
        Combined loss value with adaptive gradient regularization
    """
    bce_loss = binary_cross_entropy(logits, labels, weights)
    grad_reg_loss = adaptive_gradient_regularization_loss(
        model, params, x, transition_sensitivity, rngs
    )
    
    return bce_loss + reg_strength * grad_reg_loss, (bce_loss, grad_reg_loss)