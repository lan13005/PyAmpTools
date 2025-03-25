import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
from orbax import checkpoint as orbax_checkpoint
import matplotlib.pyplot as plt

loss_type_map = {"bce": 0, "mse": 1, "mlc": 2, "sqrt": 3}
loss_type_map_reverse = {v: k for k, v in loss_type_map.items()}

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
    
    # Compute L2 norm of gradients with clipping to prevent extreme values
    batch_gradients = jnp.clip(batch_gradients, -1e3, 1e3)
    gradient_norms = jnp.sum(batch_gradients**2, axis=1)
    
    # Return mean of gradient norms as the regularization loss
    return jnp.mean(gradient_norms)

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
        params: Model parameters
        x: Input data
        reg_strength: Regularization strength hyperparameter
        transition_sensitivity: Controls how quickly regularization decreases in high-gradient regions
        weights: Optional sample weights
        rngs: Random number generators for stochastic operations
        loss_type_code: integer code for the loss type
    
    Returns:
        Combined loss value with adaptive gradient regularization
    """
    main_loss = likelihood_ratio_loss(model_outputs, labels, loss_type_code=loss_type_code, weights=weights)
    
    grad_reg_loss = lax.cond(
        reg_strength > 0,
        lambda _: adaptive_gradient_regularization_loss(
            model, params, x, transition_sensitivity, rngs
        ),
        lambda _: jnp.array(0.0, dtype=jnp.float32),
        operand=None
    )
    
    return main_loss + reg_strength * grad_reg_loss, (main_loss, grad_reg_loss)

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
    try:
        # Load the checkpoint
        restored_state = load_checkpoint(checkpoint_dir, step)
        if restored_state is None:
            raise ValueError("Failed to load checkpoint - restored state is None")
        model_params = restored_state['params']['params'] # extract params
    except Exception as e:
        print(f"Error loading model from checkpoint: {e}")
        print("Falling back to current model parameters")
        model_params = jax.tree_util.tree_map(lambda x: x[0], state.params)['params']
        
    model_outputs = model.apply({'params': model_params}, X_data, training=False)
    
    preds = lax.cond(
        loss_type_code < 2,
        lambda _: jnp.clip(jax.nn.sigmoid(model_outputs), 1e-7, 1 - 1e-7),  # Maps to (0,1)
        lambda _: jnp.exp(model_outputs),                                   # Maps to (0,∞),
        operand=None
    )
    return preds

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