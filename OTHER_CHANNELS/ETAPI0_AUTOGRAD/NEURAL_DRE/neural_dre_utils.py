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

def create_corner_plot(X1, X2, labels, feature_names, title, filename, checkpoint_dir, n_samples=10000):
    """
    Create a corner plot showing pairwise relationships between dimensions.
    - Diagonal: 1D histograms showing marginal distributions
    - Upper triangle: Pairwise scatter plots
    - Lower triangle: Turned off
    
    Args:
        X1, X2: Arrays of shape (n_samples, n_dims)
        labels: List of labels for X1 and X2, i.e. ['accepted', 'generated']
        feature_names: List of feature names for each dimension
        title: Title for the plot
        filename: Output filename
        checkpoint_dir: Directory to save the plot
        n_samples: Number of samples to plot (subsampling for large datasets)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    n_dims = X1.shape[1]
    
    # Subsample if necessary
    if X1.shape[0] > n_samples:
        idx1 = np.random.choice(X1.shape[0], size=n_samples, replace=False)
        X1_plot = X1[idx1]
    else:
        X1_plot = X1
        
    if X2.shape[0] > n_samples:
        idx2 = np.random.choice(X2.shape[0], size=n_samples, replace=False)
        X2_plot = X2[idx2]
    else:
        X2_plot = X2
    
    # Create figure
    fig, axes = plt.subplots(n_dims, n_dims, figsize=(15, 15))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    # Colors for the two distributions
    colors = ['black', 'orange']
    alphas = [0.8, 0.5]
    zorders = [1, 2]
    
    # Loop through dimensions
    for i in range(n_dims):
        for j in range(n_dims):
            ax = axes[i, j]
            
            # Turn off lower triangle
            if i > j:
                ax.set_visible(False)
                continue
                
            # Diagonal: 1D histograms
            if i == j:
                ax.hist(X1_plot[:, i], bins=50, alpha=alphas[0], color=colors[0], density=True, label=labels[0], zorder=zorders[0])
                ax.hist(X2_plot[:, i], bins=50, alpha=alphas[1], color=colors[1], density=True, label=labels[1], zorder=zorders[1])
                
                # Only show x-label for bottom row
                if i == n_dims - 1:
                    ax.set_xlabel(feature_names[j])
                    
                # Only show y-label for left column
                if j == 0:
                    ax.set_ylabel('Density')
                    
                # Remove y-ticks for clarity
                ax.set_yticks([])
                
                # Add feature name as text in the plot
                if i == 0:
                    ax.set_title(feature_names[j])
            
            # Upper triangle: Scatter plots
            elif i < j:
                ax.scatter(X1_plot[:, j], X1_plot[:, i], s=1, alpha=alphas[0], color=colors[0], label=labels[0], zorder=zorders[0])
                ax.scatter(X2_plot[:, j], X2_plot[:, i], s=1, alpha=alphas[1], color=colors[1], label=labels[1], zorder=zorders[1])
                
                # Only show x-label for bottom row
                if i == n_dims - 1:
                    ax.set_xlabel(feature_names[j])
                    
                # Only show y-label for left column
                if j == 0:
                    ax.set_ylabel(feature_names[i])
                    
                # Remove ticks for cleaner look
                ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Add legend to the top-right plot
    handles, labels_legend = axes[0, n_dims-1].get_legend_handles_labels()
    if handles:  # Check if legend handles exist
        fig.legend(handles, labels_legend, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    
    # Add title
    plt.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f'{checkpoint_dir}/{filename}', dpi=150)
    print(f"Corner plot saved to {checkpoint_dir}/{filename}")
    plt.close()

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
        # Ensure weights are normalized
        normalized_weights = sample_weights / jnp.sum(sample_weights)
        # Compute weighted negative log likelihood
        return -jnp.sum(log_probs * normalized_weights)
    else:
        # Standard unweighted negative log likelihood
        return -jnp.mean(log_probs)

def update_step(params, opt_state, maf, x, sample_weights, optimizer):
    """Single optimization step for MAF training."""
    # Define a jitted inner function that doesn't take maf as an argument
    @jax.jit
    def _update_step(params, opt_state, x, sample_weights):
        if sample_weights is not None:
            loss_fn = lambda p: weighted_maximum_likelihood_loss(p, maf, x, sample_weights)
        else:
            loss_fn = lambda p: weighted_maximum_likelihood_loss(p, maf, x, None)
        
        loss_value, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_value
    
    return _update_step(params, opt_state, x, sample_weights)

def fit_maf(maf, params, data, batch_size=1024, learning_rate=1e-3, num_epochs=100, 
           sample_weights=None, key=None):
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    
    # Initialize parameters and optimizer state
    opt_state = optimizer.init(params)
    
    # Get data size and prepare for batching
    data_size = data.shape[0]
    steps_per_epoch = data_size // batch_size
    # Calculate the actual batch size for the last batch
    last_batch_size = data_size - (steps_per_epoch * batch_size)
    if last_batch_size > 0:
        steps_per_epoch += 1
    total_steps = steps_per_epoch * num_epochs
    
    # Initialize losses list
    losses = []
    epoch_losses = []
    
    # Get number of devices for parallelization
    num_devices = jax.device_count()
    print(f"Training MAF using {num_devices} devices")
    
    # Define a pmapped update function to parallelize across devices
    @jax.pmap
    def parallel_update(params_device, opt_state_device, batch_x_device, batch_weights_device):
        def loss_fn(p):
            return weighted_maximum_likelihood_loss(p, maf, batch_x_device, batch_weights_device)
        
        loss_value, grads = jax.value_and_grad(loss_fn)(params_device)
        updates, new_opt_state = optimizer.update(grads, opt_state_device)
        new_params = optax.apply_updates(params_device, updates)
        return new_params, new_opt_state, loss_value
    
    # Replicate parameters and optimizer state across devices
    params_replicated = jax.device_put_replicated(params, jax.devices())
    opt_state_replicated = jax.device_put_replicated(opt_state, jax.devices())
    
    # Training loop with tqdm over all batches
    progress_bar = tqdm(total=total_steps, desc=f"Training MAF [Epoch 1/{num_epochs}]")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        # Create a new key for each epoch
        if key is not None:
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, data_size)
        else: # Without a key, just use simple slicing for batches
            indices = np.arange(data_size)
            np.random.shuffle(indices)
            perm = indices
        
        # Process each batch
        for idx in range(steps_per_epoch):
            # Calculate batch start/end indices
            start_idx = idx * batch_size
            end_idx = min(start_idx + batch_size, data_size)
            actual_batch_size = end_idx - start_idx
            
            # Extract batch data using the permutation
            batch_idx = perm[start_idx:end_idx]
            batch_x = data[batch_idx]
            
            # Extract batch weights if provided
            batch_weights = None
            if sample_weights is not None:
                batch_weights = sample_weights[batch_idx]
            
            # Pad batch to be divisible by number of devices
            if actual_batch_size % num_devices != 0:
                pad_size = num_devices - (actual_batch_size % num_devices)
                batch_x = jnp.pad(batch_x, ((0, pad_size), (0, 0)), mode='constant')
                if batch_weights is not None:
                    batch_weights = jnp.pad(batch_weights, (0, pad_size), mode='constant')
            
            # Reshape for pmap
            batch_x_shaped = batch_x.reshape(num_devices, -1, batch_x.shape[1])
            if batch_weights is not None:
                batch_weights_shaped = batch_weights.reshape(num_devices, -1)
            else:
                batch_weights_shaped = jnp.ones((num_devices, batch_x_shaped.shape[1]))
            
            # Update model in parallel across devices
            params_replicated, opt_state_replicated, losses_device = parallel_update(
                params_replicated, opt_state_replicated, batch_x_shaped, batch_weights_shaped
            )
            
            # Average loss across devices
            loss_value = jnp.mean(losses_device)
            epoch_loss += loss_value
            
            # Update progress bar with batch info
            progress_bar.set_description(f"Training MAF [Epoch {epoch+1}/{num_epochs}, Batch {idx+1}/{steps_per_epoch}, Loss: {loss_value:.4f}]")
            progress_bar.update(1)
        
        # Record average epoch loss
        avg_loss = epoch_loss / steps_per_epoch
        epoch_losses.append(avg_loss)
        
        # Print epoch summary
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"\nEpoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Update progress bar description for next epoch
        if epoch < num_epochs - 1:
            progress_bar.set_description(f"Training MAF [Epoch {epoch+2}/{num_epochs}]")
    
    progress_bar.close()
    
    # Get the parameters from the first device (they should be synchronized)
    params = jax.tree_map(lambda x: x[0], params_replicated)
    
    maf_trained = MAF(maf.input_dim, maf.hidden_dims, maf.num_flows, key=key)
    maf_trained.made_layers = maf.made_layers  # Copy the architecture
    
    # Return the trained model and losses
    return {"maf": maf_trained, "params": params, "losses": epoch_losses}

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
