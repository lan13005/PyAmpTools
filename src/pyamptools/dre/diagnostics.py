import numpy as np
import matplotlib.pyplot as plt
from jax import lax

def compute_reference_efficiency(X_acc, X_gen, weights_acc, weights_gen, feature_names):
    efficiency_dict = {}
    mins = {}
    maxs = {}
    # 1D reference efficiencies for all features
    for i, feature_name in enumerate(feature_names):
        mins[feature_name] = np.min(X_acc[:, i])
        maxs[feature_name] = np.max(X_acc[:, i])
        bins = np.linspace(mins[feature_name], maxs[feature_name], 50)
        _acc_hist = np.histogram(X_acc[:, i], weights=weights_acc, bins=bins)
        _gen_hist = np.histogram(X_gen[:, i], weights=weights_gen, bins=bins)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        efficiency_dict[feature_name] = (
            np.divide(_acc_hist[0], _gen_hist[0], where=_gen_hist[0] > 0, out=np.zeros_like(_acc_hist[0])),
            bin_centers
        )
    # 2D reference efficiencies for all pairs of features
    for i, feature_i in enumerate(feature_names):
        for j, feature_j in enumerate(feature_names):
            if i < j:  # Only compute for upper triangle
                bins_i = np.linspace(mins[feature_i], maxs[feature_i], 50)
                bins_j = np.linspace(mins[feature_j], maxs[feature_j], 50)            
                acc_hist = np.histogram2d(
                    X_acc[:, i], X_acc[:, j], 
                    bins=[bins_i, bins_j], 
                    weights=weights_acc
                )[0]
                gen_hist= np.histogram2d(
                    X_gen[:, i], X_gen[:, j], 
                    bins=[bins_i, bins_j], 
                    weights=weights_gen
                )[0]
                eff_2d = np.divide(
                    acc_hist, gen_hist, 
                    where=gen_hist > 0, 
                    out=np.zeros_like(acc_hist)
                )
                key_2d = f"{feature_i}_{feature_j}"
                efficiency_dict[key_2d] = (eff_2d, bins_i, bins_j)
                
    return efficiency_dict
                
# Calculate and plot 1D (diagonal) and 2D (off-diagonal) efficiency for each variable (pair of variables)
def plot_efficiency_by_variable(X_data, model_output, feature_names=None, nbins=50, figsize=(15, 15), 
                                efficiency_dict=None, checkpoint_dir=None, suffix=None,
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
    density_ratios = density_ratios
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