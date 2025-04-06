import os
import matplotlib.pyplot as plt

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
