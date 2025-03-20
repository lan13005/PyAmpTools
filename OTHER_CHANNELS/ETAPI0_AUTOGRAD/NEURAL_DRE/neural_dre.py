import uproot as up
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=12"

import jax
import jax.numpy as jnp
from jax import random, pmap, jit
import flax.linen as nn
import optax
from orbax import checkpoint as orbax_checkpoint
from flax.training import train_state
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
import pickle as pkl

from neural_dre_utils import load_checkpoint, save_checkpoint, load_and_use_model, plot_efficiency_by_variable, binary_cross_entropy, gradient_regularization_loss, adaptive_gradient_regularization_loss, combined_loss_adaptive

# Set a seed for reproducibility
seed = 42
key = random.PRNGKey(seed)
num_devices = jax.device_count()

# Define neural network model
class DensityRatioEstimator(nn.Module):
    hidden_dims: list = (64, 64)
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x, training=True):
        for feat in self.hidden_dims:
            x = nn.Dense(features=feat)(x)
            # x = nn.relu(x)
            x = nn.leaky_relu(x, negative_slope=0.01)
            if training:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        x = nn.Dense(features=1)(x)
        return x

# DEEPER NETWORK DOES NOT PERFORM BETTER, WE MIGHT HAVE THE MOST POWERFUL ALREADY
# class DensityRatioEstimator(nn.Module):
#     hidden_dims: list = (128, 256, 128)
#     dropout_rate: float = 0.2
#     # TODO: consider adding batch normalization
#     @nn.compact
#     def __call__(self, x, training=True):
#         for feat in self.hidden_dims:
#             x = nn.Dense(features=feat)(x)
#             x = nn.relu(x)
#             if training:
#                 x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)                
#         x = nn.Dense(features=1)(x)
#         return x
    
##############################################
# BEGIN LOADING STUFF
##############################################

print("Loading data...")
with open('full_dump.pkl', 'rb') as f:
    results = pkl.load(f)
    
feature_names = ["mMassX", "mCosHel", "mPhiHel", "mt", "mPhi"]
metric_type = "standard"
metric_label = {"standard": "MAE", "relative": "Rel. MAE"}

# class balance
class_ratio = np.sum(results['label'] == 1) / np.sum(results['label'] == 0)
print(f"Pre-balancing class ratio: {class_ratio:0.2f}x (acc/gen)")

# randomly select 25% of the events with label 0 and drop them from the results dataset
if os.path.exists('results_balanced.pkl'):
    print("pre-balancing results already exist, loading...")
    with open('results_balanced.pkl', 'rb') as f:
        results = pkl.load(f)
else:
    gen_ids = np.where(results['label'] == 0)[0]
    acc_ids = np.where(results['label'] == 1)[0]
    drop_ids = np.random.choice(gen_ids, size= int(1 * (len(gen_ids) - len(acc_ids))), replace=False)
    results = results.drop(drop_ids)
    with open('results_balanced.pkl', 'wb') as f:
        pkl.dump(results, f)

percent = 100.0
train_size = int(percent / 100 * len(results))
print(f"Train size: {train_size}")
X_train = results.loc[:train_size, feature_names].values
weights_train = results.loc[:train_size, 'Weight'].values
y_train = results.loc[:train_size, 'label'].values

# standardize the data
print("Standardizing data with shape:", X_train.shape)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
print("Standardization complete...")

X_acc = results.loc[results['label'] == 1, feature_names].values
X_gen = results.loc[results['label'] == 0, feature_names].values
y_acc = results.loc[results['label'] == 1, 'label'].values
y_gen = results.loc[results['label'] == 0, 'label'].values
weights_acc = results.loc[results['label'] == 1, 'Weight'].values
weights_gen = results.loc[results['label'] == 0, 'Weight'].values

# Test on uniformly distributed points on the min-max domain of the generated data
#    Do not include min/max values from the accepted MC since smearing will leave to no generated events in the tails
print("Generating test data (uniform sampling on generated data domain)...")
n_test_samples = 2000000
min_vals = np.min(X_gen, axis=0)
max_vals = np.max(X_gen, axis=0)
X_test_raw = np.random.uniform(min_vals, max_vals, size=(n_test_samples, X_gen.shape[1]))
X_test = scaler.transform(X_test_raw)

weight_rescaling = 1.0
print(f"weight_rescaling: {weight_rescaling}")

# Initialize model
print("Initializing model...")
model = DensityRatioEstimator()
key, subkey = random.split(key)
params = model.init(subkey, jnp.ones((1, X_train.shape[1])))

# Create optimizer
print("Creating optimizer...")
learning_rate = 1e-3
# optimizer = optax.adam(learning_rate)
optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)  # L2 regularization
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer
)
batch_size = 1024
num_epochs = 20
num_batches = int(np.ceil(len(X_train) / batch_size))

# Create checkpoint directory
print("Creating checkpoint directory...")
cwd = os.getcwd()
checkpoint_dir = f'{cwd}/model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_frequency = 5  # Save checkpoint every checkpoint_frequency epochs
plot_diagnostics_frequency = 5  # save diagnostics plots every plot_diagnostics_frequency epochs
plot_training_frequency = 1  # Update plots every iteration

# Create a dropout PRNG key
dropout_rng = random.PRNGKey(123)

# computed efficiency using 1D histograms
efficiency_dict = {}
mins = {}
maxs = {}
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

# Check if checkpoints exist and attempt to resume training
print("Checking if checkpoints exist and attempting to resume training...")
try:
    checkpointer = orbax_checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax_checkpoint.CheckpointManager(
        directory=checkpoint_dir, 
        checkpointers={"model": checkpointer}
    )
    latest_step = checkpoint_manager.latest_step()
    
    if latest_step is not None:
        print(f"Found existing checkpoint at step {latest_step}. Resuming training...")
        restored_state = load_checkpoint(checkpoint_dir, latest_step)
        # Ensure restored_state is wrapped in TrainState
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=restored_state['params'],  # Assuming 'params' is the key for parameters
            tx=optimizer
        )
        # Replicate the restored state across devices for parallel training
        state = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_devices), state)
        start_epoch = latest_step  # Resume from the saved epoch
        if start_epoch == num_epochs:
            print("Training has already been completed.")
        else:
            print(f"Training will resume from epoch {start_epoch}")
    else:
        print("No existing checkpoints found. Starting training from scratch.")
        state = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_devices), state)
        start_epoch = 0
except Exception as e:
    print(f"Error checking for checkpoints: {e}")
    print("Starting training from scratch.")
    state = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_devices), state)
    start_epoch = 0


##############################################
# TRAINING FUNCTIONS
##############################################
@pmap
def train_step_parallel(state, batch_x, batch_y, dropout_rngs, batch_weights=None, reg_strength=None, transition_sensitivity=None, use_adaptive=None):

    def loss_fn(params):
        logits = model.apply(
            params, 
            batch_x, 
            training=True, 
            rngs={'dropout': dropout_rngs},
            mutable=['batch_stats']
        )
        logits = logits[0] if isinstance(logits, tuple) else logits
        
        # Compute adaptive loss
        adaptive_total_loss, (adaptive_bce_loss, adaptive_grad_loss) = combined_loss_adaptive(
            logits, batch_y, model, params, batch_x, 
            reg_strength=reg_strength, 
            transition_sensitivity=transition_sensitivity,
            weights=batch_weights, 
            rngs={'dropout': dropout_rngs}
        )
        
        # Return the adaptive loss (we're only using adaptive now)
        return adaptive_total_loss, (logits, adaptive_bce_loss, adaptive_grad_loss)

    (loss, (logits, bce_loss, grad_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    # Calculate accuracy
    preds = jax.nn.sigmoid(logits)
    accuracy = jnp.mean((preds > 0.5) == batch_y)
    
    return state, loss, accuracy, dropout_rngs, bce_loss, grad_loss

@jit
def eval_step(params, x, y, batch_weights=None, reg_strength=0.0001, transition_sensitivity=0.5):
    logits = model.apply(params, x, training=False)
    
    # Only use adaptive regularization for evaluation
    total_loss, (bce_loss, grad_loss) = combined_loss_adaptive(
        logits, y, model, params, x, 
        reg_strength=reg_strength,
        transition_sensitivity=transition_sensitivity,
        weights=batch_weights
    )
    
    preds = jax.nn.sigmoid(logits)
    accuracy = jnp.mean((preds > 0.5) == y)
    return total_loss, accuracy, bce_loss, grad_loss

def eval_model(params, x, y, weights=None, batch_size=1024, reg_strength=0.0001, transition_sensitivity=0.5):
    losses = []
    accuracies = []
    bce_losses = []
    grad_losses = []
    
    num_batches = int(np.ceil(len(x) / batch_size))
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(x))
        batch_x = x[start_idx:end_idx]
        batch_y = y[start_idx:end_idx]
        batch_weights = weights[start_idx:end_idx] if weights is not None else None
        loss, accuracy, bce_loss, grad_loss = eval_step(
            params, batch_x, batch_y, batch_weights, 
            reg_strength, transition_sensitivity
        )
        losses.append(loss)
        accuracies.append(accuracy)
        bce_losses.append(bce_loss)
        grad_losses.append(grad_loss)
    
    avg_loss = jnp.mean(jnp.array(losses))
    avg_accuracy = jnp.mean(jnp.array(accuracies))
    avg_bce_loss = jnp.mean(jnp.array(bce_losses))
    avg_grad_loss = jnp.mean(jnp.array(grad_losses))
    return avg_loss, avg_accuracy, avg_bce_loss, avg_grad_loss

# Initialize lists to store training and test losses
train_losses = []
bce_losses = []
grad_losses = []
test_losses = []
# Dictionary to track metrics for each feature
feature_metrics = {name: [] for name in feature_names}

# Set regularization strength hyperparameter
reg_strength = 10 # (1e-5, 1e-1)
transition_sensitivity = 100  # Controls how quickly regularization decreases in high-gradient regions (1e-1, 1e1)
print(f"Regularization strength: {reg_strength}")
print(f"Transition sensitivity: {transition_sensitivity}")
print(f"Using adaptive regularization: True")

# Broadcast hyperparameters across devices for pmap
reg_strength_broadcasted = jnp.array([reg_strength] * num_devices)
transition_sensitivity_broadcasted = jnp.array([transition_sensitivity] * num_devices)

print("Training loop...")
if start_epoch < num_epochs:
    total_iterations = num_epochs * num_batches
    pbar = tqdm(total=total_iterations, desc="Training Progress", unit="batch")

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_bce_loss = 0.0
        epoch_grad_loss = 0.0
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(X_train))
            batch_x = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            
            # Pad batch_weights to ensure it's divisible by num_devices
            remainder = (end_idx - start_idx) % num_devices
            pad_size = num_devices - remainder
            
            # Add weights to the batch if needed
            batch_weights = None
            if weights_train is not None:
                batch_weights = weights_train[start_idx:end_idx]
                if remainder > 0:
                    batch_weights = np.pad(batch_weights, (0, pad_size), mode='constant')
                batch_weights = batch_weights.reshape(num_devices, -1)  # Reshape like batch_y
            if remainder > 0:
                batch_x = np.pad(batch_x, ((0, pad_size), (0, 0)), mode='constant')
                batch_y = np.pad(batch_y, (0, pad_size), mode='constant')
                
            batch_x = batch_x.reshape(num_devices, -1, batch_x.shape[-1])  # (num_devices, batch_size_per_device, feature_dim)
            batch_y = batch_y.reshape(num_devices, -1)  # (num_devices, batch_size_per_device)
            
            dropout_rngs = random.split(dropout_rng, num_devices)

            state, loss, accuracy, dropout_rngs, bce_loss, grad_loss = train_step_parallel(
                state, batch_x, batch_y, dropout_rngs, batch_weights, 
                reg_strength_broadcasted, transition_sensitivity_broadcasted, None  # use_adaptive is no longer needed
            )
            epoch_loss += loss.mean()
            epoch_accuracy += accuracy.mean()
            epoch_bce_loss += bce_loss.mean()
            epoch_grad_loss += grad_loss.mean()
                
            pbar.update(1)

        train_losses.append(epoch_loss / num_batches)
        bce_losses.append(epoch_bce_loss / num_batches)
        grad_losses.append(epoch_grad_loss / num_batches)
        train_accuracy = epoch_accuracy / num_batches
        
        pbar.set_postfix({
            "Epoch": epoch,
            "Train Acc": f"{train_accuracy:.4f}",
            "Train Loss": f"{train_losses[-1]:.4f}",
            "BCE Loss": f"{bce_losses[-1]:.4f}",
            "Grad Loss": f"{grad_losses[-1]:.4f}",
        })

        print(f"Epoch {epoch}: Train Acc: {train_accuracy:.4f}, Train Loss: {train_losses[-1]:.4f}, "
              f"BCE Loss: {bce_losses[-1]:.4f}, Grad Loss: {grad_losses[-1]:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % checkpoint_frequency == 0 or epoch == num_epochs - 1 and checkpoint_dir is not None:
            save_checkpoint(state, epoch + 1, checkpoint_dir)
            
        # Draw diagnostics plots
        if (epoch + 1) % plot_diagnostics_frequency == 0 or epoch == num_epochs - 1 and checkpoint_dir is not None:
            single_device_params = jax.tree_util.tree_map(lambda x: x[0], state.params)
            print(f"calculating probabilities for {n_test_samples} events")
            probabilities = load_and_use_model(model, state, X_test, checkpoint_dir, step=None)
            print(f"plotting efficiency for these events")
            metrics = plot_efficiency_by_variable(X_test_raw, probabilities, feature_names=feature_names, 
                                        weight_rescaling=weight_rescaling, efficiency_dict=efficiency_dict, 
                                        checkpoint_dir=checkpoint_dir, suffix=f"epoch_{epoch+1}", metric_type=metric_type)
            
            # Store metrics for each feature
            for i, feature_name in enumerate(feature_names):
                feature_metrics[feature_name].append(metrics[i])

        # Draw training loss curve with metrics on twin axis
        if (epoch + 1) % plot_training_frequency == 0 or epoch == num_epochs - 1 and checkpoint_dir is not None:
            # Create figure with three subplots - one for loss, one for component losses, one for metrics
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True, 
                                               gridspec_kw={'height_ratios': [1, 1, 1.5]})
            
            # Plot training loss on top subplot
            color = 'tab:blue'
            ax1.set_ylabel('Total Loss', color=color)
            ax1.plot(range(start_epoch, start_epoch + len(train_losses)), train_losses, label='Total Loss', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper right')
            ax1.set_title('Training Loss')
            ax1.set_xlim(0, num_epochs * 1.2)
            
            # Plot component losses on middle subplot
            ax2.set_ylabel('Component Losses')
            ax2.plot(range(start_epoch, start_epoch + len(bce_losses)), bce_losses, label='BCE Loss', color='tab:orange')
            ax2.plot(range(start_epoch, start_epoch + len(grad_losses)), grad_losses, label='Gradient Reg Loss', color='tab:green')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right')
            ax2.set_title('Loss Components')
            
            # Plot metrics for each feature on bottom subplot
            if any(len(metrics) > 0 for metrics in feature_metrics.values()):
                # Create x-axis points for MAE (only epochs where we calculated MAE)
                metrics_epochs = []
                for i in range(max(len(metrics) for metrics in feature_metrics.values())):
                    metrics_epoch = start_epoch + i * plot_diagnostics_frequency + plot_diagnostics_frequency
                    metrics_epochs.append(metrics_epoch)
                
                # Add the final epoch if it's not already included
                if (epoch + 1) % plot_diagnostics_frequency != 0 and epoch == num_epochs - 1:
                    metrics_epochs.append(epoch + 1)  # +1 because we're at the end of the epoch
                
                # Find the epoch with the lowest MAE
                mean_metrics = np.mean(np.array(list(feature_metrics.values())), axis=0)
                min_metrics_epoch_idx = np.argmin(mean_metrics)
                min_metrics_epoch = metrics_epochs[min_metrics_epoch_idx]
                min_metrics_value = mean_metrics[min_metrics_epoch_idx]
                current_metrics_value = mean_metrics[-1]
                metrics_difference = current_metrics_value - min_metrics_value
                
                # Plot each feature's metric as a separate line
                for feature_name, metrics in feature_metrics.items():
                    if len(metrics) > 0:
                        last_metrics = metrics[-1] if metrics else 0.0
                        ax3.plot(metrics_epochs[:len(metrics)], metrics, '-', label=f'{feature_name} (Last {metric_label[metric_type]}: {last_metrics:.3f})')
                        
                # Plot average metric
                if any(len(metrics) > 0 for metrics in feature_metrics.values()):
                    ax3.plot(metrics_epochs[:len(mean_metrics)], mean_metrics, '--', label=f'Average (Last {metric_label[metric_type]}: {current_metrics_value:.3f})', color='black')

                ax3.set_xlabel('Epochs', fontsize=14)
                ax3.set_ylabel(f'{metric_label[metric_type]}', fontsize=14)
                ax3.grid(True, alpha=0.3)
                ax3.legend(loc='upper right')
                ax3.set_title(f'{metric_label[metric_type]} by Feature (Lowest {metric_label[metric_type]} at Epoch {min_metrics_epoch}: {min_metrics_value:.3f}, Current-Best: {metrics_difference:.3f})', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{checkpoint_dir}/training_curve.png')
            plt.close()

    pbar.close()

###########################################

# print("Loading and using model...")
# probabilities = load_and_use_model(model, state, _X_data, checkpoint_dir, step=None)  # step=None means latest checkpoint
# probabilities = probabilities.reshape(-1) # (nsamples, 1) -> (nsamples)

# print("Plotting efficiencies...")
# plot_efficiency_by_variable(scaler.inverse_transform(_X_data), probabilities, feature_names=feature_names, weight_rescaling=weight_rescaling, efficiency_dict=efficiency_dict, checkpoint_dir=checkpoint_dir)

# print("Saving efficiency data...")
# Save the calculated efficiencies
# try:
#     efficiency_data = {
#         'X_data': _X_data,
#         'probabilities': np.array(probabilities),
#         'weights': _weights
#     }
#     with open('efficiency_data.pkl', 'wb') as f:
#         pkl.dump(efficiency_data, f)
#     print("Efficiency data saved to 'efficiency_data.pkl'")
# except Exception as e:
#     print(f"Error saving efficiency data: {e}")

