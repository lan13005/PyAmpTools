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

from neural_dre_utils import load_checkpoint, save_checkpoint, load_and_use_model, plot_efficiency_by_variable, binary_cross_entropy

num_devices = jax.device_count()

# Define neural network model
class DensityRatioEstimator(nn.Module):
    hidden_dims: list = (64, 128, 64)
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training=True):
        for feat in self.hidden_dims:
            x = nn.Dense(features=feat)(x)
            x = nn.relu(x)
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

# class balance
class_ratio = np.sum(results['label'] == 1) / np.sum(results['label'] == 0)
print(f"Pre-balancing class ratio: {class_ratio:0.2f}x (acc/gen)")

# randomly select 25% of the events with label 0 and drop them from the results dataset
gen_ids = np.where(results['label'] == 0)[0]
acc_ids = np.where(results['label'] == 1)[0]
drop_ids = np.random.choice(gen_ids, size= int(1 * (len(gen_ids) - len(acc_ids))), replace=False)
results = results.drop(drop_ids)

# class balance
class_ratio = np.sum(results['label'] == 1) / np.sum(results['label'] == 0)
print(f"Post-balancing class ratio: {class_ratio:0.2f}x (acc/gen)")

percent = 1.0
train_size = int(percent * len(results))
print(f"Train size: {train_size}")
X_train = results.loc[:train_size, feature_names].values
weights_train = results.loc[:train_size, 'Weight'].values
y_train = results.loc[:train_size, 'label'].values


# standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_acc = results.loc[results['label'] == 1, feature_names].values
X_gen = results.loc[results['label'] == 0, feature_names].values
y_acc = results.loc[results['label'] == 1, 'label'].values
y_gen = results.loc[results['label'] == 0, 'label'].values
weights_acc = results.loc[results['label'] == 1, 'Weight'].values
weights_gen = results.loc[results['label'] == 0, 'Weight'].values

# Test on uniformly distributed points on the min-max domain of the generated data
#    Do not include min/max values from the accepted MC since smearing will leave to no generated events in the tails
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
key, subkey = random.split(random.PRNGKey(0))
params = model.init(subkey, jnp.ones((1, X_train.shape[1])))

# Create optimizer
print("Creating optimizer...")
learning_rate = 1e-3
optimizer = optax.adam(learning_rate)
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer
)
batch_size = 1024
num_epochs = 10
num_batches = int(np.ceil(len(X_train) / batch_size))

# Create checkpoint directory
print("Creating checkpoint directory...")
cwd = os.getcwd()
checkpoint_dir = f'{cwd}/model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_frequency = 5  # Save checkpoint every checkpoint_frequency epochs
plot_diagnostics_frequency = 1  # save diagnostics plots every plot_diagnostics_frequency epochs
plot_training_frequency = 1  # Update plots every iteration

# Create a dropout PRNG key
dropout_rng = random.PRNGKey(123)

# computed efficiency using 1D histograms
efficiency_dict = {}
for i, feature_name in enumerate(feature_names):
    minx = np.min(X_acc[:, i])
    maxx = np.max(X_acc[:, i])  
    bins = np.linspace(minx, maxx, 50)
    _acc_hist = np.histogram(X_acc[:, i], weights=weights_acc, bins=bins)
    _gen_hist = np.histogram(X_gen[:, i], weights=weights_gen, bins=bins)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    efficiency_dict[feature_name] = (
        np.divide(_acc_hist[0], _gen_hist[0], where=_gen_hist[0] > 0, out=np.zeros_like(_acc_hist[0])),
        bin_centers
    )

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
def train_step_parallel(state, batch_x, batch_y, dropout_rngs, batch_weights=None):

    def loss_fn(params):
        logits = model.apply(
            params, 
            batch_x, 
            training=True, 
            rngs={'dropout': dropout_rngs},
            mutable=['batch_stats']
        )
        logits = logits[0] if isinstance(logits, tuple) else logits
        loss = binary_cross_entropy(logits, batch_y, batch_weights)
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    # Calculate accuracy
    preds = jax.nn.sigmoid(logits)
    accuracy = jnp.mean((preds > 0.5) == batch_y)
    
    return state, loss, accuracy, dropout_rngs

@jit
def eval_step(params, x, y, batch_weights=None):
    logits = model.apply(params, x, training=False)
    loss = binary_cross_entropy(logits, y, batch_weights)
    preds = jax.nn.sigmoid(logits)
    accuracy = jnp.mean((preds > 0.5) == y)
    return loss, accuracy

def eval_model(params, x, y, weights=None, batch_size=1024):
    losses = []
    accuracies = []
    
    num_batches = int(np.ceil(len(x) / batch_size))
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(x))
        batch_x = x[start_idx:end_idx]
        batch_y = y[start_idx:end_idx]
        batch_weights = weights[start_idx:end_idx] if weights is not None else None
        loss, accuracy = eval_step(params, batch_x, batch_y, batch_weights)
        losses.append(loss)
        accuracies.append(accuracy)
    
    avg_loss = jnp.mean(jnp.array(losses))
    avg_accuracy = jnp.mean(jnp.array(accuracies))
    return avg_loss, avg_accuracy

# Initialize lists to store training and test losses
train_losses = []
test_losses = []
# Dictionary to track relative MAEs for each feature
feature_rel_maes = {name: [] for name in feature_names}

print("Training loop...")
if start_epoch < num_epochs:
    total_iterations = num_epochs * num_batches
    pbar = tqdm(total=total_iterations, desc="Training Progress", unit="batch")

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
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

            state, loss, accuracy, dropout_rngs = train_step_parallel(state, batch_x, batch_y, dropout_rngs, batch_weights)
            epoch_loss += loss.mean()
            epoch_accuracy += accuracy.mean()
                
            pbar.update(1)

        train_losses.append(epoch_loss / num_batches)
        train_accuracy = epoch_accuracy / num_batches
        
        pbar.set_postfix({
            "Epoch": epoch,
            "Train Acc": f"{train_accuracy:.4f}",
            "Train Loss": f"{train_losses[-1]:.4f}",
        })

        print(f"Epoch {epoch}: Train Acc: {train_accuracy:.4f}, Train Loss: {train_losses[-1]:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % checkpoint_frequency == 0 or epoch == num_epochs - 1 and checkpoint_dir is not None:
            save_checkpoint(state, epoch + 1, checkpoint_dir)
            
            
        # Draw diagnostics plots
        if (epoch + 1) % plot_diagnostics_frequency == 0 or epoch == num_epochs - 1 and checkpoint_dir is not None:
            single_device_params = jax.tree_util.tree_map(lambda x: x[0], state.params)
            print(f"calculating probabilities for {n_test_samples} events")
            probabilities = load_and_use_model(model, state, X_test, checkpoint_dir, step=None)
            print(f"plotting efficiency for these events")
            rel_maes = plot_efficiency_by_variable(X_test_raw, probabilities, feature_names=feature_names, 
                                        weight_rescaling=weight_rescaling, efficiency_dict=efficiency_dict, 
                                        checkpoint_dir=checkpoint_dir, suffix=f"epoch_{epoch+1}")
            
            # Store relative MAEs for each feature
            for i, feature_name in enumerate(feature_names):
                feature_rel_maes[feature_name].append(rel_maes[i])

        # Draw training loss curve with relative MAEs on twin axis
        if (epoch + 1) % plot_training_frequency == 0 or epoch == num_epochs - 1 and checkpoint_dir is not None:
            # Create figure with two subplots - one for loss, one for relative MAEs
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1.5]})
            
            # Plot training loss on top subplot
            color = 'tab:blue'
            ax1.set_ylabel('Loss', color=color)
            ax1.plot(range(start_epoch, start_epoch + len(train_losses)), train_losses, label='Training Loss', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper right')
            ax1.set_title('Training Loss')
            
            # Plot relative MAEs for each feature on bottom subplot
            if any(len(maes) > 0 for maes in feature_rel_maes.values()):
                # Create x-axis points for MAE (only epochs where we calculated MAE)
                mae_epochs = [start_epoch + i * plot_diagnostics_frequency for i in range(max(len(maes) for maes in feature_rel_maes.values()))]
                if (epoch + 1) % plot_diagnostics_frequency != 0 and epoch == num_epochs - 1:
                    # Add the final epoch if it's not a multiple of plot_diagnostics_frequency
                    mae_epochs.append(epoch)
                
                # Plot each feature's relative MAE as a separate line
                for feature_name, maes in feature_rel_maes.items():
                    if len(maes) > 0:
                        ax2.plot(mae_epochs[:len(maes)], maes, 'o-', label=f'{feature_name}')
                        
                # Plot average relative MAE
                if any(len(maes) > 0 for maes in feature_rel_maes.values()):
                    maes = np.array(list(feature_rel_maes.values()))
                    ax2.plot(mae_epochs, np.mean(maes, axis=0), 'o-', label='Average', color='black')

                ax2.set_xlabel('Epochs', fontsize=14)
                ax2.set_ylabel('Relative MAE', fontsize=14)
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc='upper right')
                ax2.set_title('Relative MAE by Feature', fontsize=14)
            
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

