import uproot as up
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax import random, pmap, jit
import jax.lax as lax
import flax.linen as nn
import optax
from orbax import checkpoint as orbax_checkpoint
from flax.training import train_state
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
import pickle as pkl

from neural_dre_utils import (
    load_checkpoint, save_checkpoint, load_and_use_model, plot_efficiency_by_variable, 
    combined_loss_adaptive, loss_type_map,
    create_corner_plot, fit_maf, save_maf_model, load_maf_model, MAF, batch_transform_maf
)

# Set device configuration
use_gpu = False  # Set to False to use CPU
cpu_cores = 24  # Number of CPU cores to use if not using GPU

# Configure device
if use_gpu and jax.devices('gpu'):
    print(f"Using GPU for training: {jax.devices('gpu')[0]}") # Already performs a check if GPU is available
    # No need to set XLA_FLAGS for GPU
    num_devices = 1  # Use a single GPU
else:
    # Set CPU cores for training
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cpu_cores}"
    num_devices = cpu_cores
    print(f"Using {num_devices} CPU cores for training")

# Set a seed for reproducibility
seed = 42
key = random.PRNGKey(seed)

feature_names = ["mMassX", "mCosHel", "mPhiHel", "mt", "mPhi"]
standardized_dump = "standardized_dump.pkl"

########################################################
############### Normalizing flow pretraining ###########
########################################################
use_flow_pretraining = False
flow_hidden_dims = [128, 128] # Hidden dimensions for flow networks
num_flow_layers = 5           # Number of flow transformations
flow_batch_size = 8192        
flow_learning_rate = 1e-4     
flow_num_epochs = 10         # Max epochs for flow training
n_test_samples = 2000000      # Evaluation test size
flow_patience = 10            # Early stopping patience for flow training
flow_clip_norm = 0.1          # Prevent large gradients
flow_adam_b1 = 0.9            # Adam beta1 parameter
flow_adam_b2 = 0.999          # Adam beta2 parameter
flow_adam_eps = 1e-8          # Adam epsilon parameter
flow_warmup_steps = 500       # Add warmup to stabilize initial training
flow_weight_decay = 1e-5      # Add weight decay to stabilize

########################################################
############### Classifier training ####################
########################################################
# Add a flag to control which loss function to use
# Options: "bce" (default), "mse", "mlc", or "sqrt"
loss_type = "bce"  # Change this to select the loss function
loss_type_code = loss_type_map[loss_type] # Mapping loss types to integer codes for jax device replication
metric_type = "standard"
metric_label = {"standard": "MAE", "relative": "Rel. MAE"}
# Hyperparameters for adaptive gradient regularization
reg_strength = 0.0001  # Strength of the gradient regularization, if 0 then do not add gradient regularization
transition_sensitivity = 0.5  # Controls how quickly regularization decreases in high-gradient regions
learning_rate = 1e-3
weight_decay = 1e-4
batch_size = 1024
num_epochs = 20

########################################################
############## Plotting and checkpointing ##############
########################################################
### FLOW
plot_flow_diagnostic_frequency = 10  # save diagnostics plots for flow frequency (epochs)
n_samples_flow_diagnostic = 5000  # Number of samples to use for flow diagnostic plots (triangular distribution plots)
### CLASSIFIER
checkpoint_frequency = 5  # Save checkpoint every checkpoint_frequency epochs
plot_classifier_diagnostic_frequency = 5  # save diagnostics plots for classifier frequency (epochs)
plot_training_frequency = 1  # Update plots every iteration
print(f"Using loss function: {loss_type}")

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

if os.path.exists(standardized_dump):
    print("Loading standardized data...")
    with open(standardized_dump, 'rb') as f:
        results = pkl.load(f)
    X_train = results['X_train']
    weights_train = results['weights_train']
    y_train = results['y_train']
    X_acc = results['X_acc']
    X_gen = results['X_gen']
    y_acc = results['y_acc']
    y_gen = results['y_gen']
    weights_acc = results['weights_acc']
    weights_gen = results['weights_gen']
    if 'scaler' in results:
        scaler = results['scaler']
    else:
        print("WARNING: Scaler not found in saved data. Creating a new scaler.")
        # Create and fit a new scaler using the training data
        scaler = StandardScaler()
        scaler.fit(X_train)
else:
    print("Loading data...")
    with open('full_dump.pkl', 'rb') as f:
        results = pkl.load(f)

    # class balance
    class_ratio = np.sum(results['label'] == 1) / np.sum(results['label'] == 0)
    print(f"Pre-balancing class ratio: {class_ratio:0.2f}x (acc/gen)")

    ########################################################
    ### NOTE: This block tests whether the imbalanced nature of the dataset is a problem
    #         This will directly affect the estimated efficiencvy but can be useful to probe
    #         how this model responds
    # # randomly select 25% of the events with label 0 and drop them from the results dataset
    # if os.path.exists('results_balanced.pkl'):
    #     print("pre-balancing results already exist, loading...")
    #     with open('results_balanced.pkl', 'rb') as f:728291
    
    #         results = pkl.load(f)
    # else:
    #     gen_ids = np.where(results['label'] == 0)[0]
    #     acc_ids = np.where(results['label'] == 1)[0]
    #     drop_ids = np.random.choice(gen_ids, size= int(1 * (len(gen_ids) - len(acc_ids))), replace=False)
    #     results = results.drop(drop_ids)
    #     with open('results_balanced.pkl', 'wb') as f:
    #         pkl.dump(results, f)
            
    # print(f"Shape of results: {results.shape}")
    # results = results.sample(frac=0.01).reset_index(drop=True)
    # print(f"Shape of results after sampling: {results.shape}")
    ########################################################

    percent = 100
    train_size = int(percent / 100 * len(results)) - 1 # zero indexed
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
    
    standardized_results = {
        'X_train': X_train,
        'weights_train': weights_train,
        'y_train': y_train,
        'X_acc': X_acc,
        'X_gen': X_gen,
        'y_acc': y_acc,
        'y_gen': y_gen,
        'weights_acc': weights_acc,
        'weights_gen': weights_gen,
        'scaler': scaler  # Save the scaler too
    }
    with open(standardized_dump, 'wb') as f:
        pkl.dump(standardized_results, f)

# Test on uniformly distributed points on the min-max domain of the generated data
#    Do not include min/max values from the accepted MC since smearing will leave to no generated events in the tails
print("Generating test data (uniform sampling on generated data domain)...")
min_vals = np.min(X_gen, axis=0)
max_vals = np.max(X_gen, axis=0)
X_test_raw = np.random.uniform(min_vals, max_vals, size=(n_test_samples, X_gen.shape[1]))
X_test = scaler.transform(X_test_raw)

# subsample the generated data for flow diagnostic plots
X_gen_flow_plot = X_gen[np.random.choice(X_gen.shape[0], size=n_samples_flow_diagnostic, replace=False)]
X_acc_flow_plot = X_acc[np.random.choice(X_acc.shape[0], size=n_samples_flow_diagnostic, replace=False)]

num_batches = int(np.ceil(len(X_train) / batch_size))

weight_rescaling = 1.0
print(f"weight_rescaling: {weight_rescaling}")


cwd = os.getcwd()
checkpoint_dir = f'{cwd}/model_checkpoints'
print("Creating checkpoint directory...")
os.makedirs(checkpoint_dir, exist_ok=True)


##############################################
# TRAINING FUNCTIONS
##############################################

##############################################
# BEGIN NORMALIZING FLOW PRETRAINING
##############################################

if use_flow_pretraining:
    print("Initializing normalizing flow pretraining...")
    
    # Create key for flow initialization
    key, flow_key = random.split(key)
    
    # Check if a previously trained flow model exists
    flow_save_path = f"{checkpoint_dir}/maf_model.pkl"
    if os.path.exists(flow_save_path):
        print(f"Loading pre-trained flow model from {flow_save_path}")
        try:
            maf_result = load_maf_model(flow_save_path)
            maf = maf_result["maf"]
            maf_params = maf_result["params"]
        except Exception as e:
            print(f"Error loading flow model: {e}")
            print("Initializing new flow model instead")
            # Initialize the MAF model
            print(f"Initializing MAF with {num_flow_layers} layers and {flow_hidden_dims} hidden dimensions")
            maf = MAF(X_train.shape[1], flow_hidden_dims, num_flow_layers, flow_key)
            maf_params = maf.init_params()
    else:
        # Initialize the MAF model
        print(f"Initializing MAF with {num_flow_layers} layers and {flow_hidden_dims} hidden dimensions")
        maf = MAF(X_train.shape[1], flow_hidden_dims, num_flow_layers, flow_key)
        maf_params = maf.init_params()
    
    # Create corner plot for original data
    print("Creating corner plot for original data...")
    create_corner_plot(
        X_gen_flow_plot, X_acc_flow_plot,
        labels=['Generated', 'Accepted'],
        feature_names=feature_names,
        filename='original_corner_plot.png',
        checkpoint_dir=checkpoint_dir,
    )
    
    # standard scale X_acc_flow_plot and X_gen_flow_plot for diagnostic plots, overwrite since we dont need for anything else
    X_acc_flow_plot = scaler.transform(X_acc_flow_plot)
    X_gen_flow_plot = scaler.transform(X_gen_flow_plot)
    
    # Train the flow model
    print(f"Training MAF for {flow_num_epochs} epochs...")
    
    key, train_key = random.split(key)
    maf_result = fit_maf(
        maf,
        maf_params,
        jnp.array(X_train),
        train_key,
        #### TRAINING ####
        batch_size=flow_batch_size,
        learning_rate=flow_learning_rate,
        num_epochs=flow_num_epochs,
        patience=flow_patience,
        sample_weights=jnp.array(weights_train) if weight_rescaling != 1.0 else None,
        #### FLOW OPTIMIZER ####
        clip_norm=flow_clip_norm,
        adam_b1=flow_adam_b1,
        adam_b2=flow_adam_b2,
        adam_eps=flow_adam_eps,
        #### PLOTTING ####
        plot_frequency=plot_flow_diagnostic_frequency,
        checkpoint_dir=checkpoint_dir,
        feature_names=feature_names,
        X_gen=X_gen_flow_plot,
        X_acc=X_acc_flow_plot,
        use_gpu=use_gpu,
        n_samples_flow_diagnostic=n_samples_flow_diagnostic,
        warmup_steps=flow_warmup_steps,
        weight_decay=flow_weight_decay
    )
    
    maf = maf_result["maf"]
    maf_params = maf_result["params"]
    flow_losses = maf_result["losses"]
    
    # Save the trained flow model
    save_maf_model({"maf": maf, "params": maf_params, "losses": flow_losses}, flow_save_path)
    
    # Transform data to latent space using batches to prevent OOM
    print("Transforming data to latent space in batches...")
    X_train_latent, _ = batch_transform_maf(maf, maf_params, X_train, batch_size=1024, direction="forward")
    X_acc_latent_flow_plot, _ = batch_transform_maf(maf, maf_params, X_acc_flow_plot, batch_size=1024, direction="forward")
    X_gen_latent_flow_plot, _ = batch_transform_maf(maf, maf_params, X_gen_flow_plot, batch_size=1024, direction="forward")
    X_test_latent, _ = batch_transform_maf(maf, maf_params, X_test, batch_size=1024, direction="forward")
    
    # Create corner plot for transformed data
    print("Creating corner plot for latent space data...")
    create_corner_plot(
        X_gen_latent_flow_plot, X_acc_latent_flow_plot,
        labels=['Generated', 'Accepted'],
        feature_names=[f'z{i+1}' for i in range(X_train.shape[1])],
        filename='latent_corner_plot.png',
        checkpoint_dir=checkpoint_dir,
    )
    
    # Use latent representations for training
    X_train = np.array(X_train_latent)
    X_test = np.array(X_test_latent)
else:
    print("Skipping flow pretraining, using original data directly.")

##############################################
# BEGIN NEURAL DRE TRAINING
##############################################

# Initialize model
print("Initializing neural DRE model...")
model = DensityRatioEstimator()
key, model_init_key = random.split(key)
params = model.init(model_init_key, jnp.ones((1, X_train.shape[1])))

# Create optimizer
print("Creating optimizer...")
# optimizer = optax.adam(learning_rate)
optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)  # L2 regularization
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer
)

# Create a dropout PRNG key
dropout_rng = random.PRNGKey(123)

#######################################
# COMPUTED EFFICIENCY USING HISTOGRAMS
#######################################
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

# Check if checkpoints exist and attempt to resume training
print("Checking if checkpoints exist and attempting to resume training...")
start_epoch = 0
train_losses = []
main_losses = []  # Will store main loss component (BCE, MSE, MLC, etc.)
grad_losses = []
accuracies = []
feature_metrics = {feature_name: [] for feature_name in feature_names}

try:
    # Try to load the latest checkpoint
    restored_state = load_checkpoint(checkpoint_dir)
    if restored_state is not None:
        # Extract the epoch from the checkpoint
        start_epoch = restored_state.get('epoch', 0)
        print(f"Resuming training from epoch {start_epoch}")
        
        # Extract the metrics from the checkpoint
        train_losses = restored_state.get('train_losses', [])
        main_losses = restored_state.get('main_losses', [])
        grad_losses = restored_state.get('grad_losses', [])
        accuracies = restored_state.get('accuracies', [])
        feature_metrics = restored_state.get('feature_metrics', {feature_name: [] for feature_name in feature_names})
        
        # Update the state with the restored parameters
        state = state.replace(params=restored_state['params'])
    else:
        print("No valid checkpoint found. Starting training from scratch.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    print("Starting training from scratch.")
    start_epoch = 0
    train_losses = []
    main_losses = []
    grad_losses = []
    accuracies = []
    feature_metrics = {feature_name: [] for feature_name in feature_names}

##############################################
# TRAINING FUNCTIONS
##############################################
def convert_to_probabilities(model_outputs, loss_type_code):
    """
    Depending on the activation func + loss function pair either the probabilities are learned
    or the likelihood ratio directly. Either way, there is a clear formula to convert ot 
    probabilities
    """
    preds = lax.cond(
        loss_type_code < 2,
        lambda _: jax.nn.sigmoid(model_outputs),
        lambda _: jnp.exp(model_outputs),
        operand=None
    )
    probs = lax.cond(
        loss_type_code < 2,
        lambda _: preds,
        lambda _: preds / (1 + preds), # convert to probability space
        operand=None
    )
    return probs

# Define both parallel and single-device training functions
def train_step_single(state, batch_x, batch_y, dropout_rng, batch_weights=None, 
                     reg_strength=None, transition_sensitivity=None, loss_type_code=None):
    """Single-device training step (for GPU)"""
    def loss_fn(params):
        dropout_rng_split, new_dropout_rng = random.split(dropout_rng)
        model_outputs = model.apply(
            params, 
            batch_x, 
            training=True, 
            rngs={'dropout': dropout_rng_split},
            mutable=['batch_stats']
        )
        model_outputs = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs
        
        # Compute loss with the selected loss function
        total_loss, (main_loss, grad_loss) = combined_loss_adaptive(
            model_outputs, batch_y, model, params, batch_x, 
            reg_strength=reg_strength, 
            transition_sensitivity=transition_sensitivity,
            weights=batch_weights, 
            rngs={'dropout': dropout_rng_split},
            loss_type_code=loss_type_code
        )
        
        return total_loss, (model_outputs, main_loss, grad_loss, new_dropout_rng)

    (loss, (model_outputs, main_loss, grad_loss, new_dropout_rng)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    probs = convert_to_probabilities(model_outputs, loss_type_code)
    accuracy = jnp.mean((probs > 0.5) == batch_y)
    return state, loss, accuracy, new_dropout_rng, main_loss, grad_loss

# Keep the original pmap function for multi-CPU training
@pmap
def train_step_parallel(state, batch_x, batch_y, dropout_rngs, batch_weights=None, 
                        reg_strength=None, transition_sensitivity=None, loss_type_code=None):
    """Multi-device parallel training step (for CPU)"""
    def loss_fn(params):
        model_outputs = model.apply(
            params, 
            batch_x, 
            training=True, 
            rngs={'dropout': dropout_rngs},
            mutable=['batch_stats']
        )
        model_outputs = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs
        
        # Compute loss with the selected loss function
        total_loss, (main_loss, grad_loss) = combined_loss_adaptive(
            model_outputs, batch_y, model, params, batch_x, 
            reg_strength=reg_strength, 
            transition_sensitivity=transition_sensitivity,
            weights=batch_weights, 
            rngs={'dropout': dropout_rngs},
            loss_type_code=loss_type_code
        )
        
        return total_loss, (model_outputs, main_loss, grad_loss)

    (loss, (model_outputs, main_loss, grad_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    probs = convert_to_probabilities(model_outputs, loss_type_code)
    accuracy = jnp.mean((probs > 0.5) == batch_y)
    return state, loss, accuracy, dropout_rngs, main_loss, grad_loss

# JIT the single-device training function
train_step_single_jit = jit(train_step_single)

##############################################
# TRAINING LOOP
##############################################
# Initialize dropout RNG
dropout_rng = random.PRNGKey(123)

# Prepare for training based on device type
if use_gpu and jax.devices('gpu'):
    # Single GPU training setup
    print("Setting up for single GPU training")
    # No need to replicate state or parameters
else:
    # Multi-CPU training setup
    print(f"Setting up for {num_devices} CPU cores training")
    # Replicate state across devices
    state = jax.device_put_replicated(state, jax.local_devices())
    dropout_rngs = jax.random.split(dropout_rng, num_devices)
    loss_type_code_devices = jnp.array([loss_type_code] * num_devices)
    reg_strength_devices = jnp.array([reg_strength] * num_devices)
    transition_sensitivity_devices = jnp.array([transition_sensitivity] * num_devices)

if start_epoch < num_epochs:
    print(f"Starting training from epoch {start_epoch} to {num_epochs}...")
    
    # Create progress bar
    pbar = tqdm(total=(num_epochs - start_epoch), desc=f"Epoch {start_epoch}")
    
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_main_loss = 0
        epoch_grad_loss = 0
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(X_train))
            
            # Get the current batch
            batch_x = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            batch_weights = weights_train[start_idx:end_idx]
            
            if use_gpu and jax.devices('gpu'):
                # Single GPU training step
                state, loss, accuracy, dropout_rng, main_loss, grad_loss = train_step_single_jit(
                    state, batch_x, batch_y, dropout_rng, batch_weights, 
                    reg_strength, transition_sensitivity, loss_type_code
                )
                
                # Update metrics (no need to aggregate across devices)
                epoch_loss += loss
                epoch_accuracy += accuracy
                epoch_main_loss += main_loss
                epoch_grad_loss += grad_loss
            else:
                # Multi-CPU training step
                # Pad the batch to be divisible by num_devices
                if len(batch_x) % num_devices != 0:
                    pad_size = num_devices - (len(batch_x) % num_devices)
                    batch_x = np.pad(batch_x, ((0, pad_size), (0, 0)), mode='constant')
                    batch_y = np.pad(batch_y, (0, pad_size), mode='constant')
                    batch_weights = np.pad(batch_weights, (0, pad_size), mode='constant')
                
                # Reshape for pmap
                batch_x = batch_x.reshape(num_devices, -1, X_train.shape[1])
                batch_y = batch_y.reshape(num_devices, -1)
                batch_weights = batch_weights.reshape(num_devices, -1)
                
                # train_step_parallel uses pmap so return is a list of length num_devices
                state, loss, accuracy, dropout_rngs, main_loss, grad_loss = train_step_parallel(
                    state, batch_x, batch_y, dropout_rngs, batch_weights, 
                    reg_strength_devices, transition_sensitivity_devices, loss_type_code_devices
                )
                
                # Update metrics (AGGREGATED ACROSS COMPUTE DEVICES)
                epoch_loss += jnp.mean(loss)
                epoch_accuracy += jnp.mean(accuracy)
                epoch_main_loss += jnp.mean(main_loss)
                epoch_grad_loss += jnp.mean(grad_loss)
        
        # Compute average metrics for the epoch
        epoch_loss /= num_batches
        epoch_accuracy /= num_batches
        epoch_main_loss /= num_batches
        epoch_grad_loss /= num_batches
        
        # Store metrics
        train_losses.append(float(epoch_loss))
        accuracies.append(float(epoch_accuracy))
        main_losses.append(float(epoch_main_loss))
        grad_losses.append(float(epoch_grad_loss))
        
        # Update progress bar after each epoch
        pbar.update(1)
        pbar.set_description(f"Epoch {epoch} - Loss: {epoch_loss:.4e}, Acc: {epoch_accuracy:.4f}")
        
        # Print metrics
        print(f"Epoch {epoch}: Train Acc: {epoch_accuracy:.4f}, Train Loss: {train_losses[-1]:.4f}, "
              f"Main Loss ({loss_type}): {main_losses[-1]:.4f}, Grad Loss: {grad_losses[-1]:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % checkpoint_frequency == 0 or epoch == num_epochs - 1 and checkpoint_dir is not None:
            # Add metrics to the state for resuming training
            if use_gpu and jax.devices('gpu'):
                # For GPU, no need to extract from device array
                checkpoint_state = {
                    'params': state.params,
                    'epoch': epoch + 1,
                    'train_losses': train_losses,
                    'main_losses': main_losses,
                    'grad_losses': grad_losses,
                    'accuracies': accuracies,
                    'feature_metrics': feature_metrics,
                    'loss_type': loss_type  # Save the loss type
                }
            else:
                # For multi-CPU, extract from first device
                single_device_params = jax.tree_util.tree_map(lambda x: x[0], state.params)
                checkpoint_state = {
                    'params': single_device_params,
                    'epoch': epoch + 1,
                    'train_losses': train_losses,
                    'main_losses': main_losses,
                    'grad_losses': grad_losses,
                    'accuracies': accuracies,
                    'feature_metrics': feature_metrics,
                    'loss_type': loss_type  # Save the loss type
                }
            save_checkpoint(checkpoint_state, epoch + 1, checkpoint_dir)
            
        # Draw diagnostics plots
        if (epoch + 1) % plot_classifier_diagnostic_frequency == 0 or epoch == num_epochs - 1 and checkpoint_dir is not None:
            if use_gpu and jax.devices('gpu'):
                single_device_params = state.params
            else:
                single_device_params = jax.tree_util.tree_map(lambda x: x[0], state.params)
            
            print(f"calculating density ratios for {n_test_samples} events")
            model_outputs = load_and_use_model(model, state, X_test, checkpoint_dir, step=None, loss_type_code=loss_type_code)
            print(f"plotting efficiency for these events")
            metrics = plot_efficiency_by_variable(X_test_raw, model_outputs, feature_names=feature_names, 
                                        weight_rescaling=weight_rescaling, efficiency_dict=efficiency_dict, 
                                        checkpoint_dir=checkpoint_dir, suffix=f"epoch_{epoch+1}", 
                                        loss_type_code=loss_type_code, metric_type='standard', plot_generated=True, plot_predicted=True)
            
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
            ax2.plot(range(start_epoch, start_epoch + len(main_losses)), main_losses, 
                    label=f'{loss_type.upper()} Loss', color='tab:orange')
            ax2.plot(range(start_epoch, start_epoch + len(grad_losses)), grad_losses, 
                    label='Gradient Reg Loss', color='tab:green')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right')
            ax2.set_title('Loss Components')
            
            # Plot metrics for each feature on bottom subplot
            if any(len(metrics) > 0 for metrics in feature_metrics.values()):
                # Create x-axis points for MAE (only epochs where we calculated MAE)
                metrics_epochs = []
                for i in range(max(len(metrics) for metrics in feature_metrics.values())):
                    metrics_epoch = start_epoch + i * plot_classifier_diagnostic_frequency + plot_classifier_diagnostic_frequency
                    metrics_epochs.append(metrics_epoch)
                
                # Add the final epoch if it's not already included
                if (epoch + 1) % plot_classifier_diagnostic_frequency != 0 and epoch == num_epochs - 1:
                    metrics_epochs.append(epoch + 1)  # +1 because we're at the end of the epoch
                
                # Find the epoch with the lowest MAE
                mean_metrics = np.mean(np.array(list(feature_metrics.values())), axis=0)
                min_metrics_epoch_idx = np.argmin(mean_metrics)
                min_metrics_epoch = metrics_epochs[min_metrics_epoch_idx]
                min_metrics_value = mean_metrics[min_metrics_epoch_idx]
                current_metrics_value = mean_metrics[-1]
                metrics_difference = current_metrics_value - min_metrics_value
                
                # Plot each feature's metric
                for feature_name, metrics in feature_metrics.items():
                    if len(metrics) > 0:
                        last_metrics = metrics[-1] if metrics else 0.0
                        ax3.plot(metrics_epochs[:len(metrics)], metrics, '-', 
                                label=f'{feature_name} (Last {metric_label[metric_type]}: {last_metrics:.3f})')
                        
                # Plot average metric
                if any(len(metrics) > 0 for metrics in feature_metrics.values()):
                    ax3.plot(metrics_epochs[:len(mean_metrics)], mean_metrics, '--', 
                            label=f'Average (Last {metric_label[metric_type]}: {current_metrics_value:.3f})', 
                            color='black')

                ax3.set_xlabel('Epochs', fontsize=14)
                ax3.set_ylabel(f'{metric_label[metric_type]}', fontsize=14)
                ax3.grid(True, alpha=0.3)
                ax3.legend(loc='upper right')
                ax3.set_title(f'{metric_label[metric_type]} by Feature ' + 
                             f'(Lowest {metric_label[metric_type]} at Epoch {min_metrics_epoch}: {min_metrics_value:.3f}, ' + 
                             f'Current-Best: {metrics_difference:.3f})', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{checkpoint_dir}/training_curve.png')
            plt.close()

        # After each epoch, add debugging code to check predictions
        if batch == 0 and epoch % 5 == 0:  # Every 5 epochs, first batch
            # Get a small sample for debugging
            sample_x = X_train[:10]
            if use_gpu and jax.devices('gpu'):
                params = state.params
            else:
                params = jax.tree_map(lambda x: x[0], state.params)
            
            # Get raw model outputs
            raw_outputs = model.apply({'params': params}, sample_x, training=False)
            print(f"DEBUG - Raw model outputs (first 10 samples): {raw_outputs.flatten()}")
            
            # Get probabilities 
            probs = jax.nn.sigmoid(raw_outputs)
            print(f"DEBUG - Probabilities: {probs.flatten()}")
            
            # Get density ratios
            ratios = probs / (1 - probs)
            print(f"DEBUG - Density ratios: {ratios.flatten()}")

    pbar.close()

###########################################

# print("Loading and using model...")
# model_outputs = load_and_use_model(model, state, _X_data, checkpoint_dir, step=None, loss_type=loss_type)
# model_outputs = model_outputs.reshape(-1) # (nsamples, 1) -> (nsamples)

# print("Plotting efficiencies...")
# plot_efficiency_by_variable(scaler.inverse_transform(_X_data), model_outputs, feature_names=feature_names, 
#                           weight_rescaling=weight_rescaling, efficiency_dict=efficiency_dict, 
#                           checkpoint_dir=checkpoint_dir, loss_type=loss_type)

# print("Saving efficiency data...")
# # Save the calculated efficiencies
# try:
#     efficiency_data = {
#         'X_data': _X_data,
#         'model_outputs': np.array(model_outputs),
#         'weights': _weights,
#         'loss_type': loss_type
#     }
#     with open(f'efficiency_data_{loss_type}.pkl', 'wb') as f:
#         pkl.dump(efficiency_data, f)
#     print(f"Efficiency data saved to 'efficiency_data_{loss_type}.pkl'")
# except Exception as e:
#     print(f"Error saving efficiency data: {e}")
