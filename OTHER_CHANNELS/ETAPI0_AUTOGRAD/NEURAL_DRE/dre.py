import numpy as np
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
import jax.lax as lax
from flax import nnx
import optax
import numpy as np
import pickle as pkl

from pyamptools.dre.classifier_train import DensityRatioEstimator
from pyamptools.dre.classifier_loss import likelihood_ratio_grad_regularized_loss_fn, convert_to_probabilities
from pyamptools.dre.diagnostics import compute_reference_efficiency
from rich.console import Console
from pyamptools.dre.classifier_utils import save_checkpoint, load_checkpoint, try_resume_from_checkpoint

loss_type_map = {"bce": 0, "mse": 1, "mlc": 2, "sqrt": 3}
loss_type_map_reverse = {v: k for k, v in loss_type_map.items()}
metric_labels = {"standard": "MAE", "relative": "Rel. MAE"}

console = Console()

################################################################

feature_names = ["mMassX", "mCosHel", "mPhiHel", "mt", "mPhi"]
metric_type = "standard"
loss_type = "bce" # Options: "bce" (default), "mse", "mlc", or "sqrt"

n_test_samples = 2000000 # Used for Monte Carlo integration of 1D/2D proj

loss_type_code = loss_type_map[loss_type]
console.print(f"Using loss function: {loss_type}")

classifier_dims = (len(feature_names), 64, 64)
classifier_dropout_rate = 0.2
classifier_learning_rate = 1e-3
classifier_weight_decay = 1e-4

batch_size = 16384
num_epochs = 20
checkpoint_frequency = 1  # Save checkpoint every checkpoint_frequency epochs
plot_diagnostics_frequency = 1  # save diagnostics plots every plot_diagnostics_frequency epochs
plot_training_frequency = 1  # Update plots every iteration
adaptive_gradient_reg_strength = 0.0001  # Strength of the gradient regularization
adaptive_gradient_transition_sensitivity = 0.5  # Controls how quickly regularization decreases in high-gradient regions

### CHECKPOINTING
cwd = os.getcwd()
checkpoint_dir = f'{cwd}/model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

### SEEDING
seed = 42 
rngs = nnx.Rngs(seed)
num_devices = jax.device_count()
console.print(f"Using {num_devices} devices for training")

# Create a mesh for data parallelism
devices = np.array(jax.devices()).reshape(num_devices, 1)
mesh = jax.sharding.Mesh(devices, axis_names=('data', None))
console.print(f"Created device mesh with shape: {mesh}")

# Define sharding specs for different array ranks (2D=inputs, 1D=labels, 1D=weights)
data_sharding_2d = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('data', None))
data_sharding_1d = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('data'))

def split_for_devices(array, num_devices):
    """Split numpy array across devices, padding if necessary."""
    if len(array) % num_devices != 0:
        # Calculate padding needed then pad the array by repeating the couple elements
        pad_size = num_devices - (len(array) % num_devices)
        array = np.concatenate([array, array[:pad_size]])
    return array.reshape(num_devices, -1, *array.shape[1:])

@nnx.jit
def train_step(model, optimizer, batch, loss_type_code=0, 
               reg_strength=0.0, transition_sensitivity=0.5):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(likelihood_ratio_grad_regularized_loss_fn, has_aux=True)
    (loss, (logits, main_loss, grad_loss)), grads = grad_fn(
        model, batch, loss_type_code, reg_strength, transition_sensitivity
    )    
    optimizer.update(grads)
    probs = convert_to_probabilities(logits, loss_type_code)
    accuracy = jnp.mean((probs > 0.5) == batch['y'])
    
    return loss, accuracy, main_loss, grad_loss

@nnx.jit
def eval_step(model, batch, loss_type_code=0, 
              reg_strength=0.0, transition_sensitivity=0.5):
    """Evaluate for a single step."""
    loss, (logits, main_loss, grad_loss) = likelihood_ratio_grad_regularized_loss_fn(
        model, batch, loss_type_code, reg_strength, transition_sensitivity
    )
    
    # Calculate accuracy
    probs = convert_to_probabilities(logits, loss_type_code)
    accuracy = jnp.mean((probs > 0.5) == batch['y'])
    
    return loss, accuracy, main_loss, grad_loss, probs

@nnx.jit
def predict(model, x, loss_type_code=0):
    """Generate predictions from the model."""
    logits = model(x)
    return convert_to_probabilities(logits, loss_type_code)

##############################################
# BEGIN LOADING STUFF
##############################################

if os.path.exists('subset_dump.pkl'):
    print("Loading subset data...")
    with open('subset_dump.pkl', 'rb') as f:
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
    scaler = results['scaler']
else:
    print("Loading full data...")
    with open('full_dump.pkl', 'rb') as f:
        results = pkl.load(f)
    
    # class balance
    class_ratio = np.sum(results['label'] == 1) / np.sum(results['label'] == 0)
    print(f"Pre-balancing class ratio: {class_ratio:0.2f}x (acc/gen)")

    percent = 10
    train_size = int(percent / 100 * len(results)) - 1 # zero indexed
    print(f"Train size: {train_size}")

    X_train = results.loc[:train_size, feature_names].values
    weights_train = results.loc[:train_size, 'Weight'].values
    y_train = results.loc[:train_size, 'label'].values

    # standardize the data
    print("Standardizing data with shape:", X_train.shape)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    print("Standardization complete...")

    X_acc = results.loc[results['label'] == 1, feature_names].values
    X_gen = results.loc[results['label'] == 0, feature_names].values
    y_acc = results.loc[results['label'] == 1, 'label'].values
    y_gen = results.loc[results['label'] == 0, 'label'].values
    weights_acc = results.loc[results['label'] == 1, 'Weight'].values
    weights_gen = results.loc[results['label'] == 0, 'Weight'].values
    
    _results = {}
    _results['X_train'] = X_train
    _results['weights_train'] = weights_train
    _results['y_train'] = y_train
    _results['X_acc'] = X_acc
    _results['X_gen'] = X_gen
    _results['y_acc'] = y_acc
    _results['y_gen'] = y_gen
    _results['weights_acc'] = weights_acc
    _results['weights_gen'] = weights_gen
    _results['scaler'] = scaler
    
    with open('subset_dump.pkl', 'wb') as f:
        pkl.dump(_results, f)
    
num_batches = int(np.ceil(len(X_train) / batch_size))

# Test on uniformly distributed points on the min-max domain of the generated data
#    Do not include min/max values from the accepted MC since smearing will leave no generated events in the tails
print("Generating test data (uniform sampling on generated data domain)...")
min_vals = np.min(X_gen, axis=0)
max_vals = np.max(X_gen, axis=0)
X_test_raw = np.random.uniform(min_vals, max_vals, size=(n_test_samples, X_gen.shape[1]))
X_test = scaler.transform(X_test_raw)

################################################################
###################### TRAINING MAF ############################
################################################################



################################################################
################# TRAINING CLASSIFIER ##########################
################################################################

console.print("Initializing Classifier Model...")
model = DensityRatioEstimator(dims=(X_train.shape[1], 64, 64), dropout_rate=classifier_dropout_rate, rngs=rngs)

optimizer = nnx.Optimizer(
    model, 
    optax.adamw(
        learning_rate=classifier_learning_rate,
        weight_decay=classifier_weight_decay
    )
)

# Create metrics tracker
metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average('loss'),
    accuracy=nnx.metrics.Average('accuracy'),
    main_loss=nnx.metrics.Average('main_loss'),
    grad_loss=nnx.metrics.Average('grad_loss')
)

# Try to resume from checkpoint before starting training
start_epoch, _loss_type_code = try_resume_from_checkpoint(model, optimizer, checkpoint_dir, console)
if _loss_type_code is not None:
    loss_type_code = _loss_type_code

# Calculate efficiency dictionary for reference
console.print("Computing reference efficiency...")
if os.path.exists('efficiency_dict.pkl'):
    console.print("Loading reference efficiency from cache...")
    with open('efficiency_dict.pkl', 'rb') as f:
        efficiency_dict = pkl.load(f)
else:
    console.print("Computing reference efficiency...")
    efficiency_dict = compute_reference_efficiency(X_acc, X_gen, weights_acc, weights_gen, feature_names)
    with open('efficiency_dict.pkl', 'wb') as f:
        pkl.dump(efficiency_dict, f)

# Prepare test data for sharded prediction
X_test_split = split_for_devices(X_test, num_devices)
X_test_sharded = jax.device_put(X_test_split, data_sharding_2d)

# Modify training loop to start from the resumed epoch
console.print("Starting training classifier...")
for epoch in range(start_epoch, num_epochs):
    # Reset metrics for this epoch
    metrics.reset()
    
    # Shuffle data for this epoch
    indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]
    weights_train_shuffled = weights_train[indices]
    
    # Calculate number of batches
    # Make sure batch size is divisible by number of devices
    if batch_size % num_devices != 0:
        effective_batch_size = (batch_size // num_devices + 1) * num_devices
    else:
        effective_batch_size = batch_size
    
    num_batches = int(np.ceil(len(X_train) / effective_batch_size))
    
    # Train on batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * effective_batch_size
        end_idx = min((batch_idx + 1) * effective_batch_size, len(X_train))
        
        # Get batch data
        batch_x = X_train_shuffled[start_idx:end_idx]
        batch_y = y_train_shuffled[start_idx:end_idx]
        batch_weights = weights_train_shuffled[start_idx:end_idx]
        
        # Pad batch to be divisible by num_devices if needed
        if len(batch_x) % num_devices != 0:
            pad_size = num_devices - (len(batch_x) % num_devices)
            batch_x = np.concatenate([batch_x, batch_x[:pad_size]])
            batch_y = np.concatenate([batch_y, batch_y[:pad_size]])
            batch_weights = np.concatenate([batch_weights, batch_weights[:pad_size]])
        
        # Shard the data across devices based on array rank
        batch_x = jax.device_put(batch_x, data_sharding_2d) # 2D array (samples, features)
        batch_y = jax.device_put(batch_y, data_sharding_1d) # 1D array (samples)
        batch_weights = jax.device_put(batch_weights, data_sharding_1d) # 1D array (samples)
        
        # Create batch dictionary
        batch = {
            'x': batch_x,
            'y': batch_y,
            'weights': batch_weights
        }
            
        with mesh:
            
            # Train step
            loss, accuracy, main_loss, grad_loss = train_step(
                model, 
                optimizer, 
                batch, 
                loss_type_code,
                adaptive_gradient_reg_strength,
                adaptive_gradient_transition_sensitivity
            )
            
            # Block until computation is done
            loss = loss.block_until_ready()
            
            # Update metrics
            metrics.update(
                loss=loss,
                accuracy=accuracy,
                main_loss=main_loss,
                grad_loss=grad_loss
            )
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                console.print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{num_batches}, "
                             f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # Compute and print epoch metrics
    epoch_metrics = metrics.compute()
    console.print(f"Epoch {epoch+1} complete - "
                 f"Loss: {epoch_metrics['loss']:.4f}, "
                 f"Accuracy: {epoch_metrics['accuracy']:.4f}, "
                 f"Main Loss: {epoch_metrics['main_loss']:.4f}, "
                 f"Grad Loss: {epoch_metrics['grad_loss']:.4f}")
    
    # Save checkpoint if needed
    if (epoch + 1) % checkpoint_frequency == 0:

        checkpoint_state = {
            'model': model,
            'optimizer': optimizer.opt_state,
            'epoch': epoch + 1,
            'loss_type_code': loss_type_code,
        }
        
        save_checkpoint(
            checkpoint_state, 
            f"epoch_{epoch+1}", 
            checkpoint_dir=checkpoint_dir,
            console=console
        )
        
    if (epoch + 1) % plot_diagnostics_frequency == 0 or epoch == num_epochs - 1:
        console.print("Predicting efficiency on random uniform test data (for Monte Carlo integration)...")
        with mesh:
            test_probs = predict(model, X_test_sharded, loss_type_code)
            test_probs = test_probs.block_until_ready()    
            test_probs = test_probs.reshape(-1) # flatten
        if len(test_probs) > n_test_samples: # remove any padding
            test_probs = test_probs[:n_test_samples]
        
        # Plot efficiency by variable
        from pyamptools.dre.diagnostics import plot_efficiency_by_variable
        console.print("Plotting efficiency comparison diagnostics...")
        console.print("Plotting efficiency comparison diagnostics...")
        metrics = plot_efficiency_by_variable(
            X_test_raw, 
            test_probs, 
            feature_names=feature_names,
            efficiency_dict=efficiency_dict,
            checkpoint_dir=checkpoint_dir, 
            suffix="final", 
            loss_type_code=loss_type_code, 
            metric_type=metric_type, 
            plot_generated=True, 
            plot_predicted=True
        )

console.print("Training and evaluation complete!")
