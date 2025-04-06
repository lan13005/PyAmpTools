import numpy as np
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
from jax import random
from flax import nnx
import jax.numpy as jnp
import optax
import numpy as np
import pickle as pkl

from pyamptools.dre.classifier_train import DensityRatioEstimator, classifier_train_step, classifier_predict
from pyamptools.dre.diagnostics import compute_reference_efficiency
from rich.console import Console
from pyamptools.dre.classifier_utils import save_checkpoint, try_resume_from_checkpoint

from pyamptools.dre.flow_arch import MAF
from pyamptools.dre.flow_train import load_maf_model, fit_maf, save_maf_model, batch_transform_maf

loss_type_map = {"bce": 0, "mse": 1, "mlc": 2, "sqrt": 3}
loss_type_map_reverse = {v: k for k, v in loss_type_map.items()}
metric_labels = {"standard": "MAE", "relative": "Rel. MAE"}

console = Console()

################################################################

feature_names = ["mMassX", "mCosHel", "mPhiHel", "mt", "mPhi"]
seed = 42 # main rng seed
n_test_samples = 2000000 # Used for Monte Carlo integration of 1D/2D proj

### CHECKPOINTING
cwd = os.getcwd()
checkpoint_dir = f'{cwd}/model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

### SEEDING
# NOTE: nnx.Rngs is update to jax.random.PRNGKey
# No need to split keys anymore, just ask for a different random stream as:
# 1. rngs['my_param']()
# 2. rngs.my_param()
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

########################################################
############### Normalizing flow pretraining ###########
########################################################
use_flow_pretraining = False
flow_hidden_dims = [1024] # Hidden dimensions for flow networks
flow_num_layers = 5           # Number of flow transformations
flow_batch_size = 4096        
flow_learning_rate = 0.0001   # From fDRE paper
flow_num_epochs = 200         # Max epochs for flow training
flow_patience = 20            # Early stopping patience for flow training
flow_clip_norm = 1.0          # Prevent large gradients
flow_adam_b1 = 0.9            # Adam beta1 parameter
flow_adam_b2 = 0.999          # Adam beta2 parameter
flow_adam_eps = 1e-8          # Adam epsilon parameter
flow_warmup_steps = 1000      # Add warmup to stabilize initial training
flow_weight_decay = 1e-6      # Add weight decay to stabilize
flow_plot_diagnostic_frequency = 10  # save diagnostics plots for flow frequency (epochs)
flow_n_samples_plotting = 5000  # Number of samples to use for flow diagnostic plots (triangular distribution plots)

########################################################
############### Classifier training ####################
########################################################
metric_type = "standard"
loss_type = "bce" # Options: "bce" (default), "mse", "mlc", or "sqrt"
classifier_dims = (len(feature_names), 256, 256)
classifier_dropout_rate = 0.2
classifier_learning_rate = 0.0002 # From fDRE paper
classifier_weight_decay = 0.0005  # From fDRE paper
adaptive_gradient_reg_strength = 0.0001  # Strength of the gradient regularization
adaptive_gradient_transition_sensitivity = 0.5  # Controls how quickly regularization decreases in high-gradient regions
classifier_batch_size = 4096
classifier_num_epochs = 20
classifier_checkpoint_frequency = 10  # Save checkpoint every classifier_checkpoint_frequency epochs
classifier_plot_diagnostics_frequency = 5  # save diagnostics plots every classifier_plot_diagnostics_frequency epochs
classifier_plot_training_frequency = 1  # Update plots every iteration

loss_type_code = loss_type_map[loss_type]
console.print(f"Classifier using loss function: {loss_type}")

def split_for_devices(array, num_devices):
    """Split numpy array across devices, padding if necessary."""
    if len(array) % num_devices != 0:
        # Calculate padding needed then pad the array by repeating the couple elements
        pad_size = num_devices - (len(array) % num_devices)
        array = np.concatenate([array, array[:pad_size]])
    return array.reshape(num_devices, -1, *array.shape[1:])

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
    
num_batches = int(np.ceil(len(X_train) / classifier_batch_size))

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
        print(f"Initializing MAF with {flow_num_layers} layers and {flow_hidden_dims} hidden dimensions")
        maf = MAF(X_train.shape[1], flow_hidden_dims, flow_num_layers, rngs)
        maf_params = maf.init_params()
else:
    X_gen_flow_plot = X_gen[np.random.choice(X_gen.shape[0], size=flow_n_samples_plotting, replace=False)]
    X_acc_flow_plot = X_acc[np.random.choice(X_acc.shape[0], size=flow_n_samples_plotting, replace=False)]
    
    # Initialize the MAF model
    print(f"Initializing MAF with {flow_num_layers} layers and {flow_hidden_dims} hidden dimensions")
    maf = MAF(X_train.shape[1], flow_hidden_dims, flow_num_layers, rngs)
    maf_params = maf.init_params()

    from pyamptools.dre.diagnostics import create_corner_plot
    console.print("Creating corner plot for original data...")
    create_corner_plot(
        X_gen_flow_plot, X_acc_flow_plot,
        labels=['Generated', 'Accepted'],
        feature_names=feature_names,
        filename='original_corner_plot.png',
        checkpoint_dir=checkpoint_dir,
    )

    X_acc_flow_plot = scaler.transform(X_acc_flow_plot)
    X_gen_flow_plot = scaler.transform(X_gen_flow_plot)

    console.print("Training MAF for {flow_num_epochs} epochs...")

    maf_result = fit_maf(
        maf,
        maf_params,
        X_train,
        rngs,
        #### TRAINING ####
        batch_size=flow_batch_size,
        learning_rate=flow_learning_rate,
        weight_decay=flow_weight_decay,
        num_epochs=flow_num_epochs,
        patience=flow_patience,
        sample_weights=weights_train,
        #### FLOW OPTIMIZER ####
        clip_norm=flow_clip_norm,
        adam_b1=flow_adam_b1,
        adam_b2=flow_adam_b2,
        adam_eps=flow_adam_eps,
        #### PLOTTING ####
        plot_frequency=flow_plot_diagnostic_frequency,
        checkpoint_dir=checkpoint_dir,
        feature_names=feature_names,
        X_gen=X_gen_flow_plot,
        X_acc=X_acc_flow_plot,
        use_gpu=False
    )

    maf = maf_result["maf"]
    maf_params = maf_result["params"]
    flow_losses = maf_result["losses"]

    # Save the trained flow model
    save_maf_model({"maf": maf, "params": maf_params, "losses": flow_losses}, flow_save_path)
    
    print("Creating corner plot for latent space data...")
    X_acc_latent_flow_plot, _ = batch_transform_maf(maf, maf_params, X_acc_flow_plot, batch_size=1024, direction="forward")
    X_gen_latent_flow_plot, _ = batch_transform_maf(maf, maf_params, X_gen_flow_plot, batch_size=1024, direction="forward")
    create_corner_plot(
        X_gen_latent_flow_plot, X_acc_latent_flow_plot,
        labels=['Generated', 'Accepted'],
        feature_names=feature_names,
        filename='latent_corner_plot.png',
        checkpoint_dir=checkpoint_dir,
    )

print("Transforming data to latent space in batches...")
X_train_latent, _ = batch_transform_maf(maf, maf_params, X_train, batch_size=1024, direction="forward")
X_test_latent, _ =  batch_transform_maf(maf, maf_params, X_test,  batch_size=1024, direction="forward")

# Use latent representations for training
X_train = np.array(X_train_latent)
X_test  = np.array(X_test_latent)

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
console.print("Splitting test data for sharded prediction...")
X_test_split = split_for_devices(X_test, num_devices)
X_test_sharded = jax.device_put(X_test_split, data_sharding_2d)

# Modify training loop to use the new iterator
console.print("Starting training classifier...")
from pyamptools.dre.classifier_train import DataLoader
iterator = DataLoader(
    X_train, y_train, weights_train,
    classifier_batch_size, num_devices,
    data_sharding_2d, data_sharding_1d,
    classifier_num_epochs, start_epoch
)

# NOTE: iteration has 3 possible return states:
# 1. 'epoch_start' - designates start, not much to do
# 2. 'epoch_end' - designates end, compute and print metrics, draw diagnostics
# 3. batch - designates a batch, train step
for item in iterator:
    # 1. Check if this is the start of an epoch
    if 'epoch_start' in item:
        epoch = item['epoch_start']
        num_batches = item['num_batches']
        console.print(f"Starting epoch {epoch+1}/{classifier_num_epochs}")
        continue
    
    # 2. Check if this is the end of an epoch
    if 'epoch_end' in item:
        epoch = item['epoch_end']
        
        # Compute and print epoch metrics
        epoch_metrics = metrics.compute()
        console.print(f"Epoch {epoch+1} complete - "
                     f"Loss: {epoch_metrics['loss']:.4f}, "
                     f"Accuracy: {epoch_metrics['accuracy']:.4f}, "
                     f"Main Loss: {epoch_metrics['main_loss']:.4f}, "
                     f"Grad Loss: {epoch_metrics['grad_loss']:.4f}")
        
        # Save checkpoint if needed
        if (epoch + 1) % classifier_checkpoint_frequency == 0:
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
            
        # Plot diagnostics if needed
        if (epoch + 1) % classifier_plot_diagnostics_frequency == 0 or epoch == classifier_num_epochs - 1:
            console.print("Predicting efficiency on random uniform test data (for Monte Carlo integration)...")
            with mesh:
                test_probs = classifier_predict(model, X_test_sharded, loss_type_code)
                test_probs = test_probs.block_until_ready()    
                test_probs = test_probs.reshape(-1) # flatten
            if len(test_probs) > n_test_samples: # remove any padding
                test_probs = test_probs[:n_test_samples]
            
            # Plot efficiency by variable
            from pyamptools.dre.diagnostics import plot_efficiency_by_variable
            console.print("Plotting efficiency comparison diagnostics...")
            efficiency_metrics = plot_efficiency_by_variable(
                X_test_raw, 
                test_probs, 
                feature_names=feature_names,
                efficiency_dict=efficiency_dict,
                checkpoint_dir=checkpoint_dir, 
                suffix=f"epoch_{epoch+1}", 
                loss_type_code=loss_type_code, 
                metric_type=metric_type, 
                plot_generated=True, 
                plot_predicted=True
            )
        
        continue
    
    # 3. Received a batch, perform a train step
    batch = item
    batch_idx = batch.pop('batch_idx')  # Remove metadata before passing to classifier_train_step
    epoch = batch.pop('epoch')
    
    with mesh:
        # Train step
        loss, accuracy, main_loss, grad_loss = classifier_train_step(
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
            console.print(f"Epoch {epoch+1}/{classifier_num_epochs}, Batch {batch_idx+1}/{num_batches}, "
                         f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

console.print("Training and evaluation complete!")
