import numpy as np
import os
import pandas as pd
from tqdm import trange
import time
from rich.console import Console
import matplotlib.pyplot as plt
os.environ["JAX_PLATFORMS"] = "cpu"

from pyamptools.utility.general import calculate_subplot_grid_size, prettyLabels
from pyamptools.utility.moment_projector import project_to_moments_refl, precompute_cg_coefficients_by_LM, get_moment_names
from pyamptools.utility.general import converter
import jax
from jax import vmap, jit
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.einstein import SVGD, RBFKernel
from numpyro.optim import Adam
from functools import partial

#############################################################################################
#### Calling svgd.init() takes a very long time likely since it is compiling the model
#### I was hoping jax persistent cache would help but I cannot get it to work
#############################################################################################
# jax.config.update("jax_compilation_cache_dir", os.path.expanduser("/w/halld-scshelf2101/lng/WORK/PyAmpTools9/OTHER_CHANNELS/ETAPI0_AUTOGRAD/jax_cache"))
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# from jax.experimental.compilation_cache import compilation_cache as cc
# cc.set_cache_dir("/w/halld-scshelf2101/lng/WORK/PyAmpTools9/OTHER_CHANNELS/ETAPI0_AUTOGRAD/jax_cache")

console = Console()

qn_to_amp = {tuple(v): k for k, v in converter.items()} # quantum number list to amplitude names

######################################################
num_particles = 100
num_iterations = 1000
tightness = 100 # controls how tight the stein fit is
seed = 42
l_max = 1
n_eps = 2
######################################################

# Seed both just in case
np.random.seed(seed)
rng_key = jax.random.PRNGKey(seed)

### LOAD INPUT AMPLITUDE
# flat_amplitudes should be a flattened version for shape [1, epsilon, l, m, particle]
# 1 is locked for index k = spin flip or non-flip
n_flat_amplitudes = 2 * sum((2*l + 1) for l in range(l_max + 1)) * n_eps
n_expected_moments = sum([L+1 for L in range(2*l_max + 1)])
flat_amplitudes = np.random.normal(size=n_flat_amplitudes)
flat_amplitudes = np.array(flat_amplitudes, dtype=jnp.float32)

# NOTE: project_to_moments_refl accepts a mask so you do not need to 
#       explicitly zero out the reference waves in flat_amplitudes
waves_in_epsilon = len(flat_amplitudes) // 2

# Create mask to identify reference wave indices
# First wave in each reflectivity is a reference wave (mask its imaginary part)
mask = np.ones(len(flat_amplitudes))
ref_indices = [0, waves_in_epsilon] # indices of reference waves
for ref_idx in ref_indices:
    mask[ref_idx+1] = 0.0  # Zero out imaginary part of reference waves

# Apply mask to zero out imaginary parts of reference waves
flat_amplitudes = flat_amplitudes * mask

# Precompute CG coefficients once
console.print("Precomputing Clebsch-Gordan coefficients...")
start = time.time()
cg_coeffs = precompute_cg_coefficients_by_LM(l_max, 2*l_max)
cg_time = time.time() - start
console.print(f"CG precomputation time: {cg_time:.4f} seconds")
console.print(f"Number of CG coefficients: {len(cg_coeffs)}")

# First test - project to moments
# NOTE: We cannot normalize since the inversion will not work properly
start = time.time()
console.print(f"input pwa amplitudes (shape: {flat_amplitudes.shape}): \n{flat_amplitudes}")
H012 = project_to_moments_refl(flat_amplitudes, mask=mask, l_max=l_max, cg_coeffs=cg_coeffs)
assert H012.shape[0] == n_expected_moments * 2 * 3  # 2x for real/imag, 3x for H0, H1, H2
n_moments = H012.shape[0] // 3
H0 = H012[ :n_moments:2            ] + 1j*H012[1:n_moments:2]
H1 = H012[  n_moments:2*n_moments:2] + 1j*H012[  n_moments+1:2*n_moments:2]
H2 = H012[2*n_moments:3*n_moments:2] + 1j*H012[2*n_moments+1:3*n_moments:2]
target_moments = jnp.array([H0, H1, H2])
moment_names = get_moment_names(l_max)
moment_names = np.array(moment_names).reshape(3, -1)
console.print(f"Moments calculation time: {time.time() - start:.4f} seconds")

######################################################
######### COMPARE TO BORIS'S CODE ####################
######################################################

# from pyamptools.utility.MomentCalculatorTwoPS import AmplitudeSet, AmplitudeValue, QnWaveIndex
# from pyamptools.utility.moment_projector import _get_boris_moments

# boris_moments = _get_boris_moments(flat_amplitudes, l_max)

# moments_all_agree = True
# for boris_moment in boris_moments.values:
#     i, L, M = boris_moment.qn.momentIndex, boris_moment.qn.L, boris_moment.qn.M
#     offset = sum(L + 1 for L in range(L))
#     moment = moments[i, offset + M]
#     if abs(boris_moment.val - moment) > 1e-5:
#         moments_all_agree = False
#         print(f"H{i}_{L}_{M} = {boris_moment.val}, {moment}, {abs(boris_moment.val - moment)}")
# if moments_all_agree:
#     console.print("All moments between Boris's code and this code agree!")

####################################################################
###### PLOTTING FREE PARAMETER COUNT IN AMPLITUDE VS MOMENT BASIS ##
####################################################################

# free_pars_amps = []
# free_pars_moms = []
# def get_n_pars_amps(lmax):
#     return sum([2*l+1 for l in range(lmax+1)]) * 2 * 2 - 2
# def get_n_pars_moms(lmax):
#     return 3 * sum([l+1 for l in range(2*lmax+1)])

# for lmax in range(50):
#     free_pars_amps.append(get_n_pars_amps(lmax))
#     free_pars_moms.append(get_n_pars_moms(lmax))
    
# free_pars_amps = np.array(free_pars_amps)
# free_pars_moms = np.array(free_pars_moms)

# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# ax.plot(free_pars_amps, label='Amplitude Basis')
# ax.plot(free_pars_moms, label='Moment Basis')
# ax.set_ylabel('Number of Parameters', size=18)
# ax.set_xlabel('Lmax', size=15)
# ax.set_ylim(0, ax.get_ylim()[1] * 1.4)
# ax.legend(loc='upper left', prop={'size': 18})

# ax2 = ax.twinx()
# ax2.plot(free_pars_moms/free_pars_amps, label='Amplitude Basis', color='xkcd:cerise')
# ax2.set_ylabel('Ratio', size=18)
# ax2.set_ylim(0, ax2.get_ylim()[1] * 1.4)
# ax2.axhline(1.5, color='xkcd:cerise', linestyle='--', label='Asymptote = 3/2')
# ax2.legend(loc='upper right', prop={'size': 15})
# ax2.spines['right'].set_color('xkcd:cerise')
# ax2.tick_params(colors='xkcd:cerise')
# ax2.yaxis.label.set_color('xkcd:cerise')

######################################################
######### MOMENT INVERSION USING SVGD ################
######################################################

# Define the model - this compares projected moments with observed moments
def model(target_moments):
    # Sample separately for real and complex parts
    # Real parts of reference waves
    free_params = numpyro.sample(
        "free_params",
        dist.Normal(0, 1).expand([int(sum(mask))]) # accepts an array
    )

    # Combine parameters into flat amplitude array
    flat_amps = jnp.zeros(n_flat_amplitudes)
    flat_amps = flat_amps.at[mask==1].set(free_params)
    
    # Project to moments using the mask
    projected = project_to_moments_refl(flat_amps, mask=mask, l_max=l_max, cg_coeffs=cg_coeffs)
    
    # Calculate moments just like in your original code
    n_moments = projected.shape[0] // 3
    H0 = projected[:n_moments:2] + 1j*projected[1:n_moments:2]
    H1 = projected [n_moments:2*n_moments:2]  + 1j*projected[n_moments+1:2*n_moments:2]
    H2 = projected[2*n_moments:3*n_moments:2] + 1j*projected[2*n_moments+1:3*n_moments:2]
    calc_moments = jnp.array([H0, H1, H2])
    
    # Compare with observed moments (using squared error as likelihood)
    error = jnp.sum(jnp.abs(calc_moments - target_moments))
    numpyro.factor("likelihood", -error * tightness)

# JIT-compile the verification function for speed
@jit
def process_particle(amps):
    """
    Apply mask to ensure reference waves are properly zeroed (Perhaps I should use this as a check)
    Normalize moments by H0_0_0
    """
    amps = amps * mask
    proj = project_to_moments_refl(amps, mask=mask, l_max=l_max, cg_coeffs=cg_coeffs)
    n_m = proj.shape[0] // 3
    h0  = proj[:n_m:2] + 1j*proj[1:n_m:2]
    h1  = proj[n_m:2*n_m:2] + 1j*proj[n_m+1:2*n_m:2]
    h2  = proj[2*n_m:3*n_m:2] + 1j*proj[2*n_m+1:3*n_m:2]
    h2 /= h0[0]
    h1 /= h0[0]
    h0 /= h0[0]
    return jnp.array([h0, h1, h2])

# Option 1
# kernel = RBFKernel()
# optimizer = Adam(step_size=0.01)
# svgd = SVGD(model, optimizer, kernel, num_stein_particles=num_particles)
# svgd_result = svgd.run(rng_key, num_iterations, target_moments=target_moments, progress_bar=True)
# params_dict = svgd_result.params # params attr is a dict

# Option 2
kernel = RBFKernel()
optimizer = Adam(step_size=0.01)
svgd = SVGD(model, optimizer, kernel, num_stein_particles=num_particles)
rng_key = jax.random.PRNGKey(0)

console.print("\nCompiling SVGD model...")
start_compile_svgd = time.time()
svgd_state = svgd.init(rng_key, target_moments=target_moments)
compile_svgd_time = time.time() - start_compile_svgd
console.print(f"  SVGD compilation time: {compile_svgd_time:.2f} seconds")

def make_bandwidth_factor(t, num_iterations):
    # bandwidth_factor must be a callable
    return lambda num_particles: 0.5 ** (t / num_iterations)
@jax.jit
def step(svgd_state, t, num_iterations, target_moments):
    kernel.bandwidth_factor = make_bandwidth_factor(t, num_iterations)
    return svgd.update(svgd_state, target_moments=target_moments) # Update returns a tuple where next state is first element

start_run_svgd = time.time()
for t in trange(num_iterations):
    svgd_state_output = step(svgd_state, t, num_iterations, target_moments)
    svgd_state = svgd_state_output[0]  # Extract the SteinVIState object
params_dict = svgd.get_params(svgd_state)
svgd_time = time.time() - start_run_svgd
console.print(f"SVGD completed in {svgd_time:.2f} seconds")
console.print(f"Params keys: {list(params_dict.keys())}")

# NOTE: I am not sure where this auto_loc suffix comes from
if 'free_params_auto_loc' in params_dict:
    free_params = params_dict['free_params_auto_loc']  # shape (num_particles, n_free_real_indices)
    console.print(f"Number of free parameters: {free_params.shape} ~ (n_particles, n_free_parameters)")
    
    # Reconstruct the full amplitude array for each particle
    num_particles = free_params.shape[0]
    inferred_amplitudes = []
    for i in range(num_particles):
        full_amps = jnp.zeros(n_flat_amplitudes)
        full_amps = full_amps.at[mask==1].set(free_params[i])
        
        # Check masking is done correctly
        masked_indices = jnp.where(mask == 0)[0]
        assert jnp.all(full_amps[masked_indices] == 0), f"Masked indices {masked_indices} not zero in particle {i}"
        
        inferred_amplitudes.append(full_amps)
    inferred_amplitudes = jnp.stack(inferred_amplitudes)
    console.print(f"\nReconstructed amplitudes shape: {inferred_amplitudes.shape} ~ (n_particles, n_flat_amplitudes)")
    
    # Map processing function (projecting to moment basis) over all particles
    process_batch = jit(vmap(process_particle))
    recovered_moments = process_batch(inferred_amplitudes)

         
    ###############################
    ## SAVE RESULTS IN DATAFRAME ##
    ###############################
    
    pred_df = {}
    amps = []

    # 1. Save inferred moments
    for alpha, names in enumerate(moment_names): # H_alpha(L, M)
        for moment_idx, name in enumerate(names):
            pred_df[name] = recovered_moments[:, alpha, moment_idx]
            
    # 2. Save inferred amplitudes
    idx = 0
    for eps in range(n_eps):
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                refl = -1 if eps == 0 else 1
                amp = qn_to_amp[(refl, l, m)]
                pred_df[amp] = inferred_amplitudes[:, idx] + 1j * inferred_amplitudes[:, idx + 1]
                amps.append(amp)
                idx += 2      
    amps = np.unique(amps)     
    pred_df = pd.DataFrame(pred_df)
    
    ############################################
    #### PRINTING DIAGNOSTICS ##################
    ############################################
    
    # Calculate errors for each particle
    errors = jnp.sum(jnp.abs(recovered_moments - target_moments.reshape(1, 3, -1))**2, axis=(1, 2))
    best_idx = jnp.argmin(errors)
    best_recovered = inferred_amplitudes[best_idx]
    
    # Project both original and recovered amplitudes to moments for comparison
    original_moments = np.concatenate(process_particle(flat_amplitudes), axis=0)
    recovered_best_moments = np.concatenate(process_particle(best_recovered), axis=0)
    
    console.print("\n[bold]Moment Reconstruction Error[/bold]")
    console.print(f"{'Index':>6} | {'Type':>4} | {'Original':>15} | {'Recovered':>15} | {'Difference':>15} | {'Relative Error':>15}")
    console.print("-" * 90)
    for i, (orig, rec) in enumerate(zip(original_moments, recovered_best_moments)):
        # Determine which moment type (H0, H1, H2) and index within that type
        moment_type = "H0" if i < len(original_moments)//3 else ("H1" if i < 2*len(original_moments)//3 else "H2")
        diff = orig - rec
        rel_err = 0 if np.abs(orig) < 1e-10 else np.abs(diff) / np.abs(orig)
        console.print(f"{i:6d} | {moment_type:4} | {orig:15.4f} | {rec:15.4f} | {diff:15.4f} | {rel_err:15.4f}")
        
######################################################
######### ENFORCE SYMMETRY OF MOMENTS ################
######################################################

def zero_moment_parts(df, moment_names):
    """
    Enforce H0/H1 to be real and H2 to be imaginary
    """
    name_mapping = {}
    drop_cols = []
    
    alpha_symmetry_dict = {0: 'real', 1: 'real', 2: 'imag'}
    for alpha, names in enumerate(moment_names):
        for moment_idx, name in enumerate(names):
            part = alpha_symmetry_dict[alpha]
            new_name = f"Re[{name}]" if part == 'real' else f"Im[{name}]"
            df[new_name] = np.round(getattr(np, part)(df[name]), 6)
            name_mapping[name] = new_name
            drop_cols.append(name)
    df = df.drop(columns=drop_cols)
    return df, name_mapping

pred_df, name_mapping = zero_moment_parts(pred_df, moment_names)

target_df = {}
for alpha, names in enumerate(moment_names):
    for moment_idx, name in enumerate(names):
        target_df[name] = np.round(np.array([target_moments[alpha, moment_idx]]), 6)
target_df = pd.DataFrame(target_df)
target_df, name_mapping = zero_moment_parts(target_df, moment_names) 
h0_value = target_df['Re[H0_0_0]'].values[0] # Normalize the moments
for col in target_df.columns:
    target_df[col] = target_df[col] / h0_value
    
############################################################
######### PLOT MOMENTS COMPARISON (RECON VS TARGET) ########
############################################################

idx = 0
for eps in range(n_eps):
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            refl = -1 if eps == 0 else 1
            amp = qn_to_amp[(refl, l, m)]
            target_df[amp] = flat_amplitudes[idx] + 1j * flat_amplitudes[idx + 1]
            idx += 2

fig, ax = plt.subplots(1, 1, figsize=(18, 6))
idx = 0
for alpha, names in enumerate(moment_names):
    for moment_idx, name in enumerate(names):
        new_name = name_mapping[name]
        ax.hlines(y=target_df[new_name],   
                xmin=idx-0.5,
                xmax=idx+0.5,
                colors='black',
                alpha=1.0,
                linewidth=1)
        jitter = np.random.uniform(-0.05, 0.05, len(pred_df[new_name]))
        ax.scatter(np.ones_like(pred_df[new_name])*idx + jitter, pred_df[new_name], color='xkcd:cerise', s=20, alpha=0.3)
        idx += 1

name_order = [name_mapping[name] for name in moment_names.flatten()]

ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.set_xlim(-0.5, idx-0.5)

ax.set_xticks(range(idx), name_order, rotation=90, size=14)
ax.set_ylabel('Normalized Moment Value', size=20)
ax.set_yticks(np.linspace(0, 1, 6), np.round(np.linspace(0, 1, 6), 3), size=14)
ax.set_xlabel('Moment', size=20)

plt.tight_layout()

########################################################
######### PLOT AMPLITUDES COMPARISON (HISTOGRAM) ########
########################################################

nrows, ncols = calculate_subplot_grid_size(len(amps))

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4), sharex=True, sharey=True)
axes = axes.flatten()

edges = np.linspace(0, np.max(np.abs(pred_df[amps])), 20)

for i, amp in enumerate(amps):
    axes[i].hist(np.abs(pred_df[amp]), label='Predicted', color='xkcd:cerise', alpha=0.7, bins=edges)
    axes[i].axvline(np.abs(target_df[amp])[0], color='black', linewidth=2, linestyle='--')
    axes[i].set_title(prettyLabels[amp], size=18)
    axes[i].set_xlim(0)

for i in np.arange(len(amps))[::ncols]:
    axes[i].set_ylabel('Counts', size=18)
for i in np.arange(len(amps))[(nrows-1)*ncols:]:
    axes[i].set_xlabel('Amplitude', size=18)
    
plt.tight_layout()

########################################################
######### PLOT PHASE COMPARISON (HISTOGRAM) ############
########################################################

nrows, ncols = calculate_subplot_grid_size(len(amps))

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4), sharex=True, sharey=True)
axes = axes.flatten()

edges = np.linspace(-np.pi, np.pi, 20)

for i, amp in enumerate(amps):
    angle = np.angle(pred_df[amp])
    angle = angle % np.pi * np.sign(angle)
    axes[i].hist(angle, label='Predicted', color='xkcd:cerise', alpha=0.7, bins=edges)
    axes[i].axvline(np.angle(target_df[amp])[0], color='black', linewidth=2, linestyle='--')
    axes[i].set_title(prettyLabels[amp], size=18)

for i in np.arange(len(amps))[::ncols]:
    axes[i].set_ylabel('Counts', size=18)
for i in np.arange(len(amps))[(nrows-1)*ncols:]:
    axes[i].set_xlabel('Phase', size=18)
    
plt.tight_layout()
