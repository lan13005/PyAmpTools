import os
# Set environment variables for multi-core processing
os.environ["JAX_PLATFORMS"] = "cpu"
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
# os.environ["OMP_NUM_THREADS"] = "8"

import jax
jax.config.update("jax_enable_x64", True)

# NOTE: These have to be near the top for some reason otherwise we get errors
from pyamptools.utility.resultManager import ResultManager
from pyamptools.utility.moment_projector import project_to_moments_refl, precompute_cg_coefficients_by_LM, get_moment_name_order, get_amplitude_name_order
from pyamptools.utility.general import qn_to_amp, identify_channel, load_yaml

import pickle as pkl
import numpy as np
import pandas as pd
from tqdm import trange
import time
from rich.console import Console
import argparse
            
import jax
from jax import vmap, jit
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.einstein import SVGD, RBFKernel
from numpyro.optim import Adam
from numpyro.contrib.einstein.stein_kernels import RBFKernel
from functools import partial
from multiprocessing import Pool
import multiprocessing

console = Console()

# NOTE:
# 1. H2 has another symmetry where I think the M=0 is 0 but more annoying to enforce at this point
# 2. More particles does not always result in better results, perhaps more particles could require more iterations
# 3. Always do not normalize moments, just spit out the raw moments and let the user normalize however they want, easier optimization
# 4. This code assumes a user has some results produced with some waveset then projected onto the moments. This code will
#    attempt to invert the moments back to the same waveset. I think this code should handle the case where we
#    allow unused waves to be non-zero also (untested). Currently a mask is used to ignore unused waves

# ############################################################################################
# ### Calling svgd.init() takes a very long time likely since it is compiling the model
# ### I was hoping jax persistent cache would help but I cannot get it to work
# ############################################################################################
# jax.config.update("jax_compilation_cache_dir", os.path.expanduser("/w/halld-scshelf2101/lng/WORK/PyAmpTools9/OTHER_CHANNELS/ETAPI0_AUTOGRAD/jax_cache"))
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# from jax.experimental.compilation_cache import compilation_cache as cc
# cc.set_cache_dir("/w/halld-scshelf2101/lng/WORK/PyAmpTools9/OTHER_CHANNELS/ETAPI0_AUTOGRAD/jax_cache")

def calculate_likelihood(flat_amps, moment_samples, mask, cg_coeffs, tightness, loss_exponent, l_max):
    """
    Calculates the likelihood component for a set of amplitudes against moment samples.
    
    Args:
        flat_amps: The flattened amplitude parameters
        moment_samples: Array of shape (n_samples, 3, n_moments_per_type) containing moment samples
        mask: Mask with same shape as n_flat_amplitudes indicating which parameters are free (1s and 0s)
        cg_coeffs: precomputed CG coefficients
        tightness: scales loss function to enforce tightness of fit
        loss_exponent: The exponent to use for error calculation (default: 2 = MSE)
        l_max: Maximum angular momentum
        
    Returns:
        likelihood: The scaled likelihood value (-avg_error * tightness)
    """
    # Project to moments using the masking out unused wave parts
    projected = project_to_moments_refl(flat_amps, mask=mask, l_max=l_max, cg_coeffs=cg_coeffs)
    n_moments = projected.shape[0] // 3
    H0 = projected[:n_moments:2] + 1j*projected[1:n_moments:2]
    H1 = projected[n_moments:2*n_moments:2] + 1j*projected[n_moments+1:2*n_moments:2]
    H2 = projected[2*n_moments:3*n_moments:2] + 1j*projected[2*n_moments+1:3*n_moments:2]
    calc_moments = jnp.array([H0, H1, H2], dtype=jnp.complex128)
        
    def compute_sample_error(sample):
        return jnp.sum(jnp.abs(calc_moments - sample)**loss_exponent)
    batch_errors = jax.vmap(compute_sample_error)(moment_samples)        
    avg_error = jnp.mean(batch_errors)
    
    # If using loss_exponent=2 then this is a Gaussian likelihood. Tightness is then
    #    a artificial inverse-variance to enforce deep inversion
    likelihood = -avg_error * tightness
    return likelihood

def make_bandwidth_factor(t, decay_iterations=1000, initial_scale=1.0, min_scale=0.0):
    """
    Create a bandwidth factor function with customizable exponential decay.
    This factor multiplies the median heuristic bandwidth in RBFKernel so should be O(1)
    
    Args:
        t: Current iteration
        decay_iterations: Number of iterations to decay initial_scale by 1/e
        initial_scale: Initial bandwidth multiplier (larger promotes exploration)
        min_scale: Minimum bandwidth multiplier (prevents bandwidth from becoming too small)
    
    Returns:
        A callable that returns the bandwidth factor for current iteration (must accept 1 argument that is the number of particles)
    """
    decay = initial_scale * jnp.exp(-t / decay_iterations)
    return lambda num_particles: jnp.maximum(decay, min_scale)

class ScheduledRBFKernel(RBFKernel):
    """
    RBFKernel that updates its bandwidth factor at each iteration
    We do this to remove the fake repulsive uncertainty as we are looking for deep inversion
    """
    def __init__(self, schedule_function):
        super().__init__()
        self.schedule_function = schedule_function
        self.iteration = 0

    def compute(self, rng_key, particles, particle_info, loss_fn):
        self.bandwidth_factor = self.schedule_function(self.iteration)
        self.iteration += 1
        return super().compute(rng_key, particles, particle_info, loss_fn)

# NOTE: We will jit later in the code
def process_particle(amp_array, mask, l_max, cg_coeffs):
    """
    Apply mask to ensure reference waves are properly zeroed (Perhaps I should use this as a check)
    Normalize moments by H0_0_0
    """
    amp_array = amp_array * mask
    proj = project_to_moments_refl(amp_array, mask=mask, l_max=l_max, cg_coeffs=cg_coeffs)
    n_m = proj.shape[0] // 3
    h0  = proj[:n_m:2] + 1j*proj[1:n_m:2]
    h1  = proj[n_m:2*n_m:2] + 1j*proj[n_m+1:2*n_m:2]
    h2  = proj[2*n_m:3*n_m:2] + 1j*proj[2*n_m+1:3*n_m:2]
    return jnp.array([h0, h1, h2])

def process_mass_bin(config):
    
    """
    This is the main function that is called to invert the moments in a bin-by-bin fashion.
    See code head for more details on the input config dictionary.
    """
    
    ######################################################
    # MUST BE SUPPLIED IN CONFIG DICT
    moments_df = config['moments_df'] # DataFrame containing the moments to be inverted
    waveNames = config['waveNames'] # List of waves as the target basis for inversion
    reference_waves = config['reference_waves'] # List of reference waves (imaginary parts will be zeroed out)

    # GENERAL PARAMETERS (WITH DEFAULTS)
    main_file = config.get('main_file', None)
    iftpwa_yaml = config.get('ift_yaml', None)
    mass_bin = config.get('mass_bin', 0)
    masses = config.get('masses', [1.0])
    output_dir = config.get('output_dir', 'moment_inversion_results')
    seed = config.get('seed', 42)
    
    # AMPLITUDE PARAMETERS (WITH DEFAULTS)
    n_eps = config.get('n_eps', 2)
    
    # SVGD PARAMETERS (WITH DEFAULTS)
    step_size = config.get('step_size', 0.01)
    num_iterations = config.get('num_iterations', 5000)
    bandwidth_params = config.get('bandwidth_params', {'decay_iterations': 1000, 'initial_scale': 1.0, 'min_scale': 0.0})
    num_particles = config.get('num_particles', 500)
    amplitude_scale = config.get('amplitude_scale', 1.0)
    tightness = config.get('tightness', 1000.0)
    loss_exponent = config.get('loss_exponent', 2.0)
    ######################################################
    
    #### BEGIN MOMENT INVERSION BIT ####
    
    # Identify the maximum L value based on the moment names
    moment_cols = [c for c in moments_df.columns if c[0] == 'H']
    L_max = max([int(c[3]) for c in moment_cols])
    l_max = L_max // 2
    
    # Precompute CG coefficients (If this turns out to be slow we can move this outside the process_mass_bin function)
    #   and precompute it once, passing it in as a config dict kwarg
    console.print("Precomputing Clebsch-Gordan coefficients...")
    start = time.time()
    cg_coeffs = precompute_cg_coefficients_by_LM(l_max, 2*l_max)
    cg_time = time.time() - start
    console.print(f"CG precomputation time: {cg_time:.4f} seconds")
    console.print(f"Number of CG coefficients: {len(cg_coeffs)}")    

    moment_names = get_moment_name_order(l_max) # Expected order for moment_inversion code [H0(0,0), ..., H1(0,0), ..., H2(0,0), ...]
    n_flat_amplitudes = 2 * sum((2*l + 1) for l in range(l_max + 1)) * n_eps

    channel = identify_channel(waveNames)
    assert channel == 'TwoPseudoscalar', "Only TwoPseudoscalar is supported ATM for moment inversion"
    
    # Create a mask for non-existant partial waves
    mask = np.ones(n_flat_amplitudes)
    ref_wave_idxs = [waveNames.index(ref) for ref in reference_waves]
    assert len(ref_wave_idxs) == len(reference_waves), f"Reference waves {reference_waves} not found in {waveNames}"
    assert len(ref_wave_idxs) == 2, f"Requires two reference waves (code expects both reflectivities), found {len(ref_wave_idxs)}"
    for ref_wave_idx in ref_wave_idxs:
        mask[2*ref_wave_idx+1] = 0.0 # Zero out imaginary part of reference waves

    # Ensure all waves used in yaml file are a subset of the expected amplitudes then zero unused waves
    amplitude_names = get_amplitude_name_order(l_max) # all the expected amplitudes in order given l_max
    assert all(wave in amplitude_names for wave in waveNames), f"Wave names {waveNames} not found in {amplitude_names}"
    missing_waves = set(amplitude_names) - set(waveNames)
    missing_wave_idxs = np.array([amplitude_names.index(wave) for wave in missing_waves])
    if len(missing_wave_idxs) > 0:
        mask[2*missing_wave_idxs  ] = 0.0
        mask[2*missing_wave_idxs+1] = 0.0 

    # NOTE:
    # Reorder moments in the mcmc_results DataFrame to be in the expected order
    # H0/H1 are purely real, H2 is purely imaginary
    # The algorithm must see both real/imag parts to form complex moments holding zeros for the other part
    # We also have to track the symmetry enforced moments (real numbers) as our final comparison
    if 'mass' not in moments_df.columns:
        samples_in_bin = moments_df # (nsamples, alpha * n_moments)
    else:
        samples_in_bin = moments_df.query(f'mass == {masses[mass_bin]}') # (nsamples, alpha * n_moments)
    n_samples = len(samples_in_bin)
    assert n_samples > 0, f"No samples found for mass bin {mass_bin}"
    moment_samples = np.zeros((n_samples, len(moment_names)), dtype=np.complex128)
    sym_moment_samples = np.zeros((n_samples, len(moment_names)), dtype=np.float64)
    for i, moment_name in enumerate(moment_names):
        if moment_name in samples_in_bin.columns:
            is_h2 = moment_name.startswith('H2')
            real_part = samples_in_bin[moment_name] * (1 - is_h2)
            imag_part = samples_in_bin[moment_name] * is_h2
            moment_samples[:, i] = real_part + 1j * imag_part
            sym_moment_samples[:, i] = samples_in_bin[moment_name] # Track moments with enforced symmetry
    moment_names = np.array(moment_names).reshape(3, -1) # Reshape for future use [ [H0(0,0), ..., H0(L,M)], [H1(0,0), ..., H1(L,M)], [H2(0,0), ..., H2(L,M)] ]
        
    # moment_samples ~ (n_samples, alpha * n_moments_per_alpha)
    #   reshape to (alpha, n_moments, n_samples) 
    #   swap axes to get our target shape (n_samples, alpha, n_moments)
    moment_samples = moment_samples.swapaxes(1, 0).reshape(3, -1, n_samples).swapaxes(0, 2).swapaxes(1, 2)
    sym_moment_samples = sym_moment_samples.swapaxes(1, 0).reshape(3, -1, n_samples).swapaxes(0, 2).swapaxes(1, 2)
    
    ######################################################
    ######### MOMENT INVERSION USING SVGD ################
    ######################################################
    
    def model(moment_samples, scale):
        """
        Model that infers amplitudes from a distribution of moments.
        
        Args:
            moment_samples: Array of shape (n_samples, 3, n_moments_per_type) containing
                        moment samples with noise or from real data
            n_flat_amplitudes: Number of flattened amplitude parameters
            mask: Mask with same shape as n_flat_amplitudes indicating which parameters are free (1s and 0s)
            l_max: Maximum angular momentum
            cg_coeffs: precomputed CG coefficients
            tightness: scales loss function to enforce tightness of fit
            loss_exponent: The exponent to use for error calculation (default: 2 = MSE)
            scale: Scaling factor for the amplitude parameters (default: 1.0)
        """
        
        # NOTE: There is a significant difference between sampling Normal(0, 1) then multiplying by scale
        #       vs sampling Normal(0, scale)
        # Repulsive Force: RBFKernel uses the median heuristic bandwidth and therefore scale invariant
        # Attractive Force: is the gradient of the logp which contains the likelihood term and prior term
        #                   scales inverse-squared with the scale of the normal
        # Adam Optimizer: Works better with standardized coordinates
        free_params = numpyro.sample(
            "free_params",
            dist.Normal(0, 1).expand([int(sum(mask))]) # accepts an array
        )    
        flat_amps = jnp.zeros(n_flat_amplitudes, dtype=jnp.float64)
        flat_amps = flat_amps.at[mask==1].set(jnp.array(free_params, dtype=jnp.float64) * scale)    
        likelihood = calculate_likelihood(flat_amps, moment_samples, mask, cg_coeffs, tightness, loss_exponent, l_max)
        numpyro.factor("likelihood", likelihood)
    
    partial_make_bandwidth_factor = partial(make_bandwidth_factor, **bandwidth_params)
    kernel = ScheduledRBFKernel(partial_make_bandwidth_factor)
    optimizer = Adam(step_size=step_size)
    svgd = SVGD(model, optimizer, kernel, num_stein_particles=num_particles)
    rng_key = jax.random.PRNGKey(0)

    console.print("\nCompiling prior model and initializing SVGD...")
    start_compile_svgd = time.time()
    svgd_state = svgd.init(rng_key, moment_samples=moment_samples, scale=amplitude_scale)
    compile_svgd_time = time.time() - start_compile_svgd
    console.print(f"  SVGD compilation time: {compile_svgd_time:.2f} seconds")
    
    @jax.jit
    def step(svgd_state, t, moment_samples, scale):
        return svgd.update(svgd_state, moment_samples=moment_samples, scale=scale)

    start_run_svgd = time.time()
    for t in trange(num_iterations):
        svgd_state_output = step(svgd_state, t, moment_samples, amplitude_scale)
        svgd_state = svgd_state_output[0]
    svgd_time = time.time() - start_run_svgd
    console.print(f"SVGD sampling completed in {svgd_time:.2f} seconds")
    console.print(f"Params keys: {list(svgd.get_params(svgd_state).keys())}")

    #############################################################
    ######### PROJECT INFERRED AMPLITUDES TO MOMENT BASIS #######
    #####################################s########################

    # NOTE: numpyro attaches auto_loc to the name for w.e. reason
    free_params = svgd.get_params(svgd_state)['free_params_auto_loc'] * amplitude_scale  # shape (num_particles, n_free_real_indices)
    console.print(f"Number of free parameters: {free_params.shape} ~ (n_particles, n_free_parameters)")

    # Reconstruct the full amplitude array for each particle
    num_particles = free_params.shape[0]
    inferred_amplitudes = []
    for i in range(num_particles):
        full_amps = jnp.zeros(n_flat_amplitudes)
        full_amps = full_amps.at[mask==1].set(free_params[i])
        
        # Check if masking is done correctly
        masked_indices = jnp.where(mask == 0)[0]
        assert jnp.all(full_amps[masked_indices] == 0), f"Masked indices {masked_indices} not zero in particle {i}"

        inferred_amplitudes.append(full_amps)
    inferred_amplitudes = jnp.stack(inferred_amplitudes)
    console.print(f"\nReconstructed amplitudes shape: {inferred_amplitudes.shape} ~ (n_particles, n_flat_amplitudes)")

    # Map processing function (projecting to moment basis) over all particles
    partial_process_particle = partial(process_particle, mask=mask, l_max=l_max, cg_coeffs=cg_coeffs)
    process_batch = jit(vmap(partial_process_particle))
    recovered_moments = process_batch(inferred_amplitudes)

    ###############################
    ## SAVE RESULTS IN DATAFRAME ##
    ###############################

    pred_df = {}
    ampNames = []
            
    # 1. Save inferred amplitudes
    idx = 0
    for eps in range(n_eps):
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                refl = -1 if eps == 0 else 1
                ampName = qn_to_amp[(refl, l, m)] + "_amp"
                # Note: The inferred_amplitudes are already scaled by GLOBAL_SCALE 
                pred_df[ampName] = inferred_amplitudes[:, idx] + 1j * inferred_amplitudes[:, idx + 1]
                ampNames.append(ampName)
                idx += 2      
    ampNames = np.unique(ampNames)    
    ampNames = sorted(ampNames, key=lambda x: (x[-1], x[0]))
    pred_df = pd.DataFrame(pred_df)

    # 2. Save projected inferred moments (going full circle here)
    for alpha, names in enumerate(moment_names): # H_alpha(L, M)
        for moment_idx, name in enumerate(names):
            pred_df[name] = recovered_moments[:, alpha, moment_idx]

    # 3. Calculate likelihood for inferred amplitudes
    partial_calculate_likelihood = partial(calculate_likelihood, moment_samples=moment_samples, mask=mask, cg_coeffs=cg_coeffs,
                                           tightness=tightness, loss_exponent=loss_exponent, l_max=l_max)
    calc_batch_likelihood = jax.vmap(partial_calculate_likelihood)
    likelihood_values = calc_batch_likelihood(inferred_amplitudes)
    pred_df['likelihood'] = np.array(likelihood_values)

    pred_df['mass'] = masses[mass_bin]
    pred_df['mass'] = pred_df['mass'].astype(np.float64)
    pred_df['sample'] = np.arange(len(pred_df))

    pred_df, name_mapping = zero_moment_parts(pred_df, moment_names)

    ########################################################
    ######### SAVE RESULTS TO PICKLE ######################
    ########################################################
    
    # If the yaml files needed to create the PWA manager is available we will append the
    #   intensities to the prediction dataframe with proper normalization
    if main_file is not None and iftpwa_yaml is not None:

        from iftpwa1.pwa.gluex.gluex_jax_manager import (
            GluexJaxManager,
        )
        from pyamptools.utility.opt_utils import Objective
        import logging

        pwa_manager = GluexJaxManager(comm0=None, mpi_offset=1,
                                    yaml_file=main_file,
                                    resolved_secondary=iftpwa_yaml, prior_simulation=False, sum_returned_nlls=False,
                                    logging_level=logging.WARNING)
        
        waveNames = pwa_manager.waveNames
        nPars = 2 * len(waveNames)
        nmbMasses = len(pwa_manager.masses)
        nmbTprimes = len(pwa_manager.tPrimes)
        reference_waves = reference_waves
        
        obj = Objective(pwa_manager, 0, nPars, nmbMasses, nmbTprimes, reference_waves=reference_waves)
        
        # NOTE: For some reason I think iterrows type converts my mass from float to complex, reenforce it
        intensity_dict = {}
        for irow, row in pred_df.iterrows():
            pwa_manager.set_bins(np.array([mass_bin]))
            final_params = np.zeros(nPars)
            for iw, waveName in enumerate(waveNames):
                final_params[2*iw] = np.real(row[f'{waveName}_amp'])
                # NOTE: If reference wave is not restricted to be real then this code needs to change
                if waveName in reference_waves:
                    final_params[2*iw+1] = 0.0
                else:
                    final_params[2*iw+1] = np.imag(row[f'{waveName}_amp'])
                
            # Call once to setup the objective function loading all normalization integrals
            if irow == 0:
                obj.objective(final_params)
                
            intensity, intensity_error = obj.intensity_and_error(final_params, None, waveNames, acceptance_correct=False)
            intensity_dict.setdefault('intensity', []).append(intensity)
            if intensity_error is not None:
                intensity_dict.setdefault('intensity_err', []).append(intensity_error)
            for waveName in waveNames:
                intensity, intensity_error = obj.intensity_and_error(final_params, None, [waveName], acceptance_correct=False)
                intensity_dict.setdefault(waveName, []).append(intensity)
                if intensity_error is not None:
                    intensity_dict.setdefault(f"{waveName}_err", []).append(intensity_error)
                    
        intensity_df = pd.DataFrame(intensity_dict)

        pred_df = pd.concat([pred_df, intensity_df], axis=1)
        # end config file check

    results = {
        'prediction': pred_df,
        'params': {
            'num_particles': num_particles,
            'num_iterations': num_iterations,
            'tightness': tightness,
            'loss_exponent': loss_exponent,
            'l_max': l_max,
            'seed': seed,
            'bandwidth_params': bandwidth_params,
            'amplitude_scale': amplitude_scale,
        }
    }

    output_file = os.path.join(output_dir, f"moment_inversion_massBin{mass_bin}.pkl")
    console.print(f"Saving results to {output_file}")
    with open(output_file, 'wb') as f:
        pkl.dump(results, f)

def zero_moment_parts(df, moment_names):
    """
    Enforce H0/H1 to be real and H2 to be imaginary
    
    Args:
        df: DataFrame containing the inferred amplitudes and moments
        moment_names: List of moment names ~ [ [H0(0,0), ..., H0(L,M)], [H1(0,0), ..., H1(L,M)], [H2(0,0), ..., H2(L,M)] ]
        
    Returns:
        df: DataFrame trading H0(L,M) for Re[H0(L,M)] H1(L,M) for Re[H1(L,M)] and H2(L,M) for Im[H2(L,M)]
        name_mapping: Dictionary mapping the original moment names to the new moment names
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

if __name__ == "__main__":

    ######################################################
    # Define and parse command line arguments
    parser = argparse.ArgumentParser(description='Moment inversion using SVGD')
    parser.add_argument('main_file', type=str, help='main YAML file for result manager')
    parser.add_argument('-mb', '--mass_bins', type=int, nargs='+', default=[], 
                        help='Mass bin(s) to use for moment inversion (default: run over all mass bin)')
    parser.add_argument('-src', '--source', default='mcmc', choices=['mle', 'mcmc', 'ift', 'gen'], 
                        help='Source of moments (from resultManager) to use for moment inversion, will override each other if run multiple(default: mcmc)')
    
    parser.add_argument('-as', '--amplitude_scale', type=float, default=1.0, 
                        help='Scale factor for the amplitudes, all start N(0,1) (default: 1.0)')
    parser.add_argument('-np', '--num_particles', type=int, default=500, 
                        help='Number of Stein particles for SVGD (default: 500)')
    parser.add_argument('-ni', '--num_iterations', type=int, default=1000, 
                        help='Number of SVGD iterations (default: 5000)')
    parser.add_argument('-t', '--tightness', type=float, default=1000, 
                        help='Controls how tight the Stein fit is (default: 5000)')
    parser.add_argument('-le', '--loss_exponent', type=int, default=2, 
                        help='Exponent to use for the likelihood (default: 2 = MSE)')
    parser.add_argument('-di', '--decay_iterations', type=int, default=None, 
                        help='Kernel bandwidth decay iterations (default: num_iterations//5)')
    parser.add_argument('-is', '--initial_scale', type=float, default=1.0, 
                        help='Initial kernel bandwidth scale (default: 1.0)')
    parser.add_argument('-ms', '--min_scale', type=float, default=0.0, 
                        help='Minimum kernel bandwidth scale (default: 0.0)')
    parser.add_argument('-ss', '--step_size', type=float, default=0.01, 
                        help='Adam step size (default: 0.01)')
    
    parser.add_argument('-pr', '--n_processes', type=int, default=1,
                        help='Number of processes to use for parallel processing (default: 1)')
    parser.add_argument('-o', '--output_dir', type=str, default='', 
                        help="Output directory for results (defaults to main_file['base_directory'])")
    parser.add_argument('-s', '--seed', type=int, default=43, 
                        help='Random seed for reproducibility (default: 43)')

    args = parser.parse_args()

    ######################################################
    # Set parameters from parsed arguments
    ######################################################
    
    def from_yaml_else_argparse(yaml_dict, args, key):
        if yaml_dict is not None and 'moment_inversion' in yaml_dict and key in yaml_dict['moment_inversion']:
            return yaml_dict['moment_inversion'][key]
        else:
            return getattr(args, key) # args is a namespace
    
    main_file = args.main_file
    main_dict = load_yaml(main_file)
    n_processes = from_yaml_else_argparse(main_dict, args, 'n_processes')
    source = from_yaml_else_argparse(main_dict, args, 'source')
    amplitude_scale = from_yaml_else_argparse(main_dict, args, 'amplitude_scale')
    num_particles = from_yaml_else_argparse(main_dict, args, 'num_particles')
    num_iterations = from_yaml_else_argparse(main_dict, args, 'num_iterations')
    tightness = from_yaml_else_argparse(main_dict, args, 'tightness')
    loss_exponent = from_yaml_else_argparse(main_dict, args, 'loss_exponent')
    seed = from_yaml_else_argparse(main_dict, args, 'seed')
    n_eps = 2
    decay_iterations = from_yaml_else_argparse(main_dict, args, 'decay_iterations')
    if isinstance(decay_iterations, str) and decay_iterations.lower() == 'none': # yaml file None -> 'None' str not None type
        decay_iterations = None
    bandwidth_params = {
        'decay_iterations': decay_iterations if decay_iterations is not None else num_iterations // 5,
        'initial_scale': from_yaml_else_argparse(main_dict, args, 'initial_scale'),
        'min_scale': from_yaml_else_argparse(main_dict, args, 'min_scale'),
    }
    mass_bins = from_yaml_else_argparse(main_dict, args, 'mass_bins')
    if len(mass_bins) == 0:
        mass_bins = list(range(main_dict['n_mass_bins']))
    step_size = from_yaml_else_argparse(main_dict, args, 'step_size')
    console.print(f"Number of devices used: {jax.device_count()}")

    output_dir = from_yaml_else_argparse(main_dict, args, 'output_dir')
    if output_dir == '':
        output_dir = os.path.join(main_dict['base_directory'], 'MOMENT_INVERSION')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    os.system(f"cp {args.main_file} {output_dir}/{os.path.basename(args.main_file)}")
    console.print(f"Copied {args.main_file} to {output_dir}/{os.path.basename(args.main_file)}", style="bold blue")

    # Seed both just in case
    np.random.seed(seed)
    rng_key = jax.random.PRNGKey(seed)
    ######################################################
    
   # TODO: Allow user to specify which dataset then check existence crash if moments not found
    #       For MLE we use only point estimates since no uncertainties
    resultManager = ResultManager(main_dict)
    resultManager.attempt_load_all()
    resultManager.attempt_project_moments() # safe call since it checks for existence before processing
    masses = resultManager.masses
    waveNames = resultManager.waveNames
    reference_waves = resultManager.phase_reference
    iftpwa_yaml = resultManager.ift_yaml
    
    if source == 'mcmc':
        console.print(f'\nInverting moments from [bold green]MCMC[/bold green] results')
        moments_df = resultManager.mcmc_results
    elif source == 'mle':
        console.print(f'\nInverting moments from [bold green]MLE[/bold green] results (using point estimates only)')
        moments_df = resultManager.mle_results
    elif source == 'ift':
        console.print(f'\nInverting moments from [bold green]IFTPWA[/bold green] results')
        moments_df = resultManager.ift_results[0]
    elif source == 'gen':
        console.print(f'\nInverting moments from [bold green]GENERATED[/bold green] results')
        moments_df = resultManager.gen_results[0]
    console.print(f"  Shape of loaded dataframe (includes all kinematic bins): {moments_df.shape}\n")
    if len(moments_df) == 0:
        raise ValueError(f"No moments found for source {source}")
    
    multiprocessing.set_start_method('spawn')
        
    # Process mass bins in parallel
    console.rule()
    console.print(f"Processing {len(mass_bins)} mass bins: {mass_bins}")
    n_processes = min(n_processes, len(mass_bins))
    console.print(f"Using {n_processes} processes")
    console.rule()
    
    configs = [
        {'main_file': main_file,
        'mass_bin': mb,
        'n_eps': n_eps,
        'step_size': step_size,
        'bandwidth_params': bandwidth_params,
        'num_particles': num_particles,
        'amplitude_scale': amplitude_scale,
        'tightness': tightness,
        'loss_exponent': loss_exponent,
        'num_iterations': num_iterations,
        'seed': seed,
        'output_dir': output_dir,
        'ift_yaml': iftpwa_yaml,
        'waveNames': waveNames,
        'reference_waves': reference_waves,
        'masses': masses,
        'moments_df': moments_df}
        for mb in mass_bins
    ]
    
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_mass_bin, configs)
        pool.close()
        pool.join()
        
    with open(os.path.join(output_dir, 'run_conditions.txt'), 'w') as f:
        f.write(f"main_file: {os.path.realpath(main_file)}\n")
        f.write(f"waveNames: {waveNames}\n")
        f.write(f"reference_waves: {reference_waves}\n")
        f.write(f"source: {source}\n")
        f.write(f"num_particles: {num_particles}\n")
        f.write(f"num_iterations: {num_iterations}\n")
        f.write(f"amplitude_scale: {amplitude_scale}\n")
        for k, v in bandwidth_params.items():
            f.write(f"bandwidth_params[{k}]: {v}\n")
        f.write(f"tightness: {tightness}\n")
        f.write(f"loss_exponent: {loss_exponent}\n")
        f.write(f"step_size: {step_size}\n")
        f.write(f"seed: {seed}\n")
        
    console.print(f"Completed processing all {len(mass_bins)} mass bins")
