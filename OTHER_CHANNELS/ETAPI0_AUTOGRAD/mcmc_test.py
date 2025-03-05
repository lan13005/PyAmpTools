# Configure JAX threading before setting device count
# Optimize for maximum parallel chains with controlled threading
import os
# os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# os.environ["XLA_FLAGS"] += " --xla_force_host_platform_device_count=40"
# os.environ["JAX_ENABLE_X64"] = "True"  # Enable double precision for numerical stability
# os.environ["JAX_PLATFORMS"] = "cpu"  # Explicitly use CPU
import jax
# jax.config.update('jax_default_matmul_precision', 'float32')  # Tradeoff precision for speed
import jax.numpy as jnp

from pyamptools.utility.general import load_yaml, Timer
import numpy as np
import sys
import pickle as pkl
from iminuit import Minuit
import argparse
from scipy.optimize import minimize
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
from rich.console import Console
import os

from pyamptools.utility.general import identify_channel, converter

from optimize_utility import Objective

Minuit.errordef = Minuit.LIKELIHOOD
comm0 = None
rank = 0 
mpi_offset = 1

# Things to consider:
# - COMPASS (~Boris) - lbfgs works well for binned fits but reverted to minuit for mass dependent fits since lbfgs was not able to find good minima
# - (Jiawei) Regularization - L1/L2 should be performed on the magnitude of the amplitude not the intensities (which would be L2/L4)

# Log magnitudes produces a bias in the total intensity
# - I guess this is log-normal magnitudes on each amplitude is biased upward and therefore the total intensity is biased upward
# - Linear spaced magnitudes are numerically unstable, I guess due to interference cancelling out contributions
use_log_magnitudes = False
use_phase_param = 'tan' # 'tan' = tan half angle form AND 'vonmises' = Von Mises with circular reparametrization

# init_maxiter does not appear to matter much
# - one test running lbfgs with [2, 5, 10, 20] iterations where ~30 is enough for deep convergence. Results basically were the same

console = Console()

import logging
import functools
from jax.debug import print as debug_callback

class JaxLogger:

    """ 
    JAX jit safe integration with logging module
    This function does not work with f-strings because formatting is delayed

    See jax.debug.print for more information. Printing is implemented through a callback
    https://jax.readthedocs.io/en/latest/_autosummary/jax.debug.print.html#jax-debug-print
    """

    def __init__(self, logger):
        self.logger = logger
        pass

    def _format_print_callback(self, fmt: str, level, *args, **kwargs):
        getattr(self.logger, level)(fmt.format(*args, **kwargs))

    def _jax_logger(self, fmt: str, level: str, *args, ordered: bool = False, **kwargs) -> None:
        debug_callback(functools.partial(self._format_print_callback, fmt, level.lower()), *args, **kwargs, ordered=ordered)

    def info(self, fmt: str, *args, **kwargs) -> None:
        self._jax_logger(fmt, "info", *args, **kwargs)

    def warning(self, fmt: str, *args, **kwargs) -> None:
        self._jax_logger(fmt, "warning", *args, **kwargs)

    def debug(self, fmt: str, *args, **kwargs) -> None:
        self._jax_logger(fmt, "debug", *args, **kwargs)

    def error(self, fmt: str, *args, **kwargs) -> None:
        self._jax_logger(fmt, "error", *args, **kwargs)

    def critical(self, fmt: str, *args, **kwargs) -> None:
        self._jax_logger(fmt, "critical", *args, **kwargs)
        
level = logging.INFO # ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
logger = logging.getLogger(__name__)
logger.setLevel(level)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s| %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(handler)
jax_logger = JaxLogger(logger)

def run_mcmc_inference(objective, prior_scale=100.0, n_warmup=1000, n_samples=10, n_chains=4, resume_state=None, save_path=None, ref_indices=None, cop="cartesian", init_method="L-BFGS-B", init_maxiter=100, init_gtol=1e-4, init_ftol=1e-6, wave_prior_scales=None):
    """ Run MCMC inference """
    rng_key = jax.random.PRNGKey(0)
    
    from jax import jit, vmap
    objective_fn = jit(objective.objective)
    gradient_fn  = jit(objective.gradient)
    
    # Process wave-specific prior scales if provided
    # NOTE: for cartesian only ATM
    console.print(f"\n\n***************************************************", style="bold")
    console.print(f"Default prior scale: {prior_scale}", style="bold")
    if wave_prior_scales is not None and cop == "cartesian":
        console.print(f"Using wave-specific prior scales:", style="bold")
        # Create array of prior scales for each parameter (real and imaginary parts)
        param_prior_scales = jnp.ones(objective.nPars) * prior_scale
        
        for wave_name, scale in wave_prior_scales.items():
            if wave_name in waveNames:
                wave_idx = waveNames.index(wave_name)
                param_prior_scales = param_prior_scales.at[2*wave_idx  ].set(scale)   # Real part
                param_prior_scales = param_prior_scales.at[2*wave_idx+1].set(scale)   # Imaginary part
                console.print(f"  {wave_name}: {scale}", style="bold")
            else:
                console.print(f"  Warning: Wave '{wave_name}' not found in wave list, ignoring custom prior scale", style="yellow")
    else:
        # Use uniform prior scale for all parameters
        param_prior_scales = jnp.ones(objective.nPars) * prior_scale
    console.print(f"***************************************************\n\n", style="bold")

    ###################
    ### INITIALIZATION
    ###################
    # Draw random sample and run optimization to move it to better starting location
    def init_chain(key, i, method=init_method, maxiter=init_maxiter, gtol=init_gtol, ftol=init_ftol):
        console.print(f"Initializing chain {i} (randomly + {method} optimization) ...", style="bold")
        # Use parameter-specific prior scales for initialization
        init_param = jax.random.uniform(key, shape=(objective.nPars,), minval=-100.0, maxval=100.0) # NOTE: Find better initialization of min/max range
        
        # Fix reference by rotating to be real
        for ref_idx in ref_indices:
            ref_mag = jnp.abs(init_param[2*ref_idx] + 1j * init_param[2*ref_idx+1])
            init_param = init_param.at[2*ref_idx  ].set(ref_mag)
            init_param = init_param.at[2*ref_idx+1].set(0.0)
        
        start_nll = objective_fn(init_param)
        
        # Define the objective function and its gradient for scipy.optimize
        def scipy_obj(x):
            return objective_fn(x).item()  # Convert to native Python type
            
        def scipy_grad(x):
            return np.array(gradient_fn(x))  # Convert to numpy array
        
        # Handle reference wave constraints using parameter bounds
        bounds = [(None, None)] * objective.nPars
        if ref_indices:
            for ref_idx in ref_indices:
                bounds[2*ref_idx+1] = (0.0, 0.0) # Fix imaginary part of reference wave to 0
        
        # Run optimization
        result = minimize(
            scipy_obj,
            np.array(init_param),  # Convert to numpy array for scipy
            method=method,
            jac=scipy_grad,
            bounds=bounds,
            options={
                'maxiter': maxiter,
                'gtol': gtol,
                'ftol': ftol,
                'disp': False
            }
        )
        
        params = jnp.array(result.x)  # Convert back to JAX array
        end_nll = objective_fn(params)
        
        # Check if optimization improved the likelihood
        if end_nll > start_nll:
            console.print(f"  Warning: Initialization resulted in worse NLL for chain {i}!", style="bold red")
            console.print(f"  Start: {start_nll:.4f}, End: {end_nll:.4f}", style="red")
            # Fall back to original point if optimization made things worse
            params = init_param
        else:
            console.print(f"  Chain {i}: NLL improved from {start_nll:.4e} to {end_nll:.4e} [Delta={end_nll-start_nll:.1f}, Iterations={result.nit}]") 
            
        return params
    
    keys = jax.random.split(rng_key, n_chains + 1)
    rng_key = keys[0]
    chain_keys = keys[1:]
    initial_params = [] # Shape: (n_chains, nPars)
    for i in range(n_chains):
        initial_params.append(init_chain(chain_keys[i], i))
    initial_params = jnp.array(initial_params)
        
    if cop == "cartesian":           
        # Create mask to exclude imaginary parts of reference waves
        mask = jnp.ones(objective.nPars, dtype=bool)
        for ref_idx in ref_indices:
            mask = mask.at[2*ref_idx+1].set(False)
            
        # # Basic initialization for 'params'
        # init_params = {'params': initial_params[:, mask]}  # Shape: (n_chains, nPars - len(ref_indices))
        
        # Non-centered parameterization of Horseshoe prior (better when posterior is dominated by prior term / likelihood not constraining)
        # The scale (of Cauchy) determines the location within which 50% of the distribution is contained
        # - Start with 10% of prior scale for global shrinkage (for each chain)
        # - Start with 50% of prior scale for local shrinkage  (for each parameter in the chain)
        global_scale_init = jnp.ones((n_chains, 1                                 )) * prior_scale * 0.01
        local_scale_init  = jnp.ones((n_chains, objective.nPars - len(ref_indices))) * prior_scale * 0.05
        
        param_magnitudes = jnp.abs(initial_params[:, mask])
        
        # Set raw_params to standard normal values, scaled to match optimized parameters
        # This ensures we start with parameters that reproduce approximately the same values 
        # as the optimized parameters, but with a proper hierarchical structure
        raw_params_init = jnp.clip(
            jnp.sign(initial_params[:, mask]) * 
            param_magnitudes / (global_scale_init * local_scale_init), 
            -3.0, 3.0
        )

        init_params = {
            'raw_params': raw_params_init,
            'global_scale': global_scale_init,
            'local_scale': local_scale_init,
            'params': raw_params_init * global_scale_init * local_scale_init,
        }
        
    elif cop == "polar":
        # NOTE: The initial parameters should by now have strictly applied the reference wave constraints
        _initial_params = jnp.zeros((n_chains, objective.nPars))
        _camp = initial_params[:, ::2] + 1j * initial_params[:, 1::2] # (c)omplex (amp)litude
        
        # Convert magnitudes to log-space for initialization
        _magnitudes = jnp.maximum(jnp.abs(_camp), 1e-5) # Ensure positive values (for log)
        if use_log_magnitudes:
            _magnitudes = jnp.log(_magnitudes)
        
        # Get phases from complex amplitudes
        _phases = jnp.angle(_camp)
        
        if use_phase_param == 'tan':
            # Convert phases to tangent half-angle parameter: u = tan(phase/2)
            # Add small epsilon to avoid exact π which would give infinity
            _phases_safe = jnp.where(jnp.abs(_phases) > jnp.pi - 1e-5, 
                                    _phases * (1 - 1e-5), 
                                    _phases)
            _phase_params = jnp.tan(_phases_safe / 2)
        elif use_phase_param == 'vonmises':
            # For von Mises, we just use the angles directly for initialization
            _phase_params = _phases
        else:
            raise ValueError(f"Invalid phase parameterization: {use_phase_param}")
        
        # Set magnitudes and store original phases for display
        _initial_params = _initial_params.at[:,  ::2].set(_magnitudes)
        _initial_params = _initial_params.at[:, 1::2].set(_phases)  # Still store phases for display

        # Create mask to exclude phases of reference waves
        mask = jnp.ones(len(waveNames), dtype=bool)
        for ref_idx in ref_indices:
            mask = mask.at[ref_idx].set(False)
        init_params = {
            'magnitudes': _initial_params[:, ::2],  # Shape: (n_chains, len(waveNames))
            'phase_params': _phase_params[:, mask]  # Shape: (n_chains, len(waveNames) - len(ref_indices))
        }
    else:
        raise ValueError(f"Invalid coordinate system: {cop}")

    console.print("\n\n=== INITIAL CHAIN PARAMETER VALUES ===", style="bold")
    for k, v in init_params.items():
        console.print(f"{k}: {v.shape} ~ (nChains, params)", style="bold")
        console.print(f"{v}\n", style="bold")

    ###################
    ### DISPLAY INITIAL INTENSITIES
    ###################
    console.print("\n=== INITIAL CHAIN INTENSITIES ===", style="bold")
    from rich.table import Table
    
    # Calculate intensities for each wave
    table = Table(title="Initial Chain Intensities")
    table.add_column("Chain", justify="right", style="cyan")
    table.add_column("Total", justify="right", style="green")
    table.add_column("NLL", justify="right", style="red")
    
    # Add columns for each wave
    for wave in waveNames:
        table.add_column(wave, justify="right")
    
    # Calculate and add intensities for each chain
    for i in range(n_chains):
        params = initial_params[i]
        total_intensity = objective.intensity(params)
        nll = objective_fn(params)
        
        # Get individual wave intensities
        wave_intensities = []
        for wave in waveNames:
            wave_intensity = objective.intensity(params, suffix=[wave])
            wave_intensities.append(f"{wave_intensity:.1f}")
        
        # Add row to table
        table.add_row(
            f"{i}", 
            f"{total_intensity:.1f}", 
            f"{nll:.1f}",
            *wave_intensities
        )
    
    console.print(table)
    console.print("")

    ###########################################
    ### MCMC CONFIGURATION
    ###########################################
    def model():
        
        """Do not include batch dimension in the sampling"""
        
        if cop == "cartesian":

            # Identify free parameters - exclude imaginary parts of reference waves
            free_indices = jnp.array([i for i in range(objective.nPars) if not any(i == 2*ref_idx+1 for ref_idx in ref_indices)])
            free_param_prior_scales = param_prior_scales[free_indices]

            ##### 
            # # Gaussian prior (L2 regularization)
            # free_params = numpyro.sample(
            #     "params",
            #     dist.Normal(loc=jnp.zeros((objective.nPars - len(ref_indices))), scale=free_param_prior_scales)
            # )

            # # Laplace prior (L1 regularization)
            # free_params = numpyro.sample(
            #     "params",
            #     dist.Laplace(loc=jnp.zeros((objective.nPars - len(ref_indices))), scale=free_param_prior_scales)
            # )
            
            # Non-centered Horseshoe prior (stronger sparsity than L1) with separate global and local shrinkage
            # Global shrinkage - controls overall sparsity
            global_scale = numpyro.sample(
                "global_scale",
                dist.HalfCauchy(scale=prior_scale * 0.01)  # Use 10% of prior_scale as global shrinkage base
            )
            # Local shrinkage - allows certain parameters to escape global shrinkage
            local_scale = numpyro.sample(
                "local_scale",
                dist.HalfCauchy(scale=jnp.ones(objective.nPars - len(ref_indices)) * prior_scale * 0.05)
            )
            raw_params = numpyro.sample(
                "raw_params",
                dist.Normal(jnp.zeros(objective.nPars - len(ref_indices)), jnp.ones(objective.nPars - len(ref_indices)))
            )

            # Apply scaling to get the actual parameters
            free_params = raw_params * local_scale * global_scale
            
            # Set parameters as before
            params = jnp.zeros(objective.nPars)
            params = params.at[free_indices].set(free_params)

        elif cop == "polar":

            magnitudes = numpyro.sample(
                "magnitudes",
                dist.Normal(loc=jnp.zeros(len(waveNames)), scale=jnp.log(prior_scale)) if use_log_magnitudes else
                dist.HalfNormal(scale=prior_scale * jnp.ones(len(waveNames)))
            )
            magnitudes = jnp.exp(magnitudes) if use_log_magnitudes else magnitudes
            
            # Identify free phase parameters - exclude reference waves
            free_phase_indices = jnp.array([i for i in range(len(waveNames)) if i not in ref_indices])
            
            if use_phase_param == 'tan':
                # Sample unbounded parameters for non-reference waves
                phase_params_free = numpyro.sample(
                    "phase_params",
                    dist.Normal(loc=jnp.zeros(len(waveNames) - len(ref_indices)), 
                                scale=jnp.ones(len(waveNames) - len(ref_indices)))
                )
                # Convert back to phases
                free_phases = 2 * jnp.arctan(phase_params_free)
            elif use_phase_param == 'vonmises':
                # concentration=0 makes it a uniform distribution over [-π, π]
                free_phases = numpyro.sample(
                    'phase_params',
                    dist.VonMises(loc=jnp.zeros(len(waveNames) - len(ref_indices)), 
                                    concentration=jnp.zeros(len(waveNames) - len(ref_indices)))
                )
            
            phases = jnp.zeros(len(waveNames))        
            phases = phases.at[free_phase_indices].set(free_phases)
            
            real_parts = magnitudes * jnp.cos(phases)
            imag_parts = magnitudes * jnp.sin(phases)
            
            # Move into cartesian coordinates for objective function
            params = jnp.zeros(objective.nPars)
            for i in range(len(waveNames)):
                params = params.at[2*i].set(real_parts[i])
                params = params.at[2*i+1].set(imag_parts[i])

        else:
            raise ValueError(f"Invalid coordinate system: {cop}")

        # Handle both batched and non-batched cases (multi-chain or not)
        if params.ndim > 1:
            batched_objective_fn = vmap(objective_fn)
            nll = batched_objective_fn(params)
        else:
            nll = objective_fn(params)
        
        regularization = 0.0
        # if cop == "polar": # Add regularization to prevent very large intensity (due to numerical instability)
        #     console.print("Polar coordinate warning: adding small regularization to stabilize intensity", style="bold yellow")
        #     free_magnitudes = jnp.array([i for i in range(objective.nPars) if i % 2 == 0])
        #     regularization = 0.001 * jnp.sum(params[free_magnitudes]**2)
        nll = nll + regularization
        
        numpyro.factor("likelihood", -nll)
    
    nuts_kernel = NUTS(
        model,
        target_accept_prob=0.85,       # Increase from 0.9 to encourage even smaller steps
        max_tree_depth=12,            # Allow deeper search but with more careful step size
        step_size=0.05 if cop == "polar" else 0.1,  # Much smaller step size for polar
        adapt_step_size=True,
        dense_mass=True,               # Keep this for handling correlations
        adapt_mass_matrix=True         # Explicitly enable mass matrix adaptation
    )
    
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=n_warmup if resume_state is None else 0,  # Skip warmup if resuming
        num_samples=n_samples,
        num_chains=n_chains,
        chain_method='parallel',
        progress_bar=True
    )
    
    ###########################################
    ### MCMC RUN
    ###########################################
    rng_key, rng_key_mcmc = jax.random.split(rng_key)
    
    # Load saved state if requested
    if resume_state is not None:
        console.print(f"Resuming from saved state", style="bold green")
        try:
            # Set post_warmup_state to the saved state - this will skip warmup
            mcmc.post_warmup_state = resume_state
            mcmc.run(resume_state.rng_key)
        except Exception as e:
            console.print(f"Error loading MCMC state: {e}", style="bold red")
            console.print("Falling back to fresh start", style="yellow")
            mcmc.run(rng_key) # , init_params=init_params)
    else:
        # Normal run with warmup
        mcmc.run(rng_key) # , init_params=init_params)
    
    # Save state if requested
    if save_path is not None:
        console.print(f"Saving MCMC state to: {save_path}", style="bold green")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            with open(save_path, 'wb') as f:
                pkl.dump(mcmc.last_state, f)
        except Exception as e:
            console.print(f"Error saving MCMC state: {e}", style="bold red")
    
    ################################################
    ### MCMC DIAGNOSTICS - THE REST OF THIS FUNCTION
    ################################################
    
    # Leave this call here. mcmc.run() call does not wait for results to finish so the following print statements
    #   will be flushed before fit is complete. Use this as a barrier
    divergence = mcmc.get_extra_fields()["diverging"]
    
    console.print("\n\n\n=== MCMC CONVERGENCE DIAGNOSTICS ===", style="bold")
    
    console.print("\nR-hat Reference:", style="bold")
    console.print("R-hat (Gelman-Rubin statistic) measures chain convergence by comparing within-chain and between-chain variance:", style="italic")
    console.print("  - Values close to 1.0 indicate good convergence (chains explore similar regions)")
    console.print("  - Values > 1.01 suggest potential convergence issues (chains may be exploring different regions)") 
    console.print("  - Values > 1.05 indicate poor convergence (chains have not mixed well)")
    
    console.print("\nEffective Sample Size (n_eff) Reference:", style="bold")
    console.print("n_eff estimates the number of effective independent samples after accounting for autocorrelation:", style="italic")
    console.print("  - Values should be compared to total number of MCMC samples (n_chains * n_samples)")
    console.print("  - n_eff ≈ total samples: Parameter explores well, low autocorrelation")
    console.print("  - n_eff << total samples: High autocorrelation, may need longer chains")
    console.print("  - Rule of thumb: n_eff > 100 per parameter for reliable inference")
    
    console.print("\n\nParameters:", style="bold")
    par_idx = 0
    for wave_idx, wave in enumerate(waveNames):
        console.print(f"params[{par_idx}] corresponds to Re[{wave}]", style="bold")
        par_idx += 1
        if wave_idx not in ref_indices:
            console.print(f"params[{par_idx}] corresponds to Im[{wave}]", style="bold")
            par_idx += 1
    
    # Print standard NumPyro summary
    mcmc.print_summary()
    
    # Divergence information
    n_divergent = jnp.sum(divergence)
    divergence_pct = n_divergent/n_samples*100
    console.print(f"\nNumber of divergent transitions: {n_divergent} ({divergence_pct:.1f}%)")
    if divergence_pct > 0.5:
        console.print("WARNING: Divergences detected (>0.5%)! This indicates the sampler is struggling with difficult geometry.", style="red")
        console.print("    Consider: increasing adapt_step_size, increasing target_accept_prob, or reparameterizing your model.", style="red")
    else:
        console.print("GOOD: Few or no divergences.", style="green")
    
    return mcmc

def run_fit(
    pyamptools_yaml, 
    iftpwa_yaml,
    bin_idx, 
    prior_scale=100.0,
    n_chains=4,
    n_samples=2000,
    n_warmup=1000,
    resume_path=None,
    save_path=None,
    reference_waves=None,
    cop="cartesian",
    init_method="L-BFGS-B",
    init_maxiter=100,
    init_gtol=1e-4,
    init_ftol=1e-6,
    wave_prior_scales=None,
    ):
    """
    Run fit with specified method
    
    Args:
        bin_idx: Index of bin to optimize
        resume_path: Path to saved MCMC state to resume from
        save_path: Path to save MCMC state for future resuming
        reference_waves: List of waves to use as references (fixes amplitude imag part or phase to be 0),
                        typically one per reflectivity sector
        wave_prior_scales: Dictionary mapping wave names to prior scales (only for cartesian coordinates)
    """
    
    from iftpwa1.pwa.gluex.gluex_jax_manager import (
        GluexJaxManager,
    )

    pwa_manager = GluexJaxManager(comm0=comm0, mpi_offset=mpi_offset,
                                yaml_file=pyamptools_yaml,
                                resolved_secondary=iftpwa_yaml, prior_simulation=False, sum_returned_nlls=False)
    pwa_manager.prepare_nll()
    pwa_manager.set_bins(np.array([bin_idx]))
        
    obj = Objective(pwa_manager, bin_idx, nPars, nmbMasses, nmbTprimes)
    
    # Process reference waves here once
    ref_indices = []
    channel = None
    refl_sectors = {}
    
    if reference_waves:
        if isinstance(reference_waves, str):
            reference_waves = [reference_waves]
        
        # Get reaction channel for all waves 
        channel = identify_channel(waveNames)
        console.print(f"Identified reaction channel: {channel}", style="bold")
        
        # Get reflectivity sectors and their waves
        for i, wave in enumerate(waveNames):
            # Extract reflectivity from wave using converter dictionary
            refl = converter[wave][0]  # First element is reflectivity (e)
            if refl not in refl_sectors:
                refl_sectors[refl] = []
            refl_sectors[refl].append((i, wave))
        
        # Process each reference wave
        for ref_wave in reference_waves:
            if ref_wave in waveNames:
                ref_idx = waveNames.index(ref_wave)
                ref_indices.append(ref_idx)
                refl = converter[ref_wave][0]
                console.print(f"Using '{ref_wave}' as reference wave for reflectivity sector {refl} (wave index {ref_idx})", style="bold green")
            else:
                console.print(f"Error: Reference wave '{ref_wave}' not found in wave list!", style="bold red")
                sys.exit(1)
        
        # Check if we have at least one reference wave per reflectivity sector
        for refl, waves in refl_sectors.items():
            wave_indices = [idx for idx, _ in waves]
            if not any(idx in ref_indices for idx in wave_indices):
                console.print(f"Error: No reference wave specified for reflectivity sector = {refl}!", style="bold red")
                sys.exit(1)
    
    console.print(f"Using coordinate system: {cop}", style="bold green")
    
    # If resume_path is specified, load the saved state
    resume_state = None
    if resume_path is not None:
        try:
            with open(resume_path, 'rb') as f:
                resume_state = pkl.load(f)
                console.print(f"Loaded saved state from {resume_path}", style="bold green")
        except Exception as e:
            console.print(f"Error loading state from {resume_path}: {e}", style="bold red")
    
    console.print("\n**************************************************************", style="bold")
    mcmc = run_mcmc_inference(obj, prior_scale=prior_scale, n_samples=n_samples, n_chains=n_chains, 
                             n_warmup=n_warmup, resume_state=resume_state, 
                             save_path=save_path, 
                             ref_indices=ref_indices, # is in waveName index space
                             cop=cop,
                             init_method=init_method,
                             init_maxiter=init_maxiter,
                             init_gtol=init_gtol,
                             init_ftol=init_ftol,
                             wave_prior_scales=wave_prior_scales)
    samples = mcmc.get_samples() # ~ (n_samples, nPars)
    
    final_result_dict = {}
    
    if cop == "cartesian":
    
        # n_samples = len(samples['params'])
        
        # Fix: Properly reconstruct the actual parameters from the Horseshoe components
        raw_params = samples['raw_params']
        global_scale = samples['global_scale']
        local_scale = samples['local_scale']
        
        # Compute the actual parameters from the raw values and scales
        n_samples = len(raw_params)
        free_indices = jnp.array([i for i in range(obj.nPars) if not any(i == 2*ref_idx+1 for ref_idx in ref_indices)])
        params = jnp.zeros((n_samples, obj.nPars))
        # params = params.at[:, free_indices].set(samples['params'])
        
        # Compute actual parameters: raw_params * local_scale * global_scale
        # Handle broadcasting for proper multiplication
        actual_params = raw_params * local_scale
        if global_scale.ndim == 2:  # If global_scale is (n_samples, 1)
            actual_params = actual_params * global_scale
        else:  # If global_scale is (n_samples,)
            actual_params = actual_params * global_scale[:, None]
        params = params.at[:, free_indices].set(actual_params)
    elif cop == "polar":
        n_samples = len(samples['magnitudes'])
        magnitudes = samples['magnitudes']  # Shape: (n_samples, n_waves)
        
        # Apply log-to-linear conversion ONCE here if needed
        magnitudes = jnp.exp(magnitudes) if use_log_magnitudes else magnitudes
        
        # Create full phases array with ref_idx phase set to 0
        phase_params = samples['phase_params']
        non_ref_indices = [i for i in range(len(waveNames)) if i not in ref_indices]
        phases = jnp.zeros((n_samples, len(waveNames)))
        
        if use_phase_param == 'tan':
            # Convert phase_params back to phases for tangent half-angle
            phases = phases.at[:, non_ref_indices].set(2 * jnp.arctan(phase_params))
        elif use_phase_param == 'vonmises':
            # For von Mises, the phases are directly sampled
            phases = phases.at[:, non_ref_indices].set(phase_params)
        
        # Convert to Cartesian coordinates
        params = jnp.zeros((n_samples, obj.nPars))
        for i in range(len(waveNames)):
            params = params.at[:, 2*i].set(magnitudes[:, i] * jnp.cos(phases[:, i]))     # Real part
            params = params.at[:, 2*i+1].set(magnitudes[:, i] * jnp.sin(phases[:, i]))   # Imaginary part
    
    for isample in range(n_samples): # Store intensities
        # Check if parameters contain NaN values
        if jnp.any(jnp.isnan(params[isample])):
            console.print(f"Warning: NaN detected in parameters for sample {isample}", style="bold red")
            total_intensity = jnp.nan
        else:
            total_intensity = obj.intensity(params[isample])
            # Check if intensity calculation resulted in NaN
            if jnp.isnan(total_intensity):
                console.print(f"Warning: NaN intensity calculated for sample {isample}", style="bold red")
        
        if "total" not in final_result_dict:
            final_result_dict["total"] = [total_intensity]
        else:
            final_result_dict["total"].append(total_intensity)
            
        for wave in pwa_manager.waveNames:
            if jnp.any(jnp.isnan(params[isample])):
                wave_intensity = jnp.nan
            else:
                wave_intensity = obj.intensity(params[isample], suffix=[wave])
                if jnp.isnan(wave_intensity):
                    console.print(f"Warning: NaN intensity calculated for wave {wave}, sample {isample}", style="bold red")
                    
            if wave not in final_result_dict:
                final_result_dict[f"{wave}"] = [wave_intensity]
            else:
                final_result_dict[f"{wave}"].append(wave_intensity)
                
    for iw, wave in enumerate(pwa_manager.waveNames): # Store complex amplitudes
        final_result_dict[f"{wave}_amp"] = params[:, 2*iw] + 1j * params[:, 2*iw+1]
        
    # Print some info on the resulting shape
    console.print(f"\n\nFinal result dict info (total=total intensity) ~ (nsamples={n_samples}):", style="bold")
    for k, v in final_result_dict.items():
        final_result_dict[k] = np.array(v)
        console.print(f"{k}: shape {final_result_dict[k].shape}", style="italic")
    console.print("**************************************************************\n", style="bold")

    return final_result_dict

class OptimizerHelpFormatter(argparse.ArgumentParser):
    def error(self, message):
        console.print(f"Error: {message}", style="red")
        self.print_help()
        sys.exit(2)

    def format_help(self):
        help_message = super().format_help()
        
        # method_help = "\nOptimizer Descriptions:\n"
        # method_help += "\nMinuit-based Methods:\n"
        # method_help += "  * minuit-numeric:\n"
        # method_help += "      Lets Minuit compute numerical gradients\n"
        # method_help += "  * minuit-analytic:\n"
        # method_help += "      Uses analytic gradients from PWA likelihood manager\n"
        
        # method_help += "\nSciPy-based Methods:\n"
        # method_help += "  * L-BFGS-B:\n"
        # method_help += "      Limited-memory BFGS quasi-Newton method (stores approximate Hessian)\n"
        # method_help += "      + Efficient for large-scale problems\n"
        # method_help += "      - May struggle with highly correlated parameters\n"
        
        # method_help += "  * trust-ncg:\n"
        # method_help += "      Trust-region Newton-Conjugate Gradient\n"
        # method_help += "      + Adaptively adjusts step size using local quadratic approximation\n"
        # method_help += "      + Efficient for large-scale problems\n"
        # method_help += "      - Can be unstable for ill-conditioned problems\n"
        
        # method_help += "  * trust-krylov:\n"
        # method_help += "      Trust-region method with Krylov subspace solver\n"
        # method_help += "      + Better handling of indefinite (sparse) Hessians, Kyrlov subspcae accounts for non-Euclidean geometry\n"
        # method_help += "      + More robust for highly correlated parameters\n"
        
        # return help_message + "\n" + method_help

        return help_message

if __name__ == "__main__":
    parser = OptimizerHelpFormatter(description="Run optimization fits using various methods.")
    parser.add_argument("yaml_file", type=str,
                       help="Path to PyAmpTools YAML configuration file")    
    parser.add_argument("-b", "--bins", type=int, nargs="+",
                       help="List of bin indices to process")
    parser.add_argument("-cop", "--coordinate_system", type=str, choices=["cartesian", "polar"], default="cartesian",
                       help="Coordinate system to use for the complex amplitudes")

    #### MCMC ARGS ####
    parser.add_argument("-ps", "--prior_scale", type=float, default=100.0,
                       help="Prior scale for the magnitude of the complex amplitudes")
    parser.add_argument("-nc", "--nchains", type=int, default=20,
                       help="Number of chains to use for numpyro MCMC (each chain runs on paralell process)")
    parser.add_argument("-ns", "--nsamples", type=int, default=2000,
                       help="Number of samples to draw per chain")
    parser.add_argument("-nw", "--nwarmup", type=int, default=1000,
                       help="Number of warmup samples to draw")
    
    #### SAVE/RESUME ARGS ####
    parser.add_argument("-r", "--resume", type=str, default=None,
                       help="Path to saved MCMC state to resume from, warmup will be skipped")
    parser.add_argument("-s", "--save", type=str, default="mcmc_state.pkl",
                       help="Path to save MCMC state for future resuming")
    
    #### HELPFUL ARGS ####
    parser.add_argument("--print_wave_names", action="store_true",
                       help="Print wave names")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    #### INITIALIZATION ARGS ####
    parser.add_argument("--init_method", type=str, default="L-BFGS-B",
                       choices=['L-BFGS-B', 'trust-ncg', 'trust-krylov'],
                       help="SciPy optimization method for initialization")
    parser.add_argument("--init_maxiter", type=int, default=100,
                       help="Maximum number of iterations for initialization optimizer. Standard fits typically converge within 100-200 iterations")
    parser.add_argument("--init_gtol", type=float, default=1e-4,
                       help="Gradient tolerance for initialization optimizer. Values of 1e-4 to 1e-5 are typical for standard fits; smaller values may be needed for high precision.")
    parser.add_argument("--init_ftol", type=float, default=1e-6,
                       help="Function value tolerance for initialization optimizer. Values of 1e-6 to 1e-8 are common for standard fits; smaller values ensure more precise convergence.")
    
    args = parser.parse_args()

    if args.save is None and os.path.exists(args.save):
        raise ValueError(f"Save path {args.save} already exists! Please provide a different path or remove the file.")

    # Then set the host device count
    numpyro.set_host_device_count(int(args.nchains))
    console.print(f"JAX is using {jax.local_device_count()} local devices", style="bold")
    
    np.random.seed(args.seed)
    
    pyamptools_yaml = load_yaml(args.yaml_file)
    iftpwa_yaml = pyamptools_yaml["nifty"]["yaml"]
    iftpwa_yaml = load_yaml(iftpwa_yaml)
    
    if not iftpwa_yaml:
        raise ValueError("iftpwa YAML file is required")
    if not pyamptools_yaml:
        raise ValueError("PyAmpTools YAML file is required")
    
    waveNames = pyamptools_yaml["waveset"].split("_")
    nmbMasses = pyamptools_yaml["n_mass_bins"]
    nmbTprimes = pyamptools_yaml["n_t_bins"]
    nPars = 2 * len(waveNames)
    
    if args.print_wave_names:
        console.print(f"Wave names: {waveNames}", style="bold")
        sys.exit(0)

    if args.bins is None:
        raise ValueError("list of bin indices is required")
    
    wave_prior_scales = None
    # wave_prior_scales = {
    #     "Sp0+": 150,
    #     "Dp2+": 150,
    # }
    
    reference_waves = pyamptools_yaml["phase_reference"].split("_")

    timer = Timer()
    final_result_dicts = []
    for bin_idx in args.bins:
        final_result_dict = run_fit(
            pyamptools_yaml, 
            iftpwa_yaml, 
            bin_idx, 
            prior_scale=args.prior_scale,
            n_chains=args.nchains,
            n_samples=args.nsamples,
            n_warmup=args.nwarmup,
            resume_path=args.resume,
            save_path=args.save,
            reference_waves=reference_waves,
            cop=args.coordinate_system,
            init_method=args.init_method,
            init_maxiter=args.init_maxiter,
            init_gtol=args.init_gtol,
            init_ftol=args.init_ftol,
            wave_prior_scales=wave_prior_scales,
        )
        final_result_dicts.append(final_result_dict)
    
    _save_dir = os.path.dirname(args.save) + ("" if os.path.dirname(args.save) == "" else "/")
    _save_fname = os.path.basename(args.save)
    _save_fname = os.path.splitext(_save_fname)[0] # drop extension
    with open(f"{_save_dir}{_save_fname}_samples.pkl", "wb") as f:
        pkl.dump(final_result_dicts, f)

    console.print(f"Total time elapsed: {timer.read()[2]}", style="bold")

    sys.exit(0)
