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

from optimize_utility import Objective

Minuit.errordef = Minuit.LIKELIHOOD
comm0 = None
rank = 0 
mpi_offset = 1

# Cannot get non-log magnitudes to work with polar coordinates
# - half normal mag + (von mises, xy-coord, tan half angle) phase distributions all fail, with small step size and high target accept rate
use_log_magnitudes = True

# 'tan' = tan half angle form
# 'vonmises' = Von Mises with circular reparametrization
use_phase_param = 'vonmises'

console = Console()

def run_mcmc_inference(objective, prior_scale=100.0, n_warmup=1000, n_samples=10, n_chains=4, resume_state=None, save_path=None, ref_idx=None, cop="cartesian", init_steps=100, init_lr=0.01, init_decay=0.99, init_momentum=0.9, init_grad_tol=1e-4):
    """ Run MCMC inference """
    rng_key = jax.random.PRNGKey(0)
    
    from jax import jit, vmap
    objective_fn = jit(objective.objective)
    gradient_fn = jit(objective.gradient)
    
    ###################
    ### INITIALIZATION
    ###################
    # Draw random sample and run gradient descent steps to move it to better starting location
    def init_chain(key, i, n_grad_steps=init_steps, lr=init_lr, decay=init_decay, momentum=init_momentum, grad_tol=init_grad_tol):
        console.print(f"Initializing chain {i} (randomly + short gradient descent) ...", style="bold")
        init_param = jax.random.uniform(key, shape=(objective.nPars,), minval=-prior_scale, maxval=prior_scale)
        ref_mag = jnp.abs(init_param[2*ref_idx] + 1j * init_param[2*ref_idx+1])
        
        # Fix reference wave if specified
        if ref_idx is not None:
            init_param = init_param.at[2*ref_idx].set(ref_mag)
            init_param = init_param.at[2*ref_idx+1].set(0.0)
        
        params = init_param
        start_nll = objective_fn(params)
        
        # Initialize momentum velocity
        velocity = jnp.zeros_like(params)
        
        # Gradient descent with momentum and reference wave fixed
        for step in range(n_grad_steps):
            # Current learning rate with decay
            step_lr = lr * (decay ** step)
            
            grad = gradient_fn(params)
            
            # Early stopping if gradient is small enough
            if jnp.sqrt(jnp.sum(grad**2)) < grad_tol:
                console.print(f"  Chain {i}: Early stopping at step {step}/{n_grad_steps} (gradient norm < {grad_tol})")
                break
                
            if ref_idx is not None:
                grad = grad.at[2*ref_idx+1].set(0.0) # zero ref wave phase / imag part gradient
                
            # Update with momentum
            velocity = momentum * velocity - step_lr * grad
            params = params + velocity
            
            # Fix reference wave constraint
            if ref_idx is not None:
                params = params.at[2*ref_idx+1].set(0.0)
        
        end_nll = objective_fn(params)
        if end_nll > start_nll:
            console.print(f"  Warning: Initialization resulted in worse NLL for chain {i}!", style="bold red")
            console.print(f"  Start: {start_nll:.4f}, End: {end_nll:.4f}", style="red")
            # Fall back to original point if optimization made things worse
            params = init_param
        else:
            console.print(f"  Chain {i}: NLL improved from {start_nll:.4f} to {end_nll:.4f} [Delta={end_nll-start_nll:.4f}]")
            
        return params
    
    keys = jax.random.split(rng_key, n_chains + 1)
    rng_key = keys[0]
    chain_keys = keys[1:]
    initial_params = [] # Shape: (n_chains, nPars)
    for i in range(n_chains):
        initial_params.append(init_chain(chain_keys[i], i))
    initial_params = jnp.array(initial_params)
    
    if cop == "cartesian":
        if ref_idx is None:
            init_params = {'params': initial_params}  # Shape: (n_chains, nPars = 2 * len(waveNames))
        else:            
            mask = jnp.ones(2*len(waveNames), dtype=bool).at[2*ref_idx+1].set(False)
            init_params = {'params': initial_params[:, mask]}  # Shape: (n_chains, nPars-1)
    elif cop == "polar":
        _initial_params = jnp.zeros((n_chains, 2 * len(waveNames)))
        _camp = initial_params[:, ::2] + 1j * initial_params[:, 1::2] # (c)omplex (amp)litude
        
        # Convert magnitudes to log-space for initialization
        _magnitudes = jnp.maximum(jnp.abs(_camp), 1e-5)    # Ensure positive values (for log)
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
        _initial_params = _initial_params.at[:, ::2].set(_magnitudes)
        _initial_params = _initial_params.at[:, 1::2].set(_phases)  # Still store phases for display
        
        if ref_idx is None:
            init_params = {
                'magnitudes': _initial_params[:, ::2],  # Shape: (n_chains, len(waveNames))
                'phase_params': _phase_params           # Shape: (n_chains, len(waveNames))
            }
        else:
            mask = jnp.ones(len(waveNames), dtype=bool).at[ref_idx].set(False)
            init_params = {
                'magnitudes': _initial_params[:, ::2],  # Shape: (n_chains, len(waveNames))
                'phase_params': _phase_params[:, mask]  # Shape: (n_chains, len(waveNames)-1)
            }
    else:
        raise ValueError(f"Invalid coordinate system: {cop}")

    console.print("\n\n=== INITIAL CHAIN PARAMETERS ===", style="bold")
    for k, v in init_params.items():
        console.print(f"{k}: {v}", style="bold")
    
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
            if ref_idx is None:
                params = numpyro.sample(
                    "params",
                    dist.Normal(loc=jnp.zeros((objective.nPars)), scale=prior_scale)
                )
            else:
                free_indices = jnp.array([i for i in range(objective.nPars) if i != 2*ref_idx+1])
                free_params = numpyro.sample(
                    "params",
                    dist.Normal(loc=jnp.zeros((objective.nPars - 1)), scale=prior_scale) # fixed imag part of ref wave
                )                
                params = jnp.zeros(objective.nPars)
                params = params.at[free_indices].set(free_params)
                
        elif cop == "polar":
            if ref_idx is None:
                magnitudes = numpyro.sample(
                    "magnitudes",
                    dist.Normal(loc=jnp.zeros(len(waveNames)), scale=jnp.log(prior_scale)) if use_log_magnitudes else
                    dist.HalfNormal(scale=prior_scale * jnp.ones(len(waveNames)))
                )
                magnitudes = jnp.exp(magnitudes) if use_log_magnitudes else magnitudes
                
                if use_phase_param == 'tan':
                    # Sample unbounded parameter u from normal distribution
                    phase_params = numpyro.sample(
                        "phase_params",
                        dist.Normal(loc=jnp.zeros(len(waveNames)), scale=jnp.ones(len(waveNames)))
                    )
                    # Convert back to phases: θ = 2*arctan(u)
                    phases = 2 * jnp.arctan(phase_params)
                elif use_phase_param == 'vonmises':
                    # concentration=0 makes it a uniform distribution over [-π, π]
                    phases = numpyro.sample(
                        'phase_params',
                        dist.VonMises(loc=jnp.zeros(len(waveNames)), concentration=jnp.zeros(len(waveNames)))
                    )

            else:
                magnitudes = numpyro.sample(
                    "magnitudes",
                    dist.Normal(loc=jnp.zeros(len(waveNames)), scale=jnp.log(prior_scale)) if use_log_magnitudes else
                    dist.HalfNormal(scale=prior_scale * jnp.ones(len(waveNames)))
                )
                magnitudes = jnp.exp(magnitudes) if use_log_magnitudes else magnitudes
                
                free_phase_indices = jnp.array([i for i in range(len(waveNames)) if i != ref_idx])
                
                if use_phase_param == 'tan':
                    # Sample unbounded parameters for non-reference waves
                    phase_params_free = numpyro.sample(
                        "phase_params",
                        dist.Normal(loc=jnp.zeros(len(waveNames)-1), scale=jnp.ones(len(waveNames)-1))
                    )
                    # Convert back to phases
                    free_phases = 2 * jnp.arctan(phase_params_free)
                elif use_phase_param == 'vonmises':
                    # concentration=0 makes it a uniform distribution over [-π, π]
                    free_phases = numpyro.sample(
                        'phase_params',
                        dist.VonMises(loc=jnp.zeros(len(waveNames)-1), concentration=jnp.zeros(len(waveNames)-1))
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
        target_accept_prob=0.8,       # Increase from 0.9 to encourage even smaller steps
        max_tree_depth=8,              # Allow deeper search but with more careful step size
        step_size=0.01 if cop == "polar" else 0.1,  # Much smaller step size for polar
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
            mcmc.run(rng_key, init_params=init_params)
    else:
        # Normal run with warmup
        mcmc.run(rng_key, init_params=init_params)
    
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
    reference_wave=None,
    cop="cartesian",
    init_steps=100,
    init_lr=0.01,
    init_decay=0.99,
    init_momentum=0.9,
    init_grad_tol=1e-4,
    ):
    """
    Run fit with specified method
    
    Args:
        bin_idx: Index of bin to optimize
        resume_path: Path to saved MCMC state to resume from
        save_path: Path to save MCMC state for future resuming
        reference_wave: Wave to use as reference (fixed to amplitude 1+0j)
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
    
    # Identify reference wave index if provided
    ref_idx = None
    if reference_wave is not None:
        if reference_wave in waveNames:
            ref_idx = waveNames.index(reference_wave)
            console.print(f"Using '{reference_wave}' as reference wave (par index {ref_idx})", style="bold green")
        else:
            console.print(f"Warning: Reference wave '{reference_wave}' not found in wave list. No reference will be used.", style="bold yellow")
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
                             save_path=save_path, ref_idx=ref_idx, cop=cop,
                             init_steps=init_steps, init_lr=init_lr, init_decay=init_decay,
                             init_momentum=init_momentum, init_grad_tol=init_grad_tol)
    samples = mcmc.get_samples() # ~ (n_samples, nPars)
    
    final_result_dict = {}
    
    if cop == "cartesian":
        n_samples = len(samples['params'])
        free_indices = jnp.array([i for i in range(obj.nPars) if i != 2*ref_idx+1])
        params = jnp.zeros((n_samples, obj.nPars))
        params = params.at[:, free_indices].set(samples['params'])
    elif cop == "polar":
        n_samples = len(samples['magnitudes'])
        magnitudes = samples['magnitudes']  # Shape: (n_samples, n_waves)
        
        if use_phase_param == 'tan':
            # Convert phase_params back to phases for tangent half-angle
            phase_params = samples['phase_params']
            
            if ref_idx is None:
                phases = 2 * jnp.arctan(phase_params)
            else:
                # Create full phases array with ref_idx phase set to 0
                full_phases = jnp.zeros((n_samples, len(waveNames)))
                non_ref_indices = [i for i in range(len(waveNames)) if i != ref_idx]
                full_phases = full_phases.at[:, non_ref_indices].set(2 * jnp.arctan(phase_params))
                phases = full_phases
        elif use_phase_param == 'vonmises':
            # For von Mises, the phases are directly sampled
            phase_params = samples['phase_params']
            
            if ref_idx is None:
                phases = phase_params
            else:
                # Create full phases array with ref_idx phase set to 0
                full_phases = jnp.zeros((n_samples, len(waveNames)))
                non_ref_indices = [i for i in range(len(waveNames)) if i != ref_idx]
                full_phases = full_phases.at[:, non_ref_indices].set(phase_params)
                phases = full_phases
        
        # Convert to Cartesian coordinates
        params = jnp.zeros((n_samples, obj.nPars))
        for i in range(len(waveNames)):
            magnitudes = jnp.exp(magnitudes) if use_log_magnitudes else magnitudes
            params = params.at[:, 2*i].set(magnitudes[:, i] * jnp.cos(phases[:, i]))     # Real part
            params = params.at[:, 2*i+1].set(magnitudes[:, i] * jnp.sin(phases[:, i]))   # Imaginary part
    
    for isample in range(n_samples): # Store intensities
        if "total" not in final_result_dict:
            final_result_dict["total"] = [obj.intensity(params[isample])]
        else:
            final_result_dict["total"].append(obj.intensity(params[isample]))
        for wave in pwa_manager.waveNames:
            if wave not in final_result_dict:
                final_result_dict[f"{wave}"] = [obj.intensity(params[isample], suffix=[wave])]
            else:
                final_result_dict[f"{wave}"].append(obj.intensity(params[isample], suffix=[wave]))
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
    parser.add_argument("-ref", "--reference_wave", type=str, default=None,
                       help="Wave to use as reference (fixed to amplitude 1+0j)")
    
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
    # momentum=0 and decay=1 will result in simple gradient descent
    parser.add_argument("--init_steps", type=int, default=100,
                       help="Number of gradient steps for initialization")
    parser.add_argument("--init_lr", type=float, default=0.01,
                       help="Initial learning rate for initialization")
    parser.add_argument("--init_decay", type=float, default=0.99,
                       help="Learning rate decay factor for initialization")
    parser.add_argument("--init_momentum", type=float, default=0.9,
                       help="Momentum coefficient for initialization")
    parser.add_argument("--init_grad_tol", type=float, default=1e-4,
                       help="Gradient norm tolerance for early stopping")
    
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
            reference_wave=args.reference_wave,
            cop=args.coordinate_system,
            init_steps=args.init_steps,
            init_lr=args.init_lr,
            init_decay=args.init_decay,
            init_momentum=args.init_momentum,
            init_grad_tol=args.init_grad_tol,
        )
        final_result_dicts.append(final_result_dict)
    
    _save_dir = os.path.dirname(args.save) + ("" if os.path.dirname(args.save) == "" else "/")
    _save_fname = os.path.basename(args.save)
    _save_fname = os.path.splitext(_save_fname)[0] # drop extension
    with open(f"{_save_dir}{_save_fname}_samples.pkl", "wb") as f:
        pkl.dump(final_result_dicts, f)

    console.print(f"Total time elapsed: {timer.read()[2]}", style="bold")

    sys.exit(0)
