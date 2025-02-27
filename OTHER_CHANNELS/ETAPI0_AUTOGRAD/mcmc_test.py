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

console = Console()

def run_mcmc_inference(objective, prior_scale=100.0, n_warmup=1000, n_samples=10, n_chains=4, resume_state=None, save_path=None):
    """ Run MCMC inference """
    rng_key = jax.random.PRNGKey(0)
    
    from jax import jit, vmap
    objective_fn = jit(objective.objective)
    gradient_fn = jit(objective.gradient)
    
    ###################
    ### INITIALIZATION
    ###################
    # Draw random sample and run gradient descent steps to move it to better starting location
    lr = 0.01
    n_grad_steps = 100
    def init_chain(key, i):
        console.print(f"Initializing chain {i}...", style="bold")
        init_param = jax.random.uniform(key, shape=(objective.nPars,), minval=-prior_scale, maxval=prior_scale)
        params = init_param
        start_nll = objective_fn(params)
        for _ in range(n_grad_steps):
            params = params - lr * gradient_fn(params)
        end_nll = objective_fn(params)
        if end_nll > start_nll:
            raise ValueError(f"Initialization failed for chain {i}!")
        return params
    
    keys = jax.random.split(rng_key, n_chains + 1)
    rng_key = keys[0]
    chain_keys = keys[1:]
    initial_params = []
    for i in range(n_chains):
        initial_params.append(init_chain(chain_keys[i], i))
    initial_params = jnp.array(initial_params)
    
    cop = "cartesian" # cartesian or polar
    if cop == "polar":
        _initial_params = jnp.zeros((n_chains, 2 * len(waveNames)))
        _camp = initial_params[:, ::2] + 1j * initial_params[:, 1::2] # (c)omplex (amp)litude
        _initial_params = _initial_params.at[:, ::2].set(jnp.abs(_camp))
        _initial_params = _initial_params.at[:, 1::2].set(jnp.angle(_camp))
        # for iw, wave in enumerate(waveNames):
        #     print(f"{wave}: {initial_params[0, 2*iw]:.3f} + 1j * {initial_params[0, 2*iw+1]:.3f} or in polar: {_initial_params[0, 2*iw]:.3f} * exp(1j * {_initial_params[0, 2*iw+1]:.3f})")
        initial_params = _initial_params
        init_params = {
            'magnitudes': initial_params[:, ::2],  # Shape: (n_chains, len(waveNames))
            'phases': initial_params[:, 1::2]      # Shape: (n_chains, len(waveNames))
        }
    else:
        init_params = {'params': initial_params}

    ###########################################
    ### MCMC CONFIGURATION
    ###########################################
    def model():
        
        """Do not include batch dimension in the sampling"""
        
        if cop == "cartesian":
            params = numpyro.sample(
                "params",
                dist.Normal(
                    loc=jnp.zeros((objective.nPars)),  # No batch dimension here
                    scale=prior_scale
                )
            )
        else:
            # Single chain model definition (no batch dimension)
            magnitudes = numpyro.sample(
                "magnitudes",
                dist.HalfNormal(
                    scale=prior_scale * jnp.ones(len(waveNames))
                )
            )
            phases = numpyro.sample(
                "phases",
                dist.Uniform(
                    low=jnp.zeros(len(waveNames)), 
                    high=2*jnp.pi * jnp.ones(len(waveNames))
                )
            )
            
            # Create real and imaginary parts from polar coordinates
            real_parts = magnitudes * jnp.cos(phases)
            imag_parts = magnitudes * jnp.sin(phases)
            
            # Interleave into flat params array (no batch dimension)
            params = jnp.zeros(objective.nPars)
            for i in range(len(waveNames)):
                params = params.at[2*i].set(real_parts[i])
                params = params.at[2*i+1].set(imag_parts[i])

        # Handle both batched and non-batched cases (multi-chain or not)
        if params.ndim > 1:
            batched_objective_fn = vmap(objective_fn)
            nll = batched_objective_fn(params)
        else:
            nll = objective_fn(params)
        numpyro.factor("likelihood", -nll)
    
    nuts_kernel = NUTS(
        model,
        target_accept_prob=0.8,
        max_tree_depth=10,       # Start lower to avoid deep searches in bad regions
        step_size=0.1,           # Start with smaller steps
        adapt_step_size=True,    # Explicitly enable adaptation
        dense_mass=True          # Try to capture correlations
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
    ):
    """
    Run fit with specified method
    
    Args:
        bin_idx: Index of bin to optimize
        resume_path: Path to saved MCMC state to resume from
        save_path: Path to save MCMC state for future resuming
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
                             save_path=save_path)
    samples = mcmc.get_samples() # ~ (n_samples, nPars)
    
    final_result_dict = {}
    for isample in range(len(samples['params'])): # Store intensities
        if "total" not in final_result_dict:
            final_result_dict["total"] = [obj.intensity(samples['params'][isample])]
        else:
            final_result_dict["total"].append(obj.intensity(samples['params'][isample]))
        for wave in pwa_manager.waveNames:
            if wave not in final_result_dict:
                final_result_dict[f"{wave}"] = [obj.intensity(samples['params'][isample], suffix=[wave])]
            else:
                final_result_dict[f"{wave}"].append(obj.intensity(samples['params'][isample], suffix=[wave]))
    for iw, wave in enumerate(pwa_manager.waveNames): # Store complex amplitudes
        final_result_dict[f"{wave}_amp"] = samples['params'][:, 2*iw] + 1j * samples['params'][:, 2*iw+1]
        
    # Print some info on the resulting shape
    nsamples = n_chains * n_samples
    console.print(f"\n\nFinal result dict info (total=total intensity) ~ (nsamples={nsamples}):", style="bold")
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
        
        method_help = "\nOptimizer Descriptions:\n"
        method_help += "\nMinuit-based Methods:\n"
        method_help += "  * minuit-numeric:\n"
        method_help += "      Lets Minuit compute numerical gradients\n"
        method_help += "  * minuit-analytic:\n"
        method_help += "      Uses analytic gradients from PWA likelihood manager\n"
        
        method_help += "\nSciPy-based Methods:\n"
        method_help += "  * L-BFGS-B:\n"
        method_help += "      Limited-memory BFGS quasi-Newton method (stores approximate Hessian)\n"
        method_help += "      + Efficient for large-scale problems\n"
        method_help += "      - May struggle with highly correlated parameters\n"
        
        method_help += "  * trust-ncg:\n"
        method_help += "      Trust-region Newton-Conjugate Gradient\n"
        method_help += "      + Adaptively adjusts step size using local quadratic approximation\n"
        method_help += "      + Efficient for large-scale problems\n"
        method_help += "      - Can be unstable for ill-conditioned problems\n"
        
        method_help += "  * trust-krylov:\n"
        method_help += "      Trust-region method with Krylov subspace solver\n"
        method_help += "      + Better handling of indefinite (sparse) Hessians, Kyrlov subspcae accounts for non-Euclidean geometry\n"
        method_help += "      + More robust for highly correlated parameters\n"
        
        return help_message + "\n" + method_help

if __name__ == "__main__":
    parser = OptimizerHelpFormatter(description="Run optimization fits using various methods.")
    parser.add_argument("yaml_file", type=str,
                       help="Path to PyAmpTools YAML configuration file")    
    parser.add_argument("-b", "--bins", type=int, nargs="+",
                       help="List of bin indices to process")

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
        )
        final_result_dicts.append(final_result_dict)
    
    _save_dir = os.path.dirname(args.save)
    _save_fname = os.path.basename(args.save)
    _save_fname = os.path.splitext(_save_fname)[0] # drop extension
    with open(f"{_save_dir}/{_save_fname}_samples.pkl", "wb") as f:
        pkl.dump(final_result_dicts, f)

    console.print(f"Total time elapsed: {timer.read()[2]}", style="bold")

    sys.exit(0)
