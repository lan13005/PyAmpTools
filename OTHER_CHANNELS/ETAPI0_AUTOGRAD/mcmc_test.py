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
from jax import jit, vmap

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

class MCMCManager:
    
    def __init__(self, pyamptools_yaml, iftpwa_yaml, bin_idx, prior_scale=100.0, prior_dist='laplace', n_chains=20, n_samples=2000, n_warmup=1000, 
                 resume_path=None, save_path=None, cop="cartesian", init_method="L-BFGS-B", init_maxiter=100, init_gtol=1e-4, 
                 init_ftol=1e-6, wave_prior_scales=None, target_accept_prob=0.85, max_tree_depth=12, 
                 step_size=0.1, adapt_step_size=True, dense_mass=True, adapt_mass_matrix=True):
        # GENERAL PARAMETERS
        self.pyamptools_yaml = pyamptools_yaml
        self.iftpwa_yaml = iftpwa_yaml
        self.bin_idx = bin_idx
        self.resume_path = resume_path
        self.save_path = save_path
        self.cop = cop
        console.print(f"Using coordinate system: {cop}", style="bold green")
        
        # EXTRACTED PARAMETERS FROM YAML
        self.waveNames = self.pyamptools_yaml["waveset"].split("_")
        self.nmbMasses = self.pyamptools_yaml["n_mass_bins"]
        self.nmbTprimes = self.pyamptools_yaml["n_t_bins"]
        self.nPars = 2 * len(self.waveNames)
        
        # REFERENCE WAVES
        self.reference_waves = self.pyamptools_yaml["phase_reference"].split("_")
        self.ref_indices = None
        self.channel = None
        self.refl_sectors = None
        
        # INITIALIZATION PARAMETERS
        self.init_method = init_method
        self.init_maxiter = init_maxiter
        self.init_gtol = init_gtol
        self.init_ftol = init_ftol
        self.init_params = None
        
        # PRIOR PARAMETERS
        self.prior_scale = prior_scale
        self.wave_prior_scales = wave_prior_scales
        self.prior_dist = prior_dist
                
        # MCMC RELATED
        self.n_chains = n_chains
        self.n_samples = n_samples
        self.n_warmup = n_warmup
        self.mcmc = None # will be set by self.run_mcmc()
        
        # MCMC SAMPLER PARAMETERS
        self.target_accept_prob = target_accept_prob
        self.max_tree_depth = max_tree_depth
        self.step_size = step_size
        self.adapt_step_size = adapt_step_size
        self.dense_mass = dense_mass
        self.adapt_mass_matrix = adapt_mass_matrix

        # SETUP
        self._process_reference_waves()     # determines reference waves and their associated indicies
        self._setup_objective()             # sets up objective function to optimize
        self.model = self.create_model()    # create MCMC prior model
        ### Should be ready to run MCMC now

    def create_model(self):
        
        console.print(f"\n\n***************************************************", style="bold")
        if self.prior_dist not in ['gaussian', 'laplace', 'horseshoe']:
            raise ValueError(f"Invalid prior distribution: {self.prior_dist}")
        console.print(f"Using '{self.prior_dist}' prior distribution", style="bold green")
        
        ################################################
        # Process wave-specific prior scales if provided
        # NOTE: works in cartesian coordinates only ATM
        #################################################
        console.print(f"Default prior scale: {self.prior_scale}", style="bold")
        if self.wave_prior_scales is not None and self.cop == "cartesian":
            console.print(f"Using wave-specific prior scales:", style="bold")
            param_prior_scales = jnp.ones(self.nPars) * self.prior_scale
            for wave_name, scale in self.wave_prior_scales.items():
                if wave_name in self.waveNames:
                    wave_idx = self.waveNames.index(wave_name)
                    param_prior_scales = param_prior_scales.at[2*wave_idx  ].set(scale)   # Real part
                    param_prior_scales = param_prior_scales.at[2*wave_idx+1].set(scale)   # Imaginary part
                    console.print(f"  {wave_name}: {scale}", style="bold")
                else:
                    console.print(f"  Warning: Wave '{wave_name}' not found in wave list, ignoring custom prior scale", style="yellow")
        else:
            # Use uniform prior scale for all parameters
            param_prior_scales = jnp.ones(self.nPars) * self.prior_scale
            
        self.param_prior_scales = param_prior_scales
        console.print(f"***************************************************\n\n", style="bold")
        
        def model():
            
            """Do not include batch dimension in the sampling"""
            
            if self.cop == "cartesian":

                # Identify free parameters - exclude imaginary parts of reference waves
                free_indices = jnp.array([i for i in range(self.nPars) if not any(i == 2*ref_idx+1 for ref_idx in self.ref_indices)])
                free_param_prior_scales = self.param_prior_scales[free_indices]

                ##### 
                # # Gaussian prior (L2 regularization)
                if self.prior_dist == "gaussian":
                    free_params = numpyro.sample(
                        "params",
                        dist.Normal(loc=jnp.zeros((self.nPars - len(self.ref_indices))), scale=free_param_prior_scales)
                    )

                # # Laplace prior (L1 regularization)
                if self.prior_dist == "laplace":
                    free_params = numpyro.sample(
                        "params",
                        dist.Laplace(loc=jnp.zeros((self.nPars - len(self.ref_indices))), scale=free_param_prior_scales)
                    )
                
                # Non-centered Horseshoe prior (stronger sparsity than L1) with separate global and local shrinkage
                # Global shrinkage - controls overall sparsity
                if self.prior_dist == "horseshoe":
                    global_scale = numpyro.sample(
                        "global_scale",
                        dist.HalfCauchy(scale=self.prior_scale * 0.5)  # Use 10% of prior_scale as global shrinkage base
                    )
                    # Local shrinkage - allows certain parameters to escape global shrinkage
                    local_scale = numpyro.sample(
                        "local_scale",
                        dist.HalfCauchy(scale=jnp.ones(self.nPars - len(self.ref_indices)))
                    )
                    raw_params = numpyro.sample(
                        "raw_params",
                        dist.Normal(jnp.zeros(self.nPars - len(self.ref_indices)), jnp.ones(self.nPars - len(self.ref_indices)))
                    )
                    free_params = raw_params * local_scale * global_scale
                
                # Set parameters as before
                params = jnp.zeros(self.nPars)
                params = params.at[free_indices].set(free_params)

            elif self.cop == "polar":

                if use_log_magnitudes:
                    magnitudes = numpyro.sample(
                        "magnitudes",
                        dist.Normal(loc=jnp.zeros(len(self.waveNames)), scale=jnp.log(self.prior_scale))
                    )
                else:
                    magnitudes = numpyro.sample(
                        "magnitudes",
                        dist.HalfNormal(scale=self.prior_scale * jnp.ones(len(self.waveNames)))
                    )
                magnitudes = jnp.exp(magnitudes) if use_log_magnitudes else magnitudes
                
                # Identify free phase parameters - exclude reference waves
                free_phase_indices = jnp.array([i for i in range(len(self.waveNames)) if i not in self.ref_indices])
                
                if use_phase_param == 'tan':
                    # Sample unbounded parameters for non-reference waves
                    phase_params_free = numpyro.sample(
                        "phase_params",
                        dist.Normal(loc=jnp.zeros(len(self.waveNames) - len(self.ref_indices)), 
                                    scale=jnp.ones(len(self.waveNames) - len(self.ref_indices)))
                    )
                    # Convert back to phases
                    free_phases = 2 * jnp.arctan(phase_params_free)
                elif use_phase_param == 'vonmises':
                    # concentration=0 makes it a uniform distribution over [-π, π]
                    free_phases = numpyro.sample(
                        'phase_params',
                        dist.VonMises(loc=jnp.zeros(len(self.waveNames) - len(self.ref_indices)), 
                                        concentration=jnp.zeros(len(self.waveNames) - len(self.ref_indices)))
                    )
                
                phases = jnp.zeros(len(self.waveNames))        
                phases = phases.at[free_phase_indices].set(free_phases)
                
                real_parts = magnitudes * jnp.cos(phases)
                imag_parts = magnitudes * jnp.sin(phases)
                
                # Move into cartesian coordinates for objective function
                params = jnp.zeros(self.nPars)
                for i in range(len(self.waveNames)):
                    params = params.at[2*i].set(real_parts[i])
                    params = params.at[2*i+1].set(imag_parts[i])

            else:
                raise ValueError(f"Invalid coordinate system: {self.cop}")


            ##### Return objective value
            # Handle both batched and non-batched cases (multi-chain or not)
            if params.ndim > 1:
                batched_objective_fn = vmap(self.objective_fn)
                nll = batched_objective_fn(params)
            else:
                nll = self.objective_fn(params)
            
            regularization = 0.0
            # if cop == "polar": # Add regularization to prevent very large intensity (due to numerical instability)
            #     console.print("Polar coordinate warning: adding small regularization to stabilize intensity", style="bold yellow")
            #     free_magnitudes = jnp.array([i for i in range(objective.nPars) if i % 2 == 0])
            #     regularization = 0.001 * jnp.sum(params[free_magnitudes]**2)
            nll = nll + regularization
            
            numpyro.factor("likelihood", -nll)
            
        return model

    def run_mcmc(self):
        
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        rng_key = jax.random.PRNGKey(0)

        ###########################################
        ### CONFIGURE MCMC SAMPLER
        ###########################################
        
        nuts_kernel = NUTS(
            self.model,
            target_accept_prob=self.target_accept_prob,     # Increase from 0.9 to encourage even smaller steps
            max_tree_depth=self.max_tree_depth,             # Allow deeper search but with more careful step size
            step_size=self.step_size,
            adapt_step_size=self.adapt_step_size,          
            dense_mass=self.dense_mass,                     # Keep this for handling correlations
            adapt_mass_matrix=self.adapt_mass_matrix        # Explicitly enable mass matrix adaptation
        )
        
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=self.n_warmup if self.resume_state is None else 0,  # Skip warmup if resuming
            num_samples=self.n_samples,
            num_chains=self.n_chains,
            chain_method='parallel',
            progress_bar=True
        )
        
        ###########################################
        ### RUN MCMC
        ###########################################
        rng_key, rng_key_mcmc = jax.random.split(rng_key)
        
        # Load saved state if requested
        if self.resume_state is not None:
            console.print(f"Resuming from saved state", style="bold green")
            try:
                # Set post_warmup_state to the saved state - this will skip warmup
                mcmc.post_warmup_state = self.resume_state
                mcmc.run(self.resume_state.rng_key)
            except Exception as e:
                console.print(f"Error loading MCMC state: {e}", style="bold red")
                console.print("Falling back to fresh start", style="yellow")
                mcmc.run(rng_key_mcmc) # , init_params=init_params)
        else:
            mcmc.run(rng_key_mcmc) # , init_params=init_params)
        
        ############################
        ### SAVE STATE IF REQUESTED
        ############################
        if self.save_path is not None:
            console.print(f"Saving MCMC state to: {self.save_path}", style="bold green")
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            try:
                with open(self.save_path, 'wb') as f:
                    pkl.dump(mcmc.last_state, f)
            except Exception as e:
                console.print(f"Error saving MCMC state: {e}", style="bold red")

        self.mcmc = mcmc
        final_result_dict = self._postprocess_samples()
        return final_result_dict
    
    def set_bin(self, bin_idx):
        if self.pwa_manager is None:
            raise ValueError("PWA manager not initialized. Call _setup_objective() first.")
        self.pwa_manager.set_bins(np.array([bin_idx]))
        self.obj = Objective(self.pwa_manager, bin_idx, self.nPars, self.nmbMasses, self.nmbTprimes)
        self.objective_fn = jit(self.obj.objective)
        self.gradient_fn  = jit(self.obj.gradient)

    def _postprocess_samples(self):
        
        if self.mcmc is None:
            raise ValueError("MCMC not run. Call run_mcmc() first.")
        
        ####################
        ### MCMC DIAGNOSTICS
        ####################
        
        # Leave this call here. mcmc.run() call does not wait for results to finish so the following print statements
        #   will be flushed before fit is complete. Use this as a barrier
        divergence = self.mcmc.get_extra_fields()["diverging"]
        
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
        for wave_idx, wave in enumerate(self.waveNames):
            console.print(f"params[{par_idx}] corresponds to Re[{wave}]", style="bold")
            par_idx += 1
            if wave_idx not in self.ref_indices:
                console.print(f"params[{par_idx}] corresponds to Im[{wave}]", style="bold")
                par_idx += 1
        
        # Print standard NumPyro summary
        self.mcmc.print_summary()
        
        # Divergence information
        n_divergent = jnp.sum(divergence)
        divergence_pct = n_divergent/self.n_samples*100
        console.print(f"\nNumber of divergent transitions: {n_divergent} ({divergence_pct:.1f}%)")
        if divergence_pct > 0.5:
            console.print("WARNING: Divergences detected (>0.5%)! This indicates the sampler is struggling with difficult geometry.", style="red")
            console.print("    Consider: increasing adapt_step_size, increasing target_accept_prob, or reparameterizing your model.", style="red")
        else:
            console.print("GOOD: Few or no divergences.", style="green")
    
        samples = self.mcmc.get_samples() # ~ (n_samples, nPars)
        
        ###########################################################################
        ### POSTPROCESS SAMPLES, STORE INTENSITIES AND COMPLEX AMPLITUDES IN DICT
        ###########################################################################
        final_result_dict = {}

        if self.cop == "cartesian":
            
            free_indices = jnp.array([i for i in range(self.nPars) if not any(i == 2*ref_idx+1 for ref_idx in self.ref_indices)])
            n_samples = len(next(iter(samples.values())))
            params = jnp.zeros((n_samples, self.nPars))
        
            if self.prior_dist != "horseshoe": # {Gaussian, Laplace}
                params = params.at[:, free_indices].set(samples['params'])
            else:
                raw_params   = samples['raw_params'] # ~ (n_samples, n_free_params)
                global_scale = samples['global_scale']
                local_scale  = samples['local_scale']            
                actual_params = raw_params * local_scale * global_scale[:, None] # broadcast global_scale across samples
                params = params.at[:, free_indices].set(actual_params)

        elif self.cop == "polar":
            n_samples = len(samples['magnitudes'])
            magnitudes = samples['magnitudes']  # Shape: (n_samples, n_waves)
            magnitudes = jnp.exp(magnitudes) if use_log_magnitudes else magnitudes
            
            # Create full phases array with ref_idx phase set to 0
            phase_params = samples['phase_params']
            non_ref_indices = [i for i in range(len(self.waveNames)) if i not in self.ref_indices]
            phases = jnp.zeros((n_samples, len(self.waveNames)))
            
            if use_phase_param == 'tan':
                # Convert phase_params back to phases for tangent half-angle
                phases = phases.at[:, non_ref_indices].set(2 * jnp.arctan(phase_params))
            elif use_phase_param == 'vonmises':
                # For von Mises, the phases are directly sampled
                phases = phases.at[:, non_ref_indices].set(phase_params)
            
            # Convert to Cartesian coordinates
            params = jnp.zeros((n_samples, self.nPars))
            for i in range(len(self.waveNames)):
                params = params.at[:, 2*i].set(magnitudes[:, i] * jnp.cos(phases[:, i]))     # Real part
                params = params.at[:, 2*i+1].set(magnitudes[:, i] * jnp.sin(phases[:, i]))   # Imaginary part
        
        for isample in range(n_samples):
            # Check if parameters contain NaN values
            if jnp.any(jnp.isnan(params[isample])):
                console.print(f"Warning: NaN detected in parameters for sample {isample}", style="bold red")
                total_intensity = jnp.nan
            else:
                total_intensity = self.obj.intensity(params[isample])
                # Check if intensity calculation resulted in NaN
                if jnp.isnan(total_intensity):
                    console.print(f"Warning: NaN intensity calculated for sample {isample}", style="bold red")
            
            if "total" not in final_result_dict:
                final_result_dict["total"] = [total_intensity]
            else:
                final_result_dict["total"].append(total_intensity)
                
            for wave in self.pwa_manager.waveNames:
                if jnp.any(jnp.isnan(params[isample])):
                    wave_intensity = jnp.nan
                else:
                    wave_intensity = self.obj.intensity(params[isample], suffix=[wave])
                    if jnp.isnan(wave_intensity):
                        console.print(f"Warning: NaN intensity calculated for wave {wave}, sample {isample}", style="bold red")
                        
                if wave not in final_result_dict:
                    final_result_dict[f"{wave}"] = [wave_intensity]
                else:
                    final_result_dict[f"{wave}"].append(wave_intensity)
                    
        for iw, wave in enumerate(self.pwa_manager.waveNames): # Store complex amplitudes
            final_result_dict[f"{wave}_amp"] = params[:, 2*iw] + 1j * params[:, 2*iw+1]
            
        # Print some info on the resulting shape
        console.print(f"\n\nFinal result dict info (total=total intensity) ~ (nsamples={n_samples}):", style="bold")
        for k, v in final_result_dict.items():
            final_result_dict[k] = np.array(v)
            console.print(f"{k}: shape {final_result_dict[k].shape}", style="italic")
        console.print("**************************************************************\n", style="bold")

        return final_result_dict
    
    def _setup_objective(self):
        from iftpwa1.pwa.gluex.gluex_jax_manager import GluexJaxManager

        self.pwa_manager = GluexJaxManager(comm0=comm0, mpi_offset=mpi_offset,
                                    yaml_file=pyamptools_yaml,
                                    resolved_secondary=iftpwa_yaml, prior_simulation=False, sum_returned_nlls=False)
        self.pwa_manager.prepare_nll()
        self.set_bin(self.bin_idx)

        # If resume_path is specified, load the saved state
        self.resume_state = None
        if self.resume_path is not None:
            try:
                with open(self.resume_path, 'rb') as f:
                    self.resume_state = pkl.load(f)
                    console.print(f"Loaded saved state from {self.resume_path}", style="bold green")
            except Exception as e:
                console.print(f"Error loading state from {self.resume_path}: {e}", style="bold red")
        console.print("\n**************************************************************", style="bold")

    def _process_reference_waves(self):
        ref_indices = []
        channel = None
        refl_sectors = {}
        if self.reference_waves:
            if isinstance(self.reference_waves, str):
                self.reference_waves = [self.reference_waves]
            
            # Get reaction channel for all waves 
            channel = identify_channel(self.waveNames)
            console.print(f"Identified reaction channel: {channel}", style="bold")
            
            # Get reflectivity sectors and their waves
            for i, wave in enumerate(self.waveNames):
                # Extract reflectivity from wave using converter dictionary
                refl = converter[wave][0]  # First element is reflectivity (e)
                if refl not in refl_sectors:
                    refl_sectors[refl] = []
                refl_sectors[refl].append((i, wave))
            
            # Process each reference wave
            for ref_wave in self.reference_waves:
                if ref_wave in self.waveNames:
                    ref_idx = self.waveNames.index(ref_wave)
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
        self.ref_indices = ref_indices
        self.channel = channel
        self.refl_sectors = refl_sectors
        
    def _get_initial_params(self):
        pass
        # ###################
        # ### INITIALIZATION
        # ###################
        # # Draw random sample and run optimization to move it to better starting location
        # def init_chain(key, i, method=init_method, maxiter=init_maxiter, gtol=init_gtol, ftol=init_ftol):
        #     console.print(f"Initializing chain {i} (randomly + {method} optimization) ...", style="bold")
        #     # Use parameter-specific prior scales for initialization
        #     init_param = jax.random.uniform(key, shape=(objective.nPars,), minval=-100.0, maxval=100.0) # NOTE: Find better initialization of min/max range
            
        #     # Fix reference by rotating to be real
        #     for ref_idx in ref_indices:
        #         ref_mag = jnp.abs(init_param[2*ref_idx] + 1j * init_param[2*ref_idx+1])
        #         init_param = init_param.at[2*ref_idx  ].set(ref_mag)
        #         init_param = init_param.at[2*ref_idx+1].set(0.0)
            
        #     start_nll = objective_fn(init_param)
            
        #     # Define the objective function and its gradient for scipy.optimize
        #     def scipy_obj(x):
        #         return objective_fn(x).item()  # Convert to native Python type
                
        #     def scipy_grad(x):
        #         return np.array(gradient_fn(x))  # Convert to numpy array
            
        #     # Handle reference wave constraints using parameter bounds
        #     bounds = [(None, None)] * objective.nPars
        #     if ref_indices:
        #         for ref_idx in ref_indices:
        #             bounds[2*ref_idx+1] = (0.0, 0.0) # Fix imaginary part of reference wave to 0
            
        #     # Run optimization
        #     result = minimize(
        #         scipy_obj,
        #         np.array(init_param),  # Convert to numpy array for scipy
        #         method=method,
        #         jac=scipy_grad,
        #         bounds=bounds,
        #         options={
        #             'maxiter': maxiter,
        #             'gtol': gtol,
        #             'ftol': ftol,
        #             'disp': False
        #         }
        #     )
            
        #     params = jnp.array(result.x)  # Convert back to JAX array
        #     end_nll = objective_fn(params)
            
        #     # Check if optimization improved the likelihood
        #     if end_nll > start_nll:
        #         console.print(f"  Warning: Initialization resulted in worse NLL for chain {i}!", style="bold red")
        #         console.print(f"  Start: {start_nll:.4f}, End: {end_nll:.4f}", style="red")
        #         # Fall back to original point if optimization made things worse
        #         params = init_param
        #     else:
        #         console.print(f"  Chain {i}: NLL improved from {start_nll:.4e} to {end_nll:.4e} [Delta={end_nll-start_nll:.1f}, Iterations={result.nit}]") 
                
        #     return params
        
        # keys = jax.random.split(rng_key, n_chains + 1)
        # rng_key = keys[0]
        # chain_keys = keys[1:]
        # initial_params = [] # Shape: (n_chains, nPars)
        # for i in range(n_chains):
        #     initial_params.append(init_chain(chain_keys[i], i))
        # initial_params = jnp.array(initial_params)
            
        # if cop == "cartesian":           
        #     # Create mask to exclude imaginary parts of reference waves
        #     mask = jnp.ones(objective.nPars, dtype=bool)
        #     for ref_idx in ref_indices:
        #         mask = mask.at[2*ref_idx+1].set(False)
                
        #     # # Basic initialization for 'params'
        #     # init_params = {'params': initial_params[:, mask]}  # Shape: (n_chains, nPars - len(ref_indices))
            
        #     # Non-centered parameterization of Horseshoe prior (better when posterior is dominated by prior term / likelihood not constraining)
        #     # The scale (of Cauchy) determines the location within which 50% of the distribution is contained
        #     # - Start with 10% of prior scale for global shrinkage (for each chain)
        #     # - Start with 50% of prior scale for local shrinkage  (for each parameter in the chain)
        #     global_scale_init = jnp.ones((n_chains, 1                                 )) * prior_scale * 0.01
        #     local_scale_init  = jnp.ones((n_chains, objective.nPars - len(ref_indices))) * prior_scale * 0.05
            
        #     param_magnitudes = jnp.abs(initial_params[:, mask])
            
        #     # Set raw_params to standard normal values, scaled to match optimized parameters
        #     # This ensures we start with parameters that reproduce approximately the same values 
        #     # as the optimized parameters, but with a proper hierarchical structure
        #     raw_params_init = jnp.clip(
        #         jnp.sign(initial_params[:, mask]) * 
        #         param_magnitudes / (global_scale_init * local_scale_init), 
        #         -3.0, 3.0
        #     )

        #     init_params = {
        #         'raw_params': raw_params_init,
        #         'global_scale': global_scale_init,
        #         'local_scale': local_scale_init,
        #         'params': raw_params_init * global_scale_init * local_scale_init,
        #     }
            
        # elif cop == "polar":
        #     # NOTE: The initial parameters should by now have strictly applied the reference wave constraints
        #     _initial_params = jnp.zeros((n_chains, objective.nPars))
        #     _camp = initial_params[:, ::2] + 1j * initial_params[:, 1::2] # (c)omplex (amp)litude
            
        #     # Convert magnitudes to log-space for initialization
        #     _magnitudes = jnp.maximum(jnp.abs(_camp), 1e-5) # Ensure positive values (for log)
        #     if use_log_magnitudes:
        #         _magnitudes = jnp.log(_magnitudes)
            
        #     # Get phases from complex amplitudes
        #     _phases = jnp.angle(_camp)
            
        #     if use_phase_param == 'tan':
        #         # Convert phases to tangent half-angle parameter: u = tan(phase/2)
        #         # Add small epsilon to avoid exact π which would give infinity
        #         _phases_safe = jnp.where(jnp.abs(_phases) > jnp.pi - 1e-5, 
        #                                 _phases * (1 - 1e-5), 
        #                                 _phases)
        #         _phase_params = jnp.tan(_phases_safe / 2)
        #     elif use_phase_param == 'vonmises':
        #         # For von Mises, we just use the angles directly for initialization
        #         _phase_params = _phases
        #     else:
        #         raise ValueError(f"Invalid phase parameterization: {use_phase_param}")
            
        #     # Set magnitudes and store original phases for display
        #     _initial_params = _initial_params.at[:,  ::2].set(_magnitudes)
        #     _initial_params = _initial_params.at[:, 1::2].set(_phases)  # Still store phases for display

        #     # Create mask to exclude phases of reference waves
        #     mask = jnp.ones(len(waveNames), dtype=bool)
        #     for ref_idx in ref_indices:
        #         mask = mask.at[ref_idx].set(False)
        #     init_params = {
        #         'magnitudes': _initial_params[:, ::2],  # Shape: (n_chains, len(waveNames))
        #         'phase_params': _phase_params[:, mask]  # Shape: (n_chains, len(waveNames) - len(ref_indices))
        #     }
        # else:
        #     raise ValueError(f"Invalid coordinate system: {cop}")

        # console.print("\n\n=== INITIAL CHAIN PARAMETER VALUES ===", style="bold")
        # for k, v in init_params.items():
        #     console.print(f"{k}: {v.shape} ~ (nChains, params)", style="bold")
        #     console.print(f"{v}\n", style="bold")

        # ###################
        # ### DISPLAY INITIAL INTENSITIES
        # ###################
        # console.print("\n=== INITIAL CHAIN INTENSITIES ===", style="bold")
        # from rich.table import Table
        
        # # Calculate intensities for each wave
        # table = Table(title="Initial Chain Intensities")
        # table.add_column("Chain", justify="right", style="cyan")
        # table.add_column("Total", justify="right", style="green")
        # table.add_column("NLL", justify="right", style="red")
        
        # # Add columns for each wave
        # for wave in waveNames:
        #     table.add_column(wave, justify="right")
        
        # # Calculate and add intensities for each chain
        # for i in range(n_chains):
        #     params = initial_params[i]
        #     total_intensity = objective.intensity(params)
        #     nll = objective_fn(params)
            
        #     # Get individual wave intensities
        #     wave_intensities = []
        #     for wave in waveNames:
        #         wave_intensity = objective.intensity(params, suffix=[wave])
        #         wave_intensities.append(f"{wave_intensity:.1f}")
            
        #     # Add row to table
        #     table.add_row(
        #         f"{i}", 
        #         f"{total_intensity:.1f}", 
        #         f"{nll:.1f}",
        #         *wave_intensities
        #     )
        
        # console.print(table)
        # console.print("")

class OptimizerHelpFormatter(argparse.ArgumentParser):
    def error(self, message):
        console.print(f"Error: {message}", style="red")
        self.print_help()
        sys.exit(2)

    def format_help(self):
        help_message = super().format_help()
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
    parser.add_argument("-pd", "--prior_dist", type=str, choices=['laplace', 'normal', 'horseshoe'], default='laplace', 
                       help="Prior distribution for the complex amplitudes")
    parser.add_argument("-nc", "--nchains", type=int, default=20,
                       help="Number of chains to use for numpyro MCMC (each chain runs on paralell process)")
    parser.add_argument("-ns", "--nsamples", type=int, default=2000,
                       help="Number of samples to draw per chain")
    parser.add_argument("-nw", "--nwarmup", type=int, default=1000,
                       help="Number of warmup samples to draw")
    
    #### MCMC SAMPLER ARGS ####
    parser.add_argument("-ta", "--target_accept", type=float, default=0.80,
                       help="Target acceptance probability for NUTS sampler (default: 0.80)")
    parser.add_argument("-mtd", "--max_tree_depth", type=int, default=12,
                       help="Maximum tree depth for NUTS sampler (default: 12)")
    parser.add_argument("-ss", "--step_size", type=float, default=0.1,
                       help="Initial step size for NUTS sampler (default: 0.05 for polar, 0.1 for cartesian)")
    parser.add_argument("--no_adapt_step_size", action="store_false", dest="adapt_step_size",
                       help="Disable step size adaptation")
    parser.add_argument("--no_dense_mass", action="store_false", dest="dense_mass",
                       help="Disable dense mass matrix adaptation (use diagonal instead). Speeds up sampling")
    parser.add_argument("--no_adapt_mass", action="store_false", dest="adapt_mass_matrix",
                       help="Disable mass matrix adaptation. Significantly speeds up sampling")
    parser.set_defaults(adapt_step_size=True, dense_mass=True, adapt_mass_matrix=True)
    
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
    
    wave_prior_scales = None
    # wave_prior_scales = {
    #     "Sp0+": 150,
    #     "Dp2+": 150,
    # }
    
    # Initialize MCMC Manager, preparing for the first requested bin
    mcmc_manager = MCMCManager(
        pyamptools_yaml, iftpwa_yaml, args.bins[0], 
        prior_scale=args.prior_scale, prior_dist=args.prior_dist,
        n_chains=args.nchains, n_samples=args.nsamples, n_warmup=args.nwarmup, 
        resume_path=args.resume, save_path=args.save, cop=args.coordinate_system, 
        init_method=args.init_method, init_maxiter=args.init_maxiter, init_gtol=args.init_gtol, init_ftol=args.init_ftol, 
        wave_prior_scales=wave_prior_scales,
        target_accept_prob=args.target_accept,
        max_tree_depth=args.max_tree_depth,
        step_size=args.step_size,
        adapt_step_size=args.adapt_step_size,
        dense_mass=args.dense_mass,
        adapt_mass_matrix=args.adapt_mass_matrix
    )
    
    console.print(f"\n\n***************************************************", style="bold")
    console.print(f"Using {mcmc_manager.n_chains} chains with {mcmc_manager.n_samples} samples per chain with {mcmc_manager.n_warmup} warmup samples", style="bold")
    console.print(f"NUTS: Max tree depth: {mcmc_manager.max_tree_depth}", style="bold")
    console.print(f"NUTS: Step size: {mcmc_manager.step_size}", style="bold")
    console.print(f"NUTS: Adapt step size? {mcmc_manager.adapt_step_size}", style="bold")
    console.print(f"NUTS: Dense mass? {mcmc_manager.dense_mass}", style="bold")
    console.print(f"NUTS: Adapt mass matrix? {mcmc_manager.adapt_mass_matrix}", style="bold")
    console.print(f"***************************************************\n\n", style="bold")
    if args.print_wave_names:
        console.print(f"Wave names: {mcmc_manager.waveNames}", style="bold")
        sys.exit(0)

    if args.bins is None:
        raise ValueError("list of bin indices is required")
    
    timer = Timer()
    final_result_dicts = [mcmc_manager.run_mcmc()]
    if len(args.bins) > 1:
        for bin_idx in args.bins[1:]:
            mcmc_manager.set_bin(bin_idx)
            final_result_dict = mcmc_manager.run_mcmc()
            final_result_dicts.append(final_result_dict)
    
    _save_dir = os.path.dirname(args.save) + ("" if os.path.dirname(args.save) == "" else "/")
    _save_fname = os.path.basename(args.save)
    _save_fname = os.path.splitext(_save_fname)[0] # drop extension
    with open(f"{_save_dir}{_save_fname}_samples.pkl", "wb") as f:
        pkl.dump(final_result_dicts, f)

    console.print(f"Total time elapsed: {timer.read()[2]}", style="bold")

    sys.exit(0)
