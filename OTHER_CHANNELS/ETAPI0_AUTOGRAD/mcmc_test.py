import os
import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
from jax import jit, vmap
from pyamptools.utility.general import load_yaml, Timer
import numpy as np
import sys
import pickle as pkl
import argparse
from numpyro.infer import NUTS, MCMC
from rich.console import Console
import os
from pyamptools.utility.general import identify_channel, converter
from pyamptools.utility.opt_utils import Objective
from multiprocessing import Pool

# Configure JAX for multi-threading on CPU
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=6"
# jax.config.update('jax_enable_x64', True)
# numpyro.set_platform("cpu")  # Ensure CPU usage
# numpyro.set_host_device_count(6)  # Use 6 CPU cores
# print(f"JAX is using {jax.local_device_count()} local devices")
# print(f"Configured JAX for 6 CPU devices")

# Initial attempt at using polar coordinates (performance is terrible in general, leave here for future)
#   Log magnitudes produces a bias in the total intensity
#   - I guess this is log-normal magnitudes on each amplitude is biased upward and therefore the total intensity is biased upward
#   - Linear spaced magnitudes are numerically unstable
#   Polar coordinates parametrizations
#   - tried tangent half angle form and von mises distribution
use_log_magnitudes = False
use_phase_param = 'tan' # 'tan' = tan half angle form AND 'vonmises' = Von Mises with circular reparametrization

console = Console()

class MCMCManager:
    
    def __init__(self, pyamptools_yaml, iftpwa_yaml, bin_idx, prior_scale=100.0, prior_dist='laplace', n_chains=20, n_samples=2000, n_warmup=1000, 
                 resume_path=None, output_folder=None, cop="cartesian", wave_prior_scales=None, target_accept_prob=0.85, max_tree_depth=12, 
                 step_size=0.1, adapt_step_size=True, dense_mass=True, adapt_mass_matrix=True, enforce_positive_reference=False):
        # GENERAL PARAMETERS
        self.pyamptools_yaml = pyamptools_yaml
        self.iftpwa_yaml = iftpwa_yaml
        self.bin_idx = bin_idx
        self.resume_path = resume_path
        self.output_folder = output_folder
        self.cop = cop
        self.enforce_positive_reference = enforce_positive_reference
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
        console.print(f"Prior distribution of Re\[reference_waves]: {'strictly positive' if self.enforce_positive_reference else 'allow negative'}", style="bold green")
        
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
                # Use pre-calculated indices instead of recreating them
                free_complex_indices = self.free_complex_indices
                free_real_indices = self.free_real_indices
                
                # Get prior scales
                real_prior_scales = self.param_prior_scales[free_real_indices]
                complex_prior_scales = self.param_prior_scales[free_complex_indices]

                # Gaussian prior (L2 regularization)
                if self.prior_dist == "gaussian":
                    # Allow reference waves to be strictly positive or allow negative values
                    if not self.enforce_positive_reference:
                        free_params_real = numpyro.sample(
                            "real_params",
                            dist.Normal(loc=jnp.zeros_like(real_prior_scales), scale=real_prior_scales)
                        )
                    else:
                        free_params_real = numpyro.sample(
                            "real_params",
                            dist.HalfNormal(scale=real_prior_scales)
                        )
                    # Non-reference waves allows all reals for both parts
                    free_params_complex = numpyro.sample(
                        "complex_params",
                        dist.Normal(loc=jnp.zeros_like(complex_prior_scales), scale=complex_prior_scales)
                    )

                # Laplace prior (L1 regularization)
                elif self.prior_dist == "laplace":
                    if not self.enforce_positive_reference:
                        free_params_real = numpyro.sample(
                            "real_params",
                            dist.Laplace(loc=jnp.zeros_like(real_prior_scales), scale=real_prior_scales)
                        )
                    else: # Exponential for real parts (equivalent to positive half of Laplace)
                        free_params_real = numpyro.sample(
                            "real_params",
                            dist.Exponential(rate=1.0/real_prior_scales)
                        )
                    free_params_complex = numpyro.sample(
                        "complex_params",
                        dist.Laplace(loc=jnp.zeros_like(complex_prior_scales), scale=complex_prior_scales)
                    )
                
                # NOTE: Horseshoe prior takes forever to sample, minimal testing done for this scenario
                # Non-centered Horseshoe prior (stronger sparsity than L1) with separate global and local shrinkage
                # Global shrinkage - controls overall sparsity
                elif self.prior_dist == "horseshoe":
                    # Global shrinkage - controls overall sparsity
                    global_scale = numpyro.sample(
                        "global_scale",
                        dist.HalfCauchy(scale=self.prior_scale * 0.5)  # Use 10% of prior_scale as global shrinkage base
                    )
                    # Local shrinkage - allows certain parameters to escape global shrinkage
                    #   Handle real parameters (reference waves)
                    local_scale_real = numpyro.sample(
                        "local_scale_real",
                        dist.HalfCauchy(scale=jnp.ones(len(real_prior_scales)))
                    )
                    if not self.enforce_positive_reference:
                        raw_params_real = numpyro.sample(
                            "raw_params_real",
                            dist.Normal(jnp.zeros(len(real_prior_scales)), jnp.ones(len(real_prior_scales)))
                        )
                    else:
                        raw_params_real = numpyro.sample(
                            "raw_params_real",
                            dist.HalfNormal(jnp.ones(len(real_prior_scales)))
                        )
                    #    Handle (flattened) complex parameters for non-reference waves
                    local_scale = numpyro.sample(
                        "local_scale",
                        dist.HalfCauchy(scale=jnp.ones(len(complex_prior_scales)))
                    )
                    raw_params = numpyro.sample(
                        "raw_params",
                        dist.Normal(jnp.zeros(len(complex_prior_scales)), jnp.ones(len(complex_prior_scales)))
                    )

                    # Assemble the complete parameter vector
                    free_params_real = raw_params_real * local_scale_real * global_scale
                    free_params_complex = raw_params * local_scale * global_scale
                    
                # Set the final parameters
                params = jnp.zeros(self.nPars)
                params = params.at[free_real_indices].set(free_params_real)
                params = params.at[free_complex_indices].set(free_params_complex)

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
            
            ##### Can manually add regularization BUT this will not be properly Bayesian!
            # - Free parameters must have a distribution!
            regularization = 0.0
            nll = nll + regularization
            
            numpyro.factor("likelihood", -nll)
            
        return model

    def run_mcmc(self):
        
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        rng_key = jax.random.PRNGKey(args.seed)

        ###########################################
        ### CONFIGURE MCMC SAMPLER
        ###########################################
        
        nuts_kernel = NUTS(
            self.model,
            target_accept_prob=self.target_accept_prob,     # Increase from 0.9 to encourage even smaller steps
            max_tree_depth=self.max_tree_depth,             # Allow deeper tree search, exponential in depth
            step_size=self.step_size,
            adapt_step_size=self.adapt_step_size,          
            dense_mass=self.dense_mass,                     # Keep this for handling correlations (if True, slows down sampling but more accurate)
            adapt_mass_matrix=self.adapt_mass_matrix        # Explicitly enable mass matrix adaptation (if True, slows down sampling but more accurate)
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
                mcmc.post_warmup_state = self.resume_state
                mcmc.run(self.resume_state.rng_key)
            except Exception as e:
                console.print(f"Error loading MCMC state: {e}", style="bold red")
                console.print("Falling back to fresh start", style="yellow")
                mcmc.run(rng_key_mcmc)
        else:
            mcmc.run(rng_key_mcmc)
        
        ############################
        ### SAVE STATE IF REQUESTED
        ############################
        if self.output_folder is not None:
            console.print(f"Saving MCMC state to: {self.output_folder}", style="bold green")
            os.makedirs(os.path.dirname(self.output_folder), exist_ok=True)
            try:
                with open(self.output_folder, 'wb') as f:
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
        console.print("NOTE: The following statistics are desired but the actual amplitude results can be useful already!", style="bold yellow")
        
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

        console.print("\nParameter mapping for NumPyro summary:", style="bold")
        
        if self.cop == "cartesian":
            if self.prior_dist != "horseshoe": # {Gaussian, Laplace}
                # For complex_params (non-reference waves)
                free_complex_indices = []
                complex_idx_map = {}  # Map from complex_params idx to wave and component
                idx = 0
                for wave_idx, wave in enumerate(self.waveNames):
                    if wave_idx not in self.ref_indices:
                        complex_idx_map[idx] = (wave, "Re")
                        free_complex_indices.append(2*wave_idx)
                        idx += 1
                        complex_idx_map[idx] = (wave, "Im")
                        free_complex_indices.append(2*wave_idx+1)
                        idx += 1
                
                # For real_params (reference waves real part)
                free_real_indices = []
                real_idx_map = {}  # Map from real_params idx to wave
                idx = 0
                for wave_idx, wave in enumerate(self.waveNames):
                    if wave_idx in self.ref_indices:
                        real_idx_map[idx] = (wave, "Re")
                        free_real_indices.append(2*wave_idx)
                        idx += 1
                
                # Print mappings
                for idx, (wave, component) in complex_idx_map.items():
                    console.print(f"complex_params[{idx}] corresponds to {component}[{wave}]", style="bold")
                for idx, (wave, component) in real_idx_map.items():
                    console.print(f"real_params[{idx}] corresponds to {component}[{wave}]", style="bold")
            
            else:  # Horseshoe prior
                free_complex_indices = []
                complex_idx_map = {}
                idx = 0
                for wave_idx, wave in enumerate(self.waveNames):
                    if wave_idx not in self.ref_indices:
                        complex_idx_map[idx] = (wave, "Re")
                        free_complex_indices.append(2*wave_idx)
                        idx += 1
                        complex_idx_map[idx] = (wave, "Im")
                        free_complex_indices.append(2*wave_idx+1)
                        idx += 1
                
                free_real_indices = []
                real_idx_map = {}
                idx = 0
                for wave_idx, wave in enumerate(self.waveNames):
                    if wave_idx in self.ref_indices:
                        real_idx_map[idx] = (wave, "Re")
                        free_real_indices.append(2*wave_idx)
                        idx += 1
                
                # Print mappings
                for idx, (wave, component) in complex_idx_map.items():
                    console.print(f"raw_params[{idx}] corresponds to {component}[{wave}]", style="bold")
                
                for idx, (wave, component) in real_idx_map.items():
                    console.print(f"raw_params_real[{idx}] corresponds to {component}[{wave}]", style="bold")
        
        elif self.cop == "polar":
            # For magnitudes (all waves)
            console.print("\nParameter mapping for NumPyro summary:", style="bold")
            for idx, wave in enumerate(self.waveNames):
                console.print(f"magnitudes[{idx}] corresponds to magnitude of [{wave}]", style="bold")
            
            # For phase_params (non-reference waves)
            free_phase_indices = [i for i in range(len(self.waveNames)) if i not in self.ref_indices]
            for i, wave_idx in enumerate(free_phase_indices):
                wave = self.waveNames[wave_idx]
                console.print(f"phase_params[{i}] corresponds to phase of [{wave}]", style="bold")
        
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
            
            n_samples = len(next(iter(samples.values())))
            params = jnp.zeros((n_samples, self.nPars))

            free_complex_indices = self.free_complex_indices
            free_real_indices = self.free_real_indices

            if self.prior_dist != "horseshoe": # {Gaussian, Laplace}
                complex_params = samples['complex_params']
                real_params = samples['real_params']                                
                for i, idx in enumerate(free_real_indices):
                    params = params.at[:, idx].set(real_params[:, i])                
                for i, idx in enumerate(free_complex_indices):
                    params = params.at[:, idx].set(complex_params[:, i])
            else: # Horseshoe prior handling
                raw_params_real = samples['raw_params_real']
                raw_params = samples['raw_params']
                local_scale_real = samples['local_scale_real']                
                local_scale = samples['local_scale']
                global_scale = samples['global_scale']                
                actual_params_real = raw_params_real * local_scale_real * global_scale[:, None]
                actual_params_complex = raw_params * local_scale * global_scale[:, None]                
                for i, idx in enumerate(free_real_indices):
                    params = params.at[:, idx].set(actual_params_real[:, i])
                for i, idx in enumerate(free_complex_indices):
                    params = params.at[:, idx].set(actual_params_complex[:, i])

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

        self.pwa_manager = GluexJaxManager(comm0=None, mpi_offset=1,
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
        
        # Determine free_complex_indices and free_real_indices once
        self.free_complex_indices = []
        self.free_real_indices = []
        for i in range(len(self.waveNames)):
            # Reference waves have real part as free parameter, imaginary fixed to 0
            # Non-reference waves have both real and imaginary parts as free parameters
            if i in self.ref_indices: 
                self.free_real_indices.append(2*i)
            else:
                self.free_complex_indices.append(2*i)
                self.free_complex_indices.append(2*i+1)
        self.free_complex_indices = jnp.array(self.free_complex_indices)
        self.free_real_indices = jnp.array(self.free_real_indices)
        
    def _get_initial_params(self):
        # NOTE: See commit for test code: 2f14970
        #       TEST CODE: uses lbfgs to perform short optimization to move initial parameters to better starting location
        #                  MCMC appears to perform well without this step so it was removed but you can reference it later
        pass

class OptimizerHelpFormatter(argparse.ArgumentParser):
    def error(self, message):
        console.print(f"Error: {message}", style="red")
        self.print_help()
        sys.exit(2)

    def format_help(self):
        help_message = super().format_help()
        return help_message

if __name__ == "__main__":
    parser = OptimizerHelpFormatter(description="Run MCMC fits using numpyro.")
    parser.add_argument("yaml_file", type=str,
                       help="Path to PyAmpTools YAML configuration file")    
    parser.add_argument("-b", "--bins", type=int, nargs="+",
                       help="List of bin indices to process")
    parser.add_argument("--output_folder", type=str, default=None,
                        help="Folder to save output results to. If not provided then will dump to 'MCMC' subdirectory in YAML.base_directory")
    parser.add_argument("--nprocesses", type=int, default=10,
                        help="Number of processes to distribute work under. Each fit in a bin utilizes around 5 complete cores (if nchains >= 5). Number of bins in paralell is then nprocesses / 5")

    #### MCMC ARGS ####
    parser.add_argument("-ps", "--prior_scale", type=float, default=1000.0,
                       help="Prior scale for the magnitude of the complex amplitudes, default is very large to be as non-informative as possible")
    parser.add_argument("-pd", "--prior_dist", type=str, choices=['laplace', 'gaussian', 'horseshoe'], default='gaussian', 
                       help="Prior distribution for the complex amplitudes")
    parser.add_argument("-nc", "--nchains", type=int, default=6,
                       help="Number of chains to use for numpyro MCMC (each chain runs on paralell process)")
    parser.add_argument("-ns", "--nsamples", type=int, default=1000,
                       help="Number of samples to draw per chain")
    parser.add_argument("-nw", "--nwarmup", type=int, default=500,
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
    
    #### HELPFUL ARGS ####
    parser.add_argument("--print_wave_names", action="store_true",
                       help="Print wave names")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("-cop", "--coordinate_system", type=str, choices=["cartesian", "polar"], default="cartesian",
                       help="Coordinate system to use for the complex amplitudes. polar is not really supported.")
    parser.add_argument("--enforce_positive_reference", action="store_true",
                       help="Force the real part of reference waves to be strictly positive (default: allow negative values)")
    
    args = parser.parse_args()

    pyamptools_yaml = load_yaml(args.yaml_file)
    iftpwa_yaml = pyamptools_yaml["nifty"]["yaml"]
    iftpwa_yaml = load_yaml(iftpwa_yaml)
    
    if not iftpwa_yaml:
        raise ValueError("iftpwa YAML file is required")
    if not pyamptools_yaml:
        raise ValueError("PyAmpTools YAML file is required")
    
    # TODO: Properly implement wave_prior_scales somewhere. Since its a dict we might have to put it into YAML file?
    wave_prior_scales = None
    # wave_prior_scales = {
    #     "Sp0+": 150,
    #     "Dp2+": 150,
    # }

    # TODO: remove dirname later in here and opt_mle and fix subdir
    output_folder = args.output_folder
    if output_folder is None:
        output_folder = os.path.join(os.path.dirname(pyamptools_yaml["base_directory"]), "TEST_MCMC")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        raise ValueError(f"Output folder {args.output_folder} already exists! Please provide a different path or remove the folder.")
    
    # Initialize MCMC Manager, preparing for the first requested bin
    mcmc_manager = MCMCManager(
        pyamptools_yaml, iftpwa_yaml, args.bins[0], 
        prior_scale=args.prior_scale, prior_dist=args.prior_dist,
        n_chains=args.nchains, n_samples=args.nsamples, n_warmup=args.nwarmup, 
        resume_path=args.resume, output_folder=args.output_folder, cop=args.coordinate_system, 
        wave_prior_scales=wave_prior_scales,
        target_accept_prob=args.target_accept,
        max_tree_depth=args.max_tree_depth,
        step_size=args.step_size,
        adapt_step_size=args.adapt_step_size,
        dense_mass=args.dense_mass,
        adapt_mass_matrix=args.adapt_mass_matrix,
        enforce_positive_reference=args.enforce_positive_reference
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
    
    #### RUN MCMC ####
    timer = Timer()
    final_result_dicts = [mcmc_manager.run_mcmc()]
    if len(args.bins) > 1:
        for bin_idx in args.bins[1:]:
            mcmc_manager.set_bin(bin_idx)
            final_result_dict = mcmc_manager.run_mcmc()
            final_result_dicts.append(final_result_dict)
    
    with open(f"{output_folder}/mcmc_bin{bin_idx}_samples.pkl", "wb") as f:
        pkl.dump(final_result_dicts, f)

    console.print(f"Total time elapsed: {timer.read()[2]}", style="bold")

    sys.exit(0)
