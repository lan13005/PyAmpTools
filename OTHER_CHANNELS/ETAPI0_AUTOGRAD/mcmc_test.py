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
import jax
import jax.numpy as jnp
import pymc as pm
import pytensor.tensor as pt
import pytensor

# jax.config.update('jax_disable_jit', True)

from optimize_utility import Objective

from iftpwa1.pwa.gluex.constants import INTENSITY_AC

Minuit.errordef = Minuit.LIKELIHOOD
comm0 = None
rank = 0 
mpi_offset = 1

def create_pwa_numpyro_model(objective, prior_scale=100.0):
    """
    Create a NumPyro model using the PWA likelihood.
    
    Args:
        objective: Instance of Objective
        prior_scale: Scale for the normal prior on parameters
    """
    def model():
        params = numpyro.sample(
            "params",
            dist.Normal(
                loc=jnp.zeros(objective.nPars),
                scale=prior_scale * jnp.ones(objective.nPars)
            )
        )
        nll = objective.objective(params)
        print(f"nll: {nll}")
        numpyro.factor("likelihood", -nll)
    
    return model

def run_mcmc_inference(objective, n_warmup=0, n_samples=100, num_chains=4):
    """
    Run MCMC inference using PyMC's NUTS sampler
    """
    # Create custom likelihood class with gradient
    class PWALikelihood:
        def __init__(self, objective):
            self.objective = objective
        
        def logp(self, params):
            # Return negative because PyMC expects log probability (we have NLL)
            return -self.objective.objective(params)
        
        def dlogp(self, params):
            # Return negative because PyMC expects gradient of log probability
            return -self.objective.gradient(params)

    pwa_likelihood = PWALikelihood(objective)
    
    # Create PyMC model
    with pm.Model() as model:
        # Define parameters with normal prior
        params = pm.Normal('params', 
                         mu=0, 
                         sigma=100.0, 
                         shape=objective.nPars)
        
        # Add custom likelihood using the Potential
        pm.Potential('likelihood', 
                    pm.logp(params, pwa_likelihood.logp, 
                           grad_logp=pwa_likelihood.dlogp))
        
        # Run NUTS sampler
        trace = pm.sample(draws=n_samples,
                         tune=n_warmup,
                         chains=num_chains,
                         target_accept=0.85,
                         return_inferencedata=True,
                         progressbar=True)
        
        # Print summary
        print("\nSampling Summary:")
        print(pm.summary(trace))
        
        # Get samples
        samples = trace.posterior['params'].values
        
        # Print diagnostics
        print("\nDivergence Information:")
        n_divergent = trace.sample_stats.diverging.sum()
        print(f"Number of divergent transitions: {n_divergent}")
        
        print("\nEnergy Information:")
        energy = trace.sample_stats.energy
        print(f"Energy mean: {energy.mean()}")
        print(f"Energy std: {energy.std()}")
    
    return trace

def run_fit(
    pyamptools_yaml, 
    iftpwa_yaml,
    bin_idx, 
    ):
    """
    Run fit with specified method
    
    Args:
        bin_idx: Index of bin to optimize
    """
    
    from iftpwa1.pwa.gluex.gluex_jax_manager import (
        GluexJaxManager,
    )
    
    mbin = bin_idx % nmbMasses
    tbin = bin_idx // nmbMasses
    pwa_manager = GluexJaxManager(comm0=comm0, mpi_offset=mpi_offset,
                                yaml_file=pyamptools_yaml,
                                resolved_secondary=iftpwa_yaml, prior_simulation=False, sum_returned_nlls=False)
 
    pwa_manager.set_bins(np.array([bin_idx]))
        
    obj = Objective(pwa_manager, bin_idx, nPars, nmbMasses, nmbTprimes)
    
    print("\n**************************************************************")
    
    final_result_dict = {}
    optim_result = {}
    
    trace = run_mcmc_inference(obj, n_warmup=2000, n_samples=2000, num_chains=args.nprocesses)
    
    # Extract samples from trace
    samples = trace.posterior['params'].values
    # Reshape to (n_samples_total, nPars)
    samples = samples.reshape(-1, obj.nPars)
    
    best_fit_params = np.zeros((nPars, nmbMasses, nmbTprimes))
    best_fit_params[:, mbin, tbin] = np.mean(samples, axis=0)
    
    sample_intensities = {}
    for isample in range(len(samples)):
        if "total" not in sample_intensities:
            sample_intensities["total"] = [obj.intensity(samples[isample])]
        else:
            sample_intensities["total"].append(obj.intensity(samples[isample]))
        for wave in pwa_manager.waveNames:
            if wave not in sample_intensities:
                sample_intensities[wave] = [obj.intensity(samples[isample], suffix=[wave])]
            else:
                sample_intensities[wave].append(obj.intensity(samples[isample], suffix=[wave]))
    
    final_result_dict['sample_intensities'] = sample_intensities
    final_result_dict['likelihood'] = None
    final_result_dict['initial_likelihood'] = None
    optim_result = {'success': 'mcmc drew requested samples', 'likelihood': 'mcmc does not return a likelihood', 'message': 'no message'}

    # print(f"Intensity for bin {bin_idx}: {final_result_dict}")
    # print(f"Optimization results for bin {bin_idx}:")
    # print(f"Success: {optim_result['success']}")
    # print(f"Final likelihood: {optim_result['likelihood']}")
    # print(f"Message: {optim_result['message']}")
    print("**************************************************************\n")
    
    # final_par_values = {}
    # for iw, wave in enumerate(pwa_manager.waveNames):
    #     final_par_values[wave] = optim_result['parameters'][2*iw] + 1j * optim_result['parameters'][2*iw+1]
    
    # final_result_dict['final_par_values'] = final_par_values

    return final_result_dict

class OptimizerHelpFormatter(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f"Error: {message}\n")
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
    parser.add_argument("--bins", type=int, nargs="+",
                       help="List of bin indices to process")
    parser.add_argument("--setting", type=int, default=0)

    #### MCMC ARGS ####
    parser.add_argument("--nprocesses", type=int, default=4,
                       help="Number of processes to use for numpyro MCMC, equals num_chains")
    
    #### HELPFUL ARGS ####
    parser.add_argument("--print_wave_names", action="store_true",
                       help="Print wave names")
    
    
    args = parser.parse_args()

    numpyro.set_host_device_count(int(args.nprocesses))
    np.random.seed(42)
    
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
        print(f"Wave names: {waveNames}")
        sys.exit(0)

    timer = Timer()
    if args.bins is None:
        raise ValueError("list of bin indices is required")

    final_result_dicts = []
    for bin_idx in args.bins:
        final_result_dict = run_fit(
            pyamptools_yaml, 
            iftpwa_yaml, 
            bin_idx, 
        )
        final_result_dicts.append(final_result_dict)
    
    sbins = "_".join([str(b) for b in args.bins]) if isinstance(args.bins, list) else args.bins
    with open(f"COMPARISONS/{args.method}_bin{sbins}_setting{args.setting}.pkl", "wb") as f:
        pkl.dump(final_result_dicts, f)

    print(f"Total time elapsed: {timer.read()[2]}")

    sys.exit(0)
