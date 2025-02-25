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

from optimize_utility import Objective

from iftpwa1.pwa.gluex.constants import INTENSITY_AC

Minuit.errordef = Minuit.LIKELIHOOD
comm0 = None
rank = 0 
mpi_offset = 1

def create_pwa_numpyro_model(objective, prior_scale=1.0):
    """
    Create a NumPyro model using the PWA likelihood.
    
    Args:
        objective: Instance of Objective
        prior_scale: Scale for the normal prior on parameters
    """
    def model():
        # Define priors for all parameters
        params = numpyro.sample(
            "params",
            dist.Normal(
                loc=jnp.zeros(objective.nPars),
                scale=prior_scale * jnp.ones(objective.nPars)
            )
        )
        
        nll = objective.objective(params)
        numpyro.factor("likelihood", -nll)
    
    return model

def run_mcmc_inference(objective, n_warmup=1000, n_samples=2000, num_chains=4, collect_warmup=True):
    """
    Run MCMC inference using NumPyro for a single bin
    
    Args:
        pwa_manager: GlueXJaxManager instance
        bin_idx: Index of bin to analyze
        n_warmup: Number of warmup steps
        n_samples: Number of samples to collect
        num_chains: Number of MCMC chains to run
    """
    
    model = create_pwa_numpyro_model(objective, prior_scale=100)
    
    # Create random key for NUTS
    rng_key = jax.random.PRNGKey(0)
    rng_key, rng_key_inference = jax.random.split(rng_key)
    
    # Setup NUTS sampler
    kernel = NUTS(model) 
    
    # Run MCMC
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=num_chains, collect_warmup=collect_warmup)
    mcmc.run(rng_key_inference)
    
    mcmc.print_summary()
    
    return mcmc

def run_fit(
    pyamptools_yaml, 
    iftpwa_yaml,
    bin_idx, 
    initial_guess, 
    initial_guess_dict,
    method="L-BFGS-B",
    ):
    """
    Run fit with specified method
    
    Args:
        bin_idx: Index of bin to optimize
        i: Iteration number
        initial_guess: Initial parameters
        initial_guess_dict: Dictionary of initial parameters
        method: see argument parser for more information
        
        initial_guess contains full sized array across kinematic bins needed for GlueXJaxManager
        initial_guess_dict contains initial guess for this bin only stored as a comparison in the pkl file
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
    
    is_mle_method = method in ['minuit-numeric', 'minuit-analytic', 'L-BFGS-B', 'trust-ncg', 'trust-krylov']
    
    obj = Objective(pwa_manager, bin_idx, nPars, nmbMasses, nmbTprimes)
    
    initial_likelihood = obj.objective(initial_guess).item()
    print("\n**************************************************************")
    print(f"Initial likelihood: {initial_likelihood}")
    print(f"Using method: {method}")
    
    final_result_dict = {}
    optim_result = {}
    if is_mle_method:
        if method == 'minuit-analytic':
            from optimize_utility import optimize_single_bin_minuit
            optim_result = optimize_single_bin_minuit(obj, initial_guess, bin_idx, use_analytic_grad=True)
        elif method == 'minuit-numeric':
            from optimize_utility import optimize_single_bin_minuit
            optim_result = optimize_single_bin_minuit(obj, initial_guess, bin_idx, use_analytic_grad=False)
        elif method == 'L-BFGS-B' or method == 'trust-ncg' or method == 'trust-krylov':
            from optimize_utility import optimize_single_bin_scipy
            optim_result = optimize_single_bin_scipy(obj, initial_guess, bin_idx, method=method)
        else:
            raise ValueError(f"Invalid Maximum Likelihood Based method: {method}")

        final_result_dict['total'] = obj.intensity(optim_result['parameters'])
        for wave in pwa_manager.waveNames:
            final_result_dict[wave] = obj.intensity(optim_result['parameters'], suffix=[wave])
        final_result_dict['likelihood'] = optim_result['likelihood']
        final_result_dict['initial_likelihood'] = initial_likelihood
    
    # TODO: This can be simplified since we can just assume MLE fits returns the best fit parameters of size (1, nPars)
    #       and MCMC returns the full set of samples of size (n_samples, nPars)
    #       Rename likelihood and initial_likelihood to energy and initial_energy
    else:
        if method == 'numpyro':
            mcmc = run_mcmc_inference(pwa_manager, bin_idx, num_chains=args.nprocesses)
            samples = mcmc.get_samples()
            best_fit_params = np.zeros((nPars, nmbMasses, nmbTprimes))
            best_fit_params[:, mbin, tbin] = jnp.mean(samples['params'], axis=0)
            final_result_dict['total'] = pwa_manager.sendAndReceive(
                best_fit_params,
                INTENSITY_AC,
                suffix=None
            )[0].item()
            for wave in pwa_manager.waveNames:
                final_result_dict[wave] = pwa_manager.sendAndReceive(
                    best_fit_params,
                    INTENSITY_AC,
                    suffix=[wave]
                )[0].item()
            
            final_result_dict['likelihood'] = None
            final_result_dict['initial_likelihood'] = None
            optim_result = {'success': 'mcmc drew requested samples', 'likelihood': 'mcmc does not return a likelihood', 'message': 'no message'}
        else:
            raise ValueError(f"Invalid Markov Chain Monte Carlo method: {method}")

    print(f"Intensity for bin {bin_idx}: {final_result_dict}")
    print(f"Optimization results for bin {bin_idx}:")
    print(f"Success: {optim_result['success']}")
    print(f"Final likelihood: {optim_result['likelihood']}")
    print(f"Message: {optim_result['message']}")
    print("**************************************************************\n")
    
    final_par_values = {}
    for wave in pwa_manager.waveNames:
        final_par_values[wave] = optim_result['parameters'][2*iw] + 1j * optim_result['parameters'][2*iw+1]
    
    final_result_dict['initial_guess_dict'] = initial_guess_dict
    final_result_dict['final_par_values'] = final_par_values

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
    parser.add_argument("--method", type=str, 
                       choices=['minuit-numeric', 'minuit-analytic', 'L-BFGS-B', 'trust-ncg', 'trust-krylov', 'numpyro'], 
                       default='L-BFGS-B',
                       help="Optimization method to use")
    parser.add_argument("--bins", type=int, nargs="+",
                       help="List of bin indices to process")

    #### MCMC ARGS ####
    parser.add_argument("--nprocesses", type=int, default=4,
                       help="Number of processes to use for numpyro MCMC, equals num_chains")
    
    #### HELPFUL ARGS ####
    parser.add_argument("--print_wave_names", action="store_true",
                       help="Print wave names")
    
    
    args = parser.parse_args()

    numpyro.set_host_device_count(int(args.nprocesses))
    np.random.seed(42)
    scale = 50
    n_iterations = 20
    
    pyamptools_yaml = load_yaml(args.yaml_file)
    iftpwa_yaml = pyamptools_yaml["nifty"]["yaml"]
    iftpwa_yaml = load_yaml(iftpwa_yaml)
    
    settings = [
        {"lambdas": 1e2, "en_alpha": 1.0},
        {"lambdas": 1e1, "en_alpha": 1.0},
        {"lambdas": 1e0, "en_alpha": 1.0},
        {"lambdas": 1e-1, "en_alpha": 1.0},
        {"lambdas": 1e-2, "en_alpha": 1.0},
        {"lambdas": 1e2, "en_alpha": 0.0},
        {"lambdas": 1e1, "en_alpha": 0.0},
        {"lambdas": 1e0, "en_alpha": 0.0},
        {"lambdas": 1e-1, "en_alpha": 0.0},
        {"lambdas": 1e-2, "en_alpha": 0.0},
        {"lambdas": 1e2, "en_alpha": 0.5},
        {"lambdas": 1e1, "en_alpha": 0.5},
        {"lambdas": 1e0, "en_alpha": 0.5},
        {"lambdas": 1e-1, "en_alpha": 0.5},
        {"lambdas": 1e-2, "en_alpha": 0.5},
        {"lambdas": {"Dm2-": 1e2, "Dm1-": 1e2, "Dp0-": 1e2, "Dp1-": 1e2, "Dp2-": 1e2, "Dm2+": 1e2, "Dm1+": 1e2, "Dp0+": 1e2, "Dp1+": 1e2, "Dp2+": 1e2}, "en_alpha": 1.0},
        {"lambdas": {"Dm2-": 1e1, "Dm1-": 1e1, "Dp0-": 1e1, "Dp1-": 1e1, "Dp2-": 1e1, "Dm2+": 1e1, "Dm1+": 1e1, "Dp0+": 1e1, "Dp1+": 1e1, "Dp2+": 1e1}, "en_alpha": 1.0},
        {"lambdas": {"Dm2-": 1e0, "Dm1-": 1e0, "Dp0-": 1e0, "Dp1-": 1e0, "Dp2-": 1e0, "Dm2+": 1e0, "Dm1+": 1e0, "Dp0+": 1e0, "Dp1+": 1e0, "Dp2+": 1e0}, "en_alpha": 1.0},
        {"lambdas": {"Dm2-": 1e-1, "Dm1-": 1e-1, "Dp0-": 1e-1, "Dp1-": 1e-1, "Dp2-": 1e-1, "Dm2+": 1e-1, "Dm1+": 1e-1, "Dp0+": 1e-1, "Dp1+": 1e-1, "Dp2+": 1e-1}, "en_alpha": 1.0},
        {"lambdas": {"Dm2-": 1e-2, "Dm1-": 1e-2, "Dp0-": 1e-2, "Dp1-": 1e-2, "Dp2-": 1e-2, "Dm2+": 1e-2, "Dm1+": 1e-2, "Dp0+": 1e-2, "Dp1+": 1e-2, "Dp2+": 1e-2}, "en_alpha": 1.0},
        {"lambdas": {"Dm2-": 1e2, "Dm1-": 1e2, "Dp0-": 1e2, "Dp1-": 1e2, "Dp2-": 1e2, "Dm2+": 1e2, "Dm1+": 1e2, "Dp0+": 1e2, "Dp1+": 1e2, "Dp2+": 1e2}, "en_alpha": 0.0},
        {"lambdas": {"Dm2-": 1e1, "Dm1-": 1e1, "Dp0-": 1e1, "Dp1-": 1e1, "Dp2-": 1e1, "Dm2+": 1e1, "Dm1+": 1e1, "Dp0+": 1e1, "Dp1+": 1e1, "Dp2+": 1e1}, "en_alpha": 0.0},
        {"lambdas": {"Dm2-": 1e0, "Dm1-": 1e0, "Dp0-": 1e0, "Dp1-": 1e0, "Dp2-": 1e0, "Dm2+": 1e0, "Dm1+": 1e0, "Dp0+": 1e0, "Dp1+": 1e0, "Dp2+": 1e0}, "en_alpha": 0.0},
        {"lambdas": {"Dm2-": 1e-1, "Dm1-": 1e-1, "Dp0-": 1e-1, "Dp1-": 1e-1, "Dp2-": 1e-1, "Dm2+": 1e-1, "Dm1+": 1e-1, "Dp0+": 1e-1, "Dp1+": 1e-1, "Dp2+": 1e-1}, "en_alpha": 0.0},
        {"lambdas": {"Dm2-": 1e-2, "Dm1-": 1e-2, "Dp0-": 1e-2, "Dp1-": 1e-2, "Dp2-": 1e-2, "Dm2+": 1e-2, "Dm1+": 1e-2, "Dp0+": 1e-2, "Dp1+": 1e-2, "Dp2+": 1e-2}, "en_alpha": 0.0},
        {"lambdas": {"Dm2-": 1e2, "Dm1-": 1e2, "Dp0-": 1e2, "Dp1-": 1e2, "Dp2-": 1e2, "Dm2+": 1e2, "Dm1+": 1e2, "Dp0+": 1e2, "Dp1+": 1e2, "Dp2+": 1e2}, "en_alpha": 0.5},
        {"lambdas": {"Dm2-": 1e1, "Dm1-": 1e1, "Dp0-": 1e1, "Dp1-": 1e1, "Dp2-": 1e1, "Dm2+": 1e1, "Dm1+": 1e1, "Dp0+": 1e1, "Dp1+": 1e1, "Dp2+": 1e1}, "en_alpha": 0.5},
        {"lambdas": {"Dm2-": 1e0, "Dm1-": 1e0, "Dp0-": 1e0, "Dp1-": 1e0, "Dp2-": 1e0, "Dm2+": 1e0, "Dm1+": 1e0, "Dp0+": 1e0, "Dp1+": 1e0, "Dp2+": 1e0}, "en_alpha": 0.5},
        {"lambdas": {"Dm2-": 1e-1, "Dm1-": 1e-1, "Dp0-": 1e-1, "Dp1-": 1e-1, "Dp2-": 1e-1, "Dm2+": 1e-1, "Dm1+": 1e-1, "Dp0+": 1e-1, "Dp1+": 1e-1, "Dp2+": 1e-1}, "en_alpha": 0.5},
        {"lambdas": {"Dm2-": 1e-2, "Dm1-": 1e-2, "Dp0-": 1e-2, "Dp1-": 1e-2, "Dp2-": 1e-2, "Dm2+": 1e-2, "Dm1+": 1e-2, "Dp0+": 1e-2, "Dp1+": 1e-2, "Dp2+": 1e-2}, "en_alpha": 0.5},
    ]
    
    for k, v in settings[args.setting].items():
        pyamptools_yaml["regularization"][k] = v
    
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

    seed_list = np.random.randint(0, 1000000, n_iterations)

    timer = Timer()
    if args.bins is None:
        raise ValueError("list of bin indices is required")

    final_result_dicts = []
    for bin_idx in args.bins:
        for i in range(n_iterations):
            np.random.seed(seed_list[i])
            initial_guess = scale * np.random.randn(nPars)
            initial_guess_dict = {} 
            for iw, wave in enumerate(waveNames):
                initial_guess_dict[wave] = initial_guess[2*iw] + 1j * initial_guess[2*iw+1]
            final_result_dict = run_fit(
                pyamptools_yaml, 
                iftpwa_yaml, 
                bin_idx, 
                initial_guess, 
                initial_guess_dict,
                method=args.method,
            )
            final_result_dicts.append(final_result_dict)
    
    sbins = "_".join([str(b) for b in args.bins]) if isinstance(args.bins, list) else args.bins
    with open(f"COMPARISONS/{args.method}_bin{sbins}_setting{args.setting}.pkl", "wb") as f:
        pkl.dump(final_result_dicts, f)

    print(f"Total time elapsed: {timer.read()[2]}")

    sys.exit(0)
