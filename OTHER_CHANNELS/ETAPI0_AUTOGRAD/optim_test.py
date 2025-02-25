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

from iftpwa1.pwa.gluex.constants import (
    LIKELIHOOD,
    LIKELIHOOD,
    GRAD,
    HESSIAN,
    INTENSITY_AC,
)

numpyro.set_host_device_count(4)

Minuit.errordef = Minuit.LIKELIHOOD
comm0 = None
rank = 0 
mpi_offset = 1

##############################################
########### MIRRORS MINUIT_TEST.PY ###########
np.random.seed(42)
scale = 50
n_iterations = 30
pyamptools_yaml = "/w/halld-scshelf2101/lng/WORK/PyAmpTools9/OTHER_CHANNELS/ETAPI0_AUTOGRAD/pyamptools.yaml"
iftpwa_yaml = "/w/halld-scshelf2101/lng/WORK/PyAmpTools9/OTHER_CHANNELS/ETAPI0_AUTOGRAD/iftpwa.yaml"

pyamptools_yaml = load_yaml(pyamptools_yaml)
waveNames = pyamptools_yaml["waveset"].split("_")
nmbMasses = pyamptools_yaml["n_mass_bins"]
nmbTprimes = pyamptools_yaml["n_t_bins"]
nPars = 2 * len(waveNames)

seed_list = np.random.randint(0, 1000000, n_iterations)
#############################################
##############################################

class Objective:

    """
    This is our objective function that utilizes GlueXJaxManager to obtain NLL ands its derivatives
    """
    
    def __init__(self, pwa_manager, bin_idx, nPars, nmbMasses, nmbTprimes):
        self.pwa_manager = pwa_manager
        self.bin_idx = bin_idx
        self.nPars = nPars
        self.nmbMasses = nmbMasses
        self.nmbTprimes = nmbTprimes
        self.mbin = bin_idx % nmbMasses
        self.tbin = bin_idx // nmbMasses
        
        # flag user can modify to affect whether __call__ returns gradients also
        self._deriv_order = 0       
        
        self.lnorm = 0
        self.lambda_ = 0.0
    
    def set_regularization(self, lnorm=0, lambda_=0.0):
        """Sets hyperparameters for regularization term for objective function.
        
        Args:
            x: Parameter vector to regularize
            lnorm: Type of regularization norm (0, 1, or 2)
                0: No regularization
                1: L1 regularization (good lambda ~ 0.01-0.1)
                2: L2 regularization (good lambda ~ 0.1-1.0)
                3+: ???
            lambda_: Regularization strength coefficient
        """
        self.lnorm = lnorm
        self.lambda_ = lambda_
        
    def regularization(self, x):
        if self.lnorm == 0:
            return 0
        elif isinstance(self.lnorm, (int, float)) and self.lnorm > 0:
            return self.lambda_ * jnp.sum(jnp.abs(x)**self.lnorm)
        else:
            raise ValueError(f"Invalid regularization norm: {self.lnorm}")
        
    def insert_into_full_array(self, x):
        x_full = jnp.zeros((self.nPars, self.nmbMasses, self.nmbTprimes))
        x_full = x_full.at[:, self.mbin, self.tbin].set(x)
        return x_full

    def objective(self, x):
        """Return only the objective value"""
        x_full = self.insert_into_full_array(x)
        return self.pwa_manager.sendAndReceive(x_full, LIKELIHOOD)[0] + self.regularization(x)

    def gradient(self, x):
        """Return only the gradient"""
        x_full = self.insert_into_full_array(x)
        return self.pwa_manager.sendAndReceive(x_full, GRAD)[0]

    def hessp(self, x, p):
        """Compute the Hessian-vector product at point x with vector p"""
        x_full = self.insert_into_full_array(x)
        hess = self.pwa_manager.sendAndReceive(x_full, HESSIAN)[0]
        return jnp.dot(hess, p)

    def __call__(self, x):
        """Return both objective value and gradient for scipy.optimize"""
        nll = self.objective(x)
        if self._deriv_order == 0:
            return nll
        elif self._deriv_order == 1:
            grad = self.gradient(x)
            return nll, grad
        else:
            raise ValueError(f"Invalid derivative order: {self._deriv_order}")
        
    def intensity(self, x, suffix=None):
        x_full = self.insert_into_full_array(x)
        return self.pwa_manager.sendAndReceive(x_full, INTENSITY_AC, suffix=suffix)[0].item()

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

def run_fit(pyamptools_yaml, bin_idx, initial_guess, initial_guess_dict, suffix=0, method='L-BFGS-B'):
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
    
    global iftpwa_yaml
    
    resolved_secondary = load_yaml(iftpwa_yaml)
    pwa_manager = GluexJaxManager(comm0=comm0, mpi_offset=mpi_offset,
                                yaml_file=pyamptools_yaml,
                                resolved_secondary=resolved_secondary, prior_simulation=False, sum_returned_nlls=False)
 
    pwa_manager.set_bins(np.array([bin_idx]))
    
    is_mle_method = method in ['minuit-numeric', 'minuit-analytic', 'L-BFGS-B', 'trust-ncg', 'trust-krylov']
    
    obj = Objective(pwa_manager, bin_idx, nPars, nmbMasses, nmbTprimes)
    
    initial_likelihood = obj.objective(initial_guess)
    print("\n**************************************************************")
    print(f"Initial likelihood: {initial_likelihood}")
    print(f"Using method: {method}")
    
    intensities = {}
    result = {}
    if is_mle_method:
        if method == 'minuit-analytic':
            from optimize_utility import optimize_single_bin_minuit
            result = optimize_single_bin_minuit(obj, initial_guess, bin_idx, use_analytic_grad=True)
        elif method == 'minuit-numeric':
            from optimize_utility import optimize_single_bin_minuit
            result = optimize_single_bin_minuit(obj, initial_guess, bin_idx, use_analytic_grad=False)
        elif method == 'L-BFGS-B' or method == 'trust-ncg' or method == 'trust-krylov':
            from optimize_utility import optimize_single_bin_scipy
            result = optimize_single_bin_scipy(obj, initial_guess, bin_idx, method=method)
        else:
            raise ValueError(f"Invalid Maximum Likelihood Based method: {method}")

        intensities['total'] = obj.intensity(result['parameters'])
        for wave in pwa_manager.waveNames:
            intensities[wave] = obj.intensity(result['parameters'], suffix=[wave])
        intensities['likelihood'] = result['likelihood']
        intensities['initial_likelihood'] = initial_likelihood
    
    # TODO: This can be simplified since we can just assume MLE fits returns the best fit parameters of size (1, nPars)
    #       and MCMC returns the full set of samples of size (n_samples, nPars)
    #       Rename likelihood and initial_likelihood to energy and initial_energy
    else:
        if method == 'numpyro':
            mcmc = run_mcmc_inference(pwa_manager, bin_idx)
            samples = mcmc.get_samples()
            best_fit_params = np.zeros((nPars, nmbMasses, nmbTprimes))
            best_fit_params[:, mbin, tbin] = jnp.mean(samples['params'], axis=0)
            intensities['total'] = pwa_manager.sendAndReceive(
                best_fit_params,
                INTENSITY_AC,
                suffix=None
            )[0].item()
            for wave in pwa_manager.waveNames:
                intensities[wave] = pwa_manager.sendAndReceive(
                    best_fit_params,
                    INTENSITY_AC,
                    suffix=[wave]
                )[0].item()
            
            intensities['likelihood'] = None
            intensities['initial_likelihood'] = None
            result = {'success': 'mcmc drew requested samples', 'likelihood': 'mcmc does not return a likelihood', 'message': 'no message'}
        else:
            raise ValueError(f"Invalid Markov Chain Monte Carlo method: {method}")

    print(f"Intensity for bin {bin_idx}: {intensities}")
    print(f"Optimization results for bin {bin_idx}:")
    print(f"Success: {result['success']}")
    print(f"Final likelihood: {result['likelihood']}")
    print(f"Message: {result['message']}")
    print("**************************************************************\n")
    
    with open(f"COMPARISONS/{method}_bin{bin_idx}_{suffix}.pkl", "wb") as f:
        pkl.dump({
            'initial_guess_dict': initial_guess_dict, 
            'intensities': intensities,
        }, f)

def run_fits_in_bin(bin_idx, method='L-BFGS-B'):

    for i in range(n_iterations):
        
        np.random.seed(seed_list[i])
        initial_guess = scale * np.random.randn(nPars)
        initial_guess_dict = {} 
        for iw, wave in enumerate(waveNames):
            initial_guess_dict[wave] = initial_guess[2*iw] + 1j * initial_guess[2*iw+1]
        
        run_fit(pyamptools_yaml, bin_idx, initial_guess, initial_guess_dict, suffix=i, method=method)

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
    parser.add_argument("--method", type=str, 
                       choices=['minuit-numeric', 'minuit-analytic', 'L-BFGS-B', 'trust-ncg', 'trust-krylov', 'numpyro'], 
                       default='L-BFGS-B',
                       help="Optimization method to use")
    parser.add_argument("--bins", type=int, nargs="+",
                       help="List of bin indices to process")
    args = parser.parse_args()

    timer = Timer()
    if args.bins is None:
        raise ValueError("list of bin indices is required")

    for bins in args.bins:
        run_fits_in_bin(bins, args.method)
        
    print(f"Total time elapsed: {timer.read()[2]}")

    sys.exit(0)
