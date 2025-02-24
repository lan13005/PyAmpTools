from pyamptools.utility.general import load_yaml, Timer
import numpy as np
# from mpi4py import MPI
import sys
import os
import shutil
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
    LIKELIHOOD_AND_GRAD,
    LIKELIHOOD_GRAD_HESSIAN,
    LIKELIHOOD,
    INTENSITY_AC,
)

Minuit.errordef = Minuit.LIKELIHOOD
# comm0 = MPI.COMM_WORLD
# rank = comm0.Get_rank()
comm0 = None
rank = 0 

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

seed_list = np.random.randint(0, 1000000, (n_iterations, nmbMasses, nmbTprimes))
#############################################
##############################################

class ObjectiveWithGrad:
    def __init__(self, pwa_manager, bin_idx, nPars, nmbMasses, nmbTprimes):
        self.pwa_manager = pwa_manager
        self.bin_idx = bin_idx
        self.nPars = nPars
        self.nmbMasses = nmbMasses
        self.nmbTprimes = nmbTprimes
        self.mbin = bin_idx % nmbMasses
        self.tbin = bin_idx // nmbMasses
        
        self._last_x = None
        self._last_val = None
        self._last_grad = None
        self._last_hess = None
    
        self._calc_grads = True
        self._calc_hess = False
    
    def _compute(self, x):
        """Compute value, gradient and hessian if needed"""
        if self._last_x is None or not np.array_equal(x, self._last_x):
            self._last_x = np.array(x)
            
            x_full = np.zeros((self.nPars, self.nmbMasses, self.nmbTprimes))
            x_full[:, self.mbin, self.tbin] = x
            
            if self._calc_hess:
                tag = LIKELIHOOD_GRAD_HESSIAN
            elif self._calc_grads:
                tag = LIKELIHOOD_AND_GRAD
            else:
                tag = LIKELIHOOD
                
            result = self.pwa_manager.sendAndReceive(x_full, tag)[0]

            if self._calc_hess:
                self._last_val, self._last_grad, self._last_hess = result
            elif self._calc_grads:
                self._last_val, self._last_grad = result
            else:
                self._last_val = result

    def objective(self, x):
        """Return only the objective value"""
        self._compute(x)
        return self._last_val
    
    def gradient(self, x):
        """Return only the gradient"""
        self._compute(x)
        if self._calc_grads:
            return self._last_grad
        else:
            raise ValueError("Gradient was never calculated!")

    def hessp(self, x, p):
        """
        Compute the Hessian-vector product at point x with vector p
        """
        self._compute(x)
        
        # Ensure Hessian and p have correct dimensions
        hess = np.asarray(self._last_hess)
        p = np.asarray(p)
        
        if hess.ndim == 0:
            raise ValueError("Hessian computation failed - received empty array")
        
        return np.dot(hess, p) # Use np.dot instead of @ operator for better dimension handling

    def __call__(self, x):
        """Return both objective value and gradient for scipy.optimize"""
        self._compute(x)
        if self._calc_grads:
            return self._last_val, self._last_grad
        else:
            return self._last_val

def optimize_single_bin_minuit(pwa_manager, initial_params, bin_idx, use_analytic_grad=True):
    """
    Optimize parameters for a single kinematic bin using Minuit.
    
    Args:
        pwa_manager: GlueXJaxManager instance that calculates likelihood and gradients
        initial_params: Initial parameters for optimization with shape (nPars, nmbMasses, nmbTprimes)
        bin_idx: Index of bin to optimize
        use_analytic_grad: If True, use analytic gradients from sendAndReceive
        
    Returns:
        dict: Dictionary containing optimization results
    """
    nPars = pwa_manager.nPars
    nmbMasses = pwa_manager.nmbMasses
    nmbTprimes = pwa_manager.nmbTprimes
    
    # Calculate mass and t bin indices
    mbin = bin_idx % nmbMasses
    tbin = bin_idx // nmbMasses
    
    # Extract initial parameters for this bin
    x0 = initial_params[:, mbin, tbin]
    
    # Create instance
    obj = ObjectiveWithGrad(pwa_manager, bin_idx, nPars, nmbMasses, nmbTprimes)
    
    if use_analytic_grad:
        obj._calc_grads = True
    else:
        obj._calc_grads = False

    if not use_analytic_grad:
        print("Using numeric gradients")

    # Initialize parameter names for Minuit
    param_names = [f'x{i}' for i in range(len(x0))]
    
    if use_analytic_grad:
        # Use Minuit with analytic gradients
        m = Minuit(
            obj.objective,
            x0,
            grad=obj.gradient,
            name=param_names,
        )
    else:
        # Let Minuit compute numerical gradients
        m = Minuit(
            obj.objective,
            x0,
            name=param_names,
        )

    # Configure Minuit
    # AmpTools uses 0.001 * tol * UP (default 0.1)
    # iminuit uses  0.002 * tol * errordef (default 0.1)
    m.tol = 0.05 # of 0.05 to match implementation in AmpTools?
    m.strategy = 1  # More accurate minimization
    m.migrad()  # Run minimization
    
    return {
        'parameters': m.values,  # Updated attribute name
        'likelihood': m.fval,
        'success': m.valid,
        'message': 'Valid minimum found' if m.valid else 'Minimization failed',
        'errors': m.errors  # Updated attribute name
    }

def optimize_single_bin_scipy(pwa_manager, initial_params, bin_idx, bounds=None, method='L-BFGS-B', options=None):
    """
    Optimize parameters for a single kinematic bin using scipy.optimize methods.
    
    Args:
        pwa_manager: GlueXJaxManager instance that calculates likelihood and gradients
        initial_params: Initial parameters for optimization with shape (nPars, nmbMasses, nmbTprimes)
        bin_idx: Index of bin to optimize
        bounds: Parameter bounds for constrained optimization (optional)
        method: Optimization method to use (default: 'L-BFGS-B')
        options: Dictionary of options for the optimizer
        
    Returns:
        dict: Dictionary containing optimization results
    """
    nPars = pwa_manager.nPars
    nmbMasses = pwa_manager.nmbMasses
    nmbTprimes = pwa_manager.nmbTprimes
    
    mbin = bin_idx % nmbMasses
    tbin = bin_idx // nmbMasses
    
    x0 = initial_params[:, mbin, tbin]
    
    if options is None:
        options = {
            'maxiter': 2000,
            'ftol': 1e-10,
            'gtol': 1e-8
        }

    obj = ObjectiveWithGrad(pwa_manager, bin_idx, nPars, nmbMasses, nmbTprimes)
    
    if method == 'trust-ncg' or method == 'trust-krylov':
        obj._calc_hess = True
        result = minimize(
            obj.objective,
            x0,
            method=method,
            jac=obj.gradient,
            hessp=obj.hessp,
            options=options
        )
    elif method == 'L-BFGS-B':
        obj._calc_hess = False
        result = minimize(
            obj,
            x0,
            method=method,
            jac=True,
            bounds=bounds,
            options=options
        )
    else:
        raise ValueError(f"Invalid optimizer: {method}")
        
    return {
        'parameters': result.x,
        'likelihood': result.fun,
        'success': result.success,
        'message': result.message,
        'errors': None  # Scipy doesn't provide error estimates by default
    }

def create_pwa_numpyro_model(objective_with_grad, prior_scale=1.0):
    """
    Create a NumPyro model using the PWA likelihood.
    
    Args:
        objective_with_grad: Instance of ObjectiveWithGrad
        prior_scale: Scale for the normal prior on parameters
    """
    def model():
        # Define priors for all parameters
        # Using normal priors centered at 0 for both real and imaginary parts
        params = numpyro.sample(
            "params",
            dist.Normal(
                loc=jnp.zeros(objective_with_grad.nPars),
                scale=prior_scale * jnp.ones(objective_with_grad.nPars)
            )
        )
        
        # The negative log-likelihood from ObjectiveWithGrad
        # Note: NumPyro maximizes the log probability, so we negate the NLL
        numpyro.factor("likelihood", -objective_with_grad.objective(params))

def run_mcmc_inference(pwa_manager, bin_idx, n_warmup=1000, n_samples=2000, num_chains=4):
    """
    Run MCMC inference using NumPyro for a single bin
    
    Args:
        pwa_manager: GlueXJaxManager instance
        bin_idx: Index of bin to analyze
        n_warmup: Number of warmup steps
        n_samples: Number of samples to collect
        num_chains: Number of MCMC chains to run
    """
    nPars = pwa_manager.nPars
    nmbMasses = pwa_manager.nmbMasses
    nmbTprimes = pwa_manager.nmbTprimes
    
    # Create objective function for this bin
    objective = ObjectiveWithGrad(pwa_manager, bin_idx, nPars, nmbMasses, nmbTprimes)
    
    # Create model
    model = create_pwa_numpyro_model(objective)
    
    # Setup NUTS sampler with gradient information
    kernel = NUTS(model, target_accept_prob=0.8)
    
    # Run MCMC
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=num_chains)
    mcmc.run(jax.random.PRNGKey(0))
    
    # Get summary statistics
    mcmc.print_summary()
    
    return mcmc

def run_fit(bin_idx, i, initial_guess, initial_guess_dict, optimizer='minuit_numeric'):
    """
    Run fit with specified optimizer
    
    Args:
        bin_idx: Index of bin to optimize
        i: Iteration number
        initial_guess: Initial parameters
        initial_guess_dict: Dictionary of initial parameters
        optimizer: see argument parser for more information
    """
    
    from iftpwa1.pwa.gluex.gluex_jax_manager import (
        GluexJaxManager,
    )
    mpi_offset = 1
    
    mbin = bin_idx % nmbMasses
    tbin = bin_idx // nmbMasses
    
    global iftpwa_yaml
    
    resolved_secondary = load_yaml(iftpwa_yaml)
    pwa_manager = GluexJaxManager(comm0=comm0, mpi_offset=mpi_offset,
                                yaml_file=pyamptools_yaml,
                                resolved_secondary=resolved_secondary, prior_simulation=False, sum_returned_nlls=False)
 
    pwa_manager.set_bins(np.array([bin_idx]))
        
    if rank == 0:
        def fit_and_dump(initial_guess, initial_guess_dict, iteration, method):
            
            """
            initial_guess contains full sized array across kinematic bins needed for GlueXJaxManager
            initial_guess_dict contains initial guess for this bin only stored as a comparison in the pkl file
            """
            initial_likelihood = 2 * pwa_manager.sendAndReceive(initial_guess, LIKELIHOOD)[0].item()
            print("\n**************************************************************")
            print(f"Initial likelihood: {initial_likelihood}")
            print(f"Using optimizer: {method}")
            
            if method == 'minuit_analytic':
                result = optimize_single_bin_minuit(pwa_manager, initial_guess, bin_idx, use_analytic_grad=True)
            elif method == 'minuit_numeric':
                result = optimize_single_bin_minuit(pwa_manager, initial_guess, bin_idx, use_analytic_grad=False)
            elif method == 'L-BFGS-B' or method == 'trust-ncg' or method == 'trust-krylov':
                result = optimize_single_bin_scipy(pwa_manager, initial_guess, bin_idx, method=method)
            elif method == 'numpyro':
                mcmc = run_mcmc_inference(pwa_manager, bin_idx)
                samples = mcmc.get_samples()
                best_fit_params = np.zeros((nPars, nmbMasses, nmbTprimes))
                best_fit_params[:, mbin, tbin] = jnp.mean(samples['params'], axis=0)
                intensities = {
                    'total': pwa_manager.sendAndReceive(
                        best_fit_params,
                        INTENSITY_AC,
                        suffix=None
                    )[0].item()
                }
                for wave in pwa_manager.waveNames:
                    intensities[wave] = pwa_manager.sendAndReceive(
                        best_fit_params,
                        INTENSITY_AC,
                        suffix=[wave]
                    )[0].item()
                intensities['likelihood'] = result['likelihood']
                intensities['initial_likelihood'] = initial_likelihood
            
            print(f"Intensity for bin {bin_idx}: {intensities}")
            print(f"Optimization results for bin {bin_idx}:")
            print(f"Success: {result['success']}")
            print(f"Final likelihood: {result['likelihood']}")
            print(f"Message: {result['message']}")
            print("**************************************************************\n")
            
            with open(f"COMPARISONS/{method}_bin{bin_idx}_{iteration}.pkl", "wb") as f:
                pkl.dump({
                    'initial_guess_dict': initial_guess_dict, 
                    'intensities': intensities,
                }, f)
                
        fit_and_dump(initial_guess, initial_guess_dict, i, optimizer)
        
        pwa_manager.stop()
        
        del pwa_manager

def run_fits_in_bin(bin_idx, optimizer='minuit'):
    
    mbin = bin_idx % nmbMasses
    tbin = bin_idx // nmbMasses
        
    for i in range(n_iterations):
        
        np.random.seed(seed_list[i, mbin, tbin])
        initial_guess = scale * np.random.randn(nPars, nmbMasses, nmbTprimes)
        initial_guess_dict = {} 
        for iw, wave in enumerate(waveNames):
            initial_guess_dict[wave] = initial_guess[2*iw, mbin, tbin] + 1j * initial_guess[2*iw+1, mbin, tbin]
        
        run_fit(bin_idx, i, initial_guess, initial_guess_dict, optimizer)

class OptimizerHelpFormatter(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f"Error: {message}\n")
        self.print_help()
        sys.exit(2)

    def format_help(self):
        help_message = super().format_help()
        
        optimizer_help = "\nOptimizer Descriptions:\n"
        optimizer_help += "\nMinuit-based Methods:\n"
        optimizer_help += "  * minuit_numeric:\n"
        optimizer_help += "      Lets Minuit compute numerical gradients\n"
        optimizer_help += "  * minuit_analytic:\n"
        optimizer_help += "      Uses analytic gradients from PWA likelihood manager\n"
        
        optimizer_help += "\nSciPy-based Methods:\n"
        optimizer_help += "  * L-BFGS-B:\n"
        optimizer_help += "      Limited-memory BFGS quasi-Newton method (stores approximate Hessian)\n"
        optimizer_help += "      + Efficient for large-scale problems\n"
        optimizer_help += "      - May struggle with highly correlated parameters\n"
        
        optimizer_help += "  * trust-ncg:\n"
        optimizer_help += "      Trust-region Newton-Conjugate Gradient\n"
        optimizer_help += "      + Adaptively adjusts step size using local quadratic approximation\n"
        optimizer_help += "      + Efficient for large-scale problems\n"
        optimizer_help += "      - Can be unstable for ill-conditioned problems\n"
        
        optimizer_help += "  * trust-krylov:\n"
        optimizer_help += "      Trust-region method with Krylov subspace solver\n"
        optimizer_help += "      + Better handling of indefinite (sparse) Hessians, Kyrlov subspcae accounts for non-Euclidean geometry\n"
        optimizer_help += "      + More robust for highly correlated parameters\n"
        
        return help_message + "\n" + optimizer_help

if __name__ == "__main__":
    parser = OptimizerHelpFormatter(description="Run optimization fits using various methods.")
    parser.add_argument("--optimizer", type=str, 
                       choices=['minuit_numeric', 'minuit_analytic', 'L-BFGS-B', 'trust-ncg', 'trust-krylov', 'numpyro'], 
                       default='minuit_numeric',
                       help="Optimization method to use")
    parser.add_argument("--bins", type=int, nargs="+",
                       help="List of bin indices to process")
    args = parser.parse_args()

    timer = Timer()
    if args.bins is None:
        raise ValueError("list of bin indices is required")
    
    
    for bins in args.bins:
        run_fits_in_bin(bins, args.optimizer)
        
    print(f"Total time elapsed: {timer.read()[2]}")

    # MPI.Finalize()
    sys.exit(0)
