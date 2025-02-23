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

from iftpwa1.pwa.gluex.gluex_jax_manager import (
    GluexJaxManager,
    LIKELIHOOD,
    LIKELIHOOD_AND_GRAD,
    LIKELIHOOD_GRAD_HESSIAN,
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
    
        self._calc_grads = True
    
    def _compute(self, x):
        """Compute both value and gradient if needed"""
        if self._last_x is None or not np.array_equal(x, self._last_x):
            self._last_x = np.array(x)
            
            x_full = np.zeros((self.nPars, self.nmbMasses, self.nmbTprimes))
            x_full[:, self.mbin, self.tbin] = x
            LIKELIHOOD_TAG = LIKELIHOOD_AND_GRAD if self._calc_grads else LIKELIHOOD
            result = self.pwa_manager.sendAndReceive(x_full, LIKELIHOOD_TAG)[0]
            if self._calc_grads:
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
    
    result = minimize(
        obj,
        x0,
        method=method,
        jac=True,
        bounds=bounds,
        options=options
    )
    
    return {
        'parameters': result.x,
        'likelihood': result.fun,
        'success': result.success,
        'message': result.message,
        'errors': None  # Scipy doesn't provide error estimates by default
    }

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
            elif method == 'lbfgs':
                result = optimize_single_bin_scipy(pwa_manager, initial_guess, bin_idx, method='L-BFGS-B')
            
            best_fit_params = np.zeros((nPars, nmbMasses, nmbTprimes))
            best_fit_params[:, mbin, tbin] = result['parameters']
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
        
        if optimizer == 'minuit_analytic': # analytic gradients provided by jax autodiff
            fit_and_dump(initial_guess, initial_guess_dict, i, 'minuit_analytic')
        elif optimizer == 'minuit_numeric': # numeric gradients provided by Minuit
            fit_and_dump(initial_guess, initial_guess_dict, i, 'minuit_numeric')
        elif optimizer == 'lbfgs':
            fit_and_dump(initial_guess, initial_guess_dict, i, 'lbfgs')
        
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    parser.add_argument("--optimizer", type=str, choices=['minuit_numeric', 'minuit_analytic', 'lbfgs'], default='minuit_numeric',
                      help="Optimization method to use")
    parser.add_argument("--bins", type=int, nargs="+")
    args = parser.parse_args()

    timer = Timer()
    if args.bins is None:
        raise ValueError("list of bin indices is required")
    
    
    for bins in args.bins:
        run_fits_in_bin(bins, args.optimizer)
        
    print(f"Total time elapsed: {timer.read()[2]}")

    # MPI.Finalize()
    sys.exit(0)
