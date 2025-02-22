from pyamptools.utility.general import load_yaml, Timer
import numpy as np
from mpi4py import MPI
import sys
from scipy.optimize import minimize
import os
import shutil
import pickle as pkl

from iftpwa1.pwa.gluex.gluex_jax_manager import (
    GluexJaxManager,
    LIKELIHOOD,
    LIKELIHOOD_AND_GRAD,
    LIKELIHOOD_GRAD_HESSIAN,
    INTENSITY_AC,
)

comm0 = MPI.COMM_WORLD
rank = comm0.Get_rank()
np.random.seed(42)
bin_idx = 0
scale = 50
n_iterations = 1
pyamptools_yaml_file = "/w/halld-scshelf2101/lng/WORK/PyAmpTools9/OTHER_CHANNELS/ETAPI0_AUTOGRAD/pyamptools.yaml"
iftpwa_yaml_file = "/w/halld-scshelf2101/lng/WORK/PyAmpTools9/OTHER_CHANNELS/ETAPI0_AUTOGRAD/iftpwa.yaml"

def optimize_single_bin(pwa_manager, initial_params, bin_idx, bounds=None, method='L-BFGS-B', options=None):
    """
    Optimize parameters for a single kinematic bin using L-BFGS-B or other methods.
    
    Args:
        pwa_manager: GlueXJaxManager instance that calculates likelihood and gradients
        initial_params: Initial parameters for optimization with shape (nPars, nmbMasses, nmbTprimes)
        bin_idx: Index of bin to optimize
        bounds: Parameter bounds for constrained optimization (optional)
        method: Optimization method to use (default: 'L-BFGS-B')
        options: Dictionary of options for the optimizer
        
    Returns:
        dict: Dictionary containing:
            - 'parameters': Optimized parameters for the bin
            - 'likelihood': Final likelihood value
            - 'success': Boolean indicating convergence
            - 'message': Status message from optimizer
    """
    
    nPars = pwa_manager.nPars
    nmbMasses = pwa_manager.nmbMasses
    nmbTprimes = pwa_manager.nmbTprimes
    
    # Calculate mass and t bin indices
    mbin = bin_idx % nmbMasses
    tbin = bin_idx // nmbMasses
    
    # Extract initial parameters for this bin
    x0 = initial_params[:, mbin, tbin]
    
    # Default options if none provided
    if options is None:
        options = {
            'maxiter': 1000,
            'ftol': 1e-8,
            'gtol': 1e-8
        }
    
    def objective_function(params):
        """Wrapper for likelihood calculation that returns scalar and gradient for specific bin"""
        # Reshape params to full array with zeros except for current bin
        full_params = np.zeros((nPars, nmbMasses, nmbTprimes))
        full_params[:, mbin, tbin] = params
        
        # Get likelihood and gradient using sendAndReceive
        results = pwa_manager.sendAndReceive(full_params, LIKELIHOOD_AND_GRAD)
        nll, nll_grad = results[0]  # Only one result since we set single bin
        
        # Negative because we want to maximize likelihood
        return nll, nll_grad
    
    # Apply optimization
    result = minimize(
        objective_function,
        x0,
        method=method,
        jac=True,  # Gradient is returned by objective function
        bounds=bounds,
        options=options
    )
    
    return {
        'parameters': result.x,
        'likelihood': result.fun,  # Convert back to positive likelihood
        'success': result.success,
        'message': result.message
    }

def main():
    mpi_offset = 1
    
    global iftpwa_yaml_file
    
    # Only leader process needs to load the yaml file
    iftpwa_yaml = load_yaml(iftpwa_yaml_file)
    pwa_manager = GluexJaxManager(comm0=comm0, mpi_offset=mpi_offset,
                                yaml_file=pyamptools_yaml_file,
                                resolved_secondary=iftpwa_yaml, prior_simulation=False, sum_returned_nlls=False)
    result = None
        
    if rank == 0: # Only leader process calculates likelihood
        
        def fit_and_dump(initial_guess, initial_guess_dict, iteration):
            
            initial_likelihood = pwa_manager.sendAndReceive(initial_guess, LIKELIHOOD)[0].item()
            print("\n**************************************************************")
            print(f"Initial likelihood: {initial_likelihood}")
                
            result = optimize_single_bin(
                pwa_manager,
                initial_guess,
                bin_idx,
                bounds=None,
                options={'maxiter': 2000, 'ftol': 1e-10}
            )
        
            # pwa_manager expects 3D array of parameters, but since only care about 1 bin then we 
            #   can just fill the bin we want and leave the rest as 0s
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
        
            print(f"Intensity for bin {bin_idx}: {intensities}")
            print(f"Optimization results for bin {bin_idx}:")
            print(f"Success: {result['success']}")
            print(f"Final likelihood: {result['likelihood']}")
            print(f"Message: {result['message']}")
            print("**************************************************************\n")
            
            with open(f"COMPARISONS/lbfgs_bin{bin_idx}_{iteration}.pkl", "wb") as f:
                pkl.dump({'initial_guess_dict': initial_guess_dict, 'intensities': intensities}, f)
        
        nPars = pwa_manager.nmbFitParametersSingle
        nmbMasses = pwa_manager.nmbMasses
        nmbTprimes = pwa_manager.nmbTprimes
        
        # bin information
        mbin = bin_idx % nmbMasses
        tbin = bin_idx // nmbMasses
        
        # Set bin for likelihood calculation
        pwa_manager.set_bins(np.array([bin_idx]))
            
        for i in range(n_iterations):
            initial_guess = scale * np.random.randn(nPars, nmbMasses, nmbTprimes)
            initial_guess = np.ones_like(initial_guess)
            initial_guess_dict = {} 
            for iw, wave in enumerate(pwa_manager.waveNames):
                initial_guess_dict[wave] = initial_guess[2*iw, mbin, tbin] + 1j * initial_guess[2*iw+1, mbin, tbin]
            fit_and_dump(initial_guess, initial_guess_dict, i)

    # Stop all daemon processes
    pwa_manager.stop()
    
    # Clean exit
    MPI.Finalize()

if __name__ == "__main__":
    timer = Timer()
    result = main()
    if rank == 0:
        print(f"Time taken: {timer.read()[2]}")
    sys.exit(0)
    