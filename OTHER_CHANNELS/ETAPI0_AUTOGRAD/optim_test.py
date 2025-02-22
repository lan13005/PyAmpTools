from pyamptools.utility.general import load_yaml, Timer
import numpy as np
from mpi4py import MPI
import sys
import os
import shutil
import pickle as pkl
from iminuit import Minuit

from iftpwa1.pwa.gluex.gluex_jax_manager import (
    GluexJaxManager,
    LIKELIHOOD,
    LIKELIHOOD_AND_GRAD,
    LIKELIHOOD_GRAD_HESSIAN,
    INTENSITY_AC,
)

Minuit.errordef = Minuit.LIKELIHOOD

comm0 = MPI.COMM_WORLD
rank = comm0.Get_rank()
np.random.seed(42)
bin_idx = 1
scale = 50
n_iterations = 1
pyamptools_yaml_file = "/w/halld-scshelf2101/lng/WORK/PyAmpTools9/OTHER_CHANNELS/ETAPI0_AUTOGRAD/pyamptools.yaml"
iftpwa_yaml_file = "/w/halld-scshelf2101/lng/WORK/PyAmpTools9/OTHER_CHANNELS/ETAPI0_AUTOGRAD/iftpwa.yaml"

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
    
    def _compute(self, x):
        """Compute both value and gradient if needed"""
        if self._last_x is None or not np.array_equal(x, self._last_x):
            self._last_x = np.array(x)
            
            x_full = np.zeros((self.nPars, self.nmbMasses, self.nmbTprimes))
            x_full[:, self.mbin, self.tbin] = x
            self._last_val, self._last_grad = self.pwa_manager.sendAndReceive(x_full, LIKELIHOOD_AND_GRAD)[0]
            self._last_val  = 2 * self._last_val.item()
            self._last_grad = 2 * np.array(self._last_grad)

    def objective(self, x):
        """Return only the objective value"""
        self._compute(x)
        return self._last_val
    
    def gradient(self, x):
        """Return only the gradient"""
        self._compute(x)
        return self._last_grad

def optimize_single_bin(pwa_manager, initial_params, bin_idx, use_analytic_grad=True):
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
        
        def fit_and_dump(initial_guess, initial_guess_dict, iteration, use_analytic_grad):
            
            initial_likelihood = pwa_manager.sendAndReceive(initial_guess, LIKELIHOOD)[0].item()
            initial_likelihood = 2 * initial_likelihood
            print("\n**************************************************************")
            print(f"Initial likelihood: {initial_likelihood}")
            print(f"Using analytic gradients: {use_analytic_grad}")
                
            result = optimize_single_bin(
                pwa_manager,
                initial_guess,
                bin_idx,
                use_analytic_grad=use_analytic_grad
            )
        
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
            # print(f"Parameter errors: {result['errors']}")
            print("**************************************************************\n")
            
            suffix = 'analytic' if use_analytic_grad else 'numeric'
            with open(f"COMPARISONS/minuit_{suffix}_bin{bin_idx}_{iteration}.pkl", "wb") as f:
                pkl.dump({
                    'initial_guess_dict': initial_guess_dict, 
                    'intensities': intensities,
                }, f)
        
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
            
            # Run with analytic gradients
            fit_and_dump(initial_guess, initial_guess_dict, i, use_analytic_grad=True)
            
            # Run with numeric gradients
            fit_and_dump(initial_guess, initial_guess_dict, i, use_analytic_grad=False)

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
    