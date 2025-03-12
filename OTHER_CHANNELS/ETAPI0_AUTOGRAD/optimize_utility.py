from iminuit import Minuit
from scipy.optimize import minimize

import jax.numpy as jnp
import numpy as np

from iftpwa1.pwa.gluex.constants import (
    LIKELIHOOD,
    LIKELIHOOD,
    GRAD,
    HESSIAN,
    INTENSITY_AC,
)
from pyamptools.utility.general import identify_channel, converter

class Objective:

    """
    This is our objective function with some expected functions for optimization frameworks 
        that utilizes GlueXJaxManager to obtain NLL ands its derivatives
    """
    
    def __init__(self, pwa_manager, bin_idx, nPars, nmbMasses, nmbTprimes, reference_waves=None):
        self.pwa_manager = pwa_manager
        self.bin_idx = bin_idx
        self.nPars = nPars
        self.nmbMasses = nmbMasses
        self.nmbTprimes = nmbTprimes
        self.mbin = bin_idx % nmbMasses
        self.tbin = bin_idx // nmbMasses
        
        # flag user can modify to affect whether __call__ returns gradients also
        self._deriv_order = 0
        
        # Reference wave handling
        self.reference_waves = reference_waves
        self.ref_indices = None
        self.refl_sectors = None
        if reference_waves:
            self._process_reference_waves()
        
    def check_reference_wave_constraints(self, x):
        """Check reference wave constraints and raise error if violated"""
        if self.ref_indices:
            for ref_idx in self.ref_indices:
                if x[2*ref_idx] < 0:
                    raise ValueError(f"Reference wave {self.pwa_manager.waveNames[ref_idx]} has negative real part: {x[2*ref_idx]}")
                if x[2*ref_idx+1] != 0:
                    raise ValueError(f"Reference wave {self.pwa_manager.waveNames[ref_idx]} has non-zero imaginary part: {x[2*ref_idx+1]}")
        
    def insert_into_full_array(self, x):
        x_full = jnp.zeros((self.nPars, self.nmbMasses, self.nmbTprimes))
        x_full = x_full.at[:, self.mbin, self.tbin].set(x)
        return x_full

    def objective(self, x):
        """Return only the objective value"""
        self.check_reference_wave_constraints(x)
        x_full = self.insert_into_full_array(x)
        _nll = self.pwa_manager.sendAndReceive(x_full, LIKELIHOOD)[0]
        return _nll
    
    def gradient(self, x):
        """Return only the gradient"""
        self.check_reference_wave_constraints(x)
        x_full = self.insert_into_full_array(x)
        grad = self.pwa_manager.sendAndReceive(x_full, GRAD)[0]
        
        # Zero out gradient for fixed parameters (imaginary parts of reference waves)
        if self.ref_indices:
            for ref_idx in self.ref_indices:
                grad = grad.at[2*ref_idx+1].set(0.0)
                
        return grad

    def hessp(self, x, p):
        """Compute the Hessian-vector product at point x with vector p"""
        self.check_reference_wave_constraints(x)
        x_full = self.insert_into_full_array(x)
        hess = self.pwa_manager.sendAndReceive(x_full, HESSIAN)[0]
        
        if self.ref_indices:
            for ref_idx in self.ref_indices:
                if isinstance(p, jnp.ndarray): p = p.at[2*ref_idx+1].set(0.0)
                else: p[2*ref_idx+1] = 0.0
                hess = hess.at[2*ref_idx+1, :].set(0.0)
                hess = hess.at[:, 2*ref_idx+1].set(0.0)
        return jnp.dot(hess, p)

    def __call__(self, x):
        """Return both objective value and gradient for scipy.optimize"""
        self.check_reference_wave_constraints(x)
        nll = self.objective(x)
        if self._deriv_order == 0:
            return nll
        elif self._deriv_order == 1:
            grad = self.gradient(x)
            return nll, grad
        else:
            raise ValueError(f"Invalid derivative order: {self._deriv_order}")
        
    def intensity(self, x, suffix=None):
        self.check_reference_wave_constraints(x)
        x_full = self.insert_into_full_array(x)
        return self.pwa_manager.sendAndReceive(x_full, INTENSITY_AC, suffix=suffix)[0].item().real
    
    def _process_reference_waves(self):
        """Process reference waves to determine their indices and reflectivity sectors"""
        ref_indices = []
        refl_sectors = {}
        
        if self.reference_waves:
            if isinstance(self.reference_waves, str):
                self.reference_waves = [self.reference_waves]
            
            # Get reflectivity sectors and their waves
            for i, wave in enumerate(self.pwa_manager.waveNames):
                # Extract reflectivity from wave name
                refl = converter[wave][0]  # First element is reflectivity (e)
                if refl not in refl_sectors:
                    refl_sectors[refl] = []
                refl_sectors[refl].append((i, wave))
            
            # Process each reference wave
            for ref_wave in self.reference_waves:
                if ref_wave in self.pwa_manager.waveNames:
                    ref_idx = self.pwa_manager.waveNames.index(ref_wave)
                    ref_indices.append(ref_idx)
                else:
                    raise ValueError(f"Reference wave '{ref_wave}' not found in wave list!")
            
            # Check if we have at least one reference wave per reflectivity sector
            for refl, waves in refl_sectors.items():
                wave_indices = [idx for idx, _ in waves]
                if not any(idx in ref_indices for idx in wave_indices):
                    raise ValueError(f"No reference wave specified for reflectivity sector = {refl}!")
        
        self.ref_indices = ref_indices
        self.refl_sectors = refl_sectors

def optimize_single_bin_minuit(objective, initial_params, bin_idx, use_analytic_grad=True):
    """
    Optimize parameters for a single kinematic bin using Minuit.
    
    Args:
        objective: Objective instance that calculates likelihood and gradients
        initial_params: Initial parameters for optimization with shape (nPars)
        bin_idx: Index of bin being optimized
        use_analytic_grad: If True, use analytic gradients from sendAndReceive
        
    Returns:
        dict: Dictionary containing optimization results
    """
    
    # Extract initial parameters for this bin
    x0 = initial_params.copy()
    objective.check_reference_wave_constraints(x0)

    # Initialize parameter names for Minuit
    param_names = [f'x{i}' for i in range(len(x0))]
    
    objective._deriv_order = 0 # objective.__call__ is not used so no need to set deriv_order
    m = Minuit(
        objective.objective,
        x0,
        grad=objective.gradient if use_analytic_grad else False,
        name=param_names,
    )
    
    # Set up fixed parameters for reference waves
    if objective.ref_indices:
        for ref_idx in objective.ref_indices:
            m.fixed[param_names[2*ref_idx+1]] = True        # fix imaginary part to zero
            m.limits[param_names[2*ref_idx]] = (0, None)    # restrict real part to be positive
            
    print(f"m.fixed: {m.fixed}")
    print(f"m.limits: {m.limits}")

    # Configure Minuit
    # AmpTools uses 0.001 * tol * UP (default 0.1) - diff is probably due to age of software?
    # iminuit uses  0.002 * tol * errordef (default 0.1)
    m.tol = 0.05 # of 0.05 to match implementation in AmpTools?
    m.strategy = 1  # More accurate minimization
    m.migrad()  # Run minimization
    
    return {
        'parameters': m.values,
        'likelihood': m.fval,
        'success': m.valid,
        'message': 'Valid minimum found' if m.valid else 'Minimization failed',
        'errors': m.errors
    }
    
def optimize_single_bin_scipy(objective, initial_params, bin_idx, method='L-BFGS-B', bounds=None, options=None):
    """
    Optimize parameters for a single kinematic bin using scipy.optimize methods.
    
    Args:
        objective: Objective instance that calculates likelihood and gradients
        initial_params: Initial parameters for optimization with shape (nPars)
        bin_idx: Index of bin being optimized
        method: Optimization method to use (default: 'L-BFGS-B')
        bounds: Parameter bounds for constrained optimization (optional)
        options: Dictionary of options for the method
        
    Returns:
        dict: Dictionary containing optimization results
    """
    
    x0 = initial_params.copy()
    objective.check_reference_wave_constraints(x0)
    
    # Handle reference wave constraints using parameter bounds
    if objective.ref_indices:
        if bounds is None: # unconstrained bounds if none provided
            bounds = [(None, None)] * len(x0)
        else:
            bounds = list(bounds)  # Make sure bounds is a list we can modify
            
        for ref_idx in objective.ref_indices:
            bounds[2*ref_idx+1] = (0.0, 0.0)  # Fix imaginary part of reference wave to 0
            bounds[2*ref_idx] = (0.0, None)   # Restrict real part to be positive
    
    if options is None:
        if method == "L-BFGS-B":
            options = {
                'maxiter': 2000,       # Maximum number of iterations
                'maxfun': 20000,       # Maximum number of function evaluations
                'ftol': 1e-10,         # Function value tolerance
                'gtol': 1e-10,         # Gradient norm tolerance
                'maxcor': 10           # Number of stored corrections
            }
        elif method == "trust-ncg":
            options = {
                'initial_trust_radius': 1.0,  # Starting trust-region radius
                'max_trust_radius': 1000.0,   # Maximum trust-region radius
                'eta': 0.15,                  # Acceptance stringency for proposed steps
                'gtol': 1e-8,                 # Gradient norm tolerance
                'maxiter': 2000               # Maximum number of iterations
            }
        elif method == "trust-krylov":
            options = {
                'inexact': False,  # Solve subproblems with high accuracy
                'gtol': 1e-8,      # Gradient norm tolerance
                'maxiter': 2000    # Maximum number of iterations
            }
        else:
            raise ValueError(f"Invalid method: {method}")
        
    if method == 'trust-ncg' or method == 'trust-krylov':
        objective._deriv_order = 0 # objective.__call__ is not used so no need to set deriv_order
        result = minimize(
            objective.objective,
            x0,
            method=method,
            jac=objective.gradient,
            hessp=objective.hessp,
            bounds=bounds,
            options=options
        )
    elif method == 'L-BFGS-B':
        objective._deriv_order = 1 # since jac=True we need the function to return both the objective and the gradient
        result = minimize(
            objective,
            x0,
            method=method,
            jac=True,
            bounds=bounds,
            options=options
        )
    else:
        raise ValueError(f"Invalid method: {method}")
        
    return {
        'parameters': result.x,
        'likelihood': result.fun,
        'success': result.success,
        'message': result.message,
        'errors': None  # Scipy doesn't provide error estimates by default
    }