from iminuit import Minuit
from scipy.optimize import minimize

import jax.numpy as jnp

from iftpwa1.pwa.gluex.constants import (
    LIKELIHOOD,
    LIKELIHOOD,
    GRAD,
    HESSIAN,
    INTENSITY_AC,
)

class Objective:

    """
    This is our objective function with some expected functions for optimization frameworks 
        that utilizes GlueXJaxManager to obtain NLL ands its derivatives
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
        
    def insert_into_full_array(self, x):
        x_full = jnp.zeros((self.nPars, self.nmbMasses, self.nmbTprimes))
        x_full = x_full.at[:, self.mbin, self.tbin].set(x)
        return x_full

    def objective(self, x):
        """Return only the objective value"""
        x_full = self.insert_into_full_array(x)
        _nll = self.pwa_manager.sendAndReceive(x_full, LIKELIHOOD)[0]
        return _nll
    
    def gradient(self, x):
        """Return only the gradient"""
        x_full = self.insert_into_full_array(x)
        grad = self.pwa_manager.sendAndReceive(x_full, GRAD)[0]
        return grad

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

def optimize_single_bin_minuit(objective, initial_params, bin_idx, use_analytic_grad=True):
    """
    Optimize parameters for a single kinematic bin using Minuit.
    
    Args:
        objective: Objective instance that calculates likelihood and gradients
        initial_params: Initial parameters for optimization with shape (nPars)
        bin_idx: Index of bin to optimize
        use_analytic_grad: If True, use analytic gradients from sendAndReceive
        
    Returns:
        dict: Dictionary containing optimization results
    """
    
    # Extract initial parameters for this bin
    x0 = initial_params

    # Initialize parameter names for Minuit
    param_names = [f'x{i}' for i in range(len(x0))]
    
    objective._deriv_order = 0 # objective.__call__ is not used so no need to set deriv_order
    m = Minuit(
        objective.objective,
        x0,
        grad=objective.gradient if use_analytic_grad else False,
        name=param_names,
    )

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
    
def optimize_single_bin_scipy(objective, initial_params, bin_idx, bounds=None, method='L-BFGS-B', options=None):
    """
    Optimize parameters for a single kinematic bin using scipy.optimize methods.
    
    Args:
        pwa_manager: GlueXJaxManager instance that calculates likelihood and gradients
        initial_params: Initial parameters for optimization with shape (nPars)
        bin_idx: Index of bin to optimize
        bounds: Parameter bounds for constrained optimization (optional)
        method: Optimization method to use (default: 'L-BFGS-B')
        options: Dictionary of options for the method
        
    Returns:
        dict: Dictionary containing optimization results
    """
    
    x0 = initial_params
    
    if options is None:
        options = {
            'maxiter': 2000,
            'ftol': 1e-10,
            'gtol': 1e-8
        }
    
    if method == 'trust-ncg' or method == 'trust-krylov':
        objective._deriv_order = 0 # objective.__call__ is not used so no need to set deriv_order
        result = minimize(
            objective.objective,
            x0,
            method=method,
            jac=objective.gradient,
            hessp=objective.hessp,
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