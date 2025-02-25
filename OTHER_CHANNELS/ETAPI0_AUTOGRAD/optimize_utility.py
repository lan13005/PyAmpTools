from iminuit import Minuit
from scipy.optimize import minimize

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