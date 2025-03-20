from iminuit import Minuit
from scipy.optimize import minimize
import scipy.linalg as linalg

import jax.numpy as jnp
import numpy as np

from iftpwa1.pwa.gluex.constants import (
    LIKELIHOOD,
    GRAD,
    HESSIAN,
    INTENSITY_AC,
    NORMINT,
    AMPINT
)
from pyamptools.utility.general import converter

class Objective:

    """
    This is our objective function with some expected functions for optimization frameworks 
        that utilizes GlueXJaxManager to obtain NLL ands its derivatives
        
    NOTE: Everything in sendAndReceive accesses the first element [0] of the tuple since this class
            only deals with a single bin. sendAndReceive normally returns a list of arrays, one for each kinematic bin
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
                if x[2*ref_idx+1] != 0:
                    raise ValueError(f"Reference wave {self.pwa_manager.waveNames[ref_idx]} has non-zero imaginary part: {x[2*ref_idx+1]}")
        
    def insert_into_full_array(self, x=None):
        x_full = jnp.zeros((self.nPars, self.nmbMasses, self.nmbTprimes))
        if x is None: return x_full
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

    def hessian(self, x):
        """Return the Hessian matrix at point x"""
        self.check_reference_wave_constraints(x)
        x_full = self.insert_into_full_array(x)
        hess = self.pwa_manager.sendAndReceive(x_full, HESSIAN)[0]
        
        # Handle reference waves by zeroing out corresponding rows/columns
        # THINK: is this correct? Can we just zero it? If it was indeed fixed then all 2nd order partials should equal 0?
        if self.ref_indices:
            for ref_idx in self.ref_indices:
                hess = hess.at[2*ref_idx+1, :].set(0.0)
                hess = hess.at[:, 2*ref_idx+1].set(0.0)
                hess = hess.at[2*ref_idx+1, 2*ref_idx+1].set(1.0) # Set diagonal element to 1 to avoid singularity
        
        return hess

    def hessp(self, x, p):
        """Compute the Hessian-vector product at point x with vector p"""
        hess = self.hessian(x)
        
        if self.ref_indices:
            for ref_idx in self.ref_indices:
                if isinstance(p, jnp.ndarray): p = p.at[2*ref_idx+1].set(0.0)
                else: p[2*ref_idx+1] = 0.0
                
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
        
    # # ORIGINAL IMPLEMENTATION, use pwa manager
    # def intensity(self, x, suffix=None):
    #     self.check_reference_wave_constraints(x)
    #     x_full = self.insert_into_full_array(x)
    #     return self.pwa_manager.sendAndReceive(x_full, INTENSITY_AC, suffix=suffix)[0].item().real
    
    def intensity_and_error(self, x, x_cov, wave_list, acceptance_correct=False):
         # intMatrix has shape (2 * nmbAmps, 2 * nmbAmps) where 2 is due to the doubling of sectors from "PosRe, NegIm, PosIm, NegRe"
         # intMatrixTerms has shape (2 * nmbAmps) of strings
        intMatrix, intMatrixTerms = self.ampInt() if acceptance_correct else self.normInt()
        intensities = []
        intensity_errors = []
        for ireaction in range(intMatrix.shape[0]):
            intensity, intensity_error = calculate_intensity_and_error(x, x_cov, intMatrix[ireaction], intMatrixTerms, wave_list, self.pwa_manager.waveNames)
            intensities.append(intensity)
            intensity_errors.append(intensity_error**2)
        return np.sum(intensities), np.sqrt(np.sum(intensity_errors))
    
    def normInt(self):
        x_full = self.insert_into_full_array() # if nothing passed, returns array of zeros (we are just getting the matrix, no calculations)
        return self.pwa_manager.sendAndReceive(x_full, NORMINT, suffix=None)[0] # (normInt, terms)
    
    def ampInt(self):
        x_full = self.insert_into_full_array()
        return self.pwa_manager.sendAndReceive(x_full, AMPINT, suffix=None)[0] # (ampInt, terms)
    
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

def regularize_hessian_for_covariance(hessian_matrix, tikhonov_delta=1e-4, ref_indices=None):
    """
    Compute regularized covariance matrices from a Hessian using two approaches.
    
    Args:
        hessian_matrix: The Hessian matrix to invert
        tikhonov_delta: Delta parameter for Tikhonov regularization to avoid singularity
    
    Returns:
        tuple: (cov_clipping, cov_tikhonov, eigenvalues, hessian_diagnostics)
    """
    
    min_eigenvalue=1e-6 # eigenvalue threshold to determine if it is "small"
    
    # Convert to numpy if it's a JAX array
    if hasattr(hessian_matrix, 'copy_to_host'):
        hessian_matrix = np.array(hessian_matrix)
    
    try:
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(hessian_matrix)
        
        if np.all(eigenvalues > 0):
            covariance = np.linalg.inv(hessian_matrix) * 2
        else:
            # NOTE: Invert the hessian to get Covariance then multiply by 2 so user can simply sqrt(covariance) to get standard errors
            #       This is because our objective is NLL instead of 2NLL
            
            # Method 1: Eigenvalue clipping - should perform better if only a small number of negative eigenvalues
            # clipped_eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
            # precision_clipped = eigenvectors @ np.diag(clipped_eigenvalues) @ eigenvectors.T
            # cov_clipping = np.linalg.inv(precision_clipped) * 2
            
            # Method 2: Tikhonov regularization - better than clipping if more small to negative eigenvalues
            # Add a small positive value to the diagonal
            non_fixed_eigenvalues = eigenvalues[np.abs(eigenvalues - 1) > 1e-10] # We set hessian elements to 1 for fixed parameters (see Objective)
            lambda_max = np.max(non_fixed_eigenvalues) if len(non_fixed_eigenvalues) > 0 else 1.0
            regularization_amount = tikhonov_delta * lambda_max # scale invariant regularization (depends on the largest eigenvalue)
            precision_tikhonov = hessian_matrix + np.eye(hessian_matrix.shape[0]) * regularization_amount
            covariance = np.linalg.inv(precision_tikhonov) * 2

        # Zero out fixed parameter (co)variances for reference waves (some fixed parameters)
        if ref_indices:
            for ref_idx in ref_indices:
                covariance[2*ref_idx+1, :] = 0.0
                covariance[:, 2*ref_idx+1] = 0.0
                covariance[2*ref_idx+1, 2*ref_idx+1] = 0.0
        
        # Diagnostics
        hessian_diagnostics = {
            'smallest_eigenvalue': float(np.min(eigenvalues)),
            'largest_eigenvalue': float(np.max(eigenvalues)),
            'condition_number': float(np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))),
            'fraction_negative_eigenvalues': np.sum(eigenvalues < 0) / len(eigenvalues),
            'fraction_small_eigenvalues':   np.sum((eigenvalues > 0) & (eigenvalues < min_eigenvalue)) / len(eigenvalues),
        }
        
        return covariance, eigenvalues, hessian_diagnostics
        
    except np.linalg.LinAlgError as e:
        print(f"Error calculating covariance matrix: {e}")
        return None, None, None, {'error': str(e)}
    
def calculate_intensity_and_error(amp, cov_amp, ampMatrix, ampMatrixTermOrder, wave_list, all_waves_list, scale_factors=None):
    """
    Calculate the intensity and its error based on amplitude values and covariance matrix.
    
    Parameters:
    - amp: flattened array of real and imaginary parts of amplitudes, shape (n_waves * 2,)
    - cov_amp: covariance matrix for the amplitude parameters, shape (n_waves * 2, n_waves * 2)
    - ampMatrix: normalization integral matrix, shape (2 * n_waves, 2 * n_waves). This 2 is due to the doubling of sectors from "PosRe, NegIm, PosIm, NegRe" not the complex parts!
    - ampMatrixTermOrder: list of keys for ampMatrix, length = 2 * n_waves
    - wave_list: list of wave names to calculate intensity for, length of list <= n_waves
    - all_waves_list: list of all wave names, length = n_waves. This must be in the order of the flattened 'amp' array
    - scale_factors: optional dictionary mapping wave names to scale factors (defaults to 1)
    
    Returns:
    - intensity: the calculated intensity
    - error: the error on the intensity
    """

    # We need to sum over both incoherent sectors (PosIm, NegRe) or (NegIm, PosRe) for each amplitude
    intMatrix_sum_amps_list = ampMatrixTermOrder # ~ ['RealNeg.Dm2-', 'ImagPos.Dm2-', 'RealNeg.Dm1-', 'ImagPos.Dm1-', 'RealNeg.Dp0-', 'ImagPos.Dp0-']
    intMatrix_pairs_dict = {} # {Dm2-: [0, 1], Dm1-: [2, 3], Dp0-: [4, 5]}
    for i, sum_amp in enumerate(intMatrix_sum_amps_list):
        _sum, _amp = sum_amp.split('.')
        if _amp not in intMatrix_pairs_dict:
            intMatrix_pairs_dict[_amp] = []
        intMatrix_pairs_dict[_amp].append(i)
    sectors_per_amp = list(set([len(intMatrix_pairs_dict[_]) for _ in intMatrix_pairs_dict]))
    
    #### Tons of shape checks ####
    for wave in wave_list:
        if wave not in all_waves_list:
            raise ValueError(f"Wave {wave} not found in all_waves_list (which should be maintained by the PWA manager)")
    if cov_amp.shape[0] != len(amp) or cov_amp.shape[1] != len(amp):
        raise ValueError(f"Covariance matrix does not match length of 'amp' array: {len(amp)} != {cov_amp.shape}")
    if ampMatrix.shape[0] != 2 * len(all_waves_list) or ampMatrix.shape[1] != 2 * len(all_waves_list):
        raise ValueError(f"Normalization integral matrix does not match length of 'all_waves_list': {2 * len(all_waves_list)} != {ampMatrix.shape}")
    if len(all_waves_list) != len(amp) // 2:
        raise ValueError(f"'all_waves_list' should have the 2x length of 'amp' array: {len(all_waves_list)} != {len(amp) // 2}")
    if len(sectors_per_amp) != 1:
        raise ValueError(f"All amplitudes should have the same number of incoherent sectors: {intMatrix_pairs_dict}")    
    if sectors_per_amp[0] > 2:  # 1 sector if polarization magnitude is 1.0 else 2 sectors (PosIm, NegRe) and (NegIm, PosRe)
        raise ValueError(f"Each amplitude is expected to have less than 2 incoherent sectors: {intMatrix_pairs_dict}")
        
    # TODO: Implement this properly but just a dictionary of 1s for now, only needed for intensity calculation
    if scale_factors is None:
        scale_factors = {wave: 1.0 for wave in all_waves_list}
    
    # Initialize intensity and derivatives vector    
    #   For now, we only consider derivatives with respect to real and imaginary parts
    #   When we add scale factors, we need to extend this to 3*n_waves instead of amps which is 2*n_waves
    deriv = np.zeros_like(amp)
    intensity = 0.0
    
    # Calculate intensity and derivatives
    for wave_i in wave_list:
        i = all_waves_list.index(wave_i)
        i_re = 2 * i
        i_im = 2 * i + 1
        scale_i = scale_factors[wave_i]
        
        for wave_j in wave_list:
            j = all_waves_list.index(wave_j)
            j_re = 2 * j
            j_im = 2 * j + 1
            scale_j = scale_factors[wave_j]
            
            # Real and imaginary parts of the amplitudes
            a_re = amp[i_re]
            a_im = amp[i_im]
            b_re = amp[j_re]
            b_im = amp[j_im]
                        
            # Each amplitude can have 2 incoherent sectors (PosIm, NegRe) or (NegIm, PosRe), loop over all of them here
            #   Integral matrix will have 0s when they are in difference sectors
            for matrix_element_i in intMatrix_pairs_dict[wave_i]:
                for matrix_element_j in intMatrix_pairs_dict[wave_j]:
                    amp_int = ampMatrix[matrix_element_i, matrix_element_j] # generally complex valued
                    if np.isclose(np.abs(amp_int), 0.0):
                        continue
            
                    # Contribution to intensity
                    intensity_contrib = scale_i * scale_j * (
                        (a_re * b_re + a_im * b_im) * np.real(amp_int) - 
                        (a_im * b_re - a_re * b_im) * np.imag(amp_int)
                    )
                    
                    intensity += intensity_contrib
                    
                    # Derivatives with respect to real and imaginary parts
                    deriv[i_re] += 2 * scale_i * scale_j * (b_re * np.real(amp_int) + b_im * np.imag(amp_int))
                    deriv[i_im] += 2 * scale_i * scale_j * (b_im * np.real(amp_int) - b_re * np.imag(amp_int))
                    
                    # NOTE: If scale factors were variable parameters, we would add derivatives with respect to them here
    
    # Calculate variance using the derivatives and covariance matrix
    variance = 0.0
    for i in range(len(deriv)):
        for j in range(len(deriv)):
            variance += deriv[i] * deriv[j] * cov_amp[i, j]
    
    error = np.sqrt(variance)
    
    return intensity, error

def calculate_relative_phase_and_error(amp, cov_amp, wave1, wave2, all_waves_list):
    """
    Calculate the relative phases between two amplitudes and its error.
    
    Parameters:
    - amp: flattened array of real and imaginary parts of amplitudes, shape (n_waves * 2,)
    - cov_amp: covariance matrix for the amplitude parameters, shape (n_waves * 2, n_waves * 2)
    - wave1: name of the first wave, i.e. Dp2+
    - wave2: name of the second wave, i.e. Sp0+
    - all_waves_list: list of all wave names, length n_waves. This must be in the order of the flattened 'amp' array
    
    Returns:
    - phase_diff: relative phases in radians
    - error: error on the relative phases
    """

    if wave1 not in all_waves_list:
        raise ValueError(f"Wave {wave1} not found in all_waves_list (which should be maintained by the PWA manager)")
    if wave2 not in all_waves_list:
        raise ValueError(f"Wave {wave2} not found in all_waves_list (which should be maintained by the PWA manager)")
    if cov_amp.shape[0] != len(amp) or cov_amp.shape[1] != len(amp):
        raise ValueError(f"Covariance matrix does not match length of 'amp' array: {len(amp)} != {cov_amp.shape}")
    if len(all_waves_list) != len(amp) // 2:
        raise ValueError(f"'all_waves_list' should have the 2x length of 'amp' array: {len(all_waves_list)} != {2 * len(amp)}")
    
    # Get indices of the waves
    idx1 = all_waves_list.index(wave1)
    idx2 = all_waves_list.index(wave2)
    
    # Get real and imaginary parts of the amplitudes
    a1_re = amp[2 * idx1]
    a1_im = amp[2 * idx1 + 1]
    a2_re = amp[2 * idx2]
    a2_im = amp[2 * idx2 + 1]
    a1_complex = amp[2 * idx1] + 1j * amp[2 * idx1 + 1]
    a2_complex = amp[2 * idx2] + 1j * amp[2 * idx2 + 1]
    a2_angle = np.angle(a2_complex)
    a1_complex *= np.exp(-1j * a2_angle)    
    phase_diff = np.angle(a1_complex)
    
    # Calculate derivatives of relative phases with respect to parameters
    p_deriv = np.zeros(4)
    p_deriv[0] = -a1_im / np.abs(a1_complex)**2  # d(phase)/d(a1_re)
    p_deriv[1] =  a1_re / np.abs(a1_complex)**2  # d(phase)/d(a1_im)
    p_deriv[2] =  a2_im / np.abs(a2_complex)**2  # d(phase)/d(a2_re)
    p_deriv[3] = -a2_re / np.abs(a2_complex)**2  # d(phase)/d(a2_im)
    
    # Get indices in the covariance matrix
    idx = [2 * idx1, 2 * idx1 + 1, 2 * idx2, 2 * idx2 + 1]
    
    # Calculate variance
    variance = 0.0
    for i in range(4):
        for j in range(4):
            variance += p_deriv[i] * p_deriv[j] * cov_amp[idx[i], idx[j]]
    
    error = np.sqrt(variance)
    
    return phase_diff, error

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
        bounds = [(None, None)] * len(x0) # unused, just to dump into results dictionary
        if objective.ref_indices:
            for ref_idx in objective.ref_indices:
                bounds[2*ref_idx+1] = (0, 0)
                m.fixed[param_names[2*ref_idx+1]] = True # fix imaginary part to zero

    # Configure Minuit
    # AmpTools uses 0.001 * tol * UP (default 0.1) - diff is probably due to age of software?
    # iminuit uses  0.002 * tol * errordef (default 0.1)
    m.tol = 0.05 # of 0.05 to match implementation in AmpTools?
    m.strategy = 1  # More accurate minimization
    m.migrad()  # Run minimization
    
    # Get covariance from Minuit, always dump it even if minimization failed
    minuit_covariance = m.covariance
    
    # Calculate covariance from objective's Hessian for comparison
    hessian_matrix = objective.hessian(m.values)
    covariance, eigenvalues, hessian_diagnostics = regularize_hessian_for_covariance(hessian_matrix, ref_indices=objective.ref_indices)
    
    return {
        'parameters': m.values,
        'likelihood': m.fval,
        'success': m.valid,
        'message': 'Valid minimum found' if m.valid else 'Minimization failed',
        'covariance': {
            'optimizer': minuit_covariance,
            'tikhonov': covariance
        },
        'eigenvalues': eigenvalues.tolist() if eigenvalues is not None else None,
        'hessian_diagnostics': hessian_diagnostics,
        'bounds': bounds,
    }
    
def optimize_single_bin_scipy(objective, initial_params, bin_idx, method='L-BFGS-B', options=None):
    """
    Optimize parameters for a single kinematic bin using scipy.optimize methods.
    
    Args:
        objective: Objective instance that calculates likelihood and gradients
        initial_params: Initial parameters for optimization with shape (nPars)
        bin_idx: Index of bin being optimized
        method: Optimization method to use (default: 'L-BFGS-B')
        options: Dictionary of options for the method
        
    Returns:
        dict: Dictionary containing optimization results
    """
    
    x0 = initial_params.copy()
    objective.check_reference_wave_constraints(x0)
    
    # Set up fixed parameters for reference waves
    if objective.ref_indices:
        bounds = [(None, None)] * len(x0) # unused, just to dump into results dictionary
        for ref_idx in objective.ref_indices:
            bounds[2*ref_idx+1] = (0.0, 0.0)  # Fix imaginary part of reference wave to 0
    
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
        
    # Always calculate Hessian and covariance at the solution
    hessian_matrix = objective.hessian(result.x)
    covariance, eigenvalues, hessian_diagnostics = regularize_hessian_for_covariance(hessian_matrix, ref_indices=objective.ref_indices)
    
    # For comparison, try to get scipy's approximated inverse Hessian if available, always dump it if possible
    scipy_covariance = None
    if hasattr(result, 'hess_inv'):
        scipy_covariance = result.hess_inv
        
    return {
        'parameters': result.x,
        'likelihood': result.fun,
        'success': result.success,
        'message': result.message,
        'covariance': {
            'optimizer': scipy_covariance,
            'tikhonov': covariance
        },
        'eigenvalues': eigenvalues.tolist() if eigenvalues is not None else None,
        'hessian_diagnostics': hessian_diagnostics,
        'bounds': bounds
    }