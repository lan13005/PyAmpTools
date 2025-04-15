import numpy as np
import time
from functools import partial
from rich.console import Console
import logging

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pyamptools.utility.clebsch import clebsch_gordan

from pyamptools.utility.MomentCalculatorTwoPS import AmplitudeSet, AmplitudeValue, QnWaveIndex
from iftpwa1.utilities.helpers import JaxLogger

n_k_values = 1

gbl_ctype = jnp.complex128
gbl_ftype = jnp.float64

console = Console()
        
_logger_level = logging.INFO # ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
_logger = logging.getLogger('pwa_manager')
_logger.setLevel(_logger_level)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter('%(asctime)s| %(message)s', datefmt='%H:%M:%S'))
_logger.addHandler(_handler)

# create compatible _logger for jax jit compiled code
_jax_logger = JaxLogger(_logger)

# TODO: Introduce masks to remove unused partial waves. i.e user only choose a subset of the full (l,m,e) up to l_max
#       jax should ignore the masked elements and not compute them in forward or backward

### NOTE: Most computation uses complex64, increase if we want more precision at cost of performance

@partial(jax.jit, static_argnames=['l_max'])
def build_reflectivity_amplitudes(flat_amplitudes, mask=None, l_max=3):
    """
    Reshape flattened real-valued amplitude array into sturctured complex amplitude array
    
    Args:
        flat_amplitudes: Flat JAX array of real and imaginary parts with gbl_ftype dtype
                        [Re(T_0), Im(T_0), Re(T_1), Im(T_1), ...]
                        NOTE: Order is [m cycles fastest, then l, then epsilon, then k]
        mask: JAX array of boolean or 0/1 values with same shape as flat_amplitudes
              When multiplied with flat_amplitudes, it zeros out specific elements
              without changing array dimensions, allowing JAX to optimize calculations
        l_max: Maximum orbital angular momentum (default: 3)
        
    Returns:
        Array of complex amplitudes [l]^{epsilon}_{m,k} with gbl_ctype dtype.
        Shape: [n_l_values, n_m_max, n_epsilon_values, n_k_values], where:
        - n_l_values = l_max + 1
        - n_m_max = 2 * l_max + 1 (m runs from -l to +l, offset by l_max in the array)
        - n_epsilon_values = 2 (epsilon = +1/-1, indexed as 0/1)
        - n_k_values = 1 (refers to spin-non-flip/spin-flip, we assume one is dominant)
    """
    
    # Apply mask if provided
    if mask is not None:
        flat_amplitudes = flat_amplitudes * mask
    
    # NOTE: This setup is kind of wasteful since we create a large array dim shape [n_l_values, n_m_max, n_epsilon_values, n_k_values]
    #       If we care about lmax=2 then there are 3 l-values, and 5 m-projections resulting in [3, 5] array (ignoring epsilon and k)
    #       Only l=2 can populate all 5 m-projections. The l=0 wave will only populate 1 m-proj leaving 4 0s.
    #       This setup forms like a triangular array of the form (x is non-zero)
    #       0 0 x
    #       0 x x 
    #       x x x
    #       0 x x
    #       0 0 x
    
    l_max = int(l_max)
    n_l_values =  l_max + 1
    n_m_max = 2 * l_max + 1
    n_eps = 2

    complex_amplitudes = flat_amplitudes[::2] + 1j * flat_amplitudes[1::2]
    total_m = sum(2 * l + 1 for l in range(n_l_values))
    
    # If this happens you might have set up your l_max wrong, check arg passing order
    assert complex_amplitudes.shape[0] == total_m * n_eps * n_k_values, f"complex_amplitudes.shape[0] = {complex_amplitudes.shape[0]} != total_m * n_eps * n_k_values = {total_m * n_eps * n_k_values}"

    T = jnp.zeros((n_l_values, n_m_max, n_eps, n_k_values), dtype=gbl_ctype)
    flat_idx = 0
    
    # Ordering: k (slowest) -> epsilon -> l -> m (fastest)
    for k in range(n_k_values):
        for eps in range(n_eps):
            for l in range(n_l_values):
                m_range = 2 * l + 1
                for m_offset in range(m_range):
                    m = m_offset - l   # Convert offset to actual m value (-l to +l)
                    m_idx = m + l_max  # Adjust to array index (0 to 2*l_max)
                    T = T.at[l, m_idx, eps, k].set(complex_amplitudes[flat_idx])
                    flat_idx += 1
                    
    return T

@partial(jax.jit, static_argnames=['l_max'])
def compute_spin_density_matrices_refl(T, l_max=3):
    """
    Compute the three spin-density matrices from the reflectivity-basis amplitudes using JAX.
    NOTE: The factor of epsilon is not included in this function and will be absorbed into the moment projection

    Args:
        T: Complex array of shape (l_max+1, 2*l_max+1, 2, n_k_values) with gbl_ctype dtype
           where T[l, m, epsilon, k], with m offset by +l_max
        l_max: Maximum orbital angular momentum

    Returns:
        Tuple of (rho0, rho1, rho2), each of shape (2, l_max+1, l_max+1, 2*l_max+1, 2*l_max+1)
        with gbl_ctype dtype, where:
        - First dimension (2): Epsilon values (+1/-1, indexed as 0/1)
        - Second dimension (l_max+1): l  index for first amplitude
        - Third dimension  (l_max+1): l' index for second amplitude
        - Fourth dimension (2*l_max+1): m  index for first amplitude, offset by l_max
        - Fifth dimension  (2*l_max+1): m' index for second amplitude, offset by l_max
        
        These moments have the following symmetry properties:
        - H0 (should have only real values)
        - H1 (should have only real values)
        - H2 (should have only imaginary values)
    """
    L = l_max + 1
    M = 2 * l_max + 1
    m_vals = jnp.arange(-l_max, l_max + 1)
    m, m_prime = m_vals[:, None], m_vals[None, :]

    phase_m = (-1.0) ** m[:, 0]                 # shape (M, )
    phase_m_prime = (-1.0) ** m_prime[0, :]     # shape (M, )
    phase_m_mprime = (-1.0) ** (m - m_prime)    # shape (M, M) - broadcasted
    
    phase_m = phase_m[None, None, None, :, None]                # shape (1,1,1,M,1)
    phase_m_prime = phase_m_prime[None, None, None, None, :]    # shape (1,1,1,1,M)
    phase_m_mprime = phase_m_mprime[None, None, None, :, :]     # shape (1,1,1,M,M)

    T_flipped = jnp.flip(T, axis=1)            # Flip m → -m
    T_conj = jnp.conj(T)
    T_conj_flipped = jnp.flip(T_conj, axis=1) # Flip m' → -m'

    # NOTE: SDMEs contracts index k and fixed on epsilon
    #       We also do not include the factor of epsilon in the returned rho matrices (reabsorbed into moment calculation)
    #       T   ~ [l, m, epsilon, k]
    #       rho ~ [epsilon, l, l', m, m']
    
    # Construct SDMEs
    term1_rho0 = jnp.einsum('lmek,inek->elimn', T, T_conj)
    term2_rho0 = jnp.einsum('lmek,inek->elimn', T_flipped, T_conj_flipped)
    rho0 = term1_rho0 + phase_m_mprime * term2_rho0
    
    term1_rho1 = jnp.einsum('lmek,inek->elimn', T_flipped, T_conj)
    term2_rho1 = jnp.einsum('lmek,inek->elimn', T, T_conj_flipped)
    rho1 = -1  * ((phase_m * term1_rho1) + (phase_m_prime * term2_rho1))
    
    rho2 = -1j * ((phase_m * term1_rho1) - (phase_m_prime * term2_rho1))

    # Check hermiticity of the spin density matrices
    def check_hermiticity(rho, name):
        # Swap l,l' and m,m' indices and take conjugate (Conjugate Transpose)
        rho_hermitian = jnp.conj(jnp.transpose(rho, axes=(0, 2, 1, 4, 3)))
        diff = rho - rho_hermitian
        max_diff = jnp.max(jnp.abs(diff))
        return max_diff

    # Apply hermiticity checks
    max_diff_rho0 = check_hermiticity(rho0, "rho0")
    max_diff_rho1 = check_hermiticity(rho1, "rho1")
    max_diff_rho2 = check_hermiticity(rho2, "rho2")
    
    # Check number of non-zero elements in rho0, rho1, rho2
    n_non_zero_rho0 = jnp.sum(jnp.abs(rho0) > 1e-6)
    n_non_zero_rho1 = jnp.sum(jnp.abs(rho1) > 1e-6)
    n_non_zero_rho2 = jnp.sum(jnp.abs(rho2) > 1e-6)
        
    _jax_logger.debug("rho0 has {}% non-zero elements", n_non_zero_rho0 / np.prod(rho0.shape) * 100)
    _jax_logger.debug("rho1 has {}% non-zero elements", n_non_zero_rho1 / np.prod(rho1.shape) * 100)
    _jax_logger.debug("rho2 has {}% non-zero elements", n_non_zero_rho2 / np.prod(rho2.shape) * 100)
    
    # Log overall result
    jax.lax.cond(
        (max_diff_rho0 > 1e-6) | (max_diff_rho1 > 1e-6) | (max_diff_rho2 > 1e-6),
        lambda _: _jax_logger.debug("Hermiticity check failed for SDMEs: max diff rho0={}, rho1={}, rho2={}", 
                                   max_diff_rho0, max_diff_rho1, max_diff_rho2),
        lambda _: _jax_logger.debug("Hermiticity check passed for all SDMEs: max diff rho0={}, rho1={}, rho2={}", 
                                  max_diff_rho0, max_diff_rho1, max_diff_rho2),
        operand=None
    )
    
    rho0 = jnp.array(rho0, dtype=gbl_ctype)
    rho1 = jnp.array(rho1, dtype=gbl_ctype)
    rho2 = jnp.array(rho2, dtype=gbl_ctype)

    return rho0, rho1, rho2

def precompute_cg_coefficients_by_LM(l_max, L_max=None):
    """Precompute and organize CG coefficients by (L,M) pairs for faster access"""
    if L_max is None:
        L_max = 2 * l_max
    
    # Dictionary to organize coefficients by (L,M)
    cg_by_LM = {}
    
    # Precompute coefficients and organize them
    # We wish to compute <l_prime, m_prime, L, M | l, m>
    #    We loop over L, M storing all non-zero coefficients
    for L in range(L_max + 1):
        for M in range(-L, L + 1):
            entries = []
            for l in range(l_max + 1):
                for m in range(-l, l + 1):
                    for l_prime in range(l_max + 1):
                        m_prime = m - M # m_prime is fixed if m, M are given

                        # Enforce selection rules
                        if abs(m_prime) > l_prime: # ensure m_prime is within bounds given by l_prime
                            continue
                        if not (abs(l - l_prime) <= L <= l + l_prime): # ensure L is within bounds given by l, l_prime
                            continue

                        cg1 = clebsch_gordan(l_prime, L, l, 0, 0, 0)
                        cg2 = clebsch_gordan(l_prime, L, l, m_prime, M, m)
                        prefactor = jnp.sqrt((2*l_prime + 1) / (2*l + 1))
                        coeff = prefactor * cg1 * cg2
                        
                        if abs(coeff) > 1e-10:  # Only store non-zero coefficients
                            entries.append((l, m, l_prime, m_prime, coeff))
            
            cg_by_LM[(L, M)] = jnp.array(entries)
    
    return cg_by_LM

# Optimize the moments computation for faster JIT compilation
@partial(jax.jit, static_argnames=['l_max', 'L_max'])
def compute_moments_refl(rho0, rho1, rho2, l_max=3, L_max=None, cg_coeffs=None):
    """
    Compute all three sets of moments (H0, H1, H2) from density matrix components with reflection symmetry.
    Only computes moments for M >= 0 due to symmetry of these moments
    
    Args:
        rho0, rho1, rho2: Density matrix components from compute_spin_density_matrices_refl
                         Each has shape (2, l_max+1, l_max+1, 2*l_max+1, 2*l_max+1) of gbl_ctype dtype
        l_max: Maximum l value
        L_max: Maximum L value (default: 2*l_max)
        cg_coeffs: Dictionary of precomputed Clebsch-Gordan coefficients
        
    Returns:
        JAX array of concatenated moments [H0, H1, H2] of gbl_ctype dtype.
        Each moment type has length sum(L+1 for L in range(L_max+1)).
        Within each moment type, values are ordered by (L,M) with M >= 0 only.
        
        According to the symmetry properties from the equations:
        - H0(L,M) should be purely real
        - H1(L,M) should be purely real
        - H2(L,M) should be purely imaginary
    """
    if L_max is None:
        L_max = 2 * l_max
    
    if cg_coeffs is None:
        cg_coeffs = precompute_cg_coefficients_by_LM(l_max, L_max)
    
    # Initialize moments arrays - one for each of H0, H1, H2
    # Only calculate for M >= 0
    num_moments_per_type = sum(L + 1 for L in range(L_max + 1))
    H0 = jnp.zeros(num_moments_per_type, dtype=gbl_ctype)
    H1 = jnp.zeros(num_moments_per_type, dtype=gbl_ctype)
    H2 = jnp.zeros(num_moments_per_type, dtype=gbl_ctype)
    
    # Loop through all moment indices (L,M) with M >= 0
    moment_idx = 0
    for L in range(L_max + 1):
        for M in range(0, L + 1):
            # Initialize accumulators for each type of moment
            h0_val = jnp.array(0.0, dtype=gbl_ctype)
            h1_val = jnp.array(0.0, dtype=gbl_ctype)
            h2_val = jnp.array(0.0, dtype=gbl_ctype)
            
            # Get the CG coefficients for this (L,M)
            entries = cg_coeffs[(L, M)]
            
            # Loop through all relevant CG coefficient entries
            for i in range(entries.shape[0]):
                entry = entries[i]
                l, m, l_prime, m_prime, cg = entry
                
                # Convert indices - JAX-friendly approach
                l_idx = jnp.array(l, dtype=jnp.int32)
                l_prime_idx = jnp.array(l_prime, dtype=jnp.int32)
                m_idx = jnp.array(m + l_max, dtype=jnp.int32)
                m_prime_idx = jnp.array(m_prime + l_max, dtype=jnp.int32)
                
                # Sum over both epsilon values (0=+1, 1=-1)
                for eps_idx in range(2):
                    # Get epsilon value for H1 and H2 calculations
                    epsilon = 1 - 2 * eps_idx  # Gives +1 for eps_idx=0, -1 for eps_idx=1
                    
                    # Get the proper rho elements
                    rho0_element = rho0[eps_idx, l_idx, l_prime_idx, m_idx, m_prime_idx]
                    rho1_element = rho1[eps_idx, l_idx, l_prime_idx, m_idx, m_prime_idx]
                    rho2_element = rho2[eps_idx, l_idx, l_prime_idx, m_idx, m_prime_idx]
                                
                    # if eps_idx == 0:
                    #     _jax_logger.debug("  term[l1={}, l2={}, L={}, m1={}, m2={}, M={}] = {}", l, l_prime, L, m, m_prime, M, cg)
                    #     _jax_logger.debug("  rhos[l1={}, l2={}, m1={}, m2={}] = {}, {}, {}", l, l_prime, m, m_prime, rho0_element, rho1_element, rho2_element)
                    
                    h0_val += cg * rho0_element                    
                    h1_val += cg * epsilon * rho1_element 
                    h2_val += cg * epsilon * rho2_element # Note rho2 includes the i factor
            
            # Set the computed moment values at specific indices
            H0 = H0.at[moment_idx].set(h0_val)
            H1 = H1.at[moment_idx].set(h1_val)
            H2 = H2.at[moment_idx].set(h2_val)
            moment_idx += 1
    
    # Concatenate all moment types and return
    return jnp.concatenate([H0, H1, H2])

@jax.jit
def flatten_moments(H):
    """
    Flatten complex moments into interleaved real and imaginary parts.
    
    Args:
        H: Array of concatenated moments [H0, H1, H2] with gbl_ctype dtype
        
    Returns:
        Flat array of real and imaginary parts with gbl_ftype dtype.
        The output is ordered as [Re(H0(0,0)), Im(H0(0,0)), Re(H0(1,0)), Im(H0(1,0)), ..., 
                                  Re(H1(0,0)), Im(H1(0,0)), ..., Re(H2(0,0)), Im(H2(0,0)), ...].
        Each H0(L,M), H1(L,M), H2(L,M) value represents a moment with specific L and M quantum numbers,
        where M >= 0 only (due to spherical harmonics symmetry).
    """
    # Reshape to flatten in the correct order
    H_flat_complex = jnp.reshape(H, (-1,))
    
    # Interleave real and imaginary parts
    real_parts = jnp.real(H_flat_complex)
    imag_parts = jnp.imag(H_flat_complex)
    
    # Interleave using vectorized operations
    flat_size = 2 * H_flat_complex.shape[0]
    flat_moments = jnp.zeros(flat_size, dtype=gbl_ftype)
    
    # Set real and imaginary parts
    flat_moments = flat_moments.at[0::2].set(real_parts)
    flat_moments = flat_moments.at[1::2].set(imag_parts)
    
    return flat_moments

@partial(jax.jit, static_argnames=['l_max', 'L_max'])
def project_to_moments_refl(flat_amplitudes, mask=None, l_max=3, L_max=None, cg_coeffs=None):
    """
    Project from reflectivity-basis partial-wave amplitudes to moments.
    Only computes moments for M >= 0 due to symmetry of these moments
    
    Args:
        flat_amplitudes: Flat JAX array of real and imaginary parts with gbl_ftype dtype
                        [Re(T_0), Im(T_0), Re(T_1), Im(T_1), ...]
                        NOTE: Order is [m cycles fastest, then l, then epsilon, then k]
        mask: JAX array of boolean values with gbl_ftype dtype that will zero out parameters (same shape as flat_amplitudes)
                intended to be used to lock a reference wave to 0
        l_max: Maximum orbital angular momentum (default: 3)
        L_max: Maximum L value for moments (default: 2*l_max)
        cg_coeffs: Dictionary of precomputed Clebsch-Gordan coefficients
        
    Returns:
        Flat array of moments [Re(H0(0,0)), Im(H0(0,0)), Re(H0(1,0)), ...] with gbl_ftype dtype.
        The output contains all three moment types (H0, H1, H2) concatenated, with real and
        imaginary parts interleaved. Within each moment type, values are ordered by (L,M)
        with M >= 0 only.
    """
    # Convert to Python ints
    l_max_int = int(l_max)
    
    if L_max is None:
        L_max = 2 * l_max_int
    L_max_int = int(L_max)
    
    # check if jnp bool mask if so convert to float32
    if mask is None:
        mask = jnp.ones_like(flat_amplitudes, dtype=gbl_ftype)
    if mask is not None and mask.dtype == jnp.bool_:
        mask = mask.astype(gbl_ftype)
    
    # Build structured array of reflectivity-basis amplitudes
    T = build_reflectivity_amplitudes(flat_amplitudes, mask, l_max_int)
    
    # Compute spin-density matrices
    rho0, rho1, rho2 = compute_spin_density_matrices_refl(T, l_max_int)
    
    # Precompute CG coefficients if not provided
    if cg_coeffs is None:
        cg_coeffs = precompute_cg_coefficients_by_LM(l_max_int, L_max_int)
    
    # Compute moments with precomputed CG coefficients
    H = compute_moments_refl(rho0, rho1, rho2, l_max_int, L_max_int, cg_coeffs)
    
    # Flatten moments
    flat_moments = flatten_moments(H)
    
    return flat_moments

def get_moment_names(l_max):
    """
    Generate names for the moments based on the maximum l value.
    """
    l_max = int(l_max)
    L_max = 2 * l_max
    names = []
    for i in range(3):
        for L in range(L_max + 1):
            for M in range(0, L + 1):
                names.append(f"H{i}({L},{M})")
    return names


#####################################################################
### BELOW THIS LINE IS FOR TESTING AND BENCHMARKING
#####################################################################

def _verify_moment_symmetry(flat_moments, l_max):
    """
    Verify the symmetry properties of the computed moments.
    
    Args:
        flat_moments: Flattened array of moments [Re(H0), Im(H0), Re(H1), Im(H1), Re(H2), Im(H2)]
        l_max: Maximum l value used in the computation
        
    Returns:
        Dictionary with statistics about the moment symmetry properties
    """
    L_max = 2 * l_max
    num_moments_per_type = sum(L + 1 for L in range(L_max + 1))
    
    # Reshape the flat array to separate real and imaginary parts
    re_im_moments = flat_moments.reshape(-1, 2)  # Shape: [3*num_moments_per_type, 2]
    
    # Split into the three moment types
    H0_re_im = re_im_moments[:num_moments_per_type]
    H1_re_im = re_im_moments[num_moments_per_type:2*num_moments_per_type]
    H2_re_im = re_im_moments[2*num_moments_per_type:]
    
    # Extract real and imaginary parts
    H0_real = H0_re_im[:, 0]
    H0_imag = H0_re_im[:, 1]
    H1_real = H1_re_im[:, 0]
    H1_imag = H1_re_im[:, 1]
    H2_real = H2_re_im[:, 0]
    H2_imag = H2_re_im[:, 1]
    
    # Calculate the ratio of imaginary to real magnitudes for H0 and H1 (should be near zero)
    # and the ratio of real to imaginary magnitudes for H2 (should be near zero)
    H0_imag_ratio = jnp.sum(jnp.abs(H0_imag)) / (jnp.sum(jnp.abs(H0_real)) + 1e-10)
    H1_imag_ratio = jnp.sum(jnp.abs(H1_imag)) / (jnp.sum(jnp.abs(H1_real)) + 1e-10)
    H2_real_ratio = jnp.sum(jnp.abs(H2_real)) / (jnp.sum(jnp.abs(H2_imag)) + 1e-10)
    
    # Create L, M indices for easier reporting
    lm_indices = []
    for L in range(L_max + 1):
        for M in range(0, L + 1):  # Only M >= 0
            lm_indices.append((L, M))
    
    # Find the largest violations
    H0_largest_imag_idx = jnp.argmax(jnp.abs(H0_imag))
    H1_largest_imag_idx = jnp.argmax(jnp.abs(H1_imag))
    H2_largest_real_idx = jnp.argmax(jnp.abs(H2_real))
    
    results = {
        "H0_imag_ratio": float(H0_imag_ratio),
        "H1_imag_ratio": float(H1_imag_ratio),
        "H2_real_ratio": float(H2_real_ratio),
        "H0_largest_imag": (lm_indices[H0_largest_imag_idx], float(H0_imag[H0_largest_imag_idx])),
        "H1_largest_imag": (lm_indices[H1_largest_imag_idx], float(H1_imag[H1_largest_imag_idx])),
        "H2_largest_real": (lm_indices[H2_largest_real_idx], float(H2_real[H2_largest_real_idx])),
        "H0_summary": {
            "real_max": float(jnp.max(jnp.abs(H0_real))),
            "imag_max": float(jnp.max(jnp.abs(H0_imag))),
        },
        "H1_summary": {
            "real_max": float(jnp.max(jnp.abs(H1_real))),
            "imag_max": float(jnp.max(jnp.abs(H1_imag))),
        },
        "H2_summary": {
            "real_max": float(jnp.max(jnp.abs(H2_real))),
            "imag_max": float(jnp.max(jnp.abs(H2_imag))),
        }
    }
    
    return results

def _get_boris_moments(flat_amplitudes, l_max=3, normalize=True):
    # Convert flat_amplitudes to AmplitudeValue objects expected by AmplitudeSet
    amplitude_values = []
    idx = 0
    for refl in [-1, 1]:  # Reflectivity: -1, +1
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                # Extract real and imaginary parts
                real_part = flat_amplitudes[idx]
                imag_part = flat_amplitudes[idx + 1]
                complex_val = complex(real_part, imag_part)                
                qn = QnWaveIndex(refl=refl, l=l, m=m)
                amplitude_values.append(AmplitudeValue(qn, val=complex_val))
                idx += 2  # Increment by 2 to skip to the next complex value

    # Create AmplitudeSet and calculate moments
    amplitude_set = AmplitudeSet(amps=amplitude_values, tolerance=1e-10)
        
    boris_moments = amplitude_set.photoProdMomentSet(
        maxL=2*l_max,  # Max L for moments is twice the max l for amplitudes
        normalize=normalize,
        printMomentFormulas=False
    )
    
    for moment in boris_moments.values:
        i, L, M = moment.qn.momentIndex, moment.qn.L, moment.qn.M
        moment_idx = f"H{i}({L},{M})"
        if abs(moment.val) > 1e-10:
            logging.debug(f"{moment_idx}: {moment.val}")
        
    return boris_moments


# Update the test function to verify moment symmetry
def _test_projection_and_gradient():
    """
    Test projection and gradient computation with performance measurements.
    """
    l_max = 2
        
    # Calculate total number of amplitudes in reflectivity basis
    total_m_values = sum(2 * l + 1 for l in range(l_max + 1))
    n_epsilon_values = 2            # +1/-1
    n_k_values = 1                  # In general we have k = [0, 1] (spin-non-flip/spin-flip), 
                                    # in practice we dont have the ability to separate (so ignore)
    
    total_complex_amplitudes = total_m_values * n_epsilon_values * n_k_values
    
    n_flat_amplitudes = 2 * total_complex_amplitudes  # Real and imaginary parts (2x number of waves)
    
    # Correct calculation for the number of flat moments
    L_max = 2 * l_max
    num_moments_per_type = sum(L + 1 for L in range(L_max + 1))
    n_flat_moments = 2 * 3 * num_moments_per_type  # 2x for real/imag, 3x for H0,H1,H2
    
    console.rule()
    console.print(f"For the given l_max: {l_max}, there are:")
    console.print(f" - total_m_values: {total_m_values}")
    console.print(f" - n_flat_amplitudes: {n_flat_amplitudes}")
    console.print(f" - n_moments_per_type: {num_moments_per_type}")
    console.print(f" - moments_output (concat H0, H1, H2) + flat real/imag: {n_flat_moments}")
    console.print(f"   - cycles real/imag faster than H0, H1, H2")
    console.rule()
    
    # Generate random amplitudes
    np.random.seed(42)
    flat_amplitudes = np.random.normal(size=n_flat_amplitudes)
    flat_amplitudes = jnp.array(flat_amplitudes, dtype=gbl_ftype)
    
    # Calculate moments using Boris's method
    boris_moments = _get_boris_moments(flat_amplitudes, l_max, normalize=True)

    # Precompute CG coefficients once
    console.print("Precomputing Clebsch-Gordan coefficients...")
    start = time.time()
    cg_coeffs = precompute_cg_coefficients_by_LM(l_max, 2*l_max)
    cg_time = time.time() - start
    console.print(f"CG precomputation time: {cg_time:.4f} seconds")
    console.print(f"Number of CG coefficients: {len(cg_coeffs)}")
    
    # First test - project to moments
    console.print("\nTesting projection to moments (reflectivity basis)...")
    start = time.time()
    moments = project_to_moments_refl(flat_amplitudes, mask=None, l_max=l_max, L_max=L_max, cg_coeffs=cg_coeffs)
    first_run_time = time.time() - start
    console.print(f"First run time (includes compilation): {first_run_time:.4f} seconds")
    
    # Multiple runs for better timing
    projection_times = []
    avg_non_zero_moments = []
    n_runs = 5
    for i in range(n_runs):
        start = time.time()
        moments = project_to_moments_refl(flat_amplitudes, mask=None, l_max=l_max, L_max=L_max, cg_coeffs=cg_coeffs)
        projection_times.append(time.time() - start)
        avg_non_zero_moments.append(jnp.sum(jnp.abs(moments) > 1e-6))
    
    avg_projection_time = sum(projection_times) / len(projection_times)
    console.print(f"Average projection time over {n_runs} runs: {avg_projection_time:.4f} seconds")
    console.print(f"Average percentage of non-zero moments: {sum(avg_non_zero_moments) / len(avg_non_zero_moments)}")
    
    # Verify moment symmetry
    console.print("\nVerifying moment symmetry...")
    symmetry_results = _verify_moment_symmetry(moments, l_max)
    if symmetry_results['H0_imag_ratio'] > 1e-6:
        console.print(f"H0 imaginary/real ratio: {symmetry_results['H0_imag_ratio']:.8f}")
        console.print(f"H0 largest imaginary component: {symmetry_results['H0_largest_imag']}")
    if symmetry_results['H1_imag_ratio'] > 1e-6:
        console.print(f"H1 imaginary/real ratio: {symmetry_results['H1_imag_ratio']:.8f}")
        console.print(f"H1 largest imaginary component: {symmetry_results['H1_largest_imag']}")
    if symmetry_results['H2_real_ratio'] > 1e-6:
        console.print(f"H2 real/imaginary ratio: {symmetry_results['H2_real_ratio']:.8f}")
        console.print(f"H2 largest real component: {symmetry_results['H2_largest_real']}")
    
    assert moments.shape == (n_flat_moments,), f"Output moments array is expected to have shape (n_flat_moments, ) = {(n_flat_moments,)}, but got {moments.shape}"

    ##################################################
    ########### COMPARE TO BORIS'S CODE ##############
    ##################################################

    # restructure moments array
    num_moments_per_type = sum(L + 1 for L in range(L_max + 1))
    num_moments_parts_per_type = num_moments_per_type * 2 # 2x for real/imag
    H0_re_im = moments[:num_moments_parts_per_type]
    H1_re_im = moments[num_moments_parts_per_type:2*num_moments_parts_per_type]
    H2_re_im = moments[2*num_moments_parts_per_type:]
    
    H0 = H0_re_im[0::2] + 1j*H0_re_im[1::2]
    H1 = H1_re_im[0::2] + 1j*H1_re_im[1::2]
    H2 = H2_re_im[0::2] + 1j*H2_re_im[1::2]
    H2 /= H0[0]
    H1 /= H0[0]
    H0 /= H0[0]
    moments = np.array([H0, H1, H2])
    
    np.set_printoptions(suppress=True, precision=6)
    moments_all_agree = True
    for boris_moment in boris_moments.values:
        i, L, M = boris_moment.qn.momentIndex, boris_moment.qn.L, boris_moment.qn.M
        offset = sum(L + 1 for L in range(L))
        moment = moments[i, offset + M]
        if abs(boris_moment.val - moment) > 1e-5:
            moments_all_agree = False
            print(f"H{i}({L},{M}) = {boris_moment.val}, {moment}, {abs(boris_moment.val - moment)}")
    
    if moments_all_agree:
        print("All moments between Boris's code and this code agree!")

if __name__ == "__main__":

    # Define loss function with precomputed CG coefficients
    @partial(jax.jit, static_argnames=['l_max', 'L_max'])
    def _loss_fn(flat_amps, target_moments, l_max, L_max, cg_coeffs):
        proj_moments = project_to_moments_refl(flat_amps, mask=None, l_max=l_max, L_max=L_max, cg_coeffs=cg_coeffs)
        return jnp.mean((proj_moments - target_moments)**2)
    
    _test_projection_and_gradient()
