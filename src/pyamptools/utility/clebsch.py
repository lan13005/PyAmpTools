import jax
import jax.numpy as jnp
import numpy as np
import time
import sympy.physics.quantum.cg as sympy_cg
import matplotlib.pyplot as plt

@jax.jit
def factorial(n: float) -> float:
    """Compute factorial using JAX."""
    n = jnp.asarray(n, dtype=jnp.float32)
    return jnp.exp(jax.lax.lgamma(n + 1))

@jax.jit
def clebsch_gordan(j1, j2, j, m1, m2, m):
    """
    Calculate the Clebsch-Gordan coefficient <j1 m1 j2 m2|j m>.
    
    Args:
        j1, j2: The angular momenta being coupled
        j: The resulting angular momentum
        m1, m2: The magnetic quantum numbers of j1 and j2
        m: The magnetic quantum number of j
        
    Returns:
        The Clebsch-Gordan coefficient as a float
    """    
    # Check if m = m1 + m2, otherwise coefficient is 0
    valid_m = jnp.abs(m - (m1 + m2)) < 1e-10
    
    # Check triangle inequality and other constraints
    valid_j = jnp.logical_and(
        jnp.abs(j1 - j2) <= j,
        jnp.logical_and(
            j <= j1 + j2,
            jnp.logical_and(
                jnp.abs(m1) <= j1,
                jnp.logical_and(
                    jnp.abs(m2) <= j2,
                    jnp.abs(m) <= j
                )
            )
        )
    )
    
    # Use scan for the summation
    def body_fun(k, acc):
        term = ((-1.0) ** k) / (factorial(k) * 
                                factorial(j1 + j2 - j - k) * 
                                factorial(j1 - m1 - k) * 
                                factorial(j2 + m2 - k) * 
                                factorial(j - j2 + m1 + k) * 
                                factorial(j - j1 - m2 + k))
        
        # Check if k is in valid range
        valid_k = jnp.logical_and(
            k >= jnp.maximum(0, jnp.maximum(j2 - j - m1, j1 + m2 - j)),
            k <= jnp.minimum(j1 + j2 - j, jnp.minimum(j1 - m1, j2 + m2))
        )
        
        return acc + jnp.where(valid_k, term, 0.0)
    
    # Use a fixed upper bound for k
    k_max = 100
    sum_result = jax.lax.fori_loop(0, k_max, body_fun, 0.0)
    
    # Calculate prefactor
    prefactor = jnp.sqrt((2*j + 1) * 
                         factorial(j + j1 - j2) * 
                         factorial(j - j1 + j2) * 
                         factorial(j1 + j2 - j) / 
                         factorial(j1 + j2 + j + 1))
    
    prefactor *= jnp.sqrt(factorial(j + m) * 
                          factorial(j - m) * 
                          factorial(j1 - m1) * 
                          factorial(j1 + m1) * 
                          factorial(j2 - m2) * 
                          factorial(j2 + m2))
    
    # Combine and apply validity conditions
    result = jnp.where(
        jnp.logical_and(valid_m, valid_j),
        prefactor * sum_result,
        0.0
    )
    
    return result

#####################################################################
### BELOW THIS LINE IS FOR TESTING AND BENCHMARKING
#####################################################################

def _test_accuracy():
    """Compare our implementation with sympy's implementation"""
    print("Testing accuracy against sympy...")
    
    # Test cases: (j1, j2, j, m1, m2, m)
    test_cases = [
        (0.5, 0.5, 1.0, 0.5, 0.5, 1.0),
        (0.5, 0.5, 0.0, 0.5, -0.5, 0.0),
        (1.0, 1.0, 2.0, 1.0, 1.0, 2.0),
        (1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        (2.0, 1.0, 3.0, 1.0, 0.0, 1.0),
        (2.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    ]
    
    print(f"{'j1':>5} {'j2':>5} {'j':>5} {'m1':>5} {'m2':>5} {'m':>5} {'Our CG':>12} {'Sympy CG':>12} {'Diff':>12}")
    print("-" * 70)
    
    for j1, j2, j, m1, m2, m in test_cases:
        # Our implementation
        our_cg = float(clebsch_gordan(j1, j2, j, m1, m2, m))
        
        # Sympy implementation
        sympy_val = float(sympy_cg.CG(j1, m1, j2, m2, j, m).doit())
        
        # Calculate difference
        diff = abs(our_cg - sympy_val)
        
        print(f"{j1:5.1f} {j2:5.1f} {j:5.1f} {m1:5.1f} {m2:5.1f} {m:5.1f} {our_cg:12.8f} {sympy_val:12.8f} {diff:12.8e}")

def _benchmark_performance():
    """Compare performance of our implementation with sympy"""
    print("\nBenchmarking performance...")
    
    # Generate random valid test cases with only integers and half-integers
    # to ensure compatibility with sympy
    np.random.seed(42)  # For reproducibility
    n_samples = 100
    
    # Generate integer and half-integer values
    possible_js = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    j1s = np.random.choice(possible_js, n_samples)
    j2s = np.random.choice(possible_js, n_samples)
    
    # Ensure j values satisfy triangle inequality
    js = []
    for j1, j2 in zip(j1s, j2s):
        # Find valid j values (integer or half-integer between |j1-j2| and j1+j2)
        min_j = abs(j1 - j2)
        max_j = j1 + j2
        valid_js = [j for j in possible_js if min_j <= j <= max_j]
        if valid_js:
            js.append(np.random.choice(valid_js))
        else:
            # Fallback if no valid j (shouldn't happen with our choices)
            js.append(min_j)
    js = np.array(js)
    
    # Generate valid m values (must be integer or half-integer between -j and j)
    m1s = []
    m2s = []
    ms = []
    
    for j1, j2, j in zip(j1s, j2s, js):
        # Generate m1 between -j1 and j1 in steps of 1 or 0.5
        possible_m1s = np.arange(-j1, j1 + 0.5, 0.5 if j1 % 1 == 0.5 else 1.0)
        m1 = np.random.choice(possible_m1s)
        
        # Generate m2 between -j2 and j2 in steps of 1 or 0.5
        possible_m2s = np.arange(-j2, j2 + 0.5, 0.5 if j2 % 1 == 0.5 else 1.0)
        m2 = np.random.choice(possible_m2s)
        
        # m must equal m1 + m2 for non-zero coefficients
        m = m1 + m2
        
        # Ensure m is within valid range for j
        if -j <= m <= j:
            m1s.append(m1)
            m2s.append(m2)
            ms.append(m)
        else:
            # If m is out of range, choose valid values
            m1s.append(0)
            m2s.append(0)
            ms.append(0)
    
    m1s = np.array(m1s)
    m2s = np.array(m2s)
    ms = np.array(ms)

    # Time our implementation (single calculation)
    _ = clebsch_gordan(j1s[0], j2s[0], js[0], m1s[0], m2s[0], ms[0]) # compiling call
    
    start = time.time()
    for i in range(n_samples):
        clebsch_gordan(j1s[i], j2s[i], js[i], m1s[i], m2s[i], ms[i])
    our_time_single = (time.time() - start) / n_samples

    # Time sympy implementation
    start = time.time()
    for i in range(n_samples):
        sympy_cg.CG(j1s[i], m1s[i], j2s[i], m2s[i], js[i], ms[i]).doit()
    sympy_time = (time.time() - start) / n_samples
    
    print("Average time per calculation:")
    print(f"Our implementation (single): {our_time_single*1000:.6f} ms")
    print(f"Sympy implementation: {sympy_time*1000:.6f} ms")
    print(f"Speedup (single vs sympy): {sympy_time/our_time_single:.2f}x")

if __name__ == "__main__":
    _test_accuracy()
    _benchmark_performance()
