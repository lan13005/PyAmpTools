# +
# Minimal imports
from dataclasses import dataclass
import numpy as np
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'  # Enable 64-bit (float64/complex128) support in JAX

import jax.numpy as jnp
import jax
from spherical import clebsch_gordan

# +
# ------------------- your cache + helpers (unchanged) -------------------------
_cg = {}
def cg(l1, l2, L, m1, m2, M):
    key = (l1, l2, L, m1, m2, M)
    if key not in _cg:
        _cg[key] = clebsch_gordan(l1, m1, l2, m2, L, M)
    return _cg[key]

def Nm(m: int) -> float:
    return (1.0/np.sqrt(2.0)) if m>0 else (0.5 if m==0 else 0.0)

def waves_m_nonneg(lmax: int):
    return [(l, m) for l in range(lmax+1) for m in range(0, l+1)]

def tri_parity_ok(lp, l, L):
    return (abs(l-lp) <= L <= l+lp) and ((l+lp+L) % 2 == 0)

@dataclass
class MomentStruct:
    lmax: int
    Lmax: int
    waves: list          # [(l,m)] with m>=0
    n: int               # number of (l,m) per reflectivity block
    M_eps: jnp.ndarray   # shape (2, Lmax+1, Lmax+1, n, n); only M<=L used

def build_dense(lmax: int) -> MomentStruct:
    Lmax  = 2*lmax
    waves = waves_m_nonneg(lmax)
    n     = len(waves)
    idx   = {wm:i for i, wm in enumerate(waves)}

    # store as REAL symmetric (float64). we'll cast to complex later.
    M_eps = np.zeros((2, Lmax+1, Lmax+1, n, n), dtype=np.float64)

    for L in range(Lmax+1):
        for M in range(L+1):
            for (lp, mp) in waves:
                for (l, m) in waves:
                    if not tri_parity_ok(lp, l, L):
                        continue

                    # all pieces are real by construction; coerce to float
                    pref = float(Nm(m)*Nm(mp) * np.sqrt((2*lp+1)/(2*l+1)) * cg(lp, l, L, 0, 0, 0))

                    T1 = float(cg(lp, L, l, mp,  M,  m)) if (m ==  mp + M) else 0.0
                    T2 = float(((-1.0)**M) * cg(lp, L, l, mp, -M,  m)) if (m ==  mp - M) else 0.0
                    T3 = float(((-1.0)**mp) * cg(lp, L, l, -mp, M,  m)) if (m == -mp + M) else 0.0
                    T4 = float(((-1.0)**m)  * cg(lp, L, l, mp,  M, -m)) if (m == -mp - M) else 0.0

                    a = idx[(l, m)]
                    b = idx[(lp, mp)]

                    # values for ε=+ and ε=−
                    val_plus  = pref * (T1 + T2 - (+1.0)*(T3 + T4))
                    val_minus = pref * (T1 + T2 - (-1.0)*(T3 + T4))

                    # write BOTH (a,b) and (b,a); keep matrices exactly symmetric
                    if a == b:
                        M_eps[0, L, M, a, b] += val_plus
                        M_eps[1, L, M, a, b] += val_minus
                    else:
                        M_eps[0, L, M, a, b] += val_plus
                        M_eps[0, L, M, b, a] += val_plus
                        M_eps[1, L, M, a, b] += val_minus
                        M_eps[1, L, M, b, a] += val_minus

    # cast to complex128 for JAX einsum
    return MomentStruct(
        lmax=lmax, Lmax=Lmax, waves=waves,
        M_eps=jnp.asarray(M_eps, dtype=jnp.complex128),
        n=n
    )

def evaluate_moments(A_plus: jnp.ndarray, A_minus: jnp.ndarray, ms: MomentStruct) -> jnp.ndarray:
    """
    A_plus, A_minus: complex amplitudes ordered like ms.waves (m>=0) for ε=+1, ε=-1.
    Returns flat H(L,M) packed as (L=0..Lmax, for each L: M=0..L).  (Eq. 63)
    """
    A_plus = A_plus.astype(jnp.complex128)
    A_minus = A_minus.astype(jnp.complex128)
    H_plus  = jnp.einsum('LMab,b,a->LM', ms.M_eps[0], A_plus,  jnp.conj(A_plus))
    H_minus = jnp.einsum('LMab,b,a->LM', ms.M_eps[1], A_minus, jnp.conj(A_minus))
    H0 = H_plus + H_minus
    H1 = H_plus - H_minus
    H2 = 1j * (H_plus - H_minus)
    H0 = jnp.concatenate([H0[L, :L+1] for L in range(ms.Lmax+1)], axis=0)
    H1 = jnp.concatenate([H1[L, :L+1] for L in range(ms.Lmax+1)], axis=0)
    H2 = jnp.concatenate([H2[L, :L+1] for L in range(ms.Lmax+1)], axis=0)
    return H0, H1, H2, 

def moment_names(Lmax):
    """
    Returns a list of moment names in the order of the output of evaluate_moments,
    i.e., ['Hi(0,0)', 'Hi(1,0)', 'Hi(1,1)', ..., 'Hi(Lmax,Lmax)']
    """
    names = []
    for L in range(Lmax+1):
        for M in range(L+1):
            names.append(f"H({L},{M})")
    return names

# ------------------------------ Example usage ---------------------------------
ell_max = 2
mom_struct = build_dense(ell_max)
waves = mom_struct.waves

# Random complex amplitudes per reflectivity (deterministic seed)
rng = np.random.default_rng(1)
A_plus  = jnp.array(rng.normal(size=mom_struct.n) + 1j * rng.normal(size=mom_struct.n), dtype=jnp.complex128)
A_minus = jnp.array(rng.normal(size=mom_struct.n) + 1j * rng.normal(size=mom_struct.n), dtype=jnp.complex128)

# Enforce reflectivity-basis convention: ε=+ has no m=0 component (it’s identically zero)
# Indices of m=0 within waves for ℓ≤2 are 0:(0,0), 1:(1,0), 3:(2,0)
m0_idx = [i for i, (l, m) in enumerate(mom_struct.waves) if m == 0]
A_plus = A_plus.at[np.array(m0_idx, dtype=int)].set(np.complex128(0.0 + 0.0j))

# Evaluate Eq. 63 (unpolarized moments)
H0, H1, H2 = evaluate_moments(A_plus, A_minus, mom_struct)
moment_labels = moment_names(mom_struct.Lmax)
# -

print(H0, H1, H2)

