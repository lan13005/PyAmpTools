import numpy as np


def breakup_momentum(mass0, mass1, mass2):
    """
    Calculate the breakup momentum for a two-body decay.
    Follows format of halld_sim/src/libraries/AMPTOOLS_AMPS/breakupMomentum.cc

    Args:
        mass0: float, mass of parent
        mass1, mass2: float, masses of the daughters

    Returns:
        float, breakup momentum
    """

    if mass0 < (mass1 + mass2):
        return 0.0

    q = np.sqrt(abs(mass0 * mass0 * mass0 * mass0 + mass1 * mass1 * mass1 * mass1 + mass2 * mass2 * mass2 * mass2 - 2.0 * mass0 * mass0 * mass1 * mass1 - 2.0 * mass0 * mass0 * mass2 * mass2 - 2.0 * mass1 * mass1 * mass2 * mass2)) / (2.0 * mass0)

    return q


def barrier_factor_qs(q, spin):
    """
    Calculate the barrier factor for a given orbital angular momentum and breakup momentum.
    Follows format at halld_sim/src/libraries/AMPTOOLS_AMPS/barrierFactor.cc
    FYI: switch statement does not exist in Python < 3.10

    Args:
        q: float, breakup momentum
        spin: int, orbital angular momentum

    Returns:
        float, barrier factor
    """
    barrier = 0.0  # default value
    z = z = (q * q) / (0.1973 * 0.1973)

    if spin == 0:
        barrier = 1.0
    elif spin == 1:
        barrier = np.sqrt((2.0 * z) / (z + 1.0))
    elif spin == 2:
        barrier = np.sqrt((13.0 * z * z) / ((z - 3.0) * (z - 3.0) + 9.0 * z))
    elif spin == 3:
        barrier = np.sqrt((277.0 * z * z * z) / (z * (z - 15.0) * (z - 15.0) + 9.0 * (2.0 * z - 5.0) * (2.0 * z - 5.0)))
    elif spin == 4:
        barrier = np.sqrt((12746.0 * z * z * z * z) / ((z * z - 45.0 * z + 105.0) * (z * z - 45.0 * z + 105.0) + 25.0 * z * (2.0 * z - 21.0) * (2.0 * z - 21.0)))
    else:
        raise ValueError(f"barrier_factor_qs| Invalid spin value passed: {spin}")

    return barrier


def barrier_factor(mass0, spin, mass1, mass2):
    q = breakup_momentum(mass0, mass1, mass2)
    return barrier_factor_qs(q, spin)


def breit_wigner_amplitude(mass, spin, mass0, Gamma0, mass1, mass2):
    """
    Calculate the relativistic Breit-Wigner amplitude including dynamical width and barrier factors.
    Follows format at halld_sim/src/libraries/AMPTOOLS_AMPS/BreitWigner.cc

    Args:
        mass: float, mass at which to calculate the amplitude
        spin: int, orbital angular momentum
        mass0: float, mass of the resonance
        Gamma0: float, natural width of the resonance
        mass1, mass2: float, masses of the decay products

    Returns:
    - float, value of the Breit-Wigner distribution
    """

    q0 = breakup_momentum(mass0, mass1, mass2)
    q = breakup_momentum(mass, mass1, mass2)

    F0 = barrier_factor_qs(q0, spin)
    F = barrier_factor_qs(q, spin)

    Gamma = Gamma0 * (mass0 / mass) * (q / q0) * (F / F0) * (F / F0)

    bwtop = complex(np.sqrt(mass0 * Gamma0 / 3.1416), 0.0)
    bwbottom = complex((mass0 * mass0 - mass * mass), -1.0 * (mass0 * Gamma))

    return F * bwtop / bwbottom


def breit_wigner(masses, spin, mass0, Gamma0, mass1, mass2):
    """
    Calculate the relativistic Breit-Wigner intensity (PDF) distribution for a given set of masses

    Args:
        masses (np.array): array of masses at which to calculate the amplitude
        L (int): orbital angular momentum
        mass0 (float): mass of the resonance
        Gamma0 (float): natural width of the resonance
        mass1 (float): mass of the first decay product
        mass2 (float): mass of the second decay product

    Returns:
        np.array: array for Breit-Wigner PDF
    """

    bw_amp = [breit_wigner_amplitude(mass, spin, mass0, Gamma0, mass1, mass2) for mass in masses]
    bw_amp = np.array(bw_amp)

    intensity = bw_amp * np.conj(bw_amp)
    intensity = intensity.real
    integral = np.trapz(intensity, masses)
    intensity /= integral  # Normalize

    phase = np.angle(bw_amp, deg=True)

    return intensity, phase


def convolve_with_resolution(y, bin_width, resolution):
    """
    Convolve a given amplitude with a Gaussian resolution function

    Args:
        y (np.array): array of intensities (y-axis)
        bin_width (float): width of the mass bins (x-axis)
        resolution (float): mass resolution (along x)

    Returns:
        np.array: array of convolved amplitudes
    """

    gx = np.arange(-3 * resolution, 3 * resolution, bin_width)
    gaussian = np.exp(-0.5 * (gx / resolution) ** 2)
    gaussian /= gaussian.sum()
    convolved = np.convolve(y, gaussian, mode="same")

    return convolved


def standard_resonance_model(masses, spin, mass0, Gamma0, mass1, mass2, bin_width, mass_resolution):
    """
    Construct Breit-Wigner and Voigt Profiles

    Args:
        masses: np.array, mass values at which to calculate the amplitude
        spin: int, orbital angular momentum
        mass0: float, mass of the resonance
        Gamma0: float, natural width of the resonance
        mass1, mass2: float, masses of the decay products
        bin_width: float, width of the mass bins
        mass_resolution: float, mass resolution

    Returns:
        np.array, np.array, np.array, np.array: Breit-Wigner, Breit-Wigner Phase, Voigt, Phase
    """
    _bw, _bw_phase = breit_wigner(masses, spin, mass0, Gamma0, mass1, mass2)
    _voigt = convolve_with_resolution(_bw, bin_width, mass_resolution)  # Voigtian
    _phase = convolve_with_resolution(_bw_phase, bin_width, mass_resolution)  # IDK if this makes sense
    return _bw, _bw_phase, _voigt, _phase


# +
# def flatte_phaseSpaceFactor(m, mass1, mass2):
#     """
#     Calculate the phase space factor for a given mass and decay products.
#     Follows format at halld_sim/src/libraries/AMPTOOLS_AMPS/Flatte.cc

#     Args:
#         m: float, mass at which to calculate the phase space factor
#         mass1, mass2: float, masses of the daughter products

#     Returns:
#         complex, phase space factor
#     """
#     if abs(m) < 1e-8:
#         print(f"Mass {m} is too close to 0. Can't calculate phasespace factor: set mass to 1.e-10")
#         m = 1.e-10

#     termPlus  = (mass1 + mass2) / m
#     termMinus = (mass1 - mass2) / m
#     tmpVal = (1. - termPlus*termPlus) * (1. - termMinus*termMinus)
#     if tmpVal >= 0:
#         result = complex(np.sqrt(tmpVal), 0.)
#     else:
#         result = complex(0., np.sqrt(-tmpVal))
#     return result

# def flatte_breakupMomentum(m, mass1, mass2):
#     """
#     Calculate the breakup momentum for a given mass and decay products.
#     Follows format at halld_sim/src/libraries/AMPTOOLS_AMPS/Flatte.cc

#     Args:
#         m: float, mass at which to calculate the breakup momentum
#         mass1, mass2: float, masses of the daughter products

#     Returns:
#         complex, breakup momentum
#     """
#     result = flatte_phaseSpaceFactor(m, mass1, mass2) * m / 2.
#     return result

# def flatte_amplitude(mass, mass0, g1, g2, mass1, mass2, channel):
#     """
#     Calculate the Flatte amplitude for a given mass and decay products.
#     Follows format at halld_sim/src/libraries/AMPTOOLS_AMPS/Flatte.cc

#     Args:
#         mass: float, mass at which to calculate the amplitude
#         mass0: float, mass of the resonance
#         g1: float, coupling of the first decay channel
#         g2: float, coupling of the second decay channel
#         mass1, mass2: float, masses of the decay products
#         channel: int, channel index

#     Returns:
#         complex, value of the Flatte amplitude
#     """
#     imag = complex(0., 1.)
#     P1 = complex(0., 0.)
#     P2 = complex(0., 0.)

#     P1 = flatte_breakupMomentum(mass, mass1, mass2)
#     P2 = flatte_breakupMomentum(mass, mass1, mass2)

#     curMass = np.sqrt(mass)
#     gamma11 = g1 * flatte_breakupMomentum(curMass, mass1, mass2)
#     gamma22 = g2 * flatte_breakupMomentum(curMass, mass1, mass2)

#     gammaLow = 0
#     if (mass1 + mass2) < (mass1 + mass2):
#         gammaLow = gamma11
#     else:
#         gammaLow = gamma22

#     gamma_j = 0
#     if channel == 1:
#         gamma_j = gamma11
#     elif channel == 2:
#         gamma_j = gamma22
#     else:
#         print("ERROR: possible channel indices for Flatte amplitude are 1 or 2!")

#     result = mass0 * np.sqrt(gammaLow * gamma_j) / (mass0*mass0 - mass*mass - imag * mass0 * (gamma11 + gamma22))
#     return result
