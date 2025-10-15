import argparse
import pickle
import os

os.environ["JAX_PLATFORMS"] = "cpu" # do need for GPU even if we have access
os.environ['JAX_ENABLE_X64'] = '1' # 64-bit

import numpy as np
from iftpwa1.model.physics_functions import barrier_factor_sq, breakup_momentum
from omegaconf import OmegaConf, ListConfig
from pyamptools.utility.general import converter

from rich.console import Console
console = Console()

def get_spin_from_parametric(config: OmegaConf, wave_name: str):
    """
    Lookup spin value for a wave from the custom model resonances
    Only works for Breit-Wigner physics function currently
    
    Procedure:
        1. Check if custom_model_path exists
        2. Yes? Check if PARAMETRIC_MODEL key exists 
        3. Yes? Check if resonances key exists
        4. Loop over dict to find mapping from the wave it populates to the spin
    """
    # try:
        
    custom_model_path = config.NIFTY.IFT_MODEL.get('custom_model_path')
    if custom_model_path is None: # Step 1
        return None            
    parametric_model = config.NIFTY.PARAMETRIC_MODEL
    if parametric_model is None: # Step 2
        return None
    resonance_list = parametric_model.get('resonances')
    if resonance_list is None: # Step 3
        return None
    
    for resonance in resonance_list: # Step 4
        if len(resonance.keys()) != 1:
            raise ValueError(f"Each list element must be a dictionary with a single key = resonance name")
        resonance_name, resonance_data = next(iter(resonance.items()))
        waves = resonance_data.get('waves', [])
        if wave_name in waves:
            static_paras = resonance_data.get('static_paras', {})
            spin = static_paras.get('spin')
            if spin is not None:
                console.print(f"Found spin {spin} for wave '{wave_name}' from resonance '{resonance_name}'", style="bold blue")
                return spin
                
    console.print(f"No spin found for wave '{wave_name}' in custom model", style="bold yellow")
    return None
        
    # except Exception as e:
    #     console.print(f"Error when attempting to extract spin for resonance parameterization '{wave_name}': {e}", style="bold red")
    #     return None

def calc_ps_ift(yaml_file):

    """
    Calculate the barrier factor for a given orbital angular momentum and breakup momentum.

    This function will create a pkl file that will be loaded into the iftpwa model_builder. A forward model
        is built, from latent space of power spectra to position/amplitude space. The forward model contains
        the phase space multipliers so that the final signal fields contain phase space factors that can depend on (mass, t, l, m) 

    Args:
        yaml_file (str): path to the primary yaml file
    """

    config = OmegaConf.load(yaml_file)
    
    base_directory = config["base_directory"]
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)
    
    # Get output path from NIFTY.IFT_MODEL.phaseSpaceMultiplier
    output = config.NIFTY.IFT_MODEL.phaseSpaceMultiplier
    if output is None:
        raise ValueError("phaseSpaceMultiplier dump location is not set in the config file")
    console.print(f"Output path from config: {output}")

    if isinstance(config["daughters"], (list, ListConfig)):
        masses = config["daughters"]
    else:
        masses = list(config["daughters"].values())

    if len(masses) == 1:
        mass1, mass2 = masses[0], masses[0]
        console.print(f"Only 1 daughter specified in yaml, calculating phase space factors for decay into 2 identical particles")
    elif len(masses) == 2:
        mass1, mass2 = masses
        console.print(f"Calculating phase space factors for decay into 2 different particles")
    else:
        console.print(f"Invalid number of daughters specified in yaml: {len(masses)}", style="bold red")
        return

    waveNames = config["waveset"].split("_")
    min_mass = config["min_mass"]
    max_mass = config["max_mass"]
    n_mass_bins = config["n_mass_bins"]
    masses_ps_edges = np.linspace(min_mass, max_mass, n_mass_bins+1)
    masses_ps = 0.5 * (masses_ps_edges[1:] + masses_ps_edges[:-1])

    spin_map = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4}

    scaling = {}

    # ---------------------------------------------------------------------------
    # In iftpwa model_builder.py, the production kinFactor is calculated as follows:
    #     proton_mass = 0.938 # proton mass (GeV)
    #     Egamma  = 8.5       # lab photon energy
    #     s0 = 2*proton_mass*Egamma + proton_mass**2
    #     flux = 1.0 / (16 * jnp.pi * (s0 - proton_mass**2)**2)
    #     kinFactor = flux * 2 / masses
    # This needs to be multiplied by the barrier factor and breakup momentum
    #     These factors depend on mass and spin which can be incorporated using this module
    #     and be included in the config file as a pkl file at key "phaseSpaceMultiplier".
    # All partial waves will be scaled by the kinFactor * phaseSpaceMultiplier
    # Breit-Wigner amplitudes defined in iftpwa physics_functions.py should use Gamma0 in the numerator
    #     since phase space factors are absorbed into the phaseSpaceMultiplier.
    # ---------------------------------------------------------------------------
    
    for iw, wave in enumerate(waveNames):
        if wave.startswith("Amp"): # Fallback to custom model lookup
            console.print(f"Wave '{wave}' not found in traditional spin_map, checking custom model...", style="bold yellow")
            spin = get_spin_from_parametric(config, wave)
            if spin is None:
                console.print(f"'{wave}' not found in model parameterization. Assume no barrier factor.", style="bold blue")
        else: # Assume traditional naming convention
            if wave not in converter:
                console.print(f"'{wave}' not in expected form, unable to determine spin for barrier factor calculation...", style="bold red")
                exit(1)
            spin = converter[wave][1] # for both TwoPseudoscalar and VectorPseudoscalar, l is the index 1
            console.print(f"Using traditional spin mapping: wave '{wave}' -> spin {spin}")
                
        factors = np.zeros(n_mass_bins)
        for i, mass in enumerate(masses_ps):
            q = breakup_momentum(mass, mass1, mass2)
            if spin is None: # assume spin-0
                factors[i] = barrier_factor_sq(q, 0) * q
            else:
                factors[i] = barrier_factor_sq(q, spin) * q
        scaling[wave] = factors
        
    # Print some diagnostics (min, max)
    for wave in waveNames:
        console.print(f"Wave: {wave} [{scaling[wave][0]:.3f}, ..., {scaling[wave][-1]:.3f}]")

    with open(output, "wb") as f:
        pickle.dump((masses_ps, scaling), f)
        console.print(f"\nSuccess! Phase space factors saved to {output}", style="bold green")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate the phase space factor for ift fits")
    parser.add_argument("yaml_file", type=str, help="Path to the primary yaml file")
    args = parser.parse_args()

    yaml_file = args.yaml_file

    calc_ps_ift(yaml_file)