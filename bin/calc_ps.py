import argparse
import pickle
import os

os.environ["JAX_PLATFORMS"] = "cpu" # do need for GPU even if we have access

import numpy as np
from iftpwa1.model.physics_functions import barrier_factor_sq, breakup_momentum
from omegaconf import OmegaConf

from rich.console import Console
console = Console()

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
    
    # Get output path from NIFTY.IFT_MODEL.phaseSpaceMultiplier
    output = config.NIFTY.IFT_MODEL.phaseSpaceMultiplier
    if output is None:
        raise ValueError("phaseSpaceMultiplier dump location is not set in the config file")
    console.print(f"Output path from config: {output}")

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
        spin = spin_map[wave[0]]
        factors = np.zeros(n_mass_bins)
        for i, mass in enumerate(masses_ps):
            q = breakup_momentum(mass, mass1, mass2)
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