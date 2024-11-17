import argparse
import pickle
import os

os.environ["JAX_PLATFORMS"] = "cpu" # do need for GPU even if we have access

import numpy as np
from iftpwa1.model.physics_functions import barrier_factor_sq, breakup_momentum
from omegaconf import OmegaConf


def calc_ps_ift(yaml_file, output):

    """
    Calculate the barrier factor for a given orbital angular momentum and breakup momentum.

    This function will create a pkl file that will be loaded into the iftpwa model_builder. A forward model
        is built, from latent space of power spectra to position/amplitude space. The forward model contains
        the phase space multipliers so that the final signal fields contain phase space factors that can depend on (mass, t, l, m) 

    Args:
        yaml_file (str): path to the primary yaml file
        output (str): pickle dump path
    """

    yaml_file = OmegaConf.load(yaml_file)

    masses = yaml_file["daughters"].values()
    if len(masses) != 2:
        raise ValueError("Only two daughter particles in the reaction are supported. Please modify 'daughters' key in source yaml")
    mass1, mass2 = masses

    waveNames = yaml_file["waveset"].split("_")
    min_mass = yaml_file["min_mass"]
    max_mass = yaml_file["max_mass"]
    n_mass_bins = yaml_file["n_mass_bins"]
    masses_ps_edges = np.linspace(min_mass, max_mass, n_mass_bins+1)
    masses_ps = 0.5 * (masses_ps_edges[1:] + masses_ps_edges[:-1])

    spin_map = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4}

    scaling = {}

    breakup_momenta = np.zeros(n_mass_bins) # just for diagnostics
    for iw, wave in enumerate(waveNames):
        spin = spin_map[wave[0]]
        barrier_factors = np.zeros(n_mass_bins)
        for i, mass in enumerate(masses_ps):
            q = breakup_momentum(mass, mass1, mass2)
            barrier_factors[i] = barrier_factor_sq(q, spin)
            if iw == 0:
                breakup_momenta[i] = q
        scaling[wave] = barrier_factors

    # for m, b in zip(masses_ps, breakup_momenta):
    #     print(m, b)

    with open(output, "wb") as f:
        pickle.dump((masses_ps, scaling), f)
        print(f"\nSuccess! Phase space factors saved to {output}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate the phase space factor for ift fits")
    parser.add_argument("yaml_file", type=str, help="Path to the primary yaml file")
    parser.add_argument("output", type=str, help="Path to the output pickle file")
    args = parser.parse_args()

    yaml_file = args.yaml_file
    output = args.output

    calc_ps_ift(yaml_file, output)