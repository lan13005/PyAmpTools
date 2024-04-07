import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import pickle as pkl
from pyamptools.utility import glob_sort_captured
from pyamptools.utility.io import loadAmpToolsResults, ConfigLoader


def load_amptools_results(cfg):
    """
    Load the AmpTools results given information in the config file

    Args:
        cfg (str): path to config file

    Returns:
        pd.DataFrame: dataframe with the amptools results
    """

    cfg = ConfigLoader(OmegaConf.load(cfg))

    n_mass_bins = cfg("n_mass_bins")
    min_mass = cfg("min_mass")
    max_mass = cfg("max_mass")
    bin_width = (max_mass - min_mass) / n_mass_bins
    mass_bins = np.arange(min_mass, max_mass + 1e-4, bin_width)  # add a bit to upper limit to include its value
    masses = (mass_bins[1:] + mass_bins[:-1]) * 0.5

    cfgfiles = cfg("amptools.cfgfiles")
    cfgfiles = glob_sort_captured(cfgfiles)  # list of sorted config files based on captured number

    n_randomizations = cfg("amptools.n_randomizations")
    mle_query_1 = cfg("amptools.mle_query_1", None)
    mle_query_2 = cfg("amptools.mle_query_2", None)

    ati_df = loadAmpToolsResults(cfgfiles, masses, n_randomizations, mle_query_1, mle_query_2)

    return ati_df


def load_sample_pkl(file):
    """
    Load a NIFTy pickled file containing the final list of amplitude samples

    Args:
        file (str): path to pickeled file of Dict {'mass': [masses], 'amplitude1': [[amplitudes]], ...}
        [[amplitudes]] is a list-of-lists with dimensions [n_samples, n_masses]

    Returns:
        pd.DataFrame: dataframe with mass and amplitude columns, and an iteration column
    """

    with open(file, "rb") as f:
        data = pkl.load(f)

    amplitude_cols = list(data.keys() - ["mass"])
    nsamples = len(data[amplitude_cols[0]])
    flat_data = {key: [] for key in data.keys()}
    flat_data["iteration"] = []

    for i in range(nsamples):
        for key in data.keys():
            if key == "mass":
                flat_data[key].extend(data[key])
            else:
                flat_data[key].extend(data[key][i])
        flat_data["iteration"].extend(np.ones(len(data["mass"]), dtype=int) * i)

    flat_data = pd.DataFrame(flat_data)
    flat_data = flat_data.round(5)

    print(f'Loaded {len(data["mass"])} kinematic bins with {nsamples} samples each')

    return flat_data
