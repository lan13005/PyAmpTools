import argparse
import glob
import os
import pickle
import re

import numpy as np
import uproot as up
from omegaconf import OmegaConf

pat_dropreac = re.compile(r'[^.]*\.(.*)')
fullname = "{r}.{t}" # (r)eactionName and (t)ermName

def load_root(fname, treename):
    with up.open(fname) as f:
        data = f[treename].arrays(library="np")
        return data

def restructure_amps(yaml_file, treename="kin", include_accmc=True, include_genmc=True):
    
    ####################################################################
    # After finalizeFit from an amptoolsinterface call a set of root
    # files will be dumped that contains the complex amplitude parts 
    # for each data source (in the same location as the data sources)

    # We use this script to pull them all into a single pickle file so
    # we can access them much faster. The complex amplitudes a flattened
    # across all (reaction, mass, t) bins. We will store the starts and
    # stop indices for each (reaction, mass, t) bin so we can reload them
    ####################################################################

    base_directory = yaml_file["base_directory"]
    output_directory = yaml_file["amptools"]["output_directory"]
    nmbMasses = yaml_file["n_mass_bins"]
    nmbTprimes = yaml_file["n_t_bins"]
    reactions = yaml_file["polarizations"] # i.e. "000", "045", "090", "135"
    share_mc = yaml_file["share_mc"]
    reactionNames = [f"reaction_{reaction}" for reaction in reactions] # i.e. "reaction_000", "reaction_045", "reaction_090", "reaction_135"
    nmbReactions = len(reactions)

    ftypes = ["data"] # bkgnd is optional
    if include_accmc:
        ftypes.append("accmc")
    if include_genmc:
        ftypes.append("genmc")
    if len(glob.glob(f"{output_directory}/bin_*/bkgnd*_amps.root")) > 0:
        ftypes.append("bkgnd")

    values = {}
    starts = {}

    for ftype in ftypes:

        ############################################################################################
        # Each kinematic bin will have a set of ampvecs root files for each (source, reaction) pair
        ############################################################################################

        # normint_files = glob.glob(f"{output_directory}/bin_*/{ftype}*_amps.root")
        # normint_files = glob.glob(f"{os.path.dirname(output_directory)}/prior_sim_DATA/{ftype}*_amps.root")
        # files = sorted(normint_files, key=extract_numbers)

        ######################################################

        starts[ftype] = [0]
        parts = None

        skip_cols = ["weight", "bin"]

        for it in range(nmbTprimes):
            for im in range(nmbMasses):
                for reactionName, reaction in zip(reactionNames, reactions):
                    k = it * nmbMasses + im

                    file = None
                    if share_mc[ftype]:
                        if reaction == "000":
                            file = f"{output_directory}/bin_{k}/{ftype}_amps.root"
                        else:
                            continue
                    else:
                        file = f"{output_directory}/bin_{k}/{ftype}{reaction}_amps.root"

                    data = load_root(file, treename)

                    nentries = len(data[list(data.keys())[0]])
                    data['bin'] = np.ones(nentries, dtype=int) * k

                    if parts is None:
                        parts = []
                        for key in data.keys():
                            match = pat_dropreac.search(key)
                            if match: parts.append(match.group(1))
                            else: parts.append(key) # i.e. weight branch

                    starts[ftype].append(starts[ftype][-1] + nentries)
                    if ftype not in values:
                        values[ftype] = {}
                        for key in parts:
                            values[ftype][key] = []
                    for key in parts:
                        if key in skip_cols:
                            data_key = key
                        else:
                            data_key = fullname.format(r=reactionName, t=key)
                        values[ftype][key].append(data[data_key])

        print(f"\n***********************\nSummary of {ftype}\n***********************\n")
        for key in values[ftype].keys():
            values[ftype][key] = np.concatenate(values[ftype][key])
            print(f"{key}: {values[ftype][key].shape}")

        print(f"Starts: {starts[ftype]}")

    print()

    # Check order of parts are exactly the same
    # this ensures the arrays are in the same order
    partNames = list(values[ftypes[0]].keys())
    for ftype in ftypes[1:]:
        if list(values[ftype].keys()) != partNames:
            raise ValueError(f"parts do not match: {list(values[ftype].keys())} != {partNames}")

    # NOTE: Check if any parts are all zeros. If so, drop them.
    # We can do this if we start running into IO problems
    drop_parts = []
    for ftype in ftypes:
        for key in values[ftype].keys():
            allzeros = np.allclose(values[ftype][key], 0)
            if allzeros:
                drop_parts.append((ftype, key))
    if len(drop_parts) > 0:
        print("**********************************************************************************")
        print("The following arrays are all zeros. Consider dropping them to save space.")
        print("  Note: This is expected for some amplitudes used by AmpTools, i.e. Zlm amplitude")
        print("  since these are the associated parts that enter in the 4 different sum")
        print("**********************************************************************************")
        for ftype, key in drop_parts:
            print(f"({ftype}, {key})")


    # We have the headers for the array and checked they are
    # consistently used. We can now convert to a giant array
    # values are concatenated over all (reaction, mass, t) datasets. We will reshape starts to the indices
    for ftype in ftypes:
        values[ftype] = np.array(list(values[ftype].values()))
        # values[ftype] = values[ftype].reshape(nmbMasses, nmbTprimes, *values[ftype].shape)

    # We will store a stops array also even though we can recreate it from starts
    # It becomes harder to track when starts are reshaped
    stops = {k: v[1:] for k, v in starts.items()}
    starts = {k: v[:-1] for k, v in starts.items()}
    starts = {k: np.array(v).reshape( (nmbReactions if not share_mc[k] else 1), nmbMasses, nmbTprimes, order="F") for k, v in starts.items()}
    stops = {k: np.array(v).reshape(  (nmbReactions if not share_mc[k] else 1), nmbMasses, nmbTprimes, order="F") for k, v in stops.items()}

    # order of partNames is the index order of the values
    data = {
        "partNames": partNames,
        "starts": starts,
        "stops": stops,
    }
    for ftype in ftypes:
        data[ftype] = values[ftype]

    with open(f"{base_directory}/amps.pkl", "wb") as f:
        pickle.dump(data, f)
    
    print(f"\nSaved amps.pkl to {base_directory}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Restructure normalization integrals')
    parser.add_argument('yaml', type=str, help='yaml file with paths')
    parser.add_argument('-ia', '--include_accmc', action='store_true', help='include accmc')
    parser.add_argument('-ig', '--include_genmc', action='store_true', help='include genmc')
    parser.add_argument('-t', '--treename', type=str, default='kin', help='name of the tree in the root file')

    args = parser.parse_args()
    yaml = args.yaml
    yaml_file = OmegaConf.load(yaml)

    include_accmc = args.include_accmc
    include_genmc = args.include_genmc

    restructure_amps(yaml_file, args.treename, include_accmc, include_genmc)
