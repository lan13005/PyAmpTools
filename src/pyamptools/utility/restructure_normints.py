import argparse
import glob
import os
import pickle
import re

import numpy as np
from omegaconf import OmegaConf

#######################################################################
# Restructure the normalization integral matrices into pkl file
#
# Normalization integrals for Zlm amps are quite sparse, with ~75% zeros
# We could (in the future) use sparse matrices to reduce this saving more 
# like 60% memory since we still have to store the indicies of the 
# non-zero elements.
#######################################################################

pat_captureParts = re.compile(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?') # captures (real, imag) parts, apparently
pat_dropReac = re.compile(r'.*::(.*)') 
pat_bin = re.compile(r'.*bin_(\d+).*')
pat_reaction = re.compile(r'.*reaction_(\d+).ni')

def extract_numbers(filepath):
    bin_match = pat_bin.search(filepath)
    reaction_match = pat_reaction.search(filepath)
    bin_num = int(bin_match.group(1)) if bin_match else None
    reaction_num = int(reaction_match.group(1)) if reaction_match else None
    return (reaction_num, bin_num)

def complexls(list_of_two_strings):
    return complex(float(list_of_two_strings[0]), float(list_of_two_strings[1]))

def coherenceMatrix(strings):
    size = len(strings)
    matrix = [[1 if strings[i] == strings[j] else 0 for j in range(size)] for i in range(size)]
    return matrix

def restructure_file(fname, verbose=False):

    with open(fname, 'r') as f:

        lines = f.readlines()
        nGen, nAcc = lines[0].split('\t')
        nGen, nAcc = float(nGen.strip()), float(nAcc.strip())
        nTerms = int(lines[1].strip())
        terms = []
        for i in range(nTerms):
            terms.append(lines[2 + i].strip())

        reaction = set([term.split('::')[0] for term in terms])
        if len(reaction) != 1:
            raise ValueError("More than one reaction in terms. Not sure how this could have happened, normIntInterfaces are reaction dependent")
        reaction = list(reaction)[0]

        ampInts = []
        for i in range(nTerms):
            values = lines[2 + nTerms + i].strip().split('\t')
            values = [complexls(re.findall(pat_captureParts, v)) for v in values]
            ampInts.append(values)
        ampInts = np.array(ampInts)

        normInts = []
        for i in range(nTerms):
            values = lines[2 + 2*nTerms + i].strip().split('\t')
            values = [complexls(re.findall(pat_captureParts, v)) for v in values]
            normInts.append(values)
        normInts = np.array(normInts)

        if np.allclose(ampInts.imag, 0):
            ampInts = ampInts.real
        if np.allclose(normInts.imag, 0):
            normInts = normInts.real

        percent_zero = np.sum(ampInts==0) / ampInts.size * 100
        percent_zero = np.sum(normInts==0) / normInts.size * 100

        if verbose:
            print("----------------------------------------")
            print(f"nGen: {nGen}, nAcc: {nAcc}, nTerms: {nTerms}")
            print(f"Percentage of zeros in ampInts: {percent_zero}")
            print(f"Percentage of zeros in normInts: {percent_zero}")
            print("----------------------------------------")

    return reaction, nGen, nAcc, nTerms, terms, ampInts, normInts

## Restructure multiple normint files
def restructure_files(
    files, 
    oshape,
    verbose=False):

    nGens = []
    nAccs = []
    nterms = []
    terms = []
    ampIntss = []
    normIntss = []
    reactions = []

    for file in files:

        reaction, nGen, nAcc, nTerms, terms, ampInts, normInts = restructure_file(file, verbose=verbose)

        if len(terms) == 0:
            terms = terms
        else:
            _terms = [pat_dropReac.match(term).group(1) for term in terms]
            if set(_terms) != set(_terms):
                raise ValueError("Terms do not match")
            if verbose:
                print("Warning: Terms agree if we ignore reaction name. This could potentially be fine if you\n  constrain your reactions to be each other as in Zlm case")

        reactions.append(reaction)
        nGens.append(nGen)
        nAccs.append(nAcc)
        nterms.append(nTerms)
        ampIntss.append(ampInts)
        normIntss.append(normInts)

    if len(set(nterms)) != 1:
        raise ValueError("nTerms must be the same for all files: ", nterms)
    nterms = nterms[0]
    terms = list(terms)

    terms = [term.replace("::",".") for term in terms]

    reactions = np.array(reactions).reshape(*oshape, order='F')
    nGens = np.array(nGens).reshape(*oshape, order='F')
    nAccs = np.array(nAccs).reshape(*oshape, order='F')
    ampIntss = np.array(ampIntss).reshape(*oshape, nterms, nterms, order='F')
    normIntss = np.array(normIntss).reshape(*oshape, nterms, nterms, order='F')
    
    return reactions, nGens, nAccs, nterms, terms, ampIntss, normIntss

def restructure_normints(yaml_file):

    ############################################
    base_directory = yaml_file["base_directory"]
    output_directory = yaml_file["amptools"]["output_directory"]
    nmbReactions = len(yaml_file["polarizations"])
    nmbMasses = yaml_file["n_mass_bins"]
    nmbTprimes = yaml_file["n_t_bins"]
    oshape = (nmbReactions, nmbMasses, nmbTprimes)

    #### Alternatively, you can manually set stuff #####
    # base_directory = "/my/base/folder"
    # output_directory = "/my/output/folder" # contains all your divded data, bin_0, bin_1, ...
    # nmbMasses = 1 # should agree with your binning
    # nmbTprimes = 1 # should agree with your binning

    ############################################
    # Globbing a certain pattern and sort by values extracted by extract_numbers
    searchFmt = f"{output_directory}/bin_*/*.ni"
    # searchFmt = f"{os.path.dirname(output_directory)}/prior_sim_DATA/*.ni"

    ############################################
    # Probably dont need to modify below here

    normint_files = glob.glob(searchFmt)
    normint_files = sorted(normint_files, key=extract_numbers)

    if len(normint_files) == 0:
        raise ValueError("No normint files found! Your search string: ", searchFmt)

    normint_files = np.array(normint_files)
    reactions, nGens, nAccs, nterms, terms, ampIntss, normIntss = restructure_files(normint_files, oshape)
    normint_files = normint_files.reshape(*oshape, order='F')

    polScales = np.full(oshape, 1.0)

    unique_sums = [tuple(x.split(".")[:2]) for x in terms]
    m_sumCoherently = coherenceMatrix(unique_sums)

    with open(f"{base_directory}/normint.pkl", "wb") as f:
        data = {
            "reactions": reactions,
            "nGens": nGens,
            "nAccs": nAccs,
            "nTerms": nterms,
            "terms": terms,
            "m_sumCoherently": m_sumCoherently,
            "ampIntss": ampIntss,
            "normIntss": normIntss,
            "normint_files": normint_files,
            "polScales": polScales
        }
        pickle.dump(data, f)

    print(f"Saved normint.pkl to {base_directory}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Restructure normalization integrals')
    parser.add_argument('yaml', type=str, help='yaml file with paths')

    args = parser.parse_args()
    yaml = args.yaml
    yaml_file = OmegaConf.load(yaml)

    restructure_normints(yaml_file)
