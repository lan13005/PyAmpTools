import glob
import itertools
import os
import re
from collections import defaultdict
from pyamptools.utility.general import glob_sort_captured

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

def get_nEventsInBin(base, acceptance_correct=True):
    """
    Load the expected number of events across all bins in a given directory. Searches metadata.txt files
    for the number of signal events (data-bkgnd). These values are calculated when the data divided using
    run_divideData + split_mass_t

    Returns:
        np.array: array of nBar values for each bin. It is up to the user to reshape to match kinematic binnings
    """

    signal_str = "nBar signal"
    if acceptance_correct:
        signal_str = "nBar corrected signal"
    intrisic_spaces = signal_str.count(" ")

    fs = glob.glob(f"{base}/bin_*/metadata.txt")
    fs = sorted(fs, key=lambda x: int(re.search(r"bin_(\d+)", x).group(1)))

    values = []
    errors = []
    for f in fs:
        with open(f) as f:
            for line in reversed(f.readlines()): # signal line is actually close to being the last line currently
                if line.startswith(signal_str):
                    line = line.strip()
                    parts = line.split(" ")
                    value, error = float(parts[intrisic_spaces+1]), float(parts[intrisic_spaces+3])
                    values.append(value)
                    errors.append(error)
                    break

    return np.array(values), np.array(errors)

# NOTE: This could in principle be done directly with amptools FitResults class but
#       when running nifty mpi fits it causes a crash. Unsure about the problem, but
#       this is a workaround for now. Parsing the file is not slow anyways
def parse_fit_file(filename):
    
    # Dictionary to store real and imaginary parts
    complex_amps_builder = defaultdict(int) # default value is 0 for ints
    status_dict = {}
    
    # Read the file
    with open(filename, 'r') as f:
        for line in f:
            line = line.replace("\s+", " ").strip()
            
            if "bestMinimum" in line:
                status_dict["bestMinimum"] = float(line.split()[1]); continue
            elif "lastMinuitCommandStatus" in line:
                status_dict["lastMinuitCommandStatus"] = int(line.split()[1]); continue
            elif "eMatrixStatus" in line:
                status_dict["eMatrixStatus"] = int(line.split()[1]); continue

            if "_re" not in line and "_im" not in line: continue
            
            # Split line into name and value
            name, value_str = line.strip().split('\t')
            try:
                value = float(value_str)
            except ValueError: continue
                
            # Parse the amplitude name
            parts = name.split('::')
            if len(parts) != 3: continue
            reaction, sum_type, amp_part = parts
            
            # Split amp_part into base name and re/im indicator
            match = re.match(r'(.+)_(re|im)$', amp_part)
            if not match: continue
            base_name, component = match.groups()
            
            # Store value
            key = (reaction, sum_type, base_name)
            complex_amps_builder[key] += value * (1j if component == 'im' else 1)
            
    complex_amps = {}
    for key, value in complex_amps_builder.items():
        if key[2] not in complex_amps:
            complex_amps[key[2]] = value
        else:
            if complex_amps[key[2]] != value:
                print(f"io| ERROR: {key[2]} is expected to be the same as it should be constrained to be the same")
                print(f"io|  {key[2]} = {value}")
    
    return complex_amps, status_dict

def loadAmpToolsResultsFromYaml(yaml):
    """ Helper function to load the amptools results from a yaml file """
    search_format = yaml['amptools']['search_format']
    cfgfiles = glob_sort_captured(yaml['amptools']['output_directory']+f"/{search_format}_*/bin_[].cfg")
    min_mass = yaml['min_mass']
    max_mass = yaml['max_mass']
    if yaml['n_mass_bins'] % yaml['amptools']['bins_per_group'] != 0:
        raise ValueError("n_mass_bins must be divisible by amptools.bins_per_group")
    bpg = yaml['amptools']['bins_per_group']
    n_mass_bins = int(yaml['n_mass_bins'] / bpg)
    massBins = np.linspace(min_mass, max_mass, n_mass_bins+1)
    masses = 0.5 * (massBins[1:] + massBins[:-1])
    min_t = yaml['min_t']
    max_t = yaml['max_t']
    n_t_bins = yaml['n_t_bins']
    tPrimeBins = np.linspace(min_t, max_t, n_t_bins+1)
    tPrimes = 0.5 * (tPrimeBins[1:] + tPrimeBins[:-1])
    # No support for multiple fits per bin, choose best
    mle_query_1 = yaml['amptools'].get('mle_query_1', '')
    mle_query_2 = yaml['amptools'].get('mle_query_2', '')
    if mle_query_1 is None: mle_query_1 = ''
    if mle_query_2 is None: mle_query_2 = ''
    n_randomizations = yaml['amptools']['n_randomizations']
    accCorrect = yaml['acceptance_correct']
    # print(f"Loading {len(cfgfiles)} fits...")
    df = loadAmpToolsResults(cfgfiles, masses, tPrimes, n_randomizations, mle_query_1, mle_query_2, accCorrect)
    unique_kin_pairs = df.value_counts(subset=['mass', 'tprime'])
    if len(unique_kin_pairs) != len(df):
        # NOTE: seems like we only need to implement t'-summed intensity section. ATM it seems like it would just sum over all fit results in a bin
        raise ValueError("Currently there is no support for plotting multiple fits per kinematic bin with 'iftpwa_plot' program. Tighten your mle_queries. For example, you can try `mle_query_1: ''` and `mle_query_2: 'delta_nll==0'` which choose the best MLE fit out of all fits (not just converged ones)")
    return df, (masses, tPrimeBins, bpg)

def loadAmpToolsResults(cfgfiles, masses, tPrimes, niters, mle_query_1, mle_query_2, accCorrect):
    """
    Load results from AmpTools

    Args:
        cfgfiles (List): list of config files to load
        masses (List/Array): list/array of masses used in the fit (does not have to match cfgfiles)
        tPrimes (List/Array): list/array of t values used in the fit (does not have to match cfgfiles)
        niters (int): number of iterations used in the fit
        mle_query_1 (str): query to apply to the DataFrame BEFORE calculating delta_nll
        mle_query_2 (str): query to apply to the DataFrame AFTER  calculating delta_nll
        accCorrect (bool): whether to use acceptance corrected signal values

    Returns:
        DataFrame: DataFrame of results
    """

    lor = 0 if accCorrect else 1 # Left or right of | symbol in the output of extract_ff. Gets corrected values or not

    df = {}

    _nlls = []
    _masses = []
    _tPrimes = []
    _iterations = []
    _statuses = []
    _ematrix = []

    for cfgfile, i in itertools.product(cfgfiles, range(niters)):

        basedir = os.path.dirname(cfgfile)
        binTag = basedir.split("/")[-1]
        binTag = "bin_" + "_".join(binTag.split("_")[1:])
        fit_file = f"{basedir}/{binTag}_{i}.fit"

        if not os.path.exists(fit_file):
            # print(f"{fit_file} expected, but not found!")
            continue

        amps, status_dict = parse_fit_file(fit_file)
        value = status_dict["bestMinimum"]
        status = status_dict["lastMinuitCommandStatus"]
        ematrix = status_dict["eMatrixStatus"]
        
        for key, value in amps.items():
            key = f"{key}_amp"
            if key in df:
                df[key].append(value)
            else:
                df[key] = [value]
    
        _nlls.append(value)
        binNum = int(binTag.split("_")[-1])
        _masses.append(masses[binNum % len(masses)])
        _tPrimes.append(tPrimes[binNum // len(masses)])
        _iterations.append(i)
        _statuses.append(status)
        _ematrix.append(ematrix)

        observed_pairs = set()
        fname = f"{basedir}/intensities_{i}.txt"
        with open(fname) as f:
            totalYield = 0
            for line in f.readlines():
                if line.startswith("TOTAL EVENTS"):
                    
                    # Store both intensity and acceptance corrected intensity
                    intensity_corr, intensity = line.split()[3].split("|")
                    if 'intensity' not in df: df['intensity'] = [float(intensity)]
                    else: df['intensity'].append(float(intensity))
                    if 'intensity_corr' not in df: df['intensity_corr'] = [float(intensity_corr)]
                    else: df['intensity_corr'].append(float(intensity_corr))

                if line.startswith("FIT FRACTION") and "::" not in line:  # Regex-merged to group Re/Im + Pols
                    amp = line.split()[2]

                    # FILL VALUES
                    if amp in df:
                        df[amp].append(float(line.split()[4].split("|")[lor]) * totalYield)
                        df[f"{amp} err"].append(float(line.split()[6].split("|")[lor]) * totalYield)
                    else:
                        df[amp] = [float(line.split()[4].split("|")[lor]) * totalYield]
                        df[f"{amp} err"] = [float(line.split()[6].split("|")[lor]) * totalYield]

                if line.startswith("PHASE DIFFERENCE"):
                    amp1 = line.split()[2].split("::")[-1]
                    amp2 = line.split()[3].split("::")[-1]

                    phaseDiff = float(line.split()[5])
                    phaseDiff_err = float(line.split()[7])

                    pair = amp1 + " " + amp2

                    if pair in df:
                        # Each file should have a unique phase diff between pairs of amps
                        #  We can check for this by checking the length of the list.
                        #  We need this check since In FitResults output file, different
                        #  polarizations and real/imag parts shares values and the phase
                        #  calculations are repeated
                        if pair not in observed_pairs:
                            df[pair].append(phaseDiff)
                            df[f"{pair} err"].append(phaseDiff_err)
                            observed_pairs.add(pair)
                    else:
                        assert i == 0  # sanity check we are on first file; create initial list
                        df[pair] = [phaseDiff]
                        df[f"{pair} err"] = [phaseDiff_err]

                    observed_pairs.add(pair)

    df["nll"] = _nlls
    df["mass"] = _masses
    df["tprime"] = _tPrimes
    df["iteration"] = _iterations
    df["status"] = _statuses
    df["ematrix"] = _ematrix

    df = pd.DataFrame(df)
        
    # reorder columns so important ones are first
    cols = ["tprime", "mass", "nll", "iteration", "status", "ematrix"]
    cols.extend([k for k in df.keys() if k not in cols])
    df = df[cols]

    # Apply Query 1
    if mle_query_1 != "":
        df = df.query(mle_query_1)
        print(f"io| Remaining number of rows after `mle_query_1` ({mle_query_1}): {len(df)}")

    # groupby mass and subtract the min nll in each kinematic bin
    # create a new column called delta_nll and subtract the min nll in each kinematic bin
    df["delta_nll"] = df.groupby(["tprime","mass"])["nll"].transform(lambda x: x - x.min())

    # Apply Query 2
    if mle_query_2 != "":
        df = df.query(mle_query_2)
        print(f"io| Remaining number of rows after `mle_query_2` ({mle_query_2}): {len(df)}")

    # This is the case if there are 0 fit result files loaded
    if len(df) == 0:
        # print("No amptools MLE fit results loaded! Returning empty DataFrame...")
        return df

    avail_t = list(df["tprime"].unique())
    avail_m = list(df["mass"].unique())
    avail_tm = list(itertools.product(avail_m, avail_t))
    needed_tm = list(itertools.product(masses, tPrimes))
    missing_tm = np.array(list(set(needed_tm) - set(avail_tm)))

    # match missing_tm to appropriate bins
    missing_tm_idxs = []
    absolute_idxs = []
    for tm in missing_tm:
        m, t = tm
        m_idx = np.where(np.abs(masses - m) < 1e-5)[0][0]
        t_idx = np.where(np.abs(tPrimes - t) < 1e-5)[0][0]
        idx = m_idx + t_idx * len(masses)
        missing_tm_idxs.append((m_idx, t_idx))
        absolute_idxs.append(idx)
    missing_tm_idxs = np.array(missing_tm_idxs)
    absolute_idxs = np.array(absolute_idxs)
    absolute_idxs = np.sort(absolute_idxs)

    if len(missing_tm) > 0:
        print("\nio| loadAmpToolsResults did not collect the correct number of masses/tPrimes, perhaps mle_queries are too strict?")
        print(f"io|  --> Missing bin indices: {absolute_idxs}")
        raise ValueError("Some kinematic bins had no converged MLE fits. This could be due to the YAML `mle_query_1` and `mle_query_2` could be too restrictive. Exiting...")

    return df
