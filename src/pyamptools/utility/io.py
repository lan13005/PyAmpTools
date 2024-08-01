import glob
import itertools
import os
import re

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

    def loadParValueFromFit(fitFile, key):
        with open(fitFile, "r") as f:
            for line in f:
                line = line.replace("\s+", " ").strip()
                if key in line:
                    return float(line.split()[1])
        return None

    for cfgfile, i in itertools.product(cfgfiles, range(niters)):

        basedir = os.path.dirname(cfgfile)
        binTag = basedir.split("/")[-1]
        fit_file = f"{basedir}/{binTag}_{i}.fit"

        if not os.path.exists(fit_file):
            raise FileNotFoundError(f"{fit_file} expected, but not found!")
    
        value = loadParValueFromFit(fit_file, "bestMinimum")
        _nlls.append(value)
        binNum = int(binTag.split("_")[-1])
        _masses.append(masses[binNum % len(masses)])
        _tPrimes.append(tPrimes[binNum // len(masses)])
        _iterations.append(i)
        status = int(loadParValueFromFit(fit_file, "lastMinuitCommandStatus"))
        _statuses.append(status)
        ematrix = int(loadParValueFromFit(fit_file, "eMatrixStatus"))
        _ematrix.append(ematrix)

        observed_pairs = set()
        fname = f"{basedir}/intensities_{i}.txt"
        with open(fname) as f:
            totalYield = 0
            for line in f.readlines():
                if line.startswith("TOTAL EVENTS"):
                    totalYield = float(line.split()[3].split("|")[lor])
                    if "total" in df:
                        df["total"].append(totalYield)
                    else:
                        df["total"] = [totalYield]

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

    # Apply Query 1
    if mle_query_1 != "":
        df = df.query(mle_query_1)

    # groupby mass and subtract the min nll in each kinematic bin
    # create a new column called delta_nll and subtract the min nll in each kinematic bin
    df["delta_nll"] = df.groupby(["mass","tprime"])["nll"].transform(lambda x: x - x.min())

    # Apply Query 2
    if mle_query_2 != "":
        df = df.query(mle_query_2)

    df = pd.DataFrame(df)

    # This is the case if there are 0 fit result files loaded
    if len(df) == 0:
        print("No amptools MLE fit results loaded! Returning empty DataFrame...")
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
        print("\nloadAmpToolsResults did not collect the correct number of masses/tPrimes, perhaps mle_queries are too strict?")
        print(f"  --> Missing bin indices: {absolute_idxs}")
        raise ValueError("Some kinematic bins had no converged MLE fits. See logs. Exiting...")

    return df
