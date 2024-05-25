import itertools
import os

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


def loadAmpToolsResults(cfgfiles, masses, niters, mle_query_1, mle_query_2):
    """
    Load results from AmpTools

    Args:
        cfgfiles (List): list of config files to load
        masses (List/Array): list/array of masses used in the fit (does not have to match cfgfiles)
        niters (int): number of iterations used in the fit
        mle_query_1 (str): query to apply to the DataFrame BEFORE calculating delta_nll
        mle_query_2 (str): query to apply to the DataFrame AFTER  calculating delta_nll

    Returns:
        DataFrame: DataFrame of results
    """

    df = {}

    _nlls = []
    _masses = []
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
            continue
    
        value = loadParValueFromFit(fit_file, "bestMinimum")
        _nlls.append(value)
        _masses.append(masses[int(binTag.split("_")[-1])])
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
                    totalYield = float(line.split()[3])
                    if "total" in df:
                        df["total"].append(totalYield)
                    else:
                        df["total"] = [totalYield]

                if line.startswith("FIT FRACTION") and "::" not in line:  # Regex-merged to group Re/Im + Pols
                    amp = line.split()[2]

                    # FILL VALUES
                    if amp in df:
                        df[amp].append(float(line.split()[4]) * totalYield)
                        df[f"{amp} err"].append(float(line.split()[6]) * totalYield)
                    else:
                        df[amp] = [float(line.split()[4]) * totalYield]
                        df[f"{amp} err"] = [float(line.split()[6]) * totalYield]

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
    df["iteration"] = _iterations
    df["status"] = _statuses
    df["ematrix"] = _ematrix

    df = pd.DataFrame(df)

    # This is the case if there are 0 fit result files loaded
    if len(df) == 0:
        print("No amptools MLE fit results loaded! Returning empty DataFrame...")
        return df

    # Apply Query 1
    if mle_query_1 != "":
        df = df.query(mle_query_1)

    # groupby mass and subtract the min nll in each kinematic bin
    # create a new column called delta_nll and subtract the min nll in each kinematic bin
    df["delta_nll"] = df.groupby("mass")["nll"].transform(lambda x: x - x.min())

    # Apply Query 2
    if mle_query_2 != "":
        df = df.query(mle_query_2)

    df = pd.DataFrame(df)

    # Check that all masses are present
    if len(df["mass"].unique()) != len(masses):
        missing_bins = set(masses) - set(df["mass"].unique())
        print(missing_bins)
        missing_bins = [round(b, 5) for b in missing_bins]
        print(missing_bins)
        print(masses)
        missing_idxs = [np.where(np.abs(masses - b) < 1e-5)[0][0] for b in missing_bins]
        print("\nloadAmpToolsResults did not collect the correct number of masses, perhaps mle_queries are too strict?")
        print(f"  --> Missing bins: {missing_idxs}")
        print(f"  --> Missing masses: {missing_bins}")
        raise ValueError("Some kinematic bins had no converged MLE fits. See logs. Exiting...")

    return df
