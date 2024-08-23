import argparse
import multiprocessing
import os

import awkward as ak
import numpy as np
import uproot


def process_bin(i, j, branches, digitized_bins, digitized_t_bins, nBins, outputBase, treeName):
    """
    Process a specific (i, j) bin, filter events, and write them to a ROOT file.

    Args:
        i (int): Mass bin index
        j (int): t bin index
        branches (dict): Branches from the ROOT file
        digitized_bins (ndarray): Array of mass bin indices an event falls into
        digitized_t_bins (ndarray): Array of t bin indices an event falls into
        nBins (int): Number of mass bins
        outputBase (str): Base name for output ROOT files
        treeName (str): Name of the TTree in the input ROOT file
    Returns:
        tuple: (i, j, nBar value, nBar_err value)
    """
    events_in_bin = branches[(digitized_bins == i) & (digitized_t_bins == j)]
    nevents = ak.sum((digitized_bins == i) & (digitized_t_bins == j))

    if nevents == 0:
        raise ValueError(f"No events in bin (massBin={i}, tBin={j}). Currently we cannot proceed with empty bins for fitting")

    output_data = {name: events_in_bin[name] for name in branches.fields}

    if "Weight" in output_data:
        nBar_val = ak.sum(output_data["Weight"])
        nBar_err_val = np.sqrt(ak.sum(output_data["Weight"] ** 2))
    else:
        nBar_val = nevents
        nBar_err_val = np.sqrt(nevents)

    k = j * nBins + i
    with uproot.recreate(f"{outputBase}_{k}.root") as outfile:
        print(f"Writing {nevents} events ({nBar_val:0.2f} weighted events) to {outputBase}_{k}.root ~> (massBin={i}, tBin={j})...")
        outfile[treeName] = output_data

    return (i, j, nBar_val, nBar_err_val)

def split_mass_t(infile, outputBase, 
                lowMass, highMass, nBins, 
                lowT, highT, nTBins,
                split_pool_size=1,
                treeName="kin", 
                keep_all_columns=True,
                maxEvents=4294967000, mass_edges=None, t_edges=None):
    """
    Split events into bins based on (invariant mass, t) and save to separate ROOT files.

    Args:
        infile (str): Input ROOT file
        outputBase (str): Base name for output ROOT files
        lowMass (float): Lower bound of mass range
        highMass (float): Upper bound of mass range
        nBins (int): Number of mass bins
        lowT (float): Lower bound of t range
        highT (float): Upper bound of t range
        nTBins (int): Number of t bins
        split_pool_size (int): Number of parallel processes to split the data, default 1
        treeName (str): Name of the TTree in the input ROOT file
        mass_edges (List[float]): Bin the data with these bin edges (nBins + 1 elements), default None
        t_edges (List[float]): Bin the data with these bin edges (nTBins + 1 elements), default None
        maxEvents (int): Maximum number of events to process
        keep_all_columns (bool): Keep all columns in the output ROOT files, default False

    Returns:
        mass_edges (List[float]): Bin edges (nBins + 1 elements)
        t_edges (List[float]): Bin edges (nTBins + 1 elements)
    """

    RELEVANT_COLS = ["Px_FinalState", "Py_FinalState", "Pz_FinalState", "E_FinalState", "Px_Beam", "Py_Beam", "Pz_Beam", "E_Beam", "Weight", "NumFinalState"]

    def compute_invariant_mass(particles):
        """
        Compute the invariant mass of a system of particles.

        Args:
            particles: awkward array of TLorentz vectors
        """

        # Axis -1 would be
        mass = np.sqrt((ak.sum(particles.E, axis=-1)) ** 2 - (ak.sum(particles.Px, axis=-1)) ** 2 - (ak.sum(particles.Py, axis=-1)) ** 2 - (ak.sum(particles.Pz, axis=-1)) ** 2)

        return mass

    def compute_mandelstam_t(particles, beam):
        """
        Compute the Mandelstam t of a system of particles.

        Args:
            particles: awkward array of TLorentz vectors
        """

        # 4-momentum mandelstam variable t
        x_sq = (ak.sum(particles.Px, axis=-1) - beam.Px) ** 2
        y_sq = (ak.sum(particles.Py, axis=-1) - beam.Py) ** 2
        z_sq = (ak.sum(particles.Pz, axis=-1) - beam.Pz) ** 2
        e_sq = (ak.sum(particles.E, axis=-1) - beam.E) ** 2
        t = x_sq + y_sq + z_sq - e_sq

        return t

    # Open the input file
    if not os.path.exists(infile):
        raise FileNotFoundError(f"File {infile} not found.")

    with uproot.open(infile) as file:
        tree = file[treeName]

        # Read the relevant branches as arrays (modify this according to your data structure)
        # Assuming the branches are named px, py, pz, e for particles
        if keep_all_columns:
            branches = tree.arrays(entry_stop=maxEvents)
        else:
            branches = tree.arrays(filter_name=RELEVANT_COLS, entry_stop=maxEvents)

        # Convert branches to TLorentzVector (use the appropriate branches from your data)
        particles = ak.zip({"Px": branches["Px_FinalState"], "Py": branches["Py_FinalState"], "Pz": branches["Pz_FinalState"], "E": branches["E_FinalState"]}, with_name="LorentzVector")

        beam = ak.zip({"Px": branches["Px_Beam"], "Py": branches["Py_Beam"], "Pz": branches["Pz_Beam"], "E": branches["E_Beam"]}, with_name="LorentzVector")

        # Compute the invariant mass for each event
        # Skipping the first FinalState particle (recoiling target) as per the original C++ code
        invariant_mass = compute_invariant_mass(particles[:, 1:])

        mandelstam_t = compute_mandelstam_t(particles[:, 1:], beam)

        # Define the number of bins and create bin edges
        if mass_edges is None:
            mass_edges = np.linspace(lowMass, highMass, nBins + 1)
        else:
            mass_edges = np.array(mass_edges)

        if t_edges is None:
            t_edges = np.linspace(lowT, highT, nTBins + 1)
        else:
            t_edges = np.array(t_edges)

        # Determine which bin each event falls into
        digitized_bins = np.digitize(invariant_mass, mass_edges) - 1  # -1 to convert bin numbers to 0-indexed
        digitized_t_bins = np.digitize(mandelstam_t, t_edges) - 1  # -1 to convert bin numbers to 0-indexed

        nBar = np.empty((nBins, nTBins), dtype=float)
        nBar_err = np.empty((nBins, nTBins), dtype=float)

        bin_indices = [(i, j) for j in range(nTBins) for i in range(nBins)]
        with multiprocessing.Pool(split_pool_size) as pool:
            results = pool.starmap(process_bin, [(i, j, branches, digitized_bins, digitized_t_bins, nBins, outputBase, treeName) for i, j in bin_indices])

        nBar = np.empty((nBins, nTBins), dtype=float)
        nBar_err = np.empty((nBins, nTBins), dtype=float)
        for i, j, nBar_val, nBar_err_val in results:
            nBar[i, j] = nBar_val
            nBar_err[i, j] = nBar_err_val

    mass_edges = [float(np.round(edge, 5)) for edge in mass_edges]
    t_edges = [float(np.round(edge, 5)) for edge in t_edges]

    return mass_edges, t_edges, nBar, nBar_err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split events into bins based on invariant mass and save to separate ROOT files.")
    parser.add_argument("infile", type=str, help="Input ROOT file")
    parser.add_argument("outputBase", type=str, help="Base name for output ROOT files")
    parser.add_argument("lowMass", type=float, help="Lower bound of mass range")
    parser.add_argument("highMass", type=float, help="Upper bound of mass range")
    parser.add_argument("nBins", type=int, help="Number of mass bins")
    parser.add_argument("lowT", type=float, help="Lower bound of t range")
    parser.add_argument("highT", type=float, help="Upper bound of t range")
    parser.add_argument("nTBins", type=int, help="Number of t bins")
    parser.add_argument("-n", "split_pool_size", type=int, default=1, help="Number of parallel processes to split the data")
    parser.add_argument("--treeName", type=str, default="kin", help="Name of the TTree in the input ROOT file")
    parser.add_argument("--mass_edges", type=list, default=None, help="Bin the data with these mass-bin edges (nBins + 1 elements). If None, will be computed based on other cfg options")
    parser.add_argument("--t_edges", type=list, default=None, help="Bin the data with these t-bin edges (nBins + 1 elements). If None, will be computed based on other cfg options")
    parser.add_argument("--maxEvents", type=int, default=4294967000, help="Maximum number of events to process")
    args = parser.parse_args()

    split_mass_t(args.infile, args.outputBase, 
                args.lowMass, args.highMass, args.nBins, 
                args.lowT, args.highT, args.nTBins,
                args.split_pool_size,
                args.treeName, args.mass_edges, args.t_edges, args.maxEvents)
