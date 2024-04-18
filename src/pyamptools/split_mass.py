import awkward as ak
import numpy as np
import uproot
import argparse


def split_mass(infile, outputBase, lowMass, highMass, nBins, treeName="kin", maxEvents=4294967000, mass_edges=None, evenly_distribute=False):
    """
    Split events into bins based on invariant mass and save to separate ROOT files.

    Args:
        infile (str): Input ROOT file
        outputBase (str): Base name for output ROOT files
        lowMass (float): Lower bound of mass range
        highMass (float): Upper bound of mass range
        nBins (int): Number of mass bins
        treeName (str): Name of the TTree in the input ROOT file
        mass_edges (List[float]): Bin the data with these bin edges (nBins + 1 elements), default None
        evenly_distribute (bool): If mass_edges is not None, choose whether to evenly distribute events across bins or not
        maxEvents (int): Maximum number of events to process

    Returns:
        mass_edges (List[float]): Bin edges (nBins + 1 elements)
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
    with uproot.open(infile) as file:
        tree = file[treeName]

        # Read the relevant branches as arrays (modify this according to your data structure)
        # Assuming the branches are named px, py, pz, e for particles
        branches = tree.arrays(filter_name=RELEVANT_COLS, entry_stop=maxEvents)

        # Convert branches to TLorentzVector (use the appropriate branches from your data)
        particles = ak.zip({"Px": branches["Px_FinalState"], "Py": branches["Py_FinalState"], "Pz": branches["Pz_FinalState"], "E": branches["E_FinalState"]}, with_name="LorentzVector")

        # beam = ak.zip({"Px": branches["Px_Beam"], "Py": branches["Py_Beam"], "Pz": branches["Pz_Beam"], "E": branches["E_Beam"]}, with_name="LorentzVector")

        # Compute the invariant mass for each event
        # Skipping the first FinalState particle (recoiling target) as per the original C++ code
        invariant_mass = compute_invariant_mass(particles[:, 1:])

        # mandelstam_t = compute_mandelstam_t(particles[:, 1:], beam)

        # Define the number of bins and create bin edges
        if mass_edges is None:
            if evenly_distribute:
                mass_edges = np.quantile(invariant_mass, np.linspace(0, 1, nBins + 1))
            else:
                mass_edges = np.linspace(lowMass, highMass, nBins + 1)
        else:
            mass_edges = np.array(mass_edges)

        # Determine which bin each event falls into
        digitized_bins = np.digitize(invariant_mass, mass_edges) - 1  # -1 to convert bin numbers to 0-indexed

        for i in range(nBins):
            # Filter events for each bin
            events_in_bin = branches[digitized_bins == i]

            nevents = ak.sum(digitized_bins == i)

            # Save filtered events to a new ROOT file
            # Create a dictionary with branches to write
            # There seems to be additional branches like nE_FinalState that are created
            #   but is not needed (basically counting entries). They do not appear in the
            #   keys of output_data either. Unsure how to remove them
            output_data = {name: events_in_bin[name] for name in branches.fields}

            assert nevents > 0, f"Error when dividing data into mass bins: nevents = {nevents} for bin_{i}. Compare the mass range specific in the submit form with your provided data."

            with uproot.recreate(f"{outputBase}_{i}.root") as outfile:
                print(f"Writing to {nevents} events to {outputBase}_{i}.root...")
                outfile[treeName] = output_data

    mass_edges = [float(np.round(edge, 5)) for edge in mass_edges]

    return mass_edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split events into bins based on invariant mass and save to separate ROOT files.")
    parser.add_argument("infile", type=str, help="Input ROOT file")
    parser.add_argument("outputBase", type=str, help="Base name for output ROOT files")
    parser.add_argument("lowMass", type=float, help="Lower bound of mass range")
    parser.add_argument("highMass", type=float, help="Upper bound of mass range")
    parser.add_argument("nBins", type=int, help="Number of mass bins")
    parser.add_argument("--treeName", type=str, default="kin", help="Name of the TTree in the input ROOT file")
    parser.add_argument("--mass_edges", type=list, default=None, help="Bin the data with these bin edges (nBins + 1 elements). If None, will be computed based on other cfg options")
    parser.add_argument("--evenly_distribute", type=bool, default=False, help="Whether to evenly distribute events across bins. Constructs bin edges so bins have ~equal num events")
    parser.add_argument("--maxEvents", type=int, default=4294967000, help="Maximum number of events to process")
    args = parser.parse_args()

    split_mass(args.infile, args.outputBase, args.lowMass, args.highMass, args.nBins, args.treeName, args.mass_edges, args.maxEvents, args.evenly_distribute)
