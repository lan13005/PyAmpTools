# Yes python, because why are people using perl...

import os
import uproot
import numpy as np
import awkward as ak
import argparse
from pyamptools.utility.general import Timer, ConfigLoader
from omegaconf import OmegaConf


def split_mass(infile, outputBase, lowMass, highMass, nBins, treeName="kin", maxEvents=4294967000, return_t=False):
    """
    Split events into bins based on invariant mass and save to separate ROOT files.
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

        # Define the number of bins and create bin edges
        bins = np.linspace(lowMass, highMass, nBins + 1)

        # Read the relevant branches as arrays (modify this according to your data structure)
        # Assuming the branches are named px, py, pz, e for particles
        branches = tree.arrays(filter_name=RELEVANT_COLS, entry_stop=maxEvents)

        # Convert branches to TLorentzVector (use the appropriate branches from your data)
        particles = ak.zip({"Px": branches["Px_FinalState"], "Py": branches["Py_FinalState"], "Pz": branches["Pz_FinalState"], "E": branches["E_FinalState"]}, with_name="LorentzVector")

        beam = ak.zip({"Px": branches["Px_Beam"], "Py": branches["Py_Beam"], "Pz": branches["Pz_Beam"], "E": branches["E_Beam"]}, with_name="LorentzVector")

        # Compute the invariant mass for each event
        # Skipping the first FinalState particle (recoiling target) as per the original C++ code
        invariant_mass = compute_invariant_mass(particles[:, 1:])

        t_min = t_max = 0
        if return_t:
            mandelstam_t = compute_mandelstam_t(particles[:, 1:], beam)
            t_min = ak.min(mandelstam_t)
            t_max = ak.max(mandelstam_t)

        # Determine which bin each event falls into
        digitized_bins = np.digitize(invariant_mass, bins) - 1  # -1 to convert bin numbers to 0-indexed

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

    return t_min, t_max


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide data into mass bins")
    parser.add_argument("yaml_name", type=str, default="conf/configuration.yaml", help="Path a configuration yaml file")
    args = parser.parse_args()
    yaml_name = args.yaml_name

    cwd = os.getcwd()
    timer = Timer()

    print("\n---------------------")
    print("Running divide_data.py")
    print(f"  yaml location: {yaml_name}")
    print("---------------------\n")

    yaml_file = OmegaConf.load(yaml_name)

    cfg = ConfigLoader(yaml_file)

    print("\n\n>>>>>>>>>>>>> ConfigLoader >>>>>>>>>>>>>>>")
    min_mass = cfg("min_mass")
    max_mass = cfg("max_mass")
    n_mass_bins = cfg("n_mass_bins")
    base_directory = cfg("base_directory")
    output_directory = cfg("amptools.output_directory")
    data_folder = cfg("data_folder")
    pols = cfg("polarizations")
    perform_split_mass = cfg("perform_split_mass", default=True)
    amptools_cfg = f"{base_directory}/amptools.cfg"
    print("<<<<<<<<<<<<<< ConfigLoader <<<<<<<<<<<<<<\n\n")

    print(f"Creating directory {output_directory}")
    os.system(f"mkdir -p {output_directory}")
    os.chdir(output_directory)

    ##########################################
    # SPLIT THE INPUT DATASETS INTO MASS BINS
    ##########################################

    # Attempt to determine the t-range by looking at the genmc file
    t_min, t_max = 1e9, -1e9

    if perform_split_mass:
        # Check if all n_mass_bins folders "bin_{}" have already been created. If not, create them
        check_for_preexisting = all([os.path.exists(f"bin_{i}") for i in range(n_mass_bins)])

        if check_for_preexisting:
            print("Binned folders already exist, skipping split_mass")
            exit(0)

        for pol in pols:
            print(f"Splitting datasets with pol: {pol}")
            for ftype in ["data", "accmc", "genmc"]:
                print(f"Splitting {ftype}{pol}.root")
                assert os.path.exists(f"{data_folder}/{ftype}{pol}.root"), f"File {data_folder}/{ftype}{pol}.root does not exist"

                # Perform split_mass but also attempt to determine t-range
                _t_min, _t_max = split_mass(f"{data_folder}/{ftype}{pol}.root", f"{ftype}{pol}", min_mass, max_mass, n_mass_bins, "kin", return_t=ftype == "genmc")
                if ftype == "genmc":
                    if _t_min < t_min:
                        t_min = _t_min
                    if _t_max > t_max:
                        t_max = _t_max

            # background file is optional, would assume data is pure weighted signal
            if os.path.exists(f"{data_folder}/bkgnd{pol}.root"):
                split_mass(f"{data_folder}/bkgnd{pol}.root", f"bkgnd{pol}", min_mass, max_mass, n_mass_bins, "kin")
            else:
                print(f"No bkgnd{pol}.root found (not required), skipping")

        # ###########################################
        # # RENAME AND COPY+MODIFY AMPTOOLS CFG FILES
        # ###########################################

        for i in range(n_mass_bins):
            print(f"Perform final preparation for mass bin: {i}")
            os.system(f"mkdir -p bin_{i}")
            os.system(f"cp -f {amptools_cfg} bin_{i}/bin_{i}.cfg")
            replace_fitname = f"sed -i 's|PLACEHOLDER_FITNAME|bin_{i}|g' bin_{i}/bin_{i}.cfg"
            os.system(replace_fitname)
            for pol in pols:
                for ftype in ["data", "accmc", "genmc", "bkgnd"]:
                    os.system(f"mv -f {ftype}{pol}_{i}.root bin_{i}/{ftype}{pol}.root > /dev/null 2>&1")  # ignore error
                    replace_cmd = "sed -i 's|{}|{}|g' bin_{}/bin_{}.cfg"
                    search = f"PLACEHOLDER_{ftype.upper()}_{pol}"
                    replace = f"{output_directory}/bin_{i}/{ftype}{pol}.root"
                    os.system(replace_cmd.format(search, replace, i, i))
            os.system(f"touch bin_{i}/seed.txt")
            os.system(f"echo '\ninclude {output_directory}/bin_{i}/seed.txt' >> bin_{i}/bin_{i}.cfg")

        ################################
        # Append timing info to metadata
        ################################
        # Read amptools.cfg and append the t-range to a metadata section using omegaconf
        os.chdir(cwd)
        fit_yaml = OmegaConf.load(yaml_name)
        t_min = float(t_min)
        t_max = float(t_max)
        if "metadata" not in fit_yaml:
            fit_yaml.metadata = {}
        fit_yaml.metadata.update({"t_min": t_min, "t_max": t_max})
        start_time, end_time, elapsed_time = timer.read()
        fit_yaml.metadata.update({"divide_data_start_time": start_time, "divide_data_end_time": end_time, "divide_data_elapsed_time": elapsed_time})
        OmegaConf.save(fit_yaml, yaml_name)
