import os
import argparse
from pyamptools.utility.general import Timer, load_yaml, dump_yaml
from omegaconf import OmegaConf
from pyamptools.split_mass import split_mass

############################################################################
# This makes calls to pyamptools' split_mass function to divide the data
# sources into separate mass bins and copies over the amptools configuration
# file. A flag can be used to distribute the data evenly across the mass bins
############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide data into mass bins")
    parser.add_argument("yaml_name", type=str, default="conf/configuration.yaml", help="Path a configuration yaml file")
    args = parser.parse_args()
    yaml_name = args.yaml_name

    cwd = os.getcwd()
    timer = Timer()

    print("\n---------------------")
    print(f"Running {__file__}")
    print(f"  yaml location: {yaml_name}")
    print("---------------------\n")

    yaml_file = load_yaml(yaml_name)

    print("\n\n>>>>>>>>>>>>> ConfigLoader >>>>>>>>>>>>>>>")
    min_mass = yaml_file["min_mass"]
    max_mass = yaml_file["max_mass"]
    n_mass_bins = yaml_file["n_mass_bins"]
    evenly_distribute = yaml_file["evenly_distribute"]
    base_directory = yaml_file["base_directory"]
    output_directory = yaml_file["amptools"]["output_directory"]
    data_folder = yaml_file["data_folder"]
    pols = yaml_file["polarizations"]
    prepare_for_nifty = bool(yaml_file["amptools"]["prepare_for_nifty"])
    amptools_cfg = f"{base_directory}/amptools.cfg"
    print("<<<<<<<<<<<<<< ConfigLoader <<<<<<<<<<<<<<\n\n")

    print(f"Creating directory {output_directory}")
    os.system(f"mkdir -p {output_directory}")
    os.chdir(output_directory)

    ##########################################
    # SPLIT THE INPUT DATASETS INTO MASS BINS
    ##########################################

    # Check if all n_mass_bins folders "bin_{}" have already been created. If not, create them
    check_for_preexisting = all([os.path.exists(f"bin_{i}") for i in range(n_mass_bins)])

    if check_for_preexisting:
        print("Binned folders already exist, skipping split_mass")
        exit(0)

    mass_edges = None  # Determine mass bin edges based on the first ftype source
    for pol in pols:
        print(f"Splitting datasets with pol: {pol}")
        for ftype in ["data", "accmc", "genmc"]:
            print(f"Splitting {ftype}{pol}.root")
            assert os.path.exists(f"{data_folder}/{ftype}{pol}.root"), f"File {data_folder}/{ftype}{pol}.root does not exist"

            # Perform split_mass
            mass_edges = split_mass(f"{data_folder}/{ftype}{pol}.root", f"{ftype}{pol}", min_mass, max_mass, n_mass_bins, "kin", mass_edges=mass_edges, evenly_distribute=evenly_distribute)

        # background file is optional, would assume data is pure weighted signal
        if os.path.exists(f"{data_folder}/bkgnd{pol}.root"):
            print(f"Splitting bkgnd{pol}.root")
            mass_edges = split_mass(f"{data_folder}/bkgnd{pol}.root", f"bkgnd{pol}", min_mass, max_mass, n_mass_bins, "kin", mass_edges=mass_edges, evenly_distribute=evenly_distribute)
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

        if prepare_for_nifty:
            os.system(f"touch bin_{i}/seed_nifty.txt")
            os.system(f"echo '\ninclude {output_directory}/bin_{i}/seed_nifty.txt' >> bin_{i}/bin_{i}.cfg")

    ################################
    # Append timing info to metadata
    ################################
    # Read amptools.cfg and append the t-range to a metadata section using omegaconf
    os.chdir(cwd)
    output_yaml = OmegaConf.load(yaml_name)
    output_yaml.update({"mass_edges": mass_edges})
    dump_yaml(output_yaml, yaml_name)
