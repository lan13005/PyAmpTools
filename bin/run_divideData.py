import argparse
import glob
import os

from omegaconf import OmegaConf
from pyamptools.split_mass import split_mass
from pyamptools.utility.general import Timer, dump_yaml, load_yaml

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
    bins_per_group = yaml_file["amptools"]["bins_per_group"] if "bins_per_group" in yaml_file["amptools"] else 1
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
    else:
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

        # Read amptools.cfg and append the t-range and mass-range to a metadata section using omegaconf
        os.chdir(cwd)
        output_yaml = OmegaConf.load(yaml_name)
        output_yaml.update({"mass_edges": mass_edges})
        dump_yaml(output_yaml, yaml_name)

    ###################################################
    # MERGE BINS INTO GROUPS TO LIMIT SPAWNED PROCESSES
    ###################################################

    if bins_per_group > 1:
        print(f"Merging bins into groups of {bins_per_group}")
        bins = glob.glob(f"{output_directory}/bin_*")
        bins = sorted(bins, key=lambda x: int(x.split("_")[-1]))
        nBins = len(bins)

        if nBins % bins_per_group != 0:
            raise ValueError(f"Number of bins ({nBins}) is not divisible by bins_per_group ({bins_per_group})")

        nGroups = nBins // bins_per_group

        # Get base information by loading in the first bin
        bin_path = bins[0]
        bin_name = bin_path.split("/")[-1]
        cfg_path = f"{bin_path}/{bin_name}.cfg"
        reactions = []
        parameters = []
        sources = []
        includes = []
        with open(cfg_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("reaction"):
                    reactions.append(line.split(" ")[1].strip())
                if line.startswith("parameter") and "parScale" in line:
                    parameters.append(line.split(" ")[1].strip())
                if any([line.startswith(source) for source in ["data", "accmc", "genmc", "bkgnd"]]):
                    sources.append(line.split(" ")[-1].strip().split("/")[-1])
                if line.startswith("include"):
                    includes.append(line.split(" ")[1].strip().split("/")[-1])


        for i in range(nGroups):
            bin_group = bins[i * bins_per_group:(i + 1) * bins_per_group]
            group_name = f"group_{i}"
            os.makedirs(f"{output_directory}/{group_name}", exist_ok=True)
            
            for j, bin in enumerate(bin_group):
                bin_name = bin.split("/")[-1]
                cfg_name = f"G{j}_{bin_name}.cfg"
                os.system(f"cp -r {bin}/{bin_name}.cfg {output_directory}/{group_name}/{cfg_name}")

                # Replace all bin names with group names
                sed_repl = f"sed -i 's|{bin_name}|group_{i}|g' {output_directory}/{group_name}/{cfg_name}"
                os.system(sed_repl)

                # Remove fit keyword lines or commented for additional files in the group as we will merge along a group
                if j != 0:
                    os.system(f"sed -i '/^fit/d' {output_directory}/{group_name}/{cfg_name}")
                    os.system(f"sed -i '/^#/d' {output_directory}/{group_name}/{cfg_name}")

                # update naming for reactions
                for reaction in reactions:
                    reaction_repl = reaction.replace("reaction", f"G{j}_reaction")
                    sed_repl = f"sed -i 's|{reaction}|{reaction_repl}|g' {output_directory}/{group_name}/{cfg_name}"
                    os.system(sed_repl)

                # update naming for parameters
                for parameter in parameters:
                    parameter_repl = parameter.replace("parScale", f"G{j}_parScale")
                    sed_repl = f"sed -i 's|{parameter}|{parameter_repl}|g' {output_directory}/{group_name}/{cfg_name}"
                    os.system(sed_repl)

                # update naming for sources
                for source in sources:
                    source_repl = source.replace("data", f"G{j}_data").replace("accmc", f"G{j}_accmc").replace("genmc", f"G{j}_genmc").replace("bkgnd", f"G{j}_bkgnd")
                    sed_repl = f"sed -i 's|{source}|{source_repl}|g' {output_directory}/{group_name}/{cfg_name}"
                    os.system(sed_repl)
                    os.system(f"cp -r {bin}/{source} {output_directory}/{group_name}/G{j}_{source}")

                # update naming
                for include in includes:
                    sed_repl = f"sed -i 's|{include}|G{j}_{include}|g' {output_directory}/{group_name}/{cfg_name}"
                    os.system(sed_repl)
                    os.system(f"cp -r {bin}/{include} {output_directory}/{group_name}/G{j}_{include}")

            # merge config files using cat
            cat_cmd = f"cat {output_directory}/{group_name}/G*bin*.cfg > {output_directory}/{group_name}/bin_{i}.cfg"
            os.system(cat_cmd)

            # clean up cfg files
            os.system(f"rm -f {output_directory}/{group_name}/G*bin*.cfg")

