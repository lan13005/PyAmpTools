import argparse
import glob
import os

from omegaconf import OmegaConf
from pyamptools.split_mass import split_mass_t
from pyamptools.utility.general import Timer, dump_yaml, load_yaml

############################################################################
# This makes calls to pyamptools' split_mass function to divide the data
# sources into separate mass bins and copies over the amptools configuration
# file. A flag can be used to distribute the data evenly across the mass bins
############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide data into mass bins")
    parser.add_argument("yaml_name", type=str, default="conf/configuration.yaml", help="Path a configuration yaml file")
    parser.add_argument("--nosplit", action="store_true", help="Skip the split_mass step")
    parser.add_argument("--nomerge", action="store_true", help="Skip the merge_bins step")
    args = parser.parse_args()
    yaml_name = args.yaml_name

    cwd = os.getcwd()
    timer = Timer()

    print("\n---------------------")
    print(f"Running {__file__}")
    print(f"  yaml location: {yaml_name}")
    print("---------------------\n")

    yaml_file = load_yaml(yaml_name)

    min_mass = yaml_file["min_mass"]
    max_mass = yaml_file["max_mass"]
    n_mass_bins = yaml_file["n_mass_bins"]
    min_t = yaml_file["min_t"]
    max_t = yaml_file["max_t"]
    n_t_bins = yaml_file["n_t_bins"]
    base_directory = yaml_file["base_directory"]
    output_directory = yaml_file["amptools"]["output_directory"]
    bins_per_group = yaml_file["bins_per_group"] if "bins_per_group" in yaml_file else 1
    data_folder = yaml_file["data_folder"]
    pols = yaml_file["polarizations"]
    prepare_for_nifty = bool(yaml_file["amptools"]["prepare_for_nifty"])
    amptools_cfg = f"{base_directory}/amptools.cfg"

    os.system(f"mkdir -p {output_directory}")
    os.chdir(output_directory)

    nBins = n_mass_bins * n_t_bins
    if nBins % bins_per_group != 0:
        raise ValueError(f"Number of bins ({nBins}) is not divisible by bins_per_group ({bins_per_group})")

    try_split_mass = not args.nosplit
    try_merge_bins = not args.nomerge

    ##########################################
    # SPLIT THE INPUT DATASETS INTO MASS BINS
    ##########################################

    # Check if all n_mass_bins folders "bin_{}" have already been created. If not, create them

    if try_split_mass:

        print("User requested data to be split into (mass, t) bins. Checking for pre-existing bins...")

        check_for_preexisting = all([os.path.exists(f"bin_{i}") for i in range(n_mass_bins * n_t_bins)])

        sharemc = False
        mc_already_shared = {"data": False, "bkgnd": False, "accmc": False, "genmc": False}
        
        if check_for_preexisting:
            delete_or_skip = input("Pre-existing bins found. Delete them? (y/n): ")
            if delete_or_skip.lower() == "y":
                os.system("rm -rf bin_*")
                print("  Deleted pre-existing bins")
            else:
                print("  Skipping split_mass")

        else:
            print("No pre-existing bins found, running split_mass")
            mass_edges = None  # Determine mass bin edges based on the first ftype source
            t_edges = None
            nBars = {}
            nBar_errs = {}
            
            for ftype in ["data", "accmc", "genmc"]:
                for pol in pols:
                    print(f"Attempting to split {ftype}{pol}.root into mass bins")
                    # AmpTools -f does not include polarization information. We can share all accmc and genmc datasets. Do so if accmc.root exists and acc{pol}.root does not exist 
                    if ftype in ["accmc", "genmc"]: # check to see if we can share the MC datasets
                        if not os.path.exists(f"{data_folder}/{ftype}{pol}.root"):
                            if not os.path.exists(f"{data_folder}/{ftype}.root"):
                                raise FileNotFoundError(f"File {data_folder}/{ftype}{pol}.root does not exist! Nor does {data_folder}/{ftype}.root (unable to share MC across polarized datasets)")
                            sharemc = True
                            if mc_already_shared[ftype]:
                                print("  * Already sharing MC datasets, will run split_mass again")
                            else:
                                print(f"  * {ftype}{pol}.root does not exist but {ftype}.root does. Will share {ftype}.root across all polarizations")
                    else:
                        if not os.path.exists(f"{data_folder}/{ftype}{pol}.root"):
                            raise FileNotFoundError(f"File {data_folder}/{ftype}{pol}.root does not exist!")

                    is_data = ftype not in ["accmc", "genmc"]
                    if is_data or not sharemc:
                        fname = f"{data_folder}/{ftype}{pol}.root"
                        oname = f"{ftype}{pol}"
                    else:
                        fname = f"{data_folder}/{ftype}.root"
                        oname = f"{ftype}"

                    # Perform split_mass
                    if is_data or not sharemc or (sharemc and not mc_already_shared[ftype]):
                        mass_edges, t_edges, nBar, nBar_err = split_mass_t(fname, oname, 
                                                    min_mass, max_mass, n_mass_bins, 
                                                    min_t, max_t, n_t_bins,
                                                    treeName="kin", mass_edges=mass_edges, t_edges=t_edges)
                        _pol = "shared" if sharemc else pol
                        nBars[(ftype, _pol)] = nBar
                        nBar_errs[(ftype, _pol)] = nBar_err
                        if sharemc:
                            mc_already_shared[ftype] = True

            # background file is optional, would assume data is pure weighted signal
            for pol in pols:
                if os.path.exists(f"{data_folder}/bkgnd{pol}.root"):
                    mass_edges, t_edges, nBar, nBar_err = split_mass_t(f"{data_folder}/bkgnd{pol}.root", f"bkgnd{pol}", 
                                                        min_mass, max_mass, n_mass_bins, 
                                                        min_t, max_t, n_t_bins,
                                                        treeName="kin", mass_edges=mass_edges, t_edges=t_edges)
                    nBars[("bkgnd", pol)] = nBar
                    nBar_errs[("bkgnd", pol)] = nBar_err
                else:
                    print(f"No bkgnd{pol}.root found (not required), skipping")

            # ###########################################
            # # RENAME AND COPY+MODIFY AMPTOOLS CFG FILES
            # ###########################################

            for j in range(n_t_bins):
                for i in range(n_mass_bins):
                    k = j * n_mass_bins + i
                    print(f"Perform final preparation for bin: {k}")
                    os.system(f"mkdir -p bin_{k}")
                    os.system(f"cp -f {amptools_cfg} bin_{k}/bin_{k}.cfg")
                    with open(f"bin_{k}/metadata.txt", "w") as f:
                        f.write("---------------------\n")
                        f.write(f"bin number: {k}\n")
                        f.write(f"mass range: {mass_edges[i]:.2f} - {mass_edges[i + 1]:.2f} GeV\n")
                        f.write(f"t range: {t_edges[j]:.2f} - {t_edges[j + 1]:.2f} GeV^2\n")
                        nBar_ftype = {pair[0]: 0 for pair in nBars.keys()} # keys ~ (ftype, pol)
                        nBar_err_ftype = {pair[0]: 0 for pair in nBars.keys()}
                        f.write("---------------------\n")
                        for (ftype, pol) in nBars:
                            f.write(f"nBar {ftype} {pol}: {nBars[(ftype, pol)][i][j]} +/- {nBar_errs[(ftype, pol)][i][j]}\n")
                            if ftype in ["data", "bkgnd"]:
                                nBar_ftype[ftype] += nBars[(ftype, pol)][i][j]
                                nBar_err_ftype[ftype] += nBar_errs[(ftype, pol)][i][j]**2
                            if ftype in ["accmc", "genmc"]:
                                share_factor = (len(pols) if pol == "shared" else 1)
                                nBar_ftype[ftype] += nBars[(ftype, pol)][i][j] * share_factor
                                nBar_err_ftype[ftype] += (nBar_errs[(ftype, pol)][i][j] * share_factor)**2
                        f.write("---------------------\n")
                        for ftype in nBar_ftype:
                            f.write(f"nBar {ftype}: {nBar_ftype[ftype]} +/- {nBar_err_ftype[ftype]**0.5}\n")
                        f.write("---------------------\n")
                        signal = nBar_ftype['data'] - nBar_ftype['bkgnd']
                        signal_err = (nBar_err_ftype['data'] + nBar_err_ftype['bkgnd'])**0.5
                        f.write(f"nBar signal: {signal} +/- {signal_err}\n")
                        eff = nBar_ftype['accmc'] / nBar_ftype['genmc']
                        eff_err = eff * ((nBar_err_ftype['accmc']**0.5 / nBar_ftype['accmc'])**2 + (nBar_err_ftype['genmc']**0.5 / nBar_ftype['genmc'])**2)**0.5
                        corr_signal = signal / eff
                        corr_signal_err = corr_signal * ((signal_err / signal)**2 + (eff_err / eff)**2)**0.5
                        f.write(f"efficiency (%): {eff*100:.3f} +/- {eff_err*100:.3f}\n")
                        f.write(f"nBar corrected signal: {corr_signal} +/- {corr_signal_err}\n")
                    replace_fitname = f"sed -i 's|PLACEHOLDER_FITNAME|bin_{k}|g' bin_{k}/bin_{k}.cfg"
                    os.system(replace_fitname)
                    for pol in pols:
                        for ftype in ["data", "accmc", "genmc", "bkgnd"]:
                            fname = f"{ftype}" if mc_already_shared[ftype] else f"{ftype}{pol}"
                            os.system(f"mv -f {fname}_{k}.root bin_{k}/{fname}.root > /dev/null 2>&1")  # ignore error
                            replace_cmd = "sed -i 's|{}|{}|g' bin_{}/bin_{}.cfg"
                            search = f"PLACEHOLDER_{ftype.upper()}_{pol}"
                            replace = f"{output_directory}/bin_{k}/{fname}.root"
                            os.system(replace_cmd.format(search, replace, k, k))

                    if prepare_for_nifty:
                        os.system(f"touch bin_{k}/seed_nifty.txt")
                        os.system(f"echo '\ninclude {output_directory}/bin_{k}/seed_nifty.txt' >> bin_{k}/bin_{k}.cfg")

            # Read amptools.cfg and append the t-range and mass-range to a metadata section using omegaconf
            os.chdir(cwd)
            output_yaml = OmegaConf.load(yaml_name)
            output_yaml.update({"mass_edges": mass_edges, "t_edges": t_edges})
            dump_yaml(output_yaml, yaml_name)


    if try_merge_bins:

        ###################################################
        # MERGE BINS INTO GROUPS TO LIMIT SPAWNED PROCESSES
        ###################################################

        prexist_groups = glob.glob(f"{output_directory}/group_*")
        n_prexist = len(prexist_groups)
        if n_prexist > 0:
            input(f"\n{n_prexist} group folders already exist. Press enter to overwrite them or Ctrl+C to exit")
            os.system(f"rm -rf {output_directory}/group_*")

        if bins_per_group > 1:
            bins = glob.glob(f"{output_directory}/bin_*")
            bins = sorted(bins, key=lambda x: int(x.split("_")[-1]))
            nBins = len(bins)

            nGroups = nBins // bins_per_group
            print(f"Merging bins into {nGroups} groups of {bins_per_group}")

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
                        source = line.split(" ")[3].strip().split("/")[-1]
                        if source not in sources: # Take care of duplicates in case we share MC datasets
                            sources.append(source) 
                    if line.startswith("include"):
                        includes.append(line.split(" ")[1].strip().split("/")[-1])

            # Merge some bins into groups on user request
            # NOTE: For t and mass binning, the mass index cycles faster: i.e. [(m1,t1), (m2,t1), ...]

            for i in range(nGroups):
                bin_group = bins[i * bins_per_group : (i + 1) * bins_per_group]
                group_name = f"group_{i}"
                os.makedirs(f"{output_directory}/{group_name}", exist_ok=True)

                metadata = open(f"{output_directory}/{group_name}/metadata.txt", "w")
                
                first_pass = True
                for j, bin in enumerate(bin_group):
                    bin_name = bin.split("/")[-1]
                    cfg_name = f"G{j}_{bin_name}.cfg"
                    os.system(f"cp -r {bin}/{bin_name}.cfg {output_directory}/{group_name}/{cfg_name}")
                    os.system(f"cat {bin}/metadata.txt >> {output_directory}/{group_name}/metadata.txt")
                    os.system(f"echo '' >> {output_directory}/{group_name}/metadata.txt")
                    os.system(f"echo '' >> {output_directory}/{group_name}/metadata.txt")

                    # Replace all bin names with group names
                    sed_repl = f"sed -i 's|{bin_name}|group_{i}|g' {output_directory}/{group_name}/{cfg_name}"
                    os.system(sed_repl)

                    # Remove fit keyword lines or commented for additional files in the group as we will merge along a group
                    if j != 0:
                        os.system(f"sed -i '/^fit/d' {output_directory}/{group_name}/{cfg_name}")
                        os.system(f"sed -i '/^#/d' {output_directory}/{group_name}/{cfg_name}")
                    os.system(f"sed -i '/^include/d' {output_directory}/{group_name}/{cfg_name}") # remove all include seed files

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

                    # append seed file parameters into group seed files
                    for include in includes:
                        cmd = f"sed 's/parScale/G{j}_parScale/g' {bin}/{include} >> {output_directory}/{group_name}/{include}"
                        os.system(cmd)

                # merge config files using cat
                cat_cmd = f"cat {output_directory}/{group_name}/G*bin*.cfg > {output_directory}/{group_name}/bin_{i}.cfg"
                os.system(cat_cmd)

                # append include seed file
                os.system(f"echo '\ninclude {output_directory}/{group_name}/seed_nifty.txt' >> {output_directory}/{group_name}/bin_{i}.cfg")

                # clean up cfg files
                os.system(f"rm -f {output_directory}/{group_name}/G*bin*.cfg")

