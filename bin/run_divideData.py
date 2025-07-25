import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
from omegaconf import OmegaConf
from pyamptools.split_mass import split_mass_t
from pyamptools.utility.general import Timer, dump_yaml, load_yaml

from rich.console import Console

# Setup pyamptools environment here
#   cannot setup inside split_mass.py nor general.py
#   PyROOT maintains global state which will lead to conflicts
from pyamptools import atiSetup
import ROOT
atiSetup.setup(globals(), use_fsroot=True)
ROOT.ROOT.EnableImplicitMT()
from pyamptools.utility.rdf_macros import loadMacros
loadMacros()

############################################################################
# This makes calls to pyamptools' split_mass function to divide the data
# sources into separate mass bins and copies over the amptools configuration
# file. A flag can be used to distribute the data evenly across the mass bins
############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide data into mass bins")
    parser.add_argument("main_yaml", type=str, default="conf/configuration.yaml", help="Path to the main yaml file")
    parser.add_argument("-e", "--use_edges", action="store_true", help="Use mass_edges and t_edges from the yaml file. Else recompute")
    parser.add_argument("-np", "--n_processes", type=int, default=-1, help="Override (spec in yaml) number of parallel processes to merge (hadd) the data")
    parser.add_argument("--nosplit", action="store_true", help="Skip the split_mass step")
    parser.add_argument("--nomerge", action="store_true", help="Skip the merge_bins step")
    parser.add_argument("--overwrite", action="store_true", help="Force recalculation of derived kinematics even if they already exist")
    args = parser.parse_args()
    main_yaml = args.main_yaml
    use_edges = args.use_edges
    
    console = Console()

    n_processes = args.n_processes

    cwd = os.getcwd()
    timer = Timer()

    console.rule()
    console.print(f"Running {__file__}")
    console.print(f"  yaml location: {main_yaml}")
    console.rule()

    main_dict = load_yaml(main_yaml)

    min_mass = main_dict["min_mass"]
    max_mass = main_dict["max_mass"]
    n_mass_bins = main_dict["n_mass_bins"]
    min_t = main_dict["min_t"]
    max_t = main_dict["max_t"]
    n_t_bins = main_dict["n_t_bins"]
    n_processes = main_dict["n_processes"] if n_processes == -1 else n_processes
    base_directory = main_dict["base_directory"]
    output_directory = f"{base_directory}/BINNED_DATA"
    bins_per_group = main_dict["bins_per_group"] if "bins_per_group" in main_dict else 1
    constrain_grouped_production = main_dict["constrain_grouped_production"] if "constrain_grouped_production" in main_dict else False
    merge_grouped_trees = main_dict["merge_grouped_trees"] if "merge_grouped_trees" in main_dict else True
    data_folder = main_dict["data_folder"]
    pols = main_dict["polarizations"]
    prepare_for_nifty = True
    amptools_cfg = f"{base_directory}/amptools.cfg"

    os.system(f"mkdir -p {output_directory}")
    os.chdir(output_directory)

    nBins = n_mass_bins * n_t_bins
    if nBins % bins_per_group != 0:
        raise ValueError(f"Number of bins ({nBins}) is not divisible by `bins_per_group` ({bins_per_group}). Please choose a different value for `bins_per_group` in your YAML file")

    try_split_mass = not args.nosplit
    try_merge_bins = not args.nomerge

    ##########################################
    # SPLIT THE INPUT DATASETS INTO MASS BINS
    ##########################################

    # Check if all n_mass_bins folders "bin_{}" have already been created. If not, create them

    if try_split_mass:

        console.print("User requested data to be split into (mass, t) bins. Checking for pre-existing bins...", style="bold yellow")

        check_for_preexisting = all([os.path.exists(f"bin_{i}") for i in range(n_mass_bins * n_t_bins)])
        
        run_split_and_cfg_ceate = True
        create_cfg = False
        if check_for_preexisting:
            delete_or_skip = input(f"Pre-existing bins found at {output_directory}. Delete them? (y/n): ")
            if delete_or_skip.lower() == "y":
                os.system("rm -rf bin_*")
                console.print("  Deleted pre-existing bins", style="bold green")
            else:
                run_split_and_cfg_ceate = False
                ans_create_cfg = input("Create new cfg files? (y/n): ")
                if ans_create_cfg.lower() == "y":
                    console.print("  Will create new cfg files", style="bold green")
                    create_cfg = True
                else:
                    console.print("  Skipping split_mass", style="bold yellow")

        # Determine mass bin edges based on the first ftype source
        mass_edges = main_dict["mass_edges"] if "mass_edges" in main_dict and use_edges else None 
        t_edges = main_dict["t_edges"] if "t_edges" in main_dict and use_edges else None
        nBars = {}
        nBar_errs = {}
        if "nBars" in main_dict:
            nBars = main_dict["nBars"]
            nBars = {key: np.array(nBars[key]) for key in nBars}
        if "nBar_errs" in main_dict:
            nBar_errs = main_dict["nBar_errs"]
            nBar_errs = {key: np.array(nBar_errs[key]) for key in nBar_errs}
        mc_already_shared = {"data": False, "bkgnd": False, "accmc": False, "genmc": False} # if "mc_already_shared" not in main_dict else main_dict["mc_already_shared"]

        if run_split_and_cfg_ceate:
            
            sharemc = False
            console.print("No pre-existing bins found, running split_mass", style="bold green")
            
            for ftype in ["data", "accmc", "genmc"]:
                for pol in pols:
                    console.print(f"Attempting to split {ftype}{pol}.root into mass bins", style="bold yellow")
                    # AmpTools -f does not include polarization information. We can share all accmc and genmc datasets. Do so if accmc.root exists and acc{pol}.root does not exist 
                    if ftype in ["accmc", "genmc"]: # check to see if we can share the MC datasets
                        if not os.path.exists(f"{data_folder}/{ftype}{pol}.root"):
                            if not os.path.exists(f"{data_folder}/{ftype}.root"):
                                raise FileNotFoundError(f"File {data_folder}/{ftype}{pol}.root does not exist! Nor does {data_folder}/{ftype}.root (unable to share MC across polarized datasets)")
                            sharemc = True
                            if mc_already_shared[ftype]:
                                console.print("  * Already sharing MC datasets, will run split_mass again", style="bold yellow")
                            else:
                                console.print(f"  * {ftype}{pol}.root does not exist but {ftype}.root does. Will share {ftype}.root across all polarizations", style="bold yellow")
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
                                                    treeName="kin", mass_edges=mass_edges, t_edges=t_edges, overwrite=args.overwrite)
                        _pol = "shared" if sharemc else pol
                        nBars[f"{ftype}_{_pol}"] = nBar
                        nBar_errs[f"{ftype}_{_pol}"] = nBar_err
                        if sharemc:
                            mc_already_shared[ftype] = True

            # background file is optional, would assume data is pure weighted signal
            for pol in pols:
                if os.path.exists(f"{data_folder}/bkgnd{pol}.root"):
                    mass_edges, t_edges, nBar, nBar_err = split_mass_t(f"{data_folder}/bkgnd{pol}.root", f"bkgnd{pol}", 
                                                        min_mass, max_mass, n_mass_bins, 
                                                        min_t, max_t, n_t_bins,
                                                        treeName="kin", mass_edges=mass_edges, t_edges=t_edges, overwrite=args.overwrite)
                    nBars[f"bkgnd_{pol}"] = nBar
                    nBar_errs[f"bkgnd_{pol}"] = nBar_err
                else:
                    console.print(f"No bkgnd{pol}.root found (not required), skipping", style="bold yellow")

            # ###########################################
            # # RENAME AND COPY+MODIFY AMPTOOLS CFG FILES
            # ###########################################

        if run_split_and_cfg_ceate or create_cfg:
            for j in range(n_t_bins):
                for i in range(n_mass_bins):
                    k = j * n_mass_bins + i
                    console.print(f"Perform final preparation for bin: {k}", style="bold yellow")
                    os.system(f"mkdir -p bin_{k}")
                    os.system(f"cp -f {amptools_cfg} bin_{k}/bin_{k}.cfg")
                    with open(f"bin_{k}/metadata.txt", "w") as f:
                        f.write("---------------------\n")
                        f.write(f"bin number: {k}\n")
                        if mass_edges is not None:
                            f.write(f"mass range: {mass_edges[i]:.2f} - {mass_edges[i + 1]:.2f} GeV\n")
                        if t_edges is not None:
                            f.write(f"t range: {t_edges[j]:.2f} - {t_edges[j + 1]:.2f} GeV^2\n")
                        nBar_ftype = {pair.split("_")[0]: 0 for pair in nBars.keys()} # keys ~ ftype_pol
                        nBar_var_ftype = {pair.split("_")[0]: 0 for pair in nBars.keys()} # variance
                        f.write("---------------------\n")
                        for ftype_pol in nBars:
                            ftype, pol = ftype_pol.split("_")
                            _nBars = nBars[f"{ftype}_{pol}"]
                            _nBar_errs = nBar_errs[f"{ftype}_{pol}"]
                            f.write(f"nBar {ftype} {pol}: {_nBars[i][j]} +/- {_nBar_errs[i][j]}\n")
                            if ftype in ["data", "bkgnd"]:
                                nBar_ftype[ftype] += _nBars[i][j]
                                nBar_var_ftype[ftype] += _nBar_errs[i][j]**2
                            if ftype in ["accmc", "genmc"]:
                                share_factor = (len(pols) if pol == "shared" else 1)
                                nBar_ftype[ftype] += _nBars[i][j] * share_factor
                                nBar_var_ftype[ftype] += (_nBar_errs[i][j] * share_factor)**2
                        f.write("---------------------\n")
                        for ftype in nBar_ftype:
                            f.write(f"nBar {ftype}: {nBar_ftype[ftype]} +/- {nBar_var_ftype[ftype]**0.5}\n")
                        f.write("---------------------\n")
                        if len(nBars) != 0:
                            if "bkgnd" in nBar_ftype: # if there is bkgnd we subtract it to get the signal
                                signal = nBar_ftype['data'] - nBar_ftype['bkgnd']
                                signal_err = (nBar_var_ftype['data'] + nBar_var_ftype['bkgnd'])**0.5
                            else:
                                signal = nBar_ftype['data']
                                signal_err = nBar_var_ftype['data']**0.5
                            f.write(f"nBar signal: {signal} +/- {signal_err}\n")
                            eff = nBar_ftype['accmc'] / nBar_ftype['genmc']                            
                            eff_err = eff * ((nBar_var_ftype['accmc']**0.5 / nBar_ftype['accmc'])**2 + (nBar_var_ftype['genmc']**0.5 / nBar_ftype['genmc'])**2)**0.5
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


                    cfg_path = "bin_{0}/bin_{0}.cfg"
                    reactions = []
                    with open(cfg_path.format(k), "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.startswith("reaction"):
                                reactions.append(line.split(" ")[1].strip())
                    with open(cfg_path.format(k), "a") as f:
                        f.write("\n") # do not assume that last line ended with a newline
                        for reaction in reactions:
                            f.write(f"normintfile {reaction} {output_directory}/bin_{k}/{reaction}.ni\n")

                    if prepare_for_nifty:
                        os.system(f"touch bin_{k}/seed_nifty.txt")
                        os.system(f"echo '\ninclude {output_directory}/bin_{k}/seed_nifty.txt' >> bin_{k}/bin_{k}.cfg")

            # Read amptools.cfg and append the t-range and mass-range to a metadata section using omegaconf
            os.chdir(cwd)
            output_dict = load_yaml(main_yaml, resolve=False) # do not resolve when updating
            nBars = {key: nBars[key].tolist() for key in nBars}
            nBar_errs = {key: nBar_errs[key].tolist() for key in nBar_errs}
            output_dict.update({"share_mc": mc_already_shared})
            dump_yaml(output_dict, main_yaml)

    if try_merge_bins:

        ###################################################
        # MERGE BINS INTO GROUPS TO LIMIT SPAWNED PROCESSES
        ###################################################

        prexist_groups = glob.glob(f"{output_directory}/group_*")
        n_prexist = len(prexist_groups)
        ans = "y"
        if n_prexist > 0:
            ans = input(f"\n{n_prexist} group folders already exist. Ovewrite? (y/n): ")
            if ans.lower() == "y":
                os.system(f"rm -rf {output_directory}/group_*")
                console.print("  Deleted pre-existing groups", style="bold green")

        if bins_per_group > 1 and ans == "y":
            bins = glob.glob(f"{output_directory}/bin_*")
            bins = sorted(bins, key=lambda x: int(x.split("_")[-1]))
            nBins = len(bins)

            nGroups = nBins // bins_per_group
            console.print(f"Merging bins into {nGroups} groups of {bins_per_group}", style="bold yellow")

            # Get base information by loading in the first bin
            bin_path = bins[0]
            bin_name = bin_path.split("/")[-1]
            cfg_path = f"{bin_path}/{bin_name}.cfg"
            reactions = []
            parameters = []
            sources = []
            includes = []
            initialized_waves = [] # amplitude names that have initialized keyword in cfg file
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
                    if line.startswith("initialize"):
                        initialized_waves.append(line.split(" ")[1].strip())

            if merge_grouped_trees:
                
                def merge_trees(i):

                    bin_group = bins[i * bins_per_group : (i + 1) * bins_per_group]
                    group_name = f"group_{i}"
                    os.makedirs(f"{output_directory}/{group_name}", exist_ok=True)

                    bin = bin_group[0]
                    bin_name = bin.split("/")[-1]
                    cfg_name = f"bin_{i}.cfg"
                    os.system(f"cp -r {bin}/{bin_name}.cfg {output_directory}/{group_name}/{cfg_name}")

                    # Replace all bin names with group names if line does not start with fit
                    sed_repl = f"sed -i '/^fit/! s|{bin_name}|group_{i}|g' {output_directory}/{group_name}/{cfg_name}"
                    os.system(sed_repl)
                    # Search for line that starts wtih fit and replace bin id with group id
                    sed_repl = f"sed -i '/^fit/ s|{bin_name}|bin_{i}|g' {output_directory}/{group_name}/{cfg_name}"
                    os.system(sed_repl)

                    # update naming for sources
                    for source in sources:
                        console.print(f"Merging {source} from {len(bin_group)} bins", style="bold yellow")
                        bin_sources = [f"{bin}/{source}" for bin in bin_group]
                        hadd_cmd = f"hadd {output_directory}/{group_name}/{source} {' '.join(bin_sources)}"
                        console.print(hadd_cmd, style="bold yellow")
                        os.system(hadd_cmd+" > /dev/null")
                    
                    os.system(f"touch {output_directory}/{group_name}/seed_nifty.txt")

                with Pool(n_processes) as p:
                    p.map(merge_trees, range(nGroups))

            # Merge some bins into groups on user request keeping bins distinct
            #   All bins in a group will have a different reaction identifier so that if you merged 2 bins there will be 2x more reactions
            #   This might not be what you want. If you merged bin_i and bin_j into one group the amplitudes will not be able to talk across
            #   bins. This should in principle result in different results when compared to dividing the data directly into the groups
            # NOTE: For t and mass binning, the mass index cycles faster: i.e. [(m1,t1), (m2,t1), ...]
            else:
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

                        # Replace all bin names with group names if line does not start with fit
                        sed_repl = f"sed -i '/^fit/! s|{bin_name}|group_{i}|g' {output_directory}/{group_name}/{cfg_name}"
                        os.system(sed_repl)
                        # Search for line that starts wtih fit and replace bin id with group id
                        sed_repl = f"sed -i '/^fit/ s|{bin_name}|bin_{i}|g' {output_directory}/{group_name}/{cfg_name}"
                        os.system(sed_repl)

                        # Remove fit keyword lines or commented for additional files in the group as we will merge along a group
                        if j != 0:
                            os.system(f"sed -i '/^fit/d' {output_directory}/{group_name}/{cfg_name}")
                            os.system(f"sed -i '/^#/d' {output_directory}/{group_name}/{cfg_name}")
                            os.system(f"sed -i '/^initialize/d' {output_directory}/{group_name}/{cfg_name}")
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

                        # constrain the production coefficients to be shared for all bins in the group
                        if constrain_grouped_production and j != 0:
                            for wave in initialized_waves:
                                cmd = f"echo 'constrain G0_{wave} G{j}_{wave}' >> {output_directory}/{group_name}/{cfg_name}"
                                os.system(cmd)

                    # merge config files using cat
                    cat_cmd = f"cat {output_directory}/{group_name}/G*bin*.cfg > {output_directory}/{group_name}/bin_{i}.cfg"
                    os.system(cat_cmd)

                    # append include seed file
                    os.system(f"echo '\ninclude {output_directory}/{group_name}/seed_nifty.txt' >> {output_directory}/{group_name}/bin_{i}.cfg")

                    # clean up cfg files
                    os.system(f"rm -f {output_directory}/{group_name}/G*bin*.cfg")

    console.print(f"\n{timer.read(return_str=False)[2]:0.2f} seconds elapsed\n", style="bold green")