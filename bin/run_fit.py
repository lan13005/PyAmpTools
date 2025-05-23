import argparse
import glob
import multiprocessing
import os
import re

from pyamptools import atiSetup
from pyamptools.utility.general import Silencer, Timer, load_yaml

pyamptools_fit_cmd = "pa fit"
pyamptools_ff_cmd = "pa fitfrac"

############################################################################
# This function performs maximum likelihood fits using AmpTools on all cfg files
# by calling pyamptools' fit and fitfrac commands on each of them.
############################################################################

seed_file = "seed_nifty.txt"

class MLE:
    def __init__(self, main_dict, dump='', main_yaml=''):
        """
        Args:
            main_dict (dict): Configuration file
        """

        print("\n\n>>>>>>>>>>>>> ConfigLoader >>>>>>>>>>>>>>>")
        self.output_directory = main_dict["amptools"]["output_directory"]
        self.n_randomizations = main_dict["amptools"]["n_randomizations"]
        self.ff_args = main_dict["amptools"]["regex_merge"]
        self.prepare_for_nifty = True
        self.devs = None
        self.dump = dump
        self.main_yaml = main_yaml
        print("<<<<<<<<<<<<<< ConfigLoader <<<<<<<<<<<<<<\n\n")

    def set_devs(self, devs):
        self.devs = int(devs)

    def __call__(self, cfgfile):

        base_fname = cfgfile.split("/")[-1].split(".")[0]
        binNum = base_fname.split("_")[-1]

        folder = os.path.dirname(cfgfile)
        print(f"Running over: {folder}")

        ###############################
        # Perform MLE fit
        ###############################
        device = int(binNum) % self.devs
        cmd = f"{pyamptools_fit_cmd} {cfgfile} --numRnd {self.n_randomizations} --seedfile seed_bin{binNum}"
        if self.devs is not None and self.devs >= 0:
            cmd += f" --device {device}"
        if self.dump == 'null':
            cmd += " > /dev/null"
        elif self.dump != '':
            if not os.path.exists(f"{self.dump}"):
                os.makedirs(f"{self.dump}")
            cmd += f" >> {self.dump}/log_{binNum}.txt 2>&1"
        os.system(cmd)

        if not os.path.exists(f"{base_fname}.fit"):
            raise Exception(f"run_mle| Error: {base_fname}.fit does not exist!")

        os.system(f"rm -f bin_{binNum}_*.ni")

        ###############################
        # Extract ff for best iteration
        ###############################

        cmd = f"{pyamptools_ff_cmd} {base_fname}.fit --outputfileName intensities_bin{binNum}.txt {self.ff_args} --main_yaml {self.main_yaml}"
        if self.dump == 'null':
            cmd += " > /dev/null"
        elif self.dump != '':
            cmd += f" >> {self.dump}/log_{binNum}.txt 2>&1"
        print(cmd)
        os.system(cmd)
        os.system(f"mv intensities_bin{binNum}.txt {folder}/intensities.txt")
        os.system(f"mv {base_fname}.fit {folder}/{base_fname}.fit")

        # Extract ff for all iterations
        for i in range(self.n_randomizations):
            cmd = f"{pyamptools_ff_cmd} {base_fname}_{i}.fit --outputfileName intensities_bin{binNum}_{i}.txt {self.ff_args} --main_yaml {self.main_yaml}"
            if self.dump == 'null':
                cmd += " > /dev/null"
            elif self.dump != '':
                cmd += f" >> {self.dump}/log_{binNum}.txt 2>&1"
            print(cmd)
            os.system(cmd)
            os.system(f"mv intensities_bin{binNum}_{i}.txt {folder}/intensities_{i}.txt")
            os.system(f"mv {base_fname}_{i}.fit {folder}/{base_fname}_{i}.fit")

        if self.prepare_for_nifty:
            ## Some parameters (like scaling between polarized datasets) need to be fixed so that
            # NIFTy does not pick it up as a free parameter. We handle this by taking the amptools
            # seed file of the best fit remove all lines related to production coefficients and
            # append "fixed" to every line that is associated with these scale parameters.
            os.system(f"mv seed_bin{binNum}.txt {folder}/{seed_file}")
            os.system(f"mv seed_bin{binNum}_*.txt {folder}")  # move all seed files to the folder
            os.system(f"sed -i '/^initialize.*$/d' {folder}/{seed_file}")  # delete whole line for initializing production coefficients
            os.system(f"sed -i 's|$| fixed|' {folder}/{seed_file}")  # append fixed to the remaning lines


pat_fit = re.compile("bin_.*.fit")
pat_txt = re.compile("seed_.*.txt")
pat_reac = re.compile("reaction_.*.ni")
pat_int = re.compile("intensities.*.txt")
def cleanup(directories):
    for root in directories:
        for file in os.listdir(root):
            if pat_fit.match(file) or pat_txt.match(file) or pat_reac.match(file) or pat_int.match(file):
                cmd = f"rm -f {os.path.join(root, file)}"
                # print(cmd)
                os.system(cmd)
        if not os.path.exists(os.path.join(root, "seed_nifty.txt")):
            os.system(f"touch {os.path.join(root, 'seed_nifty.txt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform maximum likelihood fits (using AmpTools) on all cfg files")
    parser.add_argument("main_yaml", type=str, default="conf/configuration.yaml", help="Path to the main yaml file")
    parser.add_argument("-d", type=str, default='null', help="Dump log files for each bin to this directory path. If empty str will dump to stdout")
    parser.add_argument("--clean", action="store_true", help="Cleanup all fit files, fitfrac files, and reaction files")
    args = parser.parse_args()
    main_yaml = args.main_yaml
    dump = args.d
    
    timer = Timer()
    cwd = os.getcwd()

    print("\n---------------------")
    print(f"Running {__file__}")
    print(f"  yaml location: {main_yaml}")
    print(f"  dump logs to folder: {os.path.join(cwd, dump)}")
    print("---------------------\n")

    main_dict = load_yaml(main_yaml)
    
    if args.clean:
        print(f"run_mle| Cleaning all fit files, fitfrac files, and reaction files in {main_dict['base_directory']}")
        subdirs = ['.']
        for root, dirs, files in os.walk(main_dict["base_directory"]):
            for dir in dirs:
                subdirs.append(os.path.join(root, dir))    
        cleanup(subdirs)

    mle = MLE(main_dict, dump=dump, main_yaml=main_yaml)

    output_directory = mle.output_directory
    n_randomizations = mle.n_randomizations
    bins_per_group = main_dict['amptools']["bins_per_group"]
    bins_per_group = 1 if bins_per_group < 1 or bins_per_group is None else bins_per_group
    nBins = main_dict["n_mass_bins"] * main_dict["n_t_bins"]
    if nBins % bins_per_group != 0:
        raise Exception("run_mle| Number of bins is not divisible by bins_per_group!")
    nGroups = nBins // bins_per_group

    search_format = main_dict["amptools"]["search_format"]
    if search_format not in ["bin", "group"]:
        raise Exception("run_mle| search_format must be either 'bin' or 'group'")
    if search_format == "group" and bins_per_group == 1:
        search_format = "bin"
        print("mle| search_format set to 'bin' since bins_per_group is 1 (no grouping...)")

    ###########################################
    # Perform MLE fit and extract fit fractions
    # for all cfg files if incomplete
    ###########################################

    _cfgfiles = glob.glob(f"{output_directory}/{search_format}_*/*.cfg")  # all cfg files
    cfgfiles = []  # cfg files that need to be processed
    print("mle| Will process following cfg files:")
    for cfgfile in _cfgfiles:
        folder = os.path.dirname(cfgfile)

        # Check if all randomized fits and fit fraction files are there
        fit_complete = len(glob.glob(f"{folder}/*.fit")) >= n_randomizations + 1
        ff_complete = len(glob.glob(f"{folder}/intensities*.txt")) >= n_randomizations + 1
        complete = fit_complete and ff_complete

        if not complete:
            cfgfiles.append(cfgfile)
            print(f"  {cfgfile}")

    n_processes = min(len(cfgfiles), main_dict["amptools"]["n_processes"])

    # Get number of gpu devices amptools sees
    USE_MPI, USE_GPU, RANK_MPI = atiSetup.setup(globals())

    devs = -1
    if "GPUManager" in globals():
        devs = GPUManager.getNumDevices()
    mle.set_devs(devs)

    if n_processes < 1:
        if len(cfgfiles) > 0:
            print("mle| all cfg files are complete, skipping MLE fits")
        else:
            print("mle| no cfg files to process")
    else:
        with multiprocessing.Pool(n_processes) as pool:
            pool.map(mle, cfgfiles)

    if bins_per_group > 1 and search_format == "bin":
        print("mle| collecting seed files into group subfolders")
        groups = glob.glob(f"{output_directory}/group_*")
        if len(groups) != nGroups:
            raise Exception("run_mle| Expected number of groups did not match actual groupings")
        for ig, group in enumerate(groups):
            group_name = group.split("/")[-1]
            for relBin, absBin in enumerate(range(ig * bins_per_group, (ig + 1) * bins_per_group)):
                # os.system(f"cat {output_directory}/bin_{ibin}/{seed_file} >> {group}/{seed_file}")
                src_seed = f"{output_directory}/bin_{absBin}/{seed_file}"
                dst_seed = f"{output_directory}/{group_name}/{seed_file}"
                with open(src_seed) as f:
                    lines = f.readlines()
                    paras = [line.split(" ")[1] for line in lines if line.startswith("parameter")]
                    for para in paras:
                        cmd = f"sed -n 's|{para}|G{relBin}_{para}|p' {src_seed} >> {dst_seed}"
                        os.system(cmd)

    print(f"mle| Elapsed time {timer.read()[2]}\n\n")
