import argparse
import glob
import multiprocessing
import os

from pyamptools import atiSetup
from pyamptools.utility.general import Timer, load_yaml

pyamptools_fit_cmd = "pa fit"
pyamptools_ff_cmd = "pa fitfrac"

############################################################################
# This function performs maximum likelihood fits using AmpTools on all cfg files
# by calling pyamptools' fit and fitfrac commands on each of them.
############################################################################

seed_file = "seed_nifty.txt"

class MLE:
    def __init__(self, yaml_file, dump=''):
        """
        Args:
            yaml_file (dict): Configuration file
        """

        print("\n\n>>>>>>>>>>>>> ConfigLoader >>>>>>>>>>>>>>>")
        self.output_directory = yaml_file["amptools"]["output_directory"]
        self.n_randomizations = yaml_file["amptools"]["n_randomizations"]
        self.ff_args = yaml_file["amptools"]["regex_merge"]
        self.prepare_for_nifty = bool(yaml_file["amptools"]["prepare_for_nifty"])
        self.devs = None
        self.dump = dump
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
        if self.dump != '':
            if not os.path.exists(f"{self.dump}"):
                os.makedirs(f"{self.dump}")
            cmd += f" >> {self.dump}/log_{binNum}.txt"
        os.system(cmd)

        if not os.path.exists(f"{base_fname}.fit"):
            raise Exception(f"run_mle| Error: {base_fname}.fit does not exist!")

        os.system(f"rm -f bin_{binNum}_*.ni")

        ###############################
        # Extract ff for best iteration
        ###############################

        cmd = f"{pyamptools_ff_cmd} {base_fname}.fit --outputfileName intensities_bin{binNum}.txt {self.ff_args}"
        print(cmd)
        os.system(cmd)
        os.system(f"mv intensities_bin{binNum}.txt {folder}/intensities.txt")
        os.system(f"mv {base_fname}.fit {folder}/{base_fname}.fit")

        # Extract ff for all iterations
        for i in range(self.n_randomizations):
            cmd = f"{pyamptools_ff_cmd} {base_fname}_{i}.fit --outputfileName intensities_bin{binNum}_{i}.txt {self.ff_args}"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform maximum likelihood fits (using AmpTools) on all cfg files")
    parser.add_argument("yaml_name", type=str, default="conf/configuration.yaml", help="Path a configuration yaml file")
    parser.add_argument("-d", type=str, default='', help="Dump log files for each bin to this relative path")
    args = parser.parse_args()
    yaml_name = args.yaml_name
    dump = args.d

    print("\n---------------------")
    print(f"Running {__file__}")
    print(f"  yaml location: {yaml_name}")
    print("---------------------\n")

    timer = Timer()
    cwd = os.getcwd()

    yaml_file = load_yaml(yaml_name)

    mle = MLE(yaml_file, dump=dump)

    output_directory = mle.output_directory
    n_randomizations = mle.n_randomizations
    bins_per_group = yaml_file["bins_per_group"]
    bins_per_group = 1 if bins_per_group < 1 or bins_per_group is None else bins_per_group
    nBins = yaml_file["n_mass_bins"] * yaml_file["n_t_bins"]
    if nBins % bins_per_group != 0:
        raise Exception("run_mle| Number of bins is not divisible by bins_per_group!")
    nGroups = nBins // bins_per_group

    ###########################################
    # Perform MLE fit and extract fit fractions
    # for all cfg files if incomplete
    ###########################################

    _cfgfiles = glob.glob(f"{output_directory}/bin_*/*.cfg")  # all cfg files
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

    n_processes = min(len(cfgfiles), yaml_file["amptools"]["n_processes"])

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

    if bins_per_group > 1:
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
