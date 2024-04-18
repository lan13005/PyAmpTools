import glob
import os
import multiprocessing
from pyamptools.utility.general import Timer, load_yaml

import argparse

pyamptools_fit_cmd = "pa fit"
pyamptools_ff_cmd = "pa fitfrac"


class MLE:
    def __init__(self, yaml_file):
        """
        Args:
            yaml_file (dict): Configuration file
        """

        print("\n\n>>>>>>>>>>>>> ConfigLoader >>>>>>>>>>>>>>>")
        self.output_directory = yaml_file["amptools"]["output_directory"]
        self.n_randomizations = yaml_file["amptools"]["n_randomizations"]
        self.ff_args = yaml_file["amptools"]["regex_merge"]
        print("<<<<<<<<<<<<<< ConfigLoader <<<<<<<<<<<<<<\n\n")

    def __call__(self, cfgfile):
        base_fname = cfgfile.split("/")[-1].split(".")[0]
        binNum = base_fname.split("_")[-1]

        folder = os.path.dirname(cfgfile)
        print(f"Running over: {folder}")

        ###############################
        # Perform MLE fit
        ###############################

        cmd = f"{pyamptools_fit_cmd} {cfgfile} --numRnd {self.n_randomizations} --seedfile seed_bin{binNum}"
        print(cmd)
        os.system(cmd)

        if not os.path.exists(f"{base_fname}.fit"):
            raise Exception(f"Error: {base_fname}.fit does not exist!")

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

        # Clean extra seed files
        os.system(f"mv seed_bin{binNum}.txt {folder}/seed.txt")
        os.system(f"rm seed_bin{binNum}_*.txt")

        ## Some parameters (like scaling between polarized datasets) need to be fixed so that
        # NIFTy does not pick it up as a free parameter. We handle this by taking the amptools
        # seed file of the best fit remove all lines related to production coefficients and
        # append "fixed" to every line that is associated with these scale parameters.

        cmd = f"sed -i '/^initialize.*$/d' {folder}/seed.txt"  # delete whole line for initializing production coefficients
        os.system(cmd)

        cmd = f"sed -i 's|$| fixed|' {folder}/seed.txt"  # append fixed to the remaning lines
        os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide data into mass bins")
    parser.add_argument("yaml_name", type=str, default="conf/configuration.yaml", help="Path a configuration yaml file")
    args = parser.parse_args()
    yaml_name = args.yaml_name

    print("\n---------------------")
    print(f"Running {__file__}")
    print(f"  yaml location: {yaml_name}")
    print("---------------------\n")

    timer = Timer()
    cwd = os.getcwd()

    yaml_file = load_yaml(yaml_name)

    mle = MLE(yaml_file)

    output_directory = mle.output_directory
    n_randomizations = mle.n_randomizations

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

    n_processes = len(cfgfiles)

    with multiprocessing.Pool(n_processes) as pool:
        pool.map(mle, cfgfiles)
