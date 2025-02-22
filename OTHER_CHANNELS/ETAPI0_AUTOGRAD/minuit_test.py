from pyamptools.utility.IO import get_intens_from_fitresults
from pyamptools.utility.general import load_yaml, dump_yaml
import os
import shutil
import numpy as np
import pickle as pkl
from multiprocessing import Pool
import subprocess

def run_cmd(cmd):
    # print(cmd)
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
        return output
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        return e.output
    
np.random.seed(42)
scale = 50
n_iterations = 1
n_processes = 2
pyamptools_yaml = "/w/halld-scshelf2101/lng/WORK/PyAmpTools9/OTHER_CHANNELS/ETAPI0_AUTOGRAD/pyamptools.yaml"
iftpwa_yaml = "/w/halld-scshelf2101/lng/WORK/PyAmpTools9/OTHER_CHANNELS/ETAPI0_AUTOGRAD/iftpwa.yaml"

pyamptools_yaml = load_yaml(pyamptools_yaml)
waveNames = pyamptools_yaml["waveset"].split("_")
nmbMasses = pyamptools_yaml["n_mass_bins"]
nmbTprimes = pyamptools_yaml["n_t_bins"]
nPars = 2 * len(waveNames)

seed_list = np.random.randint(0, 1000000, (n_iterations, nmbMasses, nmbTprimes))

def setup_bin_directory(bin_idx, iteration, seed_list, initialization={}):

    mbin = bin_idx % nmbMasses
    tbin = bin_idx // nmbMasses
    
    np.random.seed(seed_list[iteration, mbin, tbin])
    
    # initializaiton dictionary comes as Dict[str, complex] -> convert to Dict[str, str]
    _initialization = {}
    for k, v in initialization.items():
        sign_imag = "+" if np.imag(v) >= 0 else "-"
        _initialization[k] = f"{v.real}{sign_imag}{np.abs(np.imag(v))}j"
    initialization = _initialization

    cwd = os.getcwd()
    
    # Generate amptools configuration file with appropriate initialization
    shutil.copy2("pyamptools.yaml", f"pyamptools_{bin_idx}_{iteration}.yaml")
    yaml_file = load_yaml(f"pyamptools_{bin_idx}_{iteration}.yaml")
    yaml_file["initialization"] = initialization
    dump_yaml(yaml_file, f"pyamptools_{bin_idx}_{iteration}.yaml")
    os.system(f"pa run_cfgGen pyamptools_{bin_idx}_{iteration}.yaml -o amptools_{bin_idx}_{iteration}.cfg")    
    data_dir = f"/w/halld-scshelf2101/lng/WORK/PyAmpTools9/OTHER_CHANNELS/ETAPI0_AUTOGRAD/RESULTS/AmpToolsFits/bin_{bin_idx}"
    os.system(f"sed -i 's|PLACEHOLDER_FITNAME|bin_{bin_idx}_{iteration}|' amptools_{bin_idx}_{iteration}.cfg")
    pols = yaml_file["polarizations"].keys()
    share_mc = yaml_file["share_mc"]
    for pol in pols:
        os.system(f"sed -i 's|PLACEHOLDER_DATA_{pol}|{data_dir}/data{pol}.root|' amptools_{bin_idx}_{iteration}.cfg")
        os.system(f"sed -i 's|PLACEHOLDER_BKGND_{pol}|{data_dir}/bkgnd{pol}.root|' amptools_{bin_idx}_{iteration}.cfg")
        if share_mc["genmc"]:
            os.system(f"sed -i 's|PLACEHOLDER_GENMC_{pol}|{data_dir}/genmc.root|' amptools_{bin_idx}_{iteration}.cfg")
        else:
            os.system(f"sed -i 's|PLACEHOLDER_GENMC_{pol}|{data_dir}/genmc{pol}.root|' amptools_{bin_idx}_{iteration}.cfg")
        if share_mc["accmc"]:
            os.system(f"sed -i 's|PLACEHOLDER_ACCMC_{pol}|{data_dir}/accmc.root|' amptools_{bin_idx}_{iteration}.cfg")
        else:
            os.system(f"sed -i 's|PLACEHOLDER_ACCMC_{pol}|{data_dir}/accmc{pol}.root|' amptools_{bin_idx}_{iteration}.cfg")

    # commands to run fit
    output = run_cmd(f"pa fit amptools_{bin_idx}_{iteration}.cfg  --seedfile seed_{bin_idx}_{iteration}")
    for line in output.split('\n'):
        if "LIKELIHOOD BEFORE MINIMIZATION:" in line:
            initial_likelihood = float(line.split(':')[1].strip())

    # command to load intensities into python, `pa fit` automatically appends and _0 to the name of the fit file
    fit_file = f"bin_{bin_idx}_{iteration}_0.fit"
    intensities = get_intens_from_fitresults(fit_file, ".*::.*::")
    intensities['initial_likelihood'] = initial_likelihood
    
    with open(f"COMPARISONS/amptools_bin{bin_idx}_{iteration}.pkl", "wb") as f:
        pkl.dump({'initial_guess_dict': initialization, 'intensities': intensities}, f)
    
    # cleanup
    os.system(f"echo '\ninclude {cwd}/RESULTS/AmpToolsFits/bin_{bin_idx}/seed_{bin_idx}_{iteration}.txt' >> amptools_{bin_idx}_{iteration}.cfg")
    os.rename(f"amptools_{bin_idx}_{iteration}.cfg", f"RESULTS/AmpToolsFits/bin_{bin_idx}/amptools_{bin_idx}_{iteration}.cfg")
    os.rename(f"bin_{bin_idx}_{iteration}_0.fit", f"RESULTS/AmpToolsFits/bin_{bin_idx}/bin_{bin_idx}_{iteration}_0.fit")
    os.rename(f"seed_{bin_idx}_{iteration}_0.txt", f"RESULTS/AmpToolsFits/bin_{bin_idx}/seed_{bin_idx}_{iteration}.txt")
    run_cmd(f"sed -i '/^initialize/d' RESULTS/AmpToolsFits/bin_{bin_idx}/seed_{bin_idx}_{iteration}.txt")
    run_cmd(f"sed -i 's/$/ fixed/' RESULTS/AmpToolsFits/bin_{bin_idx}/seed_{bin_idx}_{iteration}.txt")
    run_cmd(f"rm -rf amptools_{bin_idx}_{iteration}.cfg bin_{bin_idx}_{iteration}_0.fit pyamptools_{bin_idx}_{iteration}.yaml")
    os.chdir(cwd)
    
    
def run_fits_in_bin(bin_idx):
    
    mbin = bin_idx % nmbMasses
    tbin = bin_idx // nmbMasses

    for i in range(n_iterations):
        initial_guess = scale * np.random.randn(nPars, nmbMasses, nmbTprimes)
        # initial_guess = np.ones_like(initial_guess)
        initial_guess_dict = {} 
        for iw, wave in enumerate(waveNames):
            initial_guess_dict[wave] = initial_guess[2*iw, mbin, tbin] + 1j * initial_guess[2*iw+1, mbin, tbin]
        setup_bin_directory(bin_idx, i, seed_list, initial_guess_dict)

if __name__ == "__main__":
    
    with Pool(n_processes) as p:
        p.map(run_fits_in_bin, [1, 2])  # range(nmbMasses * nmbTprimes))