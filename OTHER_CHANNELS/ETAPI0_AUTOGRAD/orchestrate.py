import os

os.environ["JAX_PLATFORMS"] = "cpu"

import random
import tempfile
import glob
from pyamptools.utility.general import load_yaml, recursive_replace, dump_yaml, execute_cmd
from pyamptools.utility.resultManager import ResultManager, plot_binned_intensities, plot_binned_complex_plane, plot_overview_across_bins, montage_and_gif_select_plots, plot_moments_across_bins
from rich.console import Console
from rich.table import Table
import sys
import time
import numpy as np
import argparse
import shutil

console = Console()

# random.seed(1425)
nseeds = 5
yaml_file = "/w/halld-scshelf2101/lng/WORK/PyAmpTools9/OTHER_CHANNELS/ETAPI0_AUTOGRAD/pyamptools_mc.yaml"

def exit_if_no_files_at(pathname_pattern, name=""):
    """
    Check if pickled results are present. Needed to manually crash if errors occurs.
        For some cannot get execute_cmd to crash if any errors show up. Probably because
        execute_cmd calls pa which calls a python script. Too much layeres of indirection?
    """
    files = glob.glob(pathname_pattern)
    if len(files) == 0:
        name = f"{name}: " if name else ""
        console.print(f"[bold red]{name}fit did not complete as expected. Pickled results not found in {pathname_pattern}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("waveset", type=str, help="Waveset to run")
    parser.add_argument("phase_reference", type=str, help="Phase reference to run")
    parser.add_argument("--output_directory", type=str, default="", help="Output directory to save results")
    args = parser.parse_args()
    waveset = args.waveset
    phase_reference = args.phase_reference
    output_directory = args.output_directory
    
    times = {}
    start_time = time.time()

    seeds = random.sample(range(10000000000), nseeds)

    # seeds = glob.glob(f"SEED_*")
    # seeds = [int(seed.split("_")[-1]) for seed in seeds]

    for seed in seeds:
        
        console.print(f"\n\n[bold red]Beginning seed {seed}[/bold red]\n")
        
        # Reload everytime to reset state (i.e. resetting folder output location)
        main_yaml = load_yaml(yaml_file)
        ift_yaml  = load_yaml(main_yaml["nifty"]["yaml"])
        base_directory = main_yaml["base_directory"]
        parent_directory = os.path.dirname(base_directory)
        
        main_yaml['waveset'] = waveset
        main_yaml['phase_reference'] = phase_reference
        n_processes = main_yaml["n_mass_bins"]

        seed_directory = f"{parent_directory}/{output_directory}/SEED_{seed}".replace("//", "/")
        recursive_replace(ift_yaml,  base_directory, seed_directory)
        recursive_replace(main_yaml, base_directory, seed_directory)
        os.makedirs(seed_directory, exist_ok=True)

        # We create temporary yaml files in case we want work with a modified version since all below commands takes a yaml files as input
        # run_priorSim will will create additional yaml files at GENERATED/*.yaml so the above ones can be safely deleted
        with tempfile.NamedTemporaryFile(delete=True) as tmp_main_yaml, \
            tempfile.NamedTemporaryFile(delete=True) as tmp_ift_yaml:
            main_yaml["nifty"]["yaml"] = tmp_ift_yaml.name
            ift_yaml["PWA_MANAGER"]["yaml"] = tmp_ift_yaml.name
            dump_yaml(main_yaml, tmp_main_yaml.name)
            dump_yaml(ift_yaml, tmp_ift_yaml.name)

            _start_time = time.time()
            execute_cmd([f"pa run_priorSim {tmp_main_yaml.name} -s {seed} -nd 250000 -np 1000000"], console=console) # Generate data, divide data, ready to fit
            exit_if_no_files_at(f"{seed_directory}/GENERATED/*pkl", "PriorSim")
            prior_sim_time = time.time() - _start_time
            times.setdefault("PriorSim", []).append(prior_sim_time)
            
            _start_time = time.time()
            execute_cmd([f"pa run_mle {seed_directory}/GENERATED/main.yaml"], console=console) # Run on new yaml files created by run_priorSim
            exit_if_no_files_at(f"{seed_directory}/MLE/*.pkl", "MLE")
            mle_time = time.time() - _start_time
            times.setdefault("MLE", []).append(mle_time)
            
            _start_time = time.time()
            execute_cmd([f"pa run_ift {seed_directory}/GENERATED/main.yaml"], console=console)
            exit_if_no_files_at(f"{seed_directory}/NiftyFits/*.pkl", "IFT")
            ift_time = time.time() - _start_time
            times.setdefault("IFT", []).append(ift_time)
            
            _start_time = time.time()
            execute_cmd([f"pa run_mcmc {seed_directory}/GENERATED/main.yaml -np {n_processes} -nc 8 -nw 500 -ns 1000"], console=console)
            exit_if_no_files_at(f"{seed_directory}/MCMC/*.pkl", "MCMC")
            mcmc_time = time.time() - _start_time
            times.setdefault("MCMC", []).append(mcmc_time)
            
            with open(f"{seed_directory}/compute_times.txt", "w") as f:
                f.write(f"prior_sim, mle, ift, mcmc\n")
                f.write(f"{prior_sim_time}, {mle_time}, {ift_time}, {mcmc_time}\n")
            
    # NOTE: If we try to run the following commands in the above for-loop we run into a problem with the run_priorSim call
    #       on the second iteration over the seeds array. The mpiexec command silently fails. 
    #       This makes it very difficult to debug. I am not sure if this is a resource management issue or something with MPI.

    for seed in seeds:
        main_yaml = load_yaml(yaml_file)
        ift_yaml  = load_yaml(main_yaml["nifty"]["yaml"])
        base_directory = main_yaml["base_directory"]
        parent_directory = os.path.dirname(base_directory)
        n_processes = main_yaml["n_mass_bins"]

        _start_time = time.time()
        resultManager = ResultManager(f"{seed_directory}/GENERATED/main.yaml")
        resultManager.attempt_load_all() # load all possible results (generated, data histogram, mle, mcmc, ift) in the 'base_directory' of the yaml file
        resultManager.attempt_project_moments(pool_size=n_processes)
        plot_binned_intensities(resultManager, file_type='pdf')
        plot_binned_complex_plane(resultManager, file_type='pdf')
        plot_overview_across_bins(resultManager, file_type='pdf')
        plot_moments_across_bins(resultManager, file_type='pdf')
        montage_and_gif_select_plots(resultManager, file_type='pdf')
        del resultManager
        times.setdefault("Plotting", []).append(time.time() - _start_time)

    ################################
    ### DUMP TIMING INFORMATION ###
    ################################
    console.rule()
    console.print(f"\n\n[bold green]Total time taken to run {len(seeds)} seeds: {time.time() - start_time:.2f} seconds[/bold green]")

    table = Table(title="Times")
    table.add_column("Stage", justify="right")
    table.add_column("Average", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Relative", justify="right")

    total_time_for_seed = np.sum(np.array(list(times.values())), axis=0)[0]
    for stage, time_in_stage in times.items():
        mean = np.mean(time_in_stage)
        min = np.min(time_in_stage)
        max = np.max(time_in_stage)
        table.add_row(stage, f"{mean:.1f}", f"{min:.1f}", f"{max:.1f}", f"{mean / total_time_for_seed:.2f}")
    console.print(table)
    console.rule()

