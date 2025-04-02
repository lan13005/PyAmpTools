import os
import random
import tempfile
from pyamptools.utility.general import load_yaml, recursive_replace, dump_yaml, execute_cmd
from pyamptools.utility.resultManager import ResultManager, plot_binned_intensities, plot_binned_complex_plane, plot_overview_across_bins, montage_and_gif_select_plots, plot_moments_across_bins
from rich.console import Console

console = Console()

random.seed(1420)
nseeds = 20
seeds = random.sample(range(10000000), nseeds)

yaml_file = "/w/halld-scshelf2101/lng/WORK/PyAmpTools9/OTHER_CHANNELS/ETAPI0_AUTOGRAD/pyamptools_mc.yaml"

for seed in seeds:
    
    console.print(f"\n\n[bold red]Beginning seed {seed}[/bold red]\n")
    
    # Reload everytime to reset state (i.e. resetting folder output location)
    main_yaml = load_yaml(yaml_file)
    ift_yaml  = load_yaml(main_yaml["nifty"]["yaml"])
    base_directory = main_yaml["base_directory"]
    parent_directory = os.path.dirname(base_directory)
    recursive_replace(ift_yaml, base_directory, f"{parent_directory}/SEED_{seed}")
    recursive_replace(main_yaml, base_directory, f"{parent_directory}/SEED_{seed}")
    
    n_processes = main_yaml["n_mass_bins"]

    seed_directory = f"{parent_directory}/SEED_{seed}"
    os.makedirs(seed_directory, exist_ok=True)

    # We create temporary yaml files in case we want work with a modified version since all below commands takes a yaml files as input
    # run_priorSim will will create additional yaml files at GENERATED/*.yaml so the above ones can be safely deleted
    with tempfile.NamedTemporaryFile(delete=True) as tmp_main_yaml, \
         tempfile.NamedTemporaryFile(delete=True) as tmp_ift_yaml:
        main_yaml["nifty"]["yaml"] = tmp_ift_yaml.name
        ift_yaml["PWA_MANAGER"]["yaml"] = tmp_ift_yaml.name
        dump_yaml(main_yaml, tmp_main_yaml.name)
        dump_yaml(ift_yaml, tmp_ift_yaml.name)
        
        execute_cmd([f"pa run_priorSim {tmp_main_yaml.name} -s {seed}"], console=console) # Generate data, divide data, ready to fit
        execute_cmd([f"pa run_mle {parent_directory}/SEED_{seed}/GENERATED/main.yaml"], console=console) # Run on new yaml files created by run_priorSim
        execute_cmd([f"pa run_ift {parent_directory}/SEED_{seed}/GENERATED/main.yaml"], console=console)
        execute_cmd([f"pa run_mcmc {parent_directory}/SEED_{seed}/GENERATED/main.yaml -np {n_processes} -nc 8 -nw 500 -ns 1000"], console=console)

# NOTE: If we try to run the following commands in the above for-loop we run into a problem with the run_priorSim call
#       on the second iteration over the seeds array. The mpiexec command silently fails. 
#       This makes it very difficult to debug. I am not sure if this is a resource management issue or something with MPI.
for seed in seeds:
    main_yaml = load_yaml(yaml_file)
    ift_yaml  = load_yaml(main_yaml["nifty"]["yaml"])
    base_directory = main_yaml["base_directory"]
    parent_directory = os.path.dirname(base_directory)

    resultManager = ResultManager(f"{parent_directory}/SEED_{seed}/GENERATED/main.yaml")
    resultManager.attempt_load_all() # load all possible results (generated, data histogram, mle, mcmc, ift) in the 'base_directory' of the yaml file
    resultManager.attempt_project_moments(pool_size=n_processes)
    plot_binned_intensities(resultManager)
    plot_binned_complex_plane(resultManager)
    plot_overview_across_bins(resultManager)
    plot_moments_across_bins(resultManager)
    montage_and_gif_select_plots(resultManager)
    del resultManager
