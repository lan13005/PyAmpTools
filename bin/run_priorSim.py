import argparse
from rich.console import Console
import os

# TODO:
# - currently only one polarization is supported with 100% pol mag

subidr = "GENERATED"
console = Console()

def simulate_from_prior(main_yaml, base_directory, data_folder, channel, verbose=True):
    
    if channel == "TwoPseudoscalar":
        simulator = "gen_amp"
    elif channel == "VectorPseudoscalar":
        simulator = "gen_vec_ps"
    else:
        raise ValueError(f"Unsupported channel: {channel}")
    
    stage1_dir = f"{base_directory}/GENERATED/prior_sim_unnormalized"
    stage2_dir = data_folder
    
    # Ask iftpwa for a sample from the prior distribution (does not consider normalization - which is why need to rerun later)
    cmd_list = []
    cmd_list.append(f"sed -i -E 's/force_load_normint: true/force_load_normint: false/' {main_yaml}")
    cmd_list.append(f"mkdir -p {stage1_dir}")
    cmd_list.append(f"pa run_ift {main_yaml} --prior_simulation")
    cmd_list.append(f"mv {base_directory}/NIFTY/niftypwa_fit.pkl {stage1_dir}/niftypwa_fit.pkl")
    
    # Need to execute immediately since sim_to_amptools_cfg needs to use it
    #   Generate amptools cfg file using piecewise amplitudes
    execute_cmd(cmd_list, console=console)
    cmd_list = []
    sim_to_amptools_cfg(stage1_dir + "/niftypwa_fit.pkl", main_yaml, output_file=f"{base_directory}/GENERATED/prior_sim_amptools.cfg")

    # Make simulations (DATA)
    cmd_list.append(f"mkdir -p {stage2_dir}")
    configuration = f"-l {min_mass} -u {max_mass} -n {n_data} -a {min_ebeam} -b {max_ebeam} -t {tslope}"
    cmd_list.append(f"pa {simulator} {base_directory}/GENERATED/prior_sim_amptools.cfg -o {stage2_dir}/data000.root {configuration}")
    cmd_list.append(f"mv {simulator}_diagnostic.root {stage2_dir}/{simulator}_diagnostic_data.root")

    # Make simulations (PHASE SPACE)
    configuration = f"-l {min_mass} -u {max_mass} -n {n_phasespace} -a {min_ebeam} -b {max_ebeam} -t {tslope} -f "
    cmd_list.append(f"pa {simulator} {base_directory}/GENERATED/prior_sim_amptools.cfg -o {stage2_dir}/accmc000.root {configuration}")
    cmd_list.append(f"ln -s {stage2_dir}/accmc000.root {stage2_dir}/genmc000.root")
    cmd_list.append(f"mv {simulator}_diagnostic.root {stage2_dir}/{simulator}_diagnostic_ps.root")
    
    # # Divide data to allow access to integral matrices
    cmd_list.append(f"pa run_cfgGen {main_yaml}")
    cmd_list.append(f"pa run_divideData {main_yaml}")
    cmd_list.append(f"pa run_processEvents {main_yaml} --verbose")

    # Recreate prior sim with proper normalization
    cmd_list.append(f"sed -i -E 's/force_load_normint: false/force_load_normint: true/' {main_yaml}")
    cmd_list.append(f"mkdir -p {base_directory}/NIFTY/prior_sim")
    cmd_list.append(f"pa run_ift {main_yaml} --prior_simulation")
    cmd_list.append(f"mv {base_directory}/NIFTY/niftypwa_fit.pkl {base_directory}/GENERATED/niftypwa_fit.pkl")
    cmd_list.append(f"rm -rf {base_directory}/NIFTY")
    
    execute_cmd(cmd_list, console=console)

def sim_to_amptools_cfg(resultFile, main_yaml, output_file):
    
    """
    Convert a NIFTy prior simulation amplitude field to an AmpTools configuration file

    Args:
        resultFile (str): Path to the NIFTy prior simulation pkl result file
        main_yaml (str): Path to the main yaml file
        output_file (str): Path to dump the output AmpTools configuration file to
    """
    
    from pyamptools.utility.cfg_gen_utils import generate_amptools_cfg, amptools_zlm_ampName, amptools_vps_ampName, help_header
    import pickle as pkl
    import numpy as np
    import pandas as pd
    from iftpwa1.utilities.helpers import load_callable_from_module, reload_fields_and_components
    from omegaconf import DictConfig
    import tempfile
    
    # NOTE: For some reason we have to include all 4 terms of the Zlm amplitude whether or not P_gamma is less than 1 or equal to 1
    #       Unsure why this is the case. When P_gamma = 1, half the terms drop off.
    #       In all scenarios the Mass indep Amptools results matches the iftpwa results, it seems like it has to do with the generation
    #       of the signal using Piecewise/Zlm amplitudes in this file.

    bkg_amps_waves_tprime = {}
    res_amps_waves_tprime = {}
    amp_field_tprime = []
    ift_res_dfs = []

    main_dict = load_yaml(main_yaml)
    min_mass =  main_dict['min_mass']
    max_mass = main_dict['max_mass']
    nmbMasses = main_dict['n_mass_bins']
    
    ########################
    # DETERMINE POLARIZATION INFO
    ########################
    
    if 'polarizations' not in main_dict:
        console.print("No polarizations found in yaml file. Assuming 100% polarization at angle 0.", style="bold yellow")
        polMag = 1.0
        polAngle = 0
    else:
        if len(main_dict['polarizations']) != 1:
            console.print("Only one polarization is supported at the moment. Update yaml `polarizations` key.", style="bold red")
            exit(1)
        polAngle = list(main_dict['polarizations'].keys())[0]
        if isinstance(main_dict['polarizations'][polAngle], (dict, DictConfig)):
            polMag = float(main_dict['polarizations'][polAngle]['pol_mag'])
        else:
            polMag = float(main_dict['polarizations'][polAngle])
        
    polAngle = int(polAngle)
    if polAngle not in [0, 45, 90, 135]:
        raise ValueError(f"Polarization angle must be 0, 45, 90, or 135 degrees. Got {polAngle}.")
    polAngle = f"{polAngle:03d}" # convert to [000, 045, 090, 135]
    react_name = f"reaction_{polAngle}"

    if polMag > 1 or polMag < 0:
        raise ValueError(f"Polarization magnitude must be between [0, 1]. Got {polMag}.")

    ########################
    # LOAD NIFTY SIMULATION
    ########################

    resultData = pkl.load(open(resultFile, "rb"))

    resonance_paramater_data_frame = {}
    for resonance_parameter in resultData["fit_parameters_dict"]:
        if "scale" not in resonance_parameter:
            resonance_paramater_data_frame[resonance_parameter] = np.array(
                resultData["fit_parameters_dict"][resonance_parameter]
            )
    resonance_paramater_data_frame = pd.DataFrame(resonance_paramater_data_frame)
    ift_res_dfs.append(resonance_paramater_data_frame)

    aux_info = resultData["pwa_manager_aux_information"]
    using_custom_intens = "calc_intensity" in aux_info and "calc_intensity_args" in aux_info
    if using_custom_intens:
        module_path, callable_name = resultData["pwa_manager_aux_information"]["calc_intensity"]
        calc_kwargs = resultData["pwa_manager_aux_information"]["calc_intensity_args"]
        calc_intens = load_callable_from_module(module_path, callable_name)(**calc_kwargs)

    # Load fields and parametric components
    _result = reload_fields_and_components(resultData=resultData)
    signal_field_sample_values, amp_field, _res_amps_waves_tprime, _bkg_amps_waves_tprime, kinematic_mutliplier = _result[:5]
    threshold_selector = kinematic_mutliplier > 0

    # Reload general information
    wave_names = resultData["pwa_manager_base_information"]["wave_names"]
    wave_names = np.array([wave.split("::")[-1] for wave in wave_names])
    nmb_waves = len(wave_names)

    mass_bins = resultData["pwa_manager_base_information"]["mass_bins"]
    masses = 0.5 * (mass_bins[1:] + mass_bins[:-1])
    mass_limits = (np.min(mass_bins), np.max(mass_bins))

    tprime_bins = resultData["pwa_manager_base_information"]["tprime_bins"]

    nmb_samples, dim_signal, nmb_masses, nmb_tprime = signal_field_sample_values.shape  # Dimensions

    for k in _bkg_amps_waves_tprime.keys():
        short_k = k.split("::")[-1]
        if short_k not in bkg_amps_waves_tprime:
            bkg_amps_waves_tprime[short_k] = []
            res_amps_waves_tprime[short_k] = []
        bkg_amps_waves_tprime[short_k].append(_bkg_amps_waves_tprime[k][0])
        res_amps_waves_tprime[short_k].append(_res_amps_waves_tprime[k][0])

    amp_field_tprime.append(amp_field)
    amp_field_tprime = np.concatenate(amp_field_tprime, axis=-1)
    
    if nmb_samples > 1:
        raise ValueError("Multiple samples found in NIFTy prior simulation. This should not happen!")
    if nmb_tprime > 1:
        raise ValueError("Multiple tprime bins found in NIFTy prior simulation. Only 1 t-bin is supported at the moment!")

    amp_field_tprime = amp_field_tprime[0, :, :, 0] # only 1 sample and 1 tbin
    
    ########################
    # CREATE AMPTOOLS CFG
    ########################
    
    # Extract quantum numbers from wave_names
    quantum_numbers = []
    for wave in wave_names:
        if wave in converter:
            quantum_numbers.append(converter[wave])
        else:
            raise ValueError(f"Unsure how to parse amplitude: '{wave}'. See pyamptools.utility.general.converter for supported amplitudes.")
    
    # Set up parameters for generate_amptools_cfg
    polAngles = [polAngle]
    polMags = [polMag]
    polFixedScales = [True]  # Fix the scale for the single polarization
    
    # No need for source files for simulation
    datas = []
    gens  = []
    accs  = []
    bkgnds = [] 
    
    # No real or fixed amplitudes for prior simulation
    realAmps  = []
    fixedAmps = []
    
    # Create a temporary file to store the basic config
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_cfg:
        temp_cfg_name = temp_cfg.name
    
    # Generate the basic config structure
    fitName = "prior_sim"
    basereactName = "reaction"
    particles = main_dict['reaction'].split()

    # Generate the basic config file
    initialization = complex(1.0, 0.0)
    generate_amptools_cfg(
        quantum_numbers,
        polAngles,
        polMags,
        polFixedScales,
        datas,
        gens,
        accs,
        bkgnds,
        realAmps,
        fixedAmps,
        fitName,
        temp_cfg_name,
        basereactName,
        particles,
        header=help_header,
        datareader=main_dict["datareader"],
        add_amp_factor=main_dict.get("add_amp_factor", "").strip(),
        append_to_cfg=main_dict.get("append_to_cfg", "").strip(),
        append_to_decay=main_dict.get("append_to_decay", "").strip(),
        initialization=initialization,
        exclude_sums_zeroed_by_polmag_value=False,
    )
    
    # Read the basic config
    with open(temp_cfg_name, 'r') as f:
        cfg = f.read()
    
    #########################################################
    # Now add the Piecewise amplitude information
    #########################################################
    
    # Copy the original config file
    new_cfg = cfg.split('\n')

    # Extract the lines where Zlm or vec_ps_refl amplitudes are defined
    amp_lines = []
    for line in cfg.split('\n'):
        if line.startswith('amplitude') and (amptools_zlm_ampName in line or amptools_vps_ampName in line):
            amp_lines.append(line)

    # Add Piecewise amplitudes
    for i, wave_name in enumerate(wave_names):
        amps = amp_field_tprime[i, :]
        
        # Add long string of parameter values to Piecewise amplitude
        amp_parTag = ""
        for j, amp in enumerate(amps):
            amp_parTag += f" {amps[j].real:0.5f}"
            amp_parTag += f" {amps[j].imag:0.5f}"
        amp_parTag = amp_parTag.lstrip()
    
        # Add Piecewise amplitude lines to config file
        relevant_amp_lines = [line for line in amp_lines if f"::{wave_name} " in line] # i.e. ["amplitude react::sum::wave Zlm ..."]
        for line in relevant_amp_lines:
            react_name, sum_name, amp_name = line.strip().split(' ')[1].split('::')
            if amp_name != wave_name:
                raise ValueError(f"Failed to properly parse amplitude name from line: {line}")
            refl_tag = "Pos" if wave_name[-1] == "+" else "Neg"
            new_cfg.append(f"amplitude {react_name}::{sum_name}::{wave_name} Piecewise {min_mass} {max_mass} {nmbMasses} 23 {refl_tag} ReIm {amp_parTag}")                
    
    # Write the final config
    with open(output_file, "w") as f:
        f.write('\n'.join(new_cfg))
    
    # Clean up temporary file
    os.unlink(temp_cfg_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw sample from NIFTy prior, create amptools cfg file, generate data using gen_amp/gen_vec_ps, and split data into kinematic bins")
    parser.add_argument("main_yaml", type=str, help="Path to main yaml file")
    parser.add_argument("-emin", "--min_ebeam", type=float, default=8.2, help="Minimum photon beam energy (default: %(default)s)")
    parser.add_argument("-emax", "--max_ebeam", type=float, default=8.8, help="Maximum photon beam energy (default: %(default)s)")
    parser.add_argument("-t", "--t_slope", type=float, default=-1, help="Slope of t distribution, reserved -1 to calculate from min_t and max_t by maximing probability mass(default: %(default)s)")
    parser.add_argument("-nd", "--n_data", type=int, default=100000, help="Number of data events to generate (default: %(default)s)")
    parser.add_argument("-np", "--n_phasespace", type=int, default=500000, help="Number of phase space events to generate (default: %(default)s)")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed (default: %(default)s)")
    parser.add_argument("-c", "--clean", action="store_true", help="Clean all directories before running (default: %(default)s)")
    args = parser.parse_args()
    
    import os
    os.environ["JAX_PLATFORMS"] = "cpu"
    import sys
    from pyamptools.utility.general import load_yaml, dump_yaml, identify_channel, converter, execute_cmd
    from pyamptools.utility.resultManager import ResultManager, plot_gen_curves
    import numpy as np

    main_yaml = args.main_yaml
    min_ebeam = args.min_ebeam
    max_ebeam = args.max_ebeam
    n_data = args.n_data
    n_phasespace = args.n_phasespace
    seed = args.seed
    
    if not os.path.exists(main_yaml):
        console.print(f"YAML file does not exist: {main_yaml}", style="bold red")
        sys.exit(1)
    
    main_dict = load_yaml(main_yaml)
    base_directory = main_dict["base_directory"]
    data_folder = main_dict["data_folder"]
    min_mass = main_dict["min_mass"]
    max_mass = main_dict["max_mass"]
    min_t = main_dict["min_t"]
    max_t = main_dict["max_t"]
    iftpwa_dict = main_dict["nifty"]["yaml"]
    wave_names = main_dict["waveset"].split("_")
    channel = identify_channel(wave_names) # this function performs a check to ensure channel is supported
    
    default_prob_mass = 0.9
    
    def f(a):
        return np.exp(-a * min_t) - np.exp(-a * max_t) - default_prob_mass
    
    # NOTE: If user does not specify a tslope this program will assume they want the most statistics as possible in the
    #       specified t-range. This is equivalent to finding the t-slope that maximizes the probability mass between tmin and tmax
    #       This assumes the t-distribution is exponentially distributed ~ exp(-a*t)
    if args.t_slope == -1:
        if min_t == 0:
            tslope = -np.log(1 - default_prob_mass) / max_t
            prob_max = 1 - np.exp(-tslope * max_t)
        else:
            tslope = np.log(max_t / min_t) / (max_t - min_t)
            prob_max = np.exp(-tslope * min_t) - np.exp(-tslope * max_t)
        console.print(f"tslope was not specified. We will choose a tslope that maximizes the probability mass between tmin and tmax")
        descrip = f"[bold yellow]- calculated (since unspecified) to max probability mass (prob_mass={prob_max:0.3f}) on (tmin={min_t:0.3f}, tmax={max_t:0.3f})[/bold yellow]"
    else:
        tslope = args.t_slope
        console.print(f"tslope was specified by the user. Using tslope={tslope}")
        descrip = ""

    console.rule()
    console.print("Prior Simulation Configuration", style="bold green")
    console.print(f"YAML file: {main_yaml}")
    console.print(f"Minimum mass: {min_mass:0.3f}")
    console.print(f"Maximum mass: {max_mass:0.3f}")
    console.print(f"Minimum t: {min_t:0.3f}")
    console.print(f"Maximum t: {max_t:0.3f}")
    console.print(f"t slope: {tslope:0.3f} {descrip}")
    console.print(f"Minimum photon beam energy: {min_ebeam:0.3f}")
    console.print(f"Maximum photon beam energy: {max_ebeam:0.3f}")
    console.print(f"Number of data events to generate: {n_data}")
    console.rule()
    console.print("\n\n")
        
    if args.clean:
        cmd_list = []
        for dirpath, dirnames, filenames in os.walk(base_directory):
            for dir in dirnames:
                cmd_list.append(f"rm -rf {dirpath}/{dir}")
        cmd_list.append(f"rm -rf {data_folder}")
        execute_cmd(cmd_list, console=console)
    
    # Guard against users overwriting data_folder accidentally when running prior simulation
    if os.path.exists(data_folder):
        console.print(f"Data Folder: {data_folder} already exists. Cannot safely dump prior simulation data here. Exiting...", style="bold red")
        exit(1)

    if not os.path.exists(f"{base_directory}/GENERATED"):
        os.makedirs(f"{base_directory}/GENERATED")
        
    iftpwa_dest_path = f"{base_directory}/GENERATED/iftpwa.yaml"
    main_dest_path = f"{base_directory}/GENERATED/main.yaml"
    
    with open(iftpwa_dest_path, "w") as iftpwa_yaml_file, \
         open(main_dest_path, "w") as main_yaml_file:

        iftpwa_dict['GENERAL']['seed'] = seed
        iftpwa_dict["PWA_MANAGER"]["yaml"] = main_dest_path
        iftpwa_dict["IFT_MODEL"]["scale"] = 'auto'
        console.print("Updated 'scale' in iftpwa yaml file to 'auto' to auto-scale amplitudes for downstream fitting", style="bold green")
        main_dict["nifty"]["yaml"] = iftpwa_dest_path
        data_folder = main_dict["data_folder"]
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        console.print(f"Dumping simulated data into yaml.data_folder: {data_folder}", style="bold blue")
        
        console.rule()
        console.print("NOTE: These YAML files contains properly updated file paths. "
                      "Since the simulations have been generated (stored in DATA_SOURCES folder), data split into kinematic bins, you are ready to run fits with this yaml pair. "
                      "User supplied YAML files have not been modified.", style="bold yellow")
        dump_yaml(iftpwa_dict, iftpwa_dest_path, console=console)
        dump_yaml(main_dict, main_dest_path, console=console)
        console.rule()
        
        # Run the prior simulation, drawing a sample from NIFTy prior, use {simulator} to generate data, split data into kinematic bins
        simulate_from_prior(main_dest_path, base_directory, data_folder, channel)

        _dumped_yaml = load_yaml(main_dest_path)
        if 'share_mc' in _dumped_yaml:
            exist_but_not_match = 'share_mc' in main_dict and main_dict['share_mc'] != _dumped_yaml['share_mc']
            not_exist = 'share_mc' not in main_dict
            if exist_but_not_match or not_exist:
                console.print("\nUpdating 'share_mc' in yaml file to match prior simulation\n", style="bold green")
                main_dict['share_mc'] = _dumped_yaml['share_mc']
                dump_yaml(main_dict, main_yaml)
            
        # Draw a plot of the generated partial waves
        resultManager = ResultManager(main_dict)
        resultManager.attempt_load_all()
        plot_gen_curves(resultManager)
        
        del resultManager
