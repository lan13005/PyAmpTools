import glob
import itertools
import os
import re
from collections import defaultdict
from pyamptools.utility.general import glob_sort_captured, identify_channel
from pyamptools.utility.MomentUtilities import MomentManagerTwoPS, MomentManagerVecPS
from iftpwa1.utilities.helpers import reload_fields_and_components
import numpy as np
import pandas as pd
import pickle as pkl
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue
from rich.console import Console

pd.options.mode.chained_assignment = None  # default='warn'

console = Console()

def get_nEventsInBin(base, acceptance_correct=True):
    """
    Load the expected number of events across all bins in a given directory. Searches metadata.txt files
    for the number of signal events (data-bkgnd). These values are calculated when the data divided using
    run_divideData + split_mass_t

    Returns:
        np.array: array of nBar values for each bin. It is up to the user to reshape to match kinematic binnings
    """

    signal_str = "nBar signal"
    if acceptance_correct:
        signal_str = "nBar corrected signal"
    intrisic_spaces = signal_str.count(" ")

    fs = glob.glob(f"{base}/bin_*/metadata.txt")
    fs = sorted(fs, key=lambda x: int(re.search(r"bin_(\d+)", x).group(1)))

    values = []
    errors = []
    for f in fs:
        with open(f) as f:
            for line in reversed(f.readlines()): # signal line is actually close to being the last line currently
                if line.startswith(signal_str):
                    line = line.strip()
                    parts = line.split(" ")
                    value, error = float(parts[intrisic_spaces+1]), float(parts[intrisic_spaces+3])
                    values.append(value)
                    errors.append(error)
                    break

    return np.array(values), np.array(errors)

# NOTE: This could in principle be done directly with amptools FitResults class but
#       when running nifty mpi fits it causes a crash. Unsure about the problem, but
#       this is a workaround for now. Parsing the file is not slow anyways
def parse_fit_file(filename):
    
    """
    Parse the fit file and return a pair of dictionaries:
    - complex amplitudes
    - convergence status + likelihood
    """
    
    # Dictionary to store real and imaginary parts
    complex_amps_builder = defaultdict(int) # default value is 0 for ints
    status_dict = {}
    
    # Read the file
    with open(filename, 'r') as f:
        for line in f:
            # Use regex sub to replace all whitespace with single space
            line = re.sub(r'\s+', ' ', line).strip()
            
            if "bestMinimum" in line:
                status_dict["bestMinimum"] = float(line.split()[1]); continue
            elif "lastMinuitCommandStatus" in line:
                status_dict["lastMinuitCommandStatus"] = int(line.split()[1]); continue
            elif "eMatrixStatus" in line:
                status_dict["eMatrixStatus"] = int(line.split()[1]); continue
                
            # you have gone too far by now
            if "+++ Normalization Integrals +++" in line:
                break

            # skip over lines that start looking like a numeric including sign -
            #    (could match the parameter covariance matrix before the normalization integrals dump)
            if re.match(r'^-?\d', line):
                continue

            # must end in space! _re could exist in vec_ps_refl
            if "_re " not in line and "_im " not in line: continue

            # Split line into name and value
            name, value_str = line.strip().split(' ')
            try:
                value = float(value_str)
            except ValueError: continue
                
            # Parse the amplitude name
            parts = name.split('::')
            if len(parts) != 3: continue
            reaction, sum_type, amp_part = parts
            
            # Split amp_part into base name and re/im indicator
            match = re.match(r'(.+)_(re|im)$', amp_part)
            if not match: continue
            base_name, component = match.groups()
            
            # Store value
            key = (reaction, sum_type, base_name)
            complex_amps_builder[key] += value * (1j if component == 'im' else 1)

    complex_amps = {}
    for key, value in complex_amps_builder.items():
        if key[2] not in complex_amps:
            complex_amps[key[2]] = value
        else:
            if complex_amps[key[2]] != value:
                console.print(f"io| ERROR: {key[2]} is expected to be the same as it should be constrained to be the same")
                console.print(f"io|  {key[2]} = {value} != {complex_amps[key[2]]}")
    
    return complex_amps, status_dict

def loadAllResultsFromYaml(yaml, pool_size=10, skip_moments=False, clean=False):
    
    """
    Loads the AmpTools and IFT results from the provided yaml file. Moments will be calculated if possible with multiprocessing.Pool with pool_size
    
    Args:
        yaml (OmegaConf): OmegaConf object (dict-like) for PyAmpTools (not iftpwa)
        pool_size (int): Number of processes to use by multiprocessing.Pool
        skip_moments (bool): If True, only load partial wave amplitudes and do not calculate moments
        clean (bool): If True, clean the output directory before running
        
    Returns:
        pd.DataFrame: AmpTools binned fit results
        pd.DataFrame: IFT binned fit results
        list: wave names
        np.array: masses (bin centers)
        np.array: tPrimeBins (bin centers)
        int: bins per group (is an additional scale factor to align IFT and AmpTools results)
        dict: latex naming of moments
    """
    
    # Be nice and accept a string path to a yaml file in case user didnt read docs
    if isinstance(yaml, str):
        yaml = OmegaConf.load(yaml)
    
    # Store a cache of the results to avoid recalculating them, calculating moments is quite time consuming
    cache = f"{yaml['base_directory']}/.moment_cache.pkl"
    if clean and os.path.exists(cache):
        console.print(f"[red]loadAllResultsFromYaml| User requested to start clean, do not use any cached results. Recalculating...[/red]\n")
        os.remove(cache)
    elif os.path.exists(cache):
        console.print(f"[green]loadAllResultsFromYaml| Loading cache from {cache}[/green]\n")
        with open(cache, "rb") as f:
            cache = pkl.load(f)
        return cache
    
    amptools_df = None
    ift_df = None
    ift_res_df = None

    ###########################################
    #### LOAD ADDITIONAL INFORMATION FROM YAML
    ###########################################
    wave_names = yaml['waveset'].split('_')
    min_mass = yaml['min_mass']
    max_mass = yaml['max_mass']
    n_mass_bins = yaml['n_mass_bins']
    min_t = yaml['min_t']
    max_t = yaml['max_t']
    n_t_bins = yaml['n_t_bins']
    masses = np.linspace(min_mass, max_mass, n_mass_bins+1)
    tPrimeBins = np.linspace(min_t, max_t, n_t_bins+1)
    masses = 0.5 * (masses[:-1] + masses[1:])
    tPrimeBins = 0.5 * (tPrimeBins[:-1] + tPrimeBins[1:])
    bpg = yaml['amptools']['bins_per_group']
    channel = identify_channel(wave_names)

    ###############################
    #### LOAD AMPTOOLS RESULTS
    ###############################
    console.print("io| Attempting to load AmpTools results...")
    loadAmpToolsResultsFromYaml(yaml, ensure_one_fit_per_bin=False)
    try:
        # hardcode check to False since hopefully if user calls this function they would want all fits
        #   across all randomizations and kinematic bins
        amptools_df, (_, _, _) = loadAmpToolsResultsFromYaml(yaml, ensure_one_fit_per_bin=False)
    except Exception as e:
        print(f"io| Error loading AmpTools results: {e}")

    ###############################
    #### LOAD NIFTY RESULTS
    ###############################
    console.print("io| Attempting to load NIFTY results...")
    try:
        ift_df, ift_res_df = loadIFTResultsFromYaml(yaml)
    except Exception as e:
        console.print(f"\n[red]io| Error loading NIFTY results: {e}[/red]")
        
    if isinstance(ift_df, pd.DataFrame) and len(ift_df) == 0:
        ift_df = None
    if isinstance(amptools_df, pd.DataFrame) and len(amptools_df) == 0:
        amptools_df = None

    if amptools_df is None:
        console.print(f"\n[red]io| AmpTools results were not found in expected location! [/red]")
    if ift_df is None:
        console.print(f"\n[red]io| NIFTy results were not found in expected location! [/red]")
    console.print(f"\n")

    if amptools_df is None and ift_df is None:
        raise ValueError("io| No results found for IFT nor AmpTools binned fits! Terminating as there is nothing to do...")

    ###########################################
    #### PROCESS THE AMPTOOLS AND IFT RESULTS
    ###########################################
    console.print("\nio| Processing AmpTools and IFT results...\n")
    latex_name_dict = None
    if not skip_moments:
        latex_name_dict_amp, latex_name_dict_ift = None, None
        if amptools_df is not None:
            if channel == "TwoPseudoscalar":
                amptools_manager = MomentManagerTwoPS(amptools_df, wave_names)
            elif channel == "VectorPseudoscalar":
                amptools_manager = MomentManagerVecPS(amptools_df, wave_names)
            # for col in amptools_df.columns:
            #     print(f"{col} {amptools_df[col].dtype}")
            amptools_df, latex_name_dict_amp = amptools_manager.process_and_return_df(normalization_scheme=1, pool_size=pool_size, append=True)

        if ift_df is not None:
            if channel == "TwoPseudoscalar":
                ift_manager = MomentManagerTwoPS(ift_df, wave_names)
            elif channel == "VectorPseudoscalar":
                ift_manager = MomentManagerVecPS(ift_df, wave_names)
            ift_df, latex_name_dict_ift = ift_manager.process_and_return_df(normalization_scheme=1, pool_size=pool_size, append=True)
        
        if latex_name_dict_amp is not None and latex_name_dict_ift is not None and latex_name_dict_amp != latex_name_dict_ift:
            raise ValueError("IFT and AmpTools moment dictionaries do not match but is expected to!")
        
        # think its safe? If they are not None by this point they must be the same dictionary
        latex_name_dict = latex_name_dict_amp if latex_name_dict_amp is not None else latex_name_dict_ift
        
    with open(cache, "wb") as f:
        console.print(f"[green]loadAllResultsFromYaml| Dumping cache to {cache}[/green]")
        cache = (amptools_df, ift_df, ift_res_df, wave_names, masses, tPrimeBins, bpg, latex_name_dict)
        pkl.dump(cache, f)
    
    return amptools_df, ift_df, ift_res_df, wave_names, masses, tPrimeBins, bpg, latex_name_dict

def loadIFTResultsFromYaml(yaml):
    
    """
    Loads the NIFTY results from the provided yaml file
    NOTE: This only loads the IFT coherent sums in each partial wave. Complex amplitudes of each component still needs to be coded up here
    
    Args:
        yaml (OmegaConf): OmegaConf object (dict-like) for PyAmpTools (not iftpwa)
        
    Returns:
        pd.DataFrame: DataFrame of results
    """

    # Be nice and accept a string path to a yaml file in case user didnt read docs
    if isinstance(yaml, str):
        yaml = OmegaConf.load(yaml)
    
    yaml_secondary = yaml['nifty']['yaml']
    yaml_secondary = OmegaConf.load(yaml_secondary)
    
    try: 
        yaml_secondary['GENERAL']['outputFolder']
    except MissingMandatoryValue:
        outputFolder = yaml['nifty']['output_directory']
        yaml_secondary['GENERAL']['outputFolder'] = outputFolder
    nifty_pkl = yaml_secondary['GENERAL']['fitResultPath']

    with open(nifty_pkl, "rb") as f:
        resultData = pkl.load(f)
        
    _result = reload_fields_and_components(resultData=resultData)

    # signal_field_sample_values, amp_field, _res_amps_waves_tprime, _bkg_amps_waves_tprime, kinematic_mutliplier = _result
    signal_field_sample_values, amp_field, res_amps_waves_tprime, bkg_amps_waves_tprime, kinematic_mutliplier, \
        threshold_selector, modelName, wave_names, wave_parametrizations, paras, resonances, calc_intens = _result

    threshold_selector = kinematic_mutliplier > 0

    # Reload general information
    wave_names = resultData["pwa_manager_base_information"]["wave_names"]
    wave_names = [wave.split("::")[-1] for wave in wave_names]
    nmb_waves = len(wave_names)

    mass_bins = resultData["pwa_manager_base_information"]["mass_bins"]
    masses = 0.5 * (mass_bins[1:] + mass_bins[:-1])
    mass_limits = (np.min(mass_bins), np.max(mass_bins))

    tprime_bins = resultData["pwa_manager_base_information"]["tprime_bins"]
    tprimes = 0.5 * (tprime_bins[1:] + tprime_bins[:-1])

    nmb_samples, dim_signal, nmb_masses, nmb_tprime = signal_field_sample_values.shape  # Dimensions

    # Stage 1:  Flatten the original dictionary structure into a single flat DataFrame
    #           Stores complex amplitude values (unnormalized)for each partial wave component, columns ending with "_amp"
    #               1. Amplitudes describe parametrically have a name <wave_name>_<resonance_name>_amp
    #               2. Residual (Correlated field) amplitudes have a name <wave_name>_cf_amp
    flat_amps_waves_tprime = {}
    for tbin in range(nmb_tprime):
        
        # For kinematics and sampling
        if 'tprime' not in flat_amps_waves_tprime:
            flat_amps_waves_tprime['tprime'] = np.repeat(tprimes[tbin], nmb_samples * nmb_masses)
        else:
            flat_amps_waves_tprime['tprime'] = np.concatenate((flat_amps_waves_tprime['tprime'], np.repeat(tprimes[tbin], nmb_samples * nmb_masses)))
        if 'mass' not in flat_amps_waves_tprime:
            flat_amps_waves_tprime['mass'] = np.tile(masses, nmb_samples)
        else:
            flat_amps_waves_tprime['mass'] = np.concatenate((flat_amps_waves_tprime['mass'], np.tile(masses, nmb_samples)))
        if 'sample' not in flat_amps_waves_tprime:
            flat_amps_waves_tprime['sample'] = np.repeat(np.arange(nmb_samples), nmb_masses)
        else:
            flat_amps_waves_tprime['sample'] = np.concatenate((flat_amps_waves_tprime['sample'], np.repeat(np.arange(nmb_samples), nmb_masses)))
            
        # For the total signal
        for wave_name in wave_names:
            if f'{wave_name}' not in flat_amps_waves_tprime:
                flat_amps_waves_tprime[f'{wave_name}_amp'] = amp_field[:, wave_names.index(wave_name), :, tbin].flatten()
            else:
                flat_amps_waves_tprime[f'{wave_name}_amp'] = np.concatenate((flat_amps_waves_tprime[f'{wave_name}'], amp_field[:, wave_names.index(wave_name), :, tbin].flatten()))

        # For the parametric models
        for wave_name in wave_names:
            if wave_name not in res_amps_waves_tprime: # there was no parameteric model for this partial wave
                continue
            for res in res_amps_waves_tprime[wave_name][tbin].keys():
                if f'{wave_name}_{res}_amp' not in flat_amps_waves_tprime:
                    flat_amps_waves_tprime[f'{wave_name}_{res}_amp'] = res_amps_waves_tprime[wave_name][tbin][res].flatten()
                else:
                    flat_amps_waves_tprime[f'{wave_name}_{res}_amp'] = np.concatenate((flat_amps_waves_tprime[f'{wave_name}_{res}_amp'], res_amps_waves_tprime[wave_name][tbin][res].flatten()))
                    
        # For the background / residual correlated field backgrounds
        for wave_name in wave_names:
            if wave_name not in bkg_amps_waves_tprime:
                continue
            if f'{wave_name}_cf_amp' not in flat_amps_waves_tprime:
                flat_amps_waves_tprime[f'{wave_name}_cf_amp'] = bkg_amps_waves_tprime[wave_name][tbin].flatten()
            else:
                flat_amps_waves_tprime[f'{wave_name}_cf_amp'] = np.concatenate((flat_amps_waves_tprime[f'{wave_name}_cf_amp'], bkg_amps_waves_tprime[wave_name][tbin].flatten()))
                
    flat_amps_waves_tprime = pd.DataFrame(flat_amps_waves_tprime)

    # Stage 2: Loop over amp columns and calculate+store the normalized intensity
    cols = list(flat_amps_waves_tprime.keys())
    flat_intens_waves_tprime = {}
    amp_cols = [col for col in cols if col.endswith('_amp')]
    for tbin, tprime in enumerate(tprimes):
        for col in amp_cols:
            wave = col.split('_')[0]
            tmp = flat_amps_waves_tprime.query('tprime == @tprime')
            intens = calc_intens([wave], tbin, [tmp[col].values.reshape(nmb_samples, nmb_masses)])
            if f'{col.strip("_amp")}' not in flat_amps_waves_tprime:
                    flat_intens_waves_tprime[f'{col.strip("_amp")}'] = intens.flatten().real
            else:
                flat_intens_waves_tprime[f'{col.strip("_amp")}'] = np.concatenate((flat_amps_waves_tprime[f'{col.strip("_amp")}'], intens.flatten().real))
                    
    # Stage 3: Calculate intensity for user requested coherent sums defined in yaml["result_dump"]["coherent_sums"]
    # NOTE: There might be something wrong with calc_intens? Directly summing intensities across GP waves (i.e. Pm1+_Pp0+_Pp1+) gives very similar results as the coherent sum P+
    #       Even if we sum BW for a given resonance I dont think this should happen since each BW has a different phase
    if "result_dump" in yaml and "coherent_sums" in yaml["result_dump"]:
        console.print('\nio| Calculating intensities for user specified coherent sums...\n')
        coherent_waves = {}
        coherent_amps = {}
        sums_dict = yaml["result_dump"]["coherent_sums"]
        for k, vs in sums_dict.items():
            vs = vs.split("_")
            for tbin, tprime in enumerate(tprimes):
                tmp = flat_amps_waves_tprime.query('tprime == @tprime')
                # Gather list of amp cols that belong in the user requested coherent sum
                for amp in amp_cols:
                    if any([v in amp for v in vs]):
                        wave, suffix = amp.partition('_')[0], amp.partition('_')[2].rstrip('_amp')
                        dest_key = f"{k}_{suffix}" if suffix != "" else k
                        tmp_amp = tmp[amp].values.reshape(nmb_samples, nmb_masses)
                        if dest_key not in coherent_waves:
                            coherent_waves[dest_key] = [wave]
                        else:
                            coherent_waves[dest_key].append(wave)
                        if dest_key not in coherent_amps:
                            coherent_amps[dest_key] = [tmp_amp]
                        else:
                            coherent_amps[dest_key].append(tmp_amp)
        # calculate intensity for the coherent sum
        for k in coherent_waves.keys():
            # print(f"{k} {coherent_waves[k]}")
            intens = calc_intens(coherent_waves[k], tbin, coherent_amps[k])
            if k not in flat_intens_waves_tprime:
                flat_intens_waves_tprime[k] = intens.flatten().real
            else:
                flat_intens_waves_tprime[k] = np.concatenate((flat_intens_waves_tprime[k], intens.flatten().real))
            
    flat_intens_waves_tprime = pd.DataFrame(flat_intens_waves_tprime)

    ift_pwa_df = pd.concat([flat_amps_waves_tprime, flat_intens_waves_tprime], axis=1)

    # Get the expected number of events (intensity)
    intensity_samples = np.array(resultData['expected_nmb_events'], dtype=float) # ~ (nmb_samples, nmb_masses, nmb_tprimes)
    intensity_samples_no_acc = np.array(resultData['expected_nmb_events_no_acc'], dtype=float) # ~ (nmb_samples, nmb_masses, nmb_tprimes)
    intensity_samples = intensity_samples.reshape(-1, 1)
    intensity_samples_no_acc = intensity_samples_no_acc.reshape(-1, 1)
    ift_pwa_df['intensity'] = intensity_samples
    ift_pwa_df['intensity_corr'] = intensity_samples_no_acc
    
    # Extract (res)onance parameter samples, dump into independent DataFrame/csv
    #   This is only an array over samples, and is the same value for all masses and tprimes
    #   Therefore, it doesn't make sense to store it in the same DataFrame as the partial wave amplitudes
    ift_res_df = {}
    for resonance_parameter in resultData["fit_parameters_dict"]:
        if "scale" not in resonance_parameter:
            ift_res_df[resonance_parameter] = np.array(
                resultData["fit_parameters_dict"][resonance_parameter]
            )
    ift_res_df = pd.DataFrame(ift_res_df)
    
    return ift_pwa_df, ift_res_df

def loadAmpToolsResultsFromYaml(yaml, ensure_one_fit_per_bin=True):
    """ 
    Helper function to load the amptools results from a yaml file 
    
    Args:
        yaml (OmegaConf): OmegaConf object (dict-like) for PyAmpTools (not iftpwa)
        ensure_one_fit_per_bin (bool):  Raise error if there are multiple fits results per kinematic bin. This flag is necessary to ensure `iftpwa_plot` program works as expected.
                                        If you are not using `iftpwa_plot` program, you can set this flag to False.
                                        NOTE: seems like we only need to implement t'-summed intensity section. ATM it seems like it would just sum over all fit results in a bin

        
    Returns:
        pd.DataFrame: DataFrame of results
        tuple: (masses, tPrimeBins, bpg) where masses and tPrimeBins correspond to the bin centers
    """
    
    # Be nice and accept a string path to a yaml file in case user didnt read docs
    if isinstance(yaml, str):
        yaml = OmegaConf.load(yaml)
    
    search_format = yaml['amptools']['search_format'] if yaml['amptools']['bins_per_group'] > 1 else 'bin'
    pattern = yaml['amptools']['output_directory']+f'/{search_format}_*/bin_[].cfg'
    console.print(f"io| Searching for amptools files using this pattern: {pattern}")
    cfgfiles = glob_sort_captured(pattern)
    min_mass = yaml['min_mass']
    max_mass = yaml['max_mass']
    if yaml['n_mass_bins'] % yaml['amptools']['bins_per_group'] != 0:
        raise ValueError("n_mass_bins must be divisible by amptools.bins_per_group")
    bpg = yaml['amptools']['bins_per_group']
    n_mass_bins = int(yaml['n_mass_bins'] / bpg)
    massBins = np.linspace(min_mass, max_mass, n_mass_bins+1)
    masses = 0.5 * (massBins[1:] + massBins[:-1])
    min_t = yaml['min_t']
    max_t = yaml['max_t']
    n_t_bins = yaml['n_t_bins']
    tPrimeBins = np.linspace(min_t, max_t, n_t_bins+1)
    tPrimes = 0.5 * (tPrimeBins[1:] + tPrimeBins[:-1])
    # No support for multiple fits per bin, choose best
    mle_query_1 = yaml['amptools'].get('mle_query_1', '')
    mle_query_2 = yaml['amptools'].get('mle_query_2', '')
    if mle_query_1 is None: mle_query_1 = ''
    if mle_query_2 is None: mle_query_2 = ''
    n_randomizations = yaml['amptools']['n_randomizations']
    accCorrect = yaml['acceptance_correct']
    
    if bpg > 1 and not yaml['amptools']['merge_grouped_trees']:
        # The error is that the checks in parse_fit_file will fail
        raise ValueError("Currently we must merge grouped trees if YAML key amptools.bins_per_group > 1. Please set amptools.merge_grouped_trees=True")
    
    df = loadAmpToolsResults(cfgfiles, masses, tPrimes, n_randomizations, mle_query_1, mle_query_2, accCorrect)
    unique_kin_pairs = df.value_counts(subset=['mass', 'tprime'])
    if ensure_one_fit_per_bin:
        if len(unique_kin_pairs) < len(df):
            raise ValueError(
                "Currently there is no support for plotting multiple fits per kinematic bin. "
                "Tighten your mle_queries. Plotting the best fit uses: mle_query_1: 'status==0 & ematrix==3' and mle_query_2: 'delta_nll==0'"
            )
        elif len(unique_kin_pairs) > len(df):
            raise ValueError(
                "Your mle_query_1 and mle_query_2 combination are too restrictive. If you were selecting fits with proper convergence codes:"
                "status=0 and ematrix=3, you can try mle_query_1: 'status==0' or mle_query_1: '' and mle_query_2: 'delta_nll==0' "
                "which would choose the best fit out of all fits that converges with any error matrix status or the entire set of fits."
            )
    return df, (masses, tPrimeBins, bpg)

def loadAmpToolsResults(cfgfiles, masses, tPrimes, niters, mle_query_1, mle_query_2, accCorrect):
    """
    Load results from AmpTools

    Args:
        cfgfiles (List): list of config files to load
        masses (List/Array): list/array of masses used in the fit (does not have to match cfgfiles)
        tPrimes (List/Array): list/array of t values used in the fit (does not have to match cfgfiles)
        niters (int): number of iterations used in the fit
        mle_query_1 (str): query to apply to the DataFrame BEFORE calculating delta_nll
        mle_query_2 (str): query to apply to the DataFrame AFTER  calculating delta_nll
        accCorrect (bool): whether to use acceptance corrected signal values

    Returns:
        DataFrame: DataFrame of results
    """

    lor = 0 if accCorrect else 1 # Left or right of | symbol in the output of extract_ff. Gets corrected values or not

    df = {}

    _nlls = []
    _masses = []
    _tPrimes = []
    _iterations = []
    _statuses = []
    _ematrix = []
    
    for cfgfile, i in itertools.product(cfgfiles, range(niters)):

        basedir = os.path.dirname(cfgfile)
        binTag = basedir.split("/")[-1]
        binTag = "bin_" + "_".join(binTag.split("_")[1:])
        fit_file = f"{basedir}/{binTag}_{i}.fit"

        if not os.path.exists(fit_file):
            # print(f"{fit_file} expected, but not found!")
            continue

        amps, status_dict = parse_fit_file(fit_file)
        
        for key, value in amps.items():
            key = f"{key}_amp"
            if key in df:
                df[key].append(value)
            else:
                df[key] = [value]
                
        value = status_dict["bestMinimum"]
        status = status_dict["lastMinuitCommandStatus"]
        ematrix = status_dict["eMatrixStatus"]
    
        _nlls.append(value)
        binNum = int(binTag.split("_")[-1])
        _masses.append(masses[binNum % len(masses)])
        _tPrimes.append(tPrimes[binNum // len(masses)])
        _iterations.append(i)
        _statuses.append(status)
        _ematrix.append(ematrix)

        observed_pairs = set()
        fname = f"{basedir}/intensities_{i}.txt"
        with open(fname) as f:
            totalYield = None
            for line in f.readlines():
                if line.startswith("TOTAL EVENTS"):
                    # Store both intensity and acceptance corrected intensity
                    intensity_corr, intensity = line.split()[3].split("|")
                    intensity_corr, intensity = float(intensity_corr), float(intensity)
                    if 'intensity' not in df: df['intensity'] = [intensity]
                    else: df['intensity'].append(intensity)
                    if 'intensity_corr' not in df: df['intensity_corr'] = [intensity_corr]
                    else: df['intensity_corr'].append(intensity_corr)
                    
                    totalYield = intensity_corr if accCorrect else intensity

                if line.startswith("FIT FRACTION") and "::" not in line:  # Regex-merged to group Re/Im + Pols
                    
                    if totalYield is None:
                        raise ValueError(f"io| totalYield is not expected to be None by now! File in question: {fname}")
                    
                    amp = line.split()[2]

                    # FILL VALUES
                    if amp in df:
                        df[amp].append(float(line.split()[4].split("|")[lor]) * totalYield)
                        df[f"{amp} err"].append(float(line.split()[6].split("|")[lor]) * totalYield)
                    else:
                        df[amp] = [float(line.split()[4].split("|")[lor]) * totalYield]
                        df[f"{amp} err"] = [float(line.split()[6].split("|")[lor]) * totalYield]

                if line.startswith("PHASE DIFFERENCE"):
                    amp1 = line.split()[2].split("::")[-1]
                    amp2 = line.split()[3].split("::")[-1]

                    phaseDiff = float(line.split()[5])
                    phaseDiff_err = float(line.split()[7])

                    pair = amp1 + " " + amp2

                    if pair in df:
                        # Each file should have a unique phase diff between pairs of amps
                        #  We can check for this by checking the length of the list.
                        #  We need this check since In FitResults output file, different
                        #  polarizations and real/imag parts shares values and the phase
                        #  calculations are repeated
                        if pair not in observed_pairs:
                            df[pair].append(phaseDiff)
                            df[f"{pair} err"].append(phaseDiff_err)
                            observed_pairs.add(pair)
                    else:
                        assert i == 0  # sanity check we are on first file; create initial list
                        df[pair] = [phaseDiff]
                        df[f"{pair} err"] = [phaseDiff_err]

                    observed_pairs.add(pair)

    df["nll"] = _nlls
    df["mass"] = _masses
    df["tprime"] = _tPrimes
    df["iteration"] = _iterations
    df["status"] = _statuses
    df["ematrix"] = _ematrix

    df = pd.DataFrame(df)
        
    # reorder columns so important ones are first
    cols = ["tprime", "mass", "nll", "iteration", "status", "ematrix"]
    cols.extend([k for k in df.keys() if k not in cols])
    df = df[cols]

    # Apply Query 1
    if mle_query_1 != "":
        df = df.query(mle_query_1)
    console.print(f"io| Remaining number of AmpTools fits (across randomizations and kinematic bins) after applying mle_query_1 = '{mle_query_1}': {len(df)}")
    console.print(f"io| Calculating delta_nll...")

    # groupby mass and subtract the min nll in each kinematic bin
    # create a new column called delta_nll and subtract the min nll in each kinematic bin
    df["delta_nll"] = df.groupby(["tprime","mass"])["nll"].transform(lambda x: x - x.min())

    # Apply Query 2
    if mle_query_2 != "":
        df = df.query(mle_query_2)
    console.print(f"io| Remaining number of AmpTools fits (across randomizations and kinematic bins) after applying mle_query_2 = '{mle_query_2}': {len(df)}")

    # This is the case if there are 0 fit result files loaded
    if len(df) == 0:
        # print("No amptools MLE fit results loaded! Returning empty DataFrame...")
        return df

    avail_t = list(df["tprime"].unique())
    avail_m = list(df["mass"].unique())
    avail_tm = list(itertools.product(avail_m, avail_t))
    needed_tm = list(itertools.product(masses, tPrimes))
    missing_tm = np.array(list(set(needed_tm) - set(avail_tm)))

    # match missing_tm to appropriate bins
    missing_tm_idxs = []
    absolute_idxs = []
    for tm in missing_tm:
        m, t = tm
        m_idx = np.where(np.abs(masses - m) < 1e-5)[0][0]
        t_idx = np.where(np.abs(tPrimes - t) < 1e-5)[0][0]
        idx = m_idx + t_idx * len(masses)
        missing_tm_idxs.append((m_idx, t_idx))
        absolute_idxs.append(idx)
    missing_tm_idxs = np.array(missing_tm_idxs)
    absolute_idxs = np.array(absolute_idxs)
    absolute_idxs = np.sort(absolute_idxs)

    if len(missing_tm) > 0:
        console.print("\nio| loadAmpToolsResults did not collect the correct number of masses/tPrimes, perhaps mle_queries are too strict?")
        console.print(f"io|  --> Missing bin indices: {absolute_idxs}")
        raise ValueError("Some kinematic bins had no converged MLE fits. This could be due to the YAML `mle_query_1` and `mle_query_2` could be too restrictive. Exiting...")

    return df
