import pickle as pkl

import numpy as np
import pandas as pd
from iftpwa1.utilities.helpers import load_callable_from_module, reload_fields_and_components
from omegaconf import OmegaConf
import os

spect_map = {"S": 0, "L": 1, "D": 2 }
sign_map = {"p": "", "m": "-"}

# +
bkg_amps_waves_tprime = {}
res_amps_waves_tprime = {}
amp_field_tprime = []

ift_res_dfs = []

PYAMPTOOLS_HOME = os.getenv("PYAMPTOOLS_HOME")
subdir = f"{PYAMPTOOLS_HOME}/demos/ift_example"

resultFile = f"{subdir}/NiftyFits/prior_sim/niftypwa_fit.pkl"
yamlFile = f"{subdir}/pyamptools.yaml"
react_name = "reaction_000"

yaml_conf = OmegaConf.load(yamlFile)
min_mass =  yaml_conf['min_mass']
max_mass = yaml_conf['max_mass']
nmbMasses = yaml_conf['n_mass_bins']
polMag = float(yaml_conf['polarizations']['000'])
polTags = f"0 {polMag}"

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

##################################################################################################
# THIS SECTION CONTAINS BASICALLY ALL YOU NEED TO START CREATING CUSTOM PLOTS
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
##################################################################################################

for k in _bkg_amps_waves_tprime.keys():
    short_k = k.split("::")[-1]
    if short_k not in bkg_amps_waves_tprime:
        bkg_amps_waves_tprime[short_k] = []
        res_amps_waves_tprime[short_k] = []
    bkg_amps_waves_tprime[short_k].append(_bkg_amps_waves_tprime[k][0])
    res_amps_waves_tprime[short_k].append(_res_amps_waves_tprime[k][0])

amp_field_tprime.append(amp_field)

amp_field_tprime = np.concatenate(amp_field_tprime, axis=-1)

amp_field_tprime = amp_field_tprime[0, :, :, 0] # only 1 sample and 1 tbin

wave_names = resultData["pwa_manager_base_information"]["wave_names"]

cfg = """
#####################################
####	THIS IS A CONFIG FILE	 ####
#####################################
##
##  Blank lines or lines beginning with a "#" are ignored.
##
##  Double colons (::) are treated like a space.
##     This is sometimes useful for grouping (for example,
##     grouping strings like "reaction::sum::amplitudeName")
##
##  All non-comment lines must begin with one of the following keywords.
##
##  (note:  <word> means necessary
##	    (word) means optional)
##
##  include	  <file>
##  define	  <word> (defn1) (defn2) (defn3) ...
##  fit 	  <fitname>
##  keyword	  <keyword> <min arguments> <max arguments>
##  reaction	  <reaction> <particle1> <particle2> (particle3) ...
##  data	  <reaction> <class> (arg1) (arg2) (arg3) ...
##  genmc	  <reaction> <class> (arg1) (arg2) (arg3) ...
##  accmc	  <reaction> <class> (arg1) (arg2) (arg3) ...
##  normintfile   <reaction> <file>
##  sum 	  <reaction> <sum> (sum2) (sum3) ...
##  amplitude	  <reaction> <sum> <amp> <class> (arg1) (arg2) ([par]) ...
##  initialize    <reaction> <sum> <amp> <"events"/"polar"/"cartesian">
##		    <value1> <value2> ("fixed"/"real")
##  scale	  <reaction> <sum> <amp> <value or [parameter]>
##  constrain	  <reaction1> <sum1> <amp1> <reaction2> <sum2> <amp2> ...
##  permute	  <reaction> <sum> <amp> <index1> <index2> ...
##  parameter	  <par> <value> ("fixed"/"bounded"/"gaussian")
##		    (lower/central) (upper/error)
##    DEPRECATED:
##  datafile	  <reaction> <file> (file2) (file3) ...
##  genmcfile	  <reaction> <file> (file2) (file3) ...
##  accmcfile	  <reaction> <file> (file2) (file3) ...
##
#####################################

### FIT CONFIGURATION ###
"""

cfg += f"\nreaction {react_name} Beam Proton Pi0 Eta"
cfg += f"\nsum {react_name} PosRe"

if polMag < 1:
    cfg += "\nsum reaction_000::PosIm"

amp_description = ""
for i, amp_name in enumerate(wave_names):
    
    try:
        L, s, m, e = amp_name
    except:
        raise ValueError(f"Could not parse amp_name: '{amp_name}'. Currently we only support Zlm amplitude naming scheme.\n")

    amps = amp_field_tprime[i, :]

    amp_parTag = ""
    for i, amp in enumerate(amps):
        amp_parTag += f" {amps[i].real:0.5f}"
        amp_parTag += f" {amps[i].imag:0.5f}"
    amp_parTag = amp_parTag.lstrip()
        
    # Angular part
    amp_description += f"\namplitude {react_name}::PosRe::{amp_name} Zlm {spect_map[L]} {sign_map[s]}{m} +1 +1 {polTags}"
    if polMag < 1:
        amp_description += f"\namplitude {react_name}::PosIm::{amp_name} Zlm {spect_map[L]} {sign_map[s]}{m} -1 -1 {polTags}"

    # Piecewise part
    refl_tag = "Pos" if e=="+" else "Neg"
    amp_description += f"\namplitude {react_name}::PosRe::{amp_name} Piecewise {min_mass} {max_mass} {nmbMasses} 23 {refl_tag} ReIm {amp_parTag}"
    if polMag < 1:
        amp_description += f"\namplitude {react_name}::PosIm::{amp_name} Piecewise {min_mass} {max_mass} {nmbMasses} 23 {refl_tag} ReIm {amp_parTag}"

    # Constraints
    if polMag < 1:
        amp_description += f"\nconstrain {react_name}::PosRe::{amp_name} {react_name}::PosIm::{amp_name}"

    # Initialize
    amp_description += f"\ninitialize {react_name}::PosRe::{amp_name} cartesian 1.0 0.0"

    amp_description += "\n"

cfg += amp_description

with open(f"{subdir}/prior_sim_gen.cfg", "w") as f:
    f.write(cfg)

