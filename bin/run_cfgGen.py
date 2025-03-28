import argparse
import array
import os

import ROOT
from pyamptools.utility.general import converter, example_vps_names, example_zlm_names, load_yaml
from pyamptools.utility.cfg_gen_utils import generate_amptools_cfg, help_header

############################################################################
# This script generates AmpTools configuration files with knobs/flags to
# append additional information to the generated file
############################################################################

from rich.console import Console
console = Console()

def generate_amptools_cfg_from_dict(yaml_file, output_location):
    #####################################################
    ################ GENERAL SPECIFICATION ##############
    #####################################################

    generate_success = False
    perform_config_checks = True

    fitName = "PLACEHOLDER_FITNAME"
    cfgFileOutputName = output_location
    basereactName = "reaction"
    data_folder = yaml_file["data_folder"]
    particles = yaml_file["reaction"].split(" ")
    console.print(f"Particles in reaction: {particles}", style="bold blue")

    cfgFileOutputFolder = os.path.dirname(cfgFileOutputName) if "/" in cfgFileOutputName else "."
    os.system(f"mkdir -p {cfgFileOutputFolder}")

    check_map = {}
    check_map["data_folder"] = [os.path.isabs(data_folder), f"Data folder, {data_folder}, is not an absolute path"]

    #####################################################
    ################# CONSTRUCT WAVESET #################
    #####################################################

    used_quantum_numbers = []

    # If user supplies waveset string we will parse and use that
    #   otherwise we will check if buttons have been clicked.
    if yaml_file["waveset"] == "":
        check_string = "You did not select any partial waves... \n Please enter a waveset string and try again."
    else:
        waveset = yaml_file["waveset"].split("_")
        # Using waveset string:  Sp0+_Dp2+
        for wave in waveset:
            if wave == "isotropic": continue
            elif wave in converter:
                used_quantum_numbers.append(converter[wave])
            else:
                check_string = f"Wave {wave} not recognized. Please check your waveset string and try again."
                check_string += f"\n\nExample partial two pseduoscalar wave names: {example_zlm_names[:10]}"
                check_string += f"\n\nExample partial vector pseudoscalar wave names: {example_vps_names[:10]}"
                return check_string, generate_success

    if yaml_file["real_waves"] == "" or yaml_file["fixed_waves"] is None:
        realAmps = yaml_file["real_waves"].split("_")
        console.print(f"Using real waves: {realAmps}", style="bold blue")

    if yaml_file["fixed_waves"] == "" or yaml_file["fixed_waves"] is None:
        fixedAmps = yaml_file["fixed_waves"].split("_")
        console.print(f"Using fixed waves: {fixedAmps}", style="bold blue")

    if len(used_quantum_numbers) == 0:
        check_string = "You did not select any partial waves... \n Please go make some selections or enter a waveset string and try again."
        check_string += f'\nYour waveset string: {yaml_file["waveset"]}'
        check_string += f"\n\nExample partial wave names: {example_zlm_names}"
        return check_string, generate_success

    #####################################################
    ############### CONSTRUCT POLARIZATIONS #############
    #####################################################

    ## Polarization related, reactNames are scaled by the scales parameters
    ##   Default: parScale0 is fixed to 1.0

    used_pols = []
    used_polMags = []
    used_polFixedScales = []
    pols = yaml_file["polarizations"] # pols can be a Dict of float values or a Dict of Dicts(storing polarization magnitude and boolean for fixed scale factor)
    for i, (polAngle, mag) in enumerate(pols.items()):
        used_pols.append(polAngle)
        if isinstance(mag, float):
            used_polMags.append(f"{mag}")
            used_polFixedScales.append(True if i == 0 else False) # scaling of first polarized data is fixed, all others have freely floating scale factors
            check_map[polAngle] = [mag >= 0 and mag <= 1, f"Polarization magnitude must be between 0 and 1 instead of {mag}"]
        elif isinstance(mag, dict):
            if "pol_mag" not in mag: raise ValueError(f"Missing 'pol_mag' key in polarization dictionary for {polAngle}. Please update YAML file `polarizations` key.")
            if "fixed_scale" not in mag: raise ValueError(f"Missing 'fixed_scale' key in polarization dictionary for {polAngle}. Please update YAML file `polarizations` key.")
            used_polMags.append(f"{mag['pol_mag']}")
            used_polFixedScales.append(bool(mag['fixed_scale']))
            check_map[polAngle] = [mag['pol_mag'] >= 0 and mag['pol_mag'] <= 1, f"Polarization magnitude must be between 0 and 1 instead of {mag['pol_mag']}"]

    #####################################################
    ################# CONSTRUCT DATASETS ################
    #####################################################

    # These are the actual locations that we can still check for
    _datas = [f"{data_folder}/data{pol:0>3}.root" for pol in used_pols]
    _bkgnds = [f"{data_folder}/bkgnd{pol:0>3}.root" for pol in used_pols]
    _gens = [f"{data_folder}/genmc{pol:0>3}.root" for pol in used_pols]
    _accs = [f"{data_folder}/accmc{pol:0>3}.root" for pol in used_pols]

    # Check if these files exist
    for ftype, sources in zip(["data", "bkgnd", "genmc", "accmc"], [_datas, _bkgnds, _gens, _accs]):
        for source in sources:
            if not os.path.isfile(source) and ftype in ["genmc", "accmc"]:
                if os.path.isfile(f"{data_folder}/{ftype}.root"):
                    if ftype == "genmc":
                        _gens = [f"{data_folder}/{ftype}.root" for pol in used_pols]
                    if ftype == "accmc":
                        _accs = [f"{data_folder}/{ftype}.root" for pol in used_pols]
                else:
                    raise FileNotFoundError(f"File {source} does not exist.")
            if not os.path.isfile(source) and ftype in ["data"]:
                raise FileNotFoundError(f"File {source} does not exist.")

    # These are placeholder locations for us to split_mass with
    datas = [f"PLACEHOLDER_DATA_{pol:0>3}" for pol in used_pols]
    gens = [f"PLACEHOLDER_GENMC_{pol:0>3}" for pol in used_pols]
    accs = [f"PLACEHOLDER_ACCMC_{pol:0>3}" for pol in used_pols]
    bkgnds = [f"PLACEHOLDER_BKGND_{pol:0>3}" for pol in used_pols]

    #####################################################
    ################# GENERATE CONFIG FILE ##############
    #####################################################

    fs_check = "Number of final state particles in {}\n Actual: {}\n Expected from Reaction: {}"
    for data, gen, acc, bkgnd in zip(_datas, _gens, _accs, _bkgnds):
        bkgnd_exists = False
        for file, _file in zip([data, gen, acc, bkgnd], ["data", "gen", "acc", "bkgnd"]):
            if _file == "bkgnd":
                bkgnd_exists = os.path.isfile(file)
            else:  # bkgnd is optional so we will not check for its existence
                check_map[file] = [os.path.isfile(file), f"File {file} does not exist."]

            if _file in ["data", "gen", "acc"] or (bkgnd_exists and _file == "bkgnd"):
                _file = ROOT.TFile.Open(file)
                _tree = _file.Get("kin")
                _branch = _tree.GetBranch("NumFinalState")
                _value = array.array("i", [0])
                _branch.SetAddress(_value)
                _tree.GetEntry(0)
                actual_num_fs_parts = _value[0]
                expect_num_fs_parts = len(particles) - 1  # ignore Beam, care only about recoil + daughters
                if "num_final_state" not in check_map:
                    check_map["num_final_state"] = [actual_num_fs_parts == expect_num_fs_parts, fs_check.format(file, actual_num_fs_parts, expect_num_fs_parts)]
                else:
                    check_map["num_final_state"][1] += "\n" + fs_check.format(file, actual_num_fs_parts, expect_num_fs_parts)

    if not bkgnd_exists:
        console.print("No background file found. AmpTools will assume data file contains pure signal", style="bold yellow")
        bkgnds = []

    result = ""
    any_check_failed = any([not check[0] for check in check_map.values()])
    if perform_config_checks and any_check_failed:
        result += "Config checks failed:\n\n"
        for check in check_map.values():
            if not check[0]:
                result += check[1] + "\n"
    else:
        result = generate_amptools_cfg(
            used_quantum_numbers,
            used_pols,
            used_polMags,
            used_polFixedScales,
            datas,
            gens,
            accs,
            bkgnds,
            realAmps,
            fixedAmps,
            fitName,
            cfgFileOutputName,
            basereactName,
            particles,
            header=help_header,
            datareader=yaml_file["datareader"],
            add_amp_factor=yaml_file.get("add_amp_factor", "").strip(),
            append_to_cfg=yaml_file.get("append_to_cfg", "").strip(),
            append_to_decay=yaml_file.get("append_to_decay", "").strip(),
            initialization=yaml_file.get("initialization", None),
        )

        generate_success = True

        #### FINAL CHECK ####
        valid_keys = ["data", "bkgnd", "accmc", "genmc", "include", "define", "fit", "keyword", "reaction", "normintfile", "sum", "amplitude", "initialize", "scale", "constrain", "permute", "parameter"]
        final_check_string = ""
        for line in result.split("\n"):
            line = line.strip()
            if not line.startswith("#") and line != "":  # if not a comment nor empty line
                if line.split(" ")[0] not in valid_keys:
                    final_check_string += f"Invalid keyword found in the config file: {line}\n"
                    generate_success = False
        if final_check_string != "":
            result = final_check_string

    return result, generate_success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an AmpTools configuration file for a Zlm fit")
    parser.add_argument("yaml_name", type=str, default="conf/configuration.yaml", help="Path a configuration yaml file")
    parser.add_argument("-o", "--output_location", type=str, default="", help="Path to the output configuration file")
    args = parser.parse_args()
    yaml_name = args.yaml_name
    output_location = args.output_location
    cwd = os.getcwd()

    console.rule()
    console.print(f"Running {__file__}", style="bold blue")
    console.print(f"  yaml location: {yaml_name}", style="bold blue")
    console.rule()

    yaml_file = load_yaml(yaml_name)
    
    output_location = f"{yaml_file['base_directory']}/amptools.cfg" if output_location == "" else output_location

    result, generate_success = generate_amptools_cfg_from_dict(yaml_file, output_location)

    if generate_success:
        with open(output_location, "w") as f:
            f.write(result)
        console.print(f"\nConfiguration file successfully generated at: {output_location}", style="bold green")
    else:
        console.print(result, style="bold red")
        console.print("\nConfiguration file generation failed!", style="bold red")
