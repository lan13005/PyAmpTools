import os
import random
from pyamptools import atiSetup
import array
import ROOT
import argparse
from pyamptools.utility.general import zlm_amp_name, vps_amp_name, converter, example_zlm_names, example_vps_names, load_yaml

help_header = """#####################################
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
#####################################\n\n
"""

amptools_zlm_ampName = "Zlm"
amptools_vps_ampName = "Vec_ps_refl"


def generate_amptools_cfg(
    quantum_numbers,
    angles,
    fractions,
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
    datareader="ROOTDataReader",
    add_amp_factor=None,
    append_to_cfg=None,
    append_to_decay=None,
):
    """
    Generate an AmpTools configuration file for a Zlm fit

    Args:
        quantum_numbers (list): List of lists of quantum numbers. For Zlm then [Reflectivity, spin, spin-projection]
        angles (list): List of polarization angles in degrees
        fractions (list): List of polarization fractions
        datas (list): List of data files
        gens (list): List of gen files
        accs (list): List of acc files
        bkgnds (list): List of bkgnd files
        realAmps (list): List of amplitude names that are real
        fixedAmps (list): List of amplitude names that are fixed
        fitName (str): A FitResults (.fit) file will be created with this name prefixed
        cfgFileOutputName (str): Name of output configuration file
        basereactName (str): Base name of reaction
        particles (list): List of particles in reaction
        header (str): Header to append to the top of the file
        datareader (str): Data reader to use
        add_amp_factor (str): Additional factor to add to the amplitude
        append_to_cfg (str): Additional string args to append to the end of the cfg file
        append_to_decay (str): Additional string args to append to the decay factor

    Returns:
        None, writes a file to cfgFileOutputName
    """

    ############## SET ENVIRONMENT VARIABLES ##############
    atiSetup.setup(globals())
    #######################################################

    cfgInfo = ConfigurationInfo(fitName)

    constraintMap = {}
    refTagMap = {1: "Pos", -1: "Neg"}  # Positive / Negative
    conjugations = {"Re": "+1", "Im": "-1"}  # Real / Imaginary

    for i in range(len(angles)):
        ####################################
        #### Create ReactionInfo object ####
        ####################################

        angle = angles[i]
        fraction = fractions[i]

        reactName = f"{basereactName}_{angle:0>3}"
        scaleName = f"parScale{angle:0>3}"
        parScale = cfgInfo.createParameter(scaleName, 1.0)
        if angle == "0":
            parScale.setFixed(True)

        reactionInfo = cfgInfo.createReaction(reactName, particles)  # ReactionInfo*

        ###################################
        #### SET DATA, GEN, ACC, BKGND ####
        ###################################

        reader_parts = datareader.split(" ")
        reader_name = reader_parts[0]

        data = datas[i]
        data_args = [data] + (reader_parts[1:] if len(reader_parts) > 1 else [])  # append args
        reactionInfo.setData(reader_name, data_args)
        gen = gens[i]
        gen_args = [gen] + (reader_parts[1:] if len(reader_parts) > 1 else [])
        reactionInfo.setGenMC(reader_name, gen_args)
        acc = accs[i]
        acc_args = [acc] + (reader_parts[1:] if len(reader_parts) > 1 else [])
        reactionInfo.setAccMC(reader_name, acc_args)

        if len(bkgnds) > 0:
            bkgnd = bkgnds[i]
            bkgnd_args = [bkgnd] + (reader_parts[1:] if len(reader_parts) > 1 else [])
            reactionInfo.setBkgnd(reader_name, bkgnd_args)

        #############################################
        #### DEFINE COHERENT SUMS AND AMPLITUDES ####
        #############################################

        for conj in conjugations.items():
            conjTag, conjVal = conj

            for quantum_number in quantum_numbers:
                ref, L, M = quantum_number[:3]
                J = quantum_number[3] if len(quantum_number) > 3 else None
                assume_zlm = J is None

                if float(fraction) == 1:  # Some terms disappear if the fraction is 1
                    if (ref == 1 and conjTag == "Im") or (ref == -1 and conjTag == "Re"):
                        continue

                refTag = refTagMap[ref]

                sumName = f"{refTag}{conjTag}"
                cfgInfo.createCoherentSum(reactName, sumName)  # returns CoherentSumInfo*

                ampName = zlm_amp_name(ref, L, M) if assume_zlm else vps_amp_name(ref, J, M, L)
                ampInfo = cfgInfo.createAmplitude(reactName, sumName, ampName)  # AmplitudeInfo*

                part = "+1" if int(ref) * int(conjVal) > 0 else "-1"
                if assume_zlm:
                    angularFactor = [f"{amptools_zlm_ampName}", f"{L}", f"{M}", conjVal, part, angle, fraction]
                else:
                    # Vec_ps_refl 1 -1 0 -1  -1  LOOPPOLANG LOOPPOLVAL omega3pi
                    angularFactor = [f"{amptools_vps_ampName}", f"{J}", f"{M}", f"{L}", conjVal, part, angle, fraction]
                if append_to_decay:
                    angularFactor.extend(append_to_decay.split())
                ampInfo.addFactor(angularFactor)
                if add_amp_factor:
                    ampInfo.addFactor(add_amp_factor.split(" "))
                ampInfo.setScale(f"[{scaleName}]")

                if ampName not in constraintMap:
                    constraintMap[ampName] = [ampInfo]
                else:
                    constraintMap[ampName].append(ampInfo)

    #####################################
    ### RANDOMLY INITALIZE AMPLITUDES ###
    #####################################

    for amp, lines in constraintMap.items():
        value = random.uniform(0.0, 1.0)
        if amp in realAmps:
            lines[0].setReal(True)
        else:
            value += random.uniform(0.0, 1.0) * 1j
        if amp in fixedAmps:
            lines[0].setFixed(True)
        for line in lines[1:]:
            lines[0].addConstraint(line)
        lines[0].setValue(value)

    #########################
    ### WRITE CONFIG FILE ###
    #########################

    cfgInfo.display()
    cfgInfo.write(cfgFileOutputName)

    with open(cfgFileOutputName, "r") as original:
        data = original.read()

    # prepend help_header to top of the file
    data = header + data
    with open(cfgFileOutputName, "w") as modified:
        modified.write(data)
    if append_to_cfg:
        data = data + append_to_cfg
        with open(cfgFileOutputName, "w") as modified:
            modified.write(data)

    return data


def generate_amptools_cfg_from_dict(yaml_file):
    #####################################################
    ################ GENERAL SPECIFICATION ##############
    #####################################################

    generate_success = False
    perform_config_checks = True

    fitName = "PLACEHOLDER_FITNAME"
    cfgFileOutputName = f'{yaml_file["base_directory"]}/amptools.cfg'
    basereactName = "reaction"
    data_folder = yaml_file["data_folder"]
    particles = yaml_file["reaction"].split(" ")
    print("Particles in reaction: ", particles)

    cfgFileOutputFolder = os.path.dirname(cfgFileOutputName)
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
        print("Using waveset string: ", waveset)
        # Using waveset string:  Sp0+_Dp2+
        for wave in waveset:
            if wave in converter:
                used_quantum_numbers.append(converter[wave])
            else:
                check_string = f"Wave {wave} not recognized. Please check your waveset string and try again."
                check_string += f"\n\nExample partial two pseduoscalar wave names: {example_zlm_names[:10]}"
                check_string += f"\n\nExample partial vector pseudoscalar wave names: {example_vps_names[:10]}"
                return check_string, generate_success

    if yaml_file["real_waves"] == "" or yaml_file["fixed_waves"] is None:
        realAmps = yaml_file["real_waves"].split("_")
        print(f"Using real waves: {realAmps}")

    if yaml_file["fixed_waves"] == "" or yaml_file["fixed_waves"] is None:
        fixedAmps = yaml_file["fixed_waves"].split("_")
        print(f"Using fixed waves: {fixedAmps}")

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
    pols = yaml_file["polarizations"]
    for angle, mag in pols.items():
        used_pols.append(angle)
        used_polMags.append(f"{mag}")
        check_map[angle] = [mag >= 0 and mag <= 1, f"Polarization magnitude must be between 0 and 1 instead of {mag}"]

    #####################################################
    ################# CONSTRUCT DATASETS ################
    #####################################################

    # These are the actual locations that we can still check for
    _datas = [f"{data_folder}/data{pol:0>3}.root" for pol in used_pols]
    _gens = [f"{data_folder}/genmc{pol:0>3}.root" for pol in used_pols]
    _accs = [f"{data_folder}/accmc{pol:0>3}.root" for pol in used_pols]
    _bkgnds = [f"{data_folder}/bkgnd{pol:0>3}.root" for pol in used_pols]

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
        print("No background file found. AmpTools will assume data file contains pure signal")
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
            add_amp_factor=yaml_file["add_amp_factor"].strip(),
            append_to_cfg=yaml_file["append_to_cfg"].strip(),
            append_to_decay=yaml_file["append_to_decay"].strip(),
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
    args = parser.parse_args()
    yaml_name = args.yaml_name

    cwd = os.getcwd()

    print("\n---------------------")
    print(f"Running {__file__}")
    print(f"  yaml location: {yaml_name}")
    print("---------------------\n")

    yaml_file = load_yaml(yaml_name)

    result, generate_success = generate_amptools_cfg_from_dict(yaml_file)

    if generate_success:
        with open(f"{yaml_file['base_directory']}/amptools.cfg", "w") as f:
            f.write(result)
        print("\nConfiguration file successfully generated")
    else:
        print(result)
        print("\nConfiguration file generation failed!")
